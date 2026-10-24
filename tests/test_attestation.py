"""Attestation verification: mandatory DCAP binding + fail-closed gating.

These tests exercise the *real* mock signing identity, quote layout and
measurement policy, faking only the three offline boundaries (DCAP, the public
measurement endpoint and NRAS, all provided by ``TeeFakes`` in ``conftest.py``).
GPU verification therefore always runs, against a fake NVIDIA SDK, so no
hardware or network is required.

The production verifier **raises** on every failure: there is no
``verified=False`` success path and no ``verify_quote``/``verify_gpu`` toggle.
"""

import asyncio
import base64
import sys
import threading
import time
import types

import pytest
from pqcrypto.kem.ml_kem_768 import generate_keypair

from chutes_litellm import attestation as att
from chutes_litellm import endpoints
from chutes_litellm.tee import EvidenceError, MeasurementMismatchError
from chutes_litellm.tee.measurements import TeeMeasurement
from chutes_litellm.tee.verifier import _verify_nvidia_gpu
from conftest import ensure_installed
from mock_chutes_server import (
    CHUTE_ID,
    INSTANCE_ID,
    MODEL_ID,
    MockState,
    build_evidence,
    build_evidence_for,
    build_quote,
)


@pytest.fixture(scope="module")
def state() -> MockState:
    """A standalone mock state (signing identity + fixed measurements)."""
    return MockState(instance_pk=b"mock-instance-public-key", instance_sk=b"")


def _valid_evidence(state: MockState, nonce: str, **kwargs) -> dict:
    pubkey = state.pubkey_b64
    return build_evidence(state, att._digest_of(nonce, pubkey), nonce, **kwargs)


# ---------------------------------------------------------------------------
# Positive path + exact check names
# ---------------------------------------------------------------------------


def test_verify_evidence_valid(state):
    nonce = "ab" * 32
    pubkey = state.pubkey_b64
    evidence = _valid_evidence(state, nonce)

    result = att.verify_evidence(evidence, pubkey, nonce=nonce)

    assert result.verified is True
    assert result.instance_id == state.instance_id
    assert result.nonce == nonce
    assert result.digest == att._digest_of(nonce, pubkey)
    assert result.dcap_status == "UpToDate"
    assert result.gpu is not None
    assert result.gpu["gpus"] == 1
    assert result.checks == {
        "signature": True,
        "quote_report_data": True,
        "certificate_binding": True,
        "measurement": True,
        "dcap_quote": True,
        "nv_gpu": True,
    }


def test_verify_evidence_calls_nras(state, tee_fakes):
    """GPU verification is unconditional: the evidence always reaches NRAS."""
    nonce = "ad" * 32
    evidence = _valid_evidence(state, nonce)

    att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)

    assert len(tee_fakes.nras_calls) == 1
    assert tee_fakes.nras_calls[0] == [
        {
            "arch": "HOPPER",
            "evidence": "ZXZpZGVuY2U=",
            "certificate": "Y2VydGlmaWNhdGU=",
        }
    ]


def test_verify_evidence_with_gpu_adds_nv_gpu_check(state, monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    nonce = "ac" * 32
    evidence = _valid_evidence(state, nonce)

    result = att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)

    assert result.checks["nv_gpu"] is True
    assert result.gpu is not None
    assert result.gpu["gpus"] == 1
    assert result.gpu["expected"] == 1
    assert fake.calls["nonce"] == result.digest


# ---------------------------------------------------------------------------
# Digest binding: wrong key, legacy/raw-nonce, zero digest
# ---------------------------------------------------------------------------


def test_verify_evidence_rejects_wrong_pubkey(state):
    """Evidence built for another key must not verify against the real one."""
    nonce = "cd" * 32
    foreign_pk, _ = generate_keypair()
    foreign_b64 = base64.b64encode(foreign_pk).decode()

    evidence = build_evidence(state, att._digest_of(nonce, foreign_b64), nonce)
    with pytest.raises(att.AttestationError, match=r"report_data\[0:32\]"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_verify_evidence_rejects_legacy_zero_digest(state):
    nonce = "ef" * 32
    evidence = build_evidence(state, "0" * 64, nonce)
    with pytest.raises(att.AttestationError, match=r"report_data\[0:32\].*legacy|legacy.*report_data"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_verify_evidence_rejects_certificate_binding_mismatch(state):
    """A valid signed body whose quote does not bind the signing cert fails."""
    nonce = "a1" * 32
    evidence = _valid_evidence(state, nonce, cert_bind=False)
    with pytest.raises(att.AttestationError, match=r"report_data\[32:64\]"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


# ---------------------------------------------------------------------------
# Signed-body requirements (no unsigned / partial evidence is accepted)
# ---------------------------------------------------------------------------


def test_verify_evidence_rejects_bad_signature(state):
    nonce = "b1" * 32
    evidence = _valid_evidence(state, nonce)
    evidence["signature"] = base64.b64encode(b"not a real signature").decode()
    with pytest.raises(EvidenceError, match="signature"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("signature", "no 'signature'"),
        ("certificate", "no 'certificate'"),
        ("attested_body", "no 'attested_body'"),
    ],
)
def test_verify_evidence_requires_signed_envelope(state, field, match):
    nonce = "b2" * 32
    evidence = _valid_evidence(state, nonce)
    evidence.pop(field)
    with pytest.raises(EvidenceError, match=match):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_verify_evidence_requires_instance_id(state):
    nonce = "b3" * 32
    evidence = _valid_evidence(state, nonce)
    evidence.pop("instance_id")
    with pytest.raises(EvidenceError, match="no instance_id"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_verify_evidence_rejects_top_level_quote_mismatch(state):
    nonce = "b4" * 32
    evidence = _valid_evidence(state, nonce)
    evidence["quote"] = base64.b64encode(build_quote(state, "ab" * 32)).decode()
    with pytest.raises(EvidenceError, match="top-level quote differs"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_verify_evidence_rejects_top_level_gpu_mismatch(state):
    nonce = "b5" * 32
    evidence = _valid_evidence(state, nonce)
    evidence["gpu_evidence"] = [{"arch": "HOPPER", "certificate": "eA==", "evidence": "eQ=="}]
    with pytest.raises(EvidenceError, match="top-level GPU evidence differs"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


# ---------------------------------------------------------------------------
# DCAP TCB policy (mandatory)
# ---------------------------------------------------------------------------


def test_dcap_debug_bit_fails(state, tee_fakes):
    tee_fakes.dcap_debug = True
    nonce = "c1" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="debug=on"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_dcap_advisories_fail(state, tee_fakes):
    tee_fakes.dcap_advisories = ["INTEL-SA-00999"]
    nonce = "c2" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="advisories=INTEL-SA-00999"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_dcap_out_of_date_status_fails(state, tee_fakes):
    tee_fakes.dcap_status = "OutOfDate"
    nonce = "c3" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="OutOfDate"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_dcap_error_fails_closed(state, tee_fakes):
    tee_fakes.dcap_error = RuntimeError("no route to PCCS")
    nonce = "c4" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="collateral fetch failed"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


# ---------------------------------------------------------------------------
# Measurement policy (mandatory)
# ---------------------------------------------------------------------------


def test_measurement_mismatch_fails(state, tee_fakes):
    tee_fakes.mrtd = "00" * 48
    nonce = "d1" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(MeasurementMismatchError, match="does not match any published"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_missing_measurement_source_fails(state, tee_fakes):
    tee_fakes.measurements_error = att.AttestationError("measurement source down")
    nonce = "d2" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="measurement source down"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_ambiguous_measurements_fail_closed(state, tee_fakes):
    """Two matching entries that disagree on GPU expectations are ambiguous."""
    tee_fakes.measurements = [
        TeeMeasurement(
            version="v1",
            name="a",
            mrtd=tee_fakes.mrtd,
            rtmr0=tee_fakes.rtmr0,
            rtmr1=tee_fakes.rtmr1,
            rtmr2=tee_fakes.rtmr2,
            runtime_rtmr3=tee_fakes.rtmr3,
            expected_gpus=["hopper"],
            gpu_count=1,
        ),
        TeeMeasurement(
            version="v1",
            name="b",
            mrtd=tee_fakes.mrtd,
            rtmr0=tee_fakes.rtmr0,
            rtmr1=tee_fakes.rtmr1,
            rtmr2=tee_fakes.rtmr2,
            runtime_rtmr3=tee_fakes.rtmr3,
            expected_gpus=["blackwell"],
            gpu_count=8,
        ),
    ]
    nonce = "d3" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(MeasurementMismatchError, match="ambiguous"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


# ---------------------------------------------------------------------------
# NVIDIA GPU verification (fake SDK, no network)
# ---------------------------------------------------------------------------


class _FakeAttestation:
    def __init__(self):
        self.result = True
        self.calls: dict = {}

    def reset(self):
        self.calls = {}

    def add_verifier(self, *args, **kwargs):
        self.calls["add_verifier"] = args

    def set_nonce(self, nonce):
        self.calls["nonce"] = nonce

    def attest(self, evidence_list):
        self.calls["evidence"] = evidence_list
        return self.result


def _install_fake_nv_sdk(monkeypatch, fake: _FakeAttestation) -> None:
    mod = types.ModuleType("nv_attestation_sdk.attestation")
    mod.Attestation = lambda *a, **k: fake
    mod.Devices = types.SimpleNamespace(GPU="GPU")
    mod.Environment = types.SimpleNamespace(REMOTE="REMOTE")
    monkeypatch.setitem(sys.modules, "nv_attestation_sdk", types.ModuleType("nv_attestation_sdk"))
    monkeypatch.setitem(sys.modules, "nv_attestation_sdk.attestation", mod)


def test_nvidia_gpu_evidence_is_normalized_before_attest(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    raw = [
        # extra keys are stripped; duplicates are collapsed
        {"arch": "HOPPER", "certificate": "cert", "evidence": "ev", "gpu_uuid": "x"},
        {"arch": "HOPPER", "certificate": "cert", "evidence": "ev", "gpu_uuid": "x"},
    ]
    out = _verify_nvidia_gpu(raw, "ab" * 32, expected_count=1)
    assert fake.calls["nonce"] == "ab" * 32
    assert fake.calls["evidence"] == [{"arch": "HOPPER", "evidence": "ev", "certificate": "cert"}]
    assert out["gpus"] == 1 and out["arch"] == "HOPPER"
    # REMOTE mode must be configured with a non-empty NRAS URL.
    assert fake.calls["add_verifier"][2]


def test_nvidia_gpu_count_must_match_measurement(state, monkeypatch, tee_fakes):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    tee_fakes.gpu_count = 2
    nonce = "e1" * 32
    evidence = _valid_evidence(state, nonce)
    with pytest.raises(att.AttestationError, match="count does not match"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)


def test_nvidia_gpu_nras_failure_fails_closed(monkeypatch):
    fake = _FakeAttestation()
    fake.result = False
    _install_fake_nv_sdk(monkeypatch, fake)
    with pytest.raises(att.AttestationError, match="did not pass"):
        _verify_nvidia_gpu(
            [{"arch": "HOPPER", "certificate": "c", "evidence": "e"}],
            "00" * 32,
            expected_count=1,
        )


def test_nvidia_gpu_rejects_missing_evidence(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    with pytest.raises(att.AttestationError, match="no GPU data"):
        _verify_nvidia_gpu([], "00" * 32, expected_count=1)


def test_nvidia_gpu_rejects_unusable_evidence(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    # The old mock {nonce, gpu_uuid} shape carries no arch/certificate/evidence.
    with pytest.raises(att.AttestationError, match="no usable NVIDIA"):
        _verify_nvidia_gpu([{"nonce": "00" * 32, "gpu_uuid": "x"}], "00" * 32, expected_count=1)


def test_nvidia_gpu_rejects_mixed_arch(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    raw = [
        {"arch": "HOPPER", "certificate": "c1", "evidence": "e1"},
        {"arch": "BLACKWELL", "certificate": "c2", "evidence": "e2"},
    ]
    with pytest.raises(att.AttestationError, match="mixes architectures"):
        _verify_nvidia_gpu(raw, "00" * 32, expected_count=2)


# ---------------------------------------------------------------------------
# End-to-end gate against the mock server
# ---------------------------------------------------------------------------


def test_verify_chute_end_to_end(mock_server):
    verified = att.verify_chute("cpk_att_ok", MODEL_ID, force=True)
    assert verified == {INSTANCE_ID: mock_server.state.pubkey_b64}


def test_verify_chute_report_surfaces_instances(mock_server):
    report = att.verify_chute_report("cpk_report", MODEL_ID)
    assert report.ok
    assert report.verified == {INSTANCE_ID: mock_server.state.pubkey_b64}
    assert report.results and report.results[0].verified
    assert report.results[0].checks == {
        "signature": True,
        "quote_report_data": True,
        "certificate_binding": True,
        "measurement": True,
        "dcap_quote": True,
        "nv_gpu": True,
    }


def test_verify_chute_report_marks_failures(mock_server):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    try:
        report = att.verify_chute_report("cpk_report_bad", MODEL_ID)
    finally:
        mock_server.state.evidence_ok = True
    assert not report.ok
    assert report.failures
    assert INSTANCE_ID in report.failures[0]
    assert "report_data" in report.failures[0]


def test_verify_chute_fails_closed_when_tampered(mock_server):
    mock_server.state.evidence_ok = False
    try:
        with pytest.raises(att.AttestationError) as excinfo:
            att.verify_chute("cpk_att_bad", MODEL_ID, force=True)
    finally:
        mock_server.state.evidence_ok = True

    message = str(excinfo.value)
    assert "attestation failed" in message
    assert "request refused" in message
    assert INSTANCE_ID in message
    assert "report_data" in message


def test_transport_gate_fails_closed(mock_server):
    """Bad evidence is refused before any /e2e/invoke, unconditionally."""
    import litellm

    ensure_installed()
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    invokes_before = len(mock_server.state.invokes)
    try:

        async def bad():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_gate_bad",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        with pytest.raises(Exception) as excinfo:
            asyncio.run(bad())
        # Fail closed: no encrypted inference request may have gone out.
        assert len(mock_server.state.invokes) == invokes_before, (
            "request was sent to /e2e/invoke despite failed attestation"
        )
        assert any("evidence" in p for p in mock_server.state.seen_paths)
        message = str(excinfo.value)
        assert "attestation failed" in message
        assert "request not sent" in message
        assert INSTANCE_ID in message
        assert "report_data" in message
    finally:
        mock_server.state.evidence_ok = True


def test_attestation_cannot_be_disabled_by_env(mock_server, monkeypatch):
    """The removed toggles are inert: setting them must not open the gate.

    Regression guard for a backdoor: every variable a user could try to use to
    turn verification off is present, yet bad evidence must still be refused
    before the encrypted invocation is sent.
    """
    import litellm

    ensure_installed()
    att._verify_cache.clear()
    monkeypatch.setenv("CHUTES_VERIFY_ATTESTATION", "false")
    monkeypatch.setenv("CHUTES_VERIFY_GPU", "false")
    monkeypatch.setenv("CHUTES_E2EE_VERIFY_SSL", "false")
    monkeypatch.setenv("CHUTES_E2EE_HOSTS", "")
    # The old endpoint overrides are gone: setting them must not redirect
    # requests away from the fixed endpoints (here, the mock).
    monkeypatch.setenv("CHUTES_E2EE_API_BASE", "https://api.private.test")
    monkeypatch.setenv("CHUTES_E2EE_MODELS_BASE", "https://llm.private.test")
    assert endpoints.API_BASE == mock_server.base_url
    assert endpoints.MODELS_BASE == mock_server.base_url
    mock_server.state.evidence_ok = False
    try:

        async def call():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_gate_env",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        with pytest.raises(Exception) as excinfo:
            asyncio.run(call())
        assert mock_server.state.invokes == [], "request reached /e2e/invoke despite bad evidence"
        assert not any("private.test" in path for path in mock_server.state.seen_paths)
        message = str(excinfo.value)
        assert "attestation failed" in message
        assert "request not sent" in message
    finally:
        mock_server.state.evidence_ok = True


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


def test_format_checks_rendering():
    rendered = att._format_checks({"signature": True, "quote_report_data": False, "nv_gpu": None})
    assert rendered == "signature=ok, tdx_binding=FAIL, nv_gpu=n/a"


def test_verify_chute_failure_message_is_actionable(mock_server):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    try:
        with pytest.raises(att.AttestationError) as excinfo:
            att.verify_chute("cpk_msg_bad", MODEL_ID, force=True)
    finally:
        mock_server.state.evidence_ok = True

    message = str(excinfo.value)
    assert "attestation failed" in message
    assert "request refused" in message
    assert MODEL_ID in message
    assert INSTANCE_ID in message
    assert "per-instance reason(s):" in message
    assert "report_data" in message
    # the counts + endpoints make it debuggable from a log alone
    assert "evidence blobs:" in message
    assert "api_base=" in message


def test_verify_chute_reports_missing_evidence(mock_server, monkeypatch):
    monkeypatch.setattr(att, "fetch_chute_evidence", lambda *a, **k: [])
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute("cpk_no_evidence", MODEL_ID, force=True)
    assert "no evidence blobs" in str(excinfo.value)


def test_verify_chute_reports_no_instances(mock_server, monkeypatch):
    monkeypatch.setattr(att, "fetch_instances", lambda *a, **k: [])
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute("cpk_no_instances", MODEL_ID, force=True)
    message = str(excinfo.value)
    assert "no E2EE instances" in message
    assert MODEL_ID in message


def test_verify_chute_reports_failed_instance_ids(mock_server, monkeypatch):
    monkeypatch.setattr(
        att,
        "fetch_chute_evidence",
        lambda *a, **k: att.ChuteEvidence(evidence=[], failed_instance_ids=[INSTANCE_ID]),
    )
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute("cpk_failed_ids", MODEL_ID, force=True)
    message = str(excinfo.value)
    assert INSTANCE_ID in message
    assert "failed_instance_ids" in message


def test_verify_chute_reports_listed_instance_without_evidence(mock_server, monkeypatch):
    monkeypatch.setattr(
        att,
        "fetch_chute_evidence",
        lambda *a, **k: att.ChuteEvidence(evidence=[], failed_instance_ids=[]),
    )
    with pytest.raises(att.AttestationError, match="no attestation evidence returned"):
        att.verify_chute("cpk_missing", MODEL_ID, force=True)


# ---------------------------------------------------------------------------
# Connection reuse / caching (no new TLS handshake per request)
# ---------------------------------------------------------------------------


def test_shared_client_is_reused_and_reset():
    att.close_shared_http()
    try:
        first = att._client()
        assert att._client() is first  # same pooled client across calls
        assert att._client()._transport is first._transport
        att.close_shared_http()
        assert att._client() is not first  # reset really drops it
    finally:
        att.close_shared_http()


def test_shared_client_is_thread_safe():
    att.close_shared_http()
    seen: list[int] = []
    seen_lock = threading.Lock()

    def worker() -> None:
        client = att._client()
        with seen_lock:
            seen.append(id(client))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert len(set(seen)) == 1, "concurrent callers created multiple clients"
    finally:
        att.close_shared_http()


def test_fetch_helpers_use_shared_client(monkeypatch):
    """A default (http=None) fetch must go through the pooled client."""
    calls: list[str] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": []}

    class _RecordingClient:
        def get(self, url, **_kwargs):
            calls.append(url)
            return _Resp()

    monkeypatch.setattr(att, "_client", lambda: _RecordingClient())
    monkeypatch.setattr(endpoints, "MODELS_BASE", "https://example.test")
    assert att.fetch_model_map("cpk_x") == {}
    assert calls == ["https://example.test/v1/models"]


def test_model_map_is_cached(mock_server, monkeypatch):
    """Two verifications must resolve the model->chute id once (cached)."""
    att._model_map_cache.clear()
    real = att.fetch_model_map
    count = {"n": 0}

    def counting(*args, **kwargs):
        count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(att, "fetch_model_map", counting)

    for _ in range(2):
        att.verify_chute("cpk_cache", MODEL_ID, force=True)
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# Fail-closed on *any* unverified instance (the transport picks one of them)
# ---------------------------------------------------------------------------

_ID_1 = "11111111-1111-4111-8111-111111111111"
_ID_2 = "22222222-2222-4222-8222-222222222222"
_PK_1 = "PUBKEY-ONE"
_PK_2 = "PUBKEY-TWO"


def _two_instance_setup(mock_server, monkeypatch, *, second_ok: bool):
    monkeypatch.setattr(
        att,
        "fetch_instances",
        lambda *a, **k: [
            att.InstanceInfo(_ID_1, _PK_1),
            att.InstanceInfo(_ID_2, _PK_2),
        ],
    )

    def fake_evidence(api_key, chute_id, nonce, **_k):
        return [
            build_evidence_for(mock_server.state, _ID_1, _PK_1, nonce, bind=True),
            build_evidence_for(mock_server.state, _ID_2, _PK_2, nonce, bind=second_ok),
        ]

    monkeypatch.setattr(att, "fetch_chute_evidence", fake_evidence)


def test_verify_chute_verifies_all_instances(mock_server, monkeypatch):
    _two_instance_setup(mock_server, monkeypatch, second_ok=True)
    verified = att.verify_chute("cpk_all_ok", MODEL_ID, force=True)
    assert verified == {_ID_1: _PK_1, _ID_2: _PK_2}


def test_verify_chute_fails_closed_if_any_instance_unverified(mock_server, monkeypatch):
    _two_instance_setup(mock_server, monkeypatch, second_ok=False)
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute("cpk_partial", MODEL_ID, force=True)
    message = str(excinfo.value)
    assert "attestation failed" in message
    assert _ID_2 in message  # the bad instance is named
    reasons = message.split("per-instance reason(s):", 1)[1]
    assert _ID_1 not in reasons
    assert "report_data" in reasons


# ---------------------------------------------------------------------------
# Cache correctness / performance
# ---------------------------------------------------------------------------


def test_repeated_verification_is_cached_and_force_bypasses(mock_server, monkeypatch):
    """The cache key is the chute id alone; ``force=True`` re-runs verification."""
    calls = {"n": 0}

    def fake_do_verify_chute(*args, **kwargs):
        calls["n"] += 1
        return {INSTANCE_ID: "pk"}

    monkeypatch.setattr(att, "_do_verify_chute", fake_do_verify_chute)

    first = att.verify_chute("cpk_cache_repeat", MODEL_ID, force=True)
    assert first == {INSTANCE_ID: "pk"}
    assert calls["n"] == 1

    second = att.verify_chute("cpk_cache_repeat", MODEL_ID)
    assert second == {INSTANCE_ID: "pk"}
    assert calls["n"] == 1, "a repeated verify for the same chute was not cached"

    att.verify_chute("cpk_cache_repeat", MODEL_ID, force=True)
    assert calls["n"] == 2, "force=True did not bypass the cache"


def test_failed_attestation_is_negatively_cached(mock_server, monkeypatch):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    calls = {"n": 0}
    real = att.fetch_chute_evidence

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(att, "fetch_chute_evidence", counting)
    try:
        with pytest.raises(att.AttestationError):
            att.verify_chute("cpk_neg", MODEL_ID, force=True)
        with pytest.raises(att.AttestationError):
            att.verify_chute("cpk_neg", MODEL_ID)
        assert calls["n"] == 1
    finally:
        mock_server.state.evidence_ok = True


def test_concurrent_cold_cache_is_single_flight(mock_server, monkeypatch):
    att._verify_cache.clear()
    att._inflight.clear()
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def slow(*args, **kwargs):
        calls["n"] += 1
        started.set()
        release.wait(10)
        return {INSTANCE_ID: "pk-sf"}

    monkeypatch.setattr(att, "_do_verify_chute", slow)
    results: list[dict[str, str]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(att.verify_chute("cpk_sf", MODEL_ID))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=worker)
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=worker)
    second.start()
    time.sleep(0.2)  # let the second caller observe the in-flight leader
    release.set()
    first.join(10)
    second.join(10)

    assert not errors
    assert calls["n"] == 1, f"expected one evidence fetch, got {calls['n']}"
    assert results and all(r == {INSTANCE_ID: "pk-sf"} for r in results)


def test_uuid_chute_with_force_verifies(mock_server):
    """``--chute <uuid> --force`` must work (force must not defeat uuid detection)."""
    att._verify_cache.clear()
    verified = att.verify_chute("cpk_uuid", CHUTE_ID, force=True)
    assert verified == {INSTANCE_ID: mock_server.state.pubkey_b64}


# ---------------------------------------------------------------------------
# Transport wiring: verification runs with no toggle kwarg at all
# ---------------------------------------------------------------------------


def test_transport_calls_verification_without_toggles(mock_server, monkeypatch):
    """The transport calls the gate with no ``verify_*`` kwargs and still sends."""
    import litellm

    from chutes_litellm import e2ee_litellm

    ensure_installed()
    e2ee_litellm._clients.clear()
    e2ee_litellm._aclients.clear()
    att._verify_cache.clear()

    model_calls: list[dict] = []
    chute_calls: list[dict] = []
    real_model = att.verify_model
    real_chute = att.verify_chute

    def spy_model(api_key, model, **kwargs):
        model_calls.append(kwargs)
        return real_model(api_key, model, **kwargs)

    def spy_chute(api_key, model, *args, **kwargs):
        chute_calls.append(kwargs)
        return real_chute(api_key, model, *args, **kwargs)

    monkeypatch.setattr(att, "verify_model", spy_model)
    monkeypatch.setattr(att, "verify_chute", spy_chute)

    async def call():
        return await litellm.acompletion(
            model=f"chutes/{MODEL_ID}",
            api_base=f"{mock_server.base_url}/v1",
            api_key="cpk_wire",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            timeout=60,
        )

    resp = asyncio.run(call())
    assert resp.choices[0].message.content == "pong from mock"
    assert model_calls, "verify_model was never called"
    assert chute_calls, "verify_chute was never called"
    for kwargs in [*model_calls, *chute_calls]:
        assert "verify_gpu" not in kwargs
        assert "verify_quote" not in kwargs
    assert any("evidence" in p for p in mock_server.state.seen_paths)
    assert mock_server.state.invokes, "attested request never reached /e2e/invoke"
