"""Attestation verification: binding checks + fail-closed gating (see chutes_litellm.attestation).

These tests use the mock's *real* signing identity and binding layout, so they
exercise the exact code paths that would verify a real Chutes TEE/GPU instance.
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
import types

import pytest

from chutes_litellm import attestation as att
from conftest import ensure_installed
from mock_chutes_server import (
    CHUTE_ID,
    INSTANCE_ID,
    MODEL_ID,
    QUOTE_LEN,
    QUOTE_REPORT_DATA_OFFSET,
    build_evidence,
)

from pqcrypto.kem.ml_kem_768 import generate_keypair


def _digest(nonce: str, pubkey_b64: str) -> str:
    return att._digest_of(nonce, pubkey_b64)


def test_verify_evidence_valid(mock_server):
    state = mock_server.state
    nonce = "ab" * 32
    evidence = build_evidence(state, _digest(nonce, state.pubkey_b64), nonce)
    res = att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)
    assert res.verified, res.detail
    assert res.checks["key_possession"] is True
    assert res.checks["pubkey_bound"] is True
    assert res.checks["quote_report_data"] is True
    assert res.checks["gpu_nonce"] is True


def test_verify_evidence_rejects_wrong_pubkey(mock_server):
    """An attacker substituting their own key must fail the binding check."""
    state = mock_server.state
    foreign_pk, _ = generate_keypair()
    foreign_b64 = base64.b64encode(foreign_pk).decode()

    nonce = "cd" * 32
    # Evidence is built for the *foreign* key digest...
    evidence = build_evidence(state, _digest(nonce, foreign_b64), nonce)
    # ...but we try to verify it as if it were the real instance key.
    res = att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)
    assert not res.verified
    assert res.checks["pubkey_bound"] is False
    assert "quote_report_data" in res.detail or "gpu_nonce" in res.detail


def test_verify_evidence_rejects_bad_signature(mock_server):
    state = mock_server.state
    nonce = "ef" * 32
    evidence = build_evidence(state, _digest(nonce, state.pubkey_b64), nonce)
    evidence["signature"] = base64.b64encode(b"garbage").decode()
    res = att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce)
    assert not res.verified
    assert res.checks["key_possession"] is False


def test_hardware_verification_requires_optional_sdks(mock_server, monkeypatch):
    state = mock_server.state
    nonce = "22" * 32
    evidence = build_evidence(state, _digest(nonce, state.pubkey_b64), nonce)

    # Force the optional deps to look absent even when `--extra attestation` is
    # installed in the dev environment, so this stays hermetic/deterministic.
    monkeypatch.setitem(sys.modules, "dcap_qvl", None)
    with pytest.raises(att.AttestationError, match="dcap-qvl"):
        att.verify_evidence(evidence, state.pubkey_b64, nonce=nonce, verify_quote=True)

    # Missing GPU evidence is reported before any SDK is touched.
    with pytest.raises(att.AttestationError, match="gpu_evidence"):
        att.verify_evidence(
            {"instance_id": "no-gpu"}, state.pubkey_b64, nonce=nonce, verify_gpu=True
        )

    # Usable evidence but no SDK: fail closed, naming the package.
    monkeypatch.setitem(sys.modules, "nv_attestation_sdk.attestation", None)
    usable = dict(evidence)
    usable["gpu_evidence"] = [{"arch": "HOPPER", "certificate": "Y2VydA==", "evidence": "ZXZpZA=="}]
    with pytest.raises(att.AttestationError, match="nv-attestation-sdk"):
        att.verify_evidence(usable, state.pubkey_b64, nonce=nonce, verify_gpu=True)


# ---------------------------------------------------------------------------
# Hardware verifiers: shape handling and fail-closed semantics with fake SDKs
# (no real SDK, no network, so they run in the offline suite)
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


class _FakeDcapReport:
    def __init__(self, status="OK", advisories=None):
        self.status = status
        self.advisory_ids = advisories or []

    def to_json(self):
        return json.dumps({"status": self.status})


def _install_fake_dcap(monkeypatch, *, status="OK", error=None) -> None:
    async def fake_verify(raw_quote, pccs_url=None):
        if error is not None:
            raise error
        return _FakeDcapReport(status)

    mod = types.ModuleType("dcap_qvl")
    mod.get_collateral_and_verify = fake_verify
    monkeypatch.setitem(sys.modules, "dcap_qvl", mod)


def test_nvidia_gpu_evidence_is_normalized_before_attest(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    raw = [
        # extra keys are stripped; duplicates are collapsed
        {"arch": "HOPPER", "certificate": "cert", "evidence": "ev", "gpu_uuid": "x"},
        {"arch": "HOPPER", "certificate": "cert", "evidence": "ev", "gpu_uuid": "x"},
    ]
    out = att._verify_nvidia_gpu(raw, "ab" * 32)
    assert fake.calls["nonce"] == "ab" * 32
    assert fake.calls["evidence"] == [
        {"arch": "HOPPER", "evidence": "ev", "certificate": "cert"}
    ]
    assert out["gpus"] == 1 and out["arch"] == "HOPPER"
    # REMOTE mode must be configured with a non-empty NRAS URL.
    assert fake.calls["add_verifier"][2]


def test_nvidia_gpu_failure_fails_whole_verification(monkeypatch):
    fake = _FakeAttestation()
    fake.result = False
    _install_fake_nv_sdk(monkeypatch, fake)
    with pytest.raises(att.AttestationError, match="did not pass"):
        att._verify_nvidia_gpu(
            [{"arch": "HOPPER", "certificate": "c", "evidence": "e"}], "00" * 32
        )


def test_nvidia_gpu_rejects_mixed_arch(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    raw = [
        {"arch": "HOPPER", "certificate": "c1", "evidence": "e1"},
        {"arch": "BLACKWELL", "certificate": "c2", "evidence": "e2"},
    ]
    with pytest.raises(att.AttestationError, match="mixes architectures"):
        att._verify_nvidia_gpu(raw, "00" * 32)


def test_nvidia_gpu_rejects_unusable_evidence(monkeypatch):
    fake = _FakeAttestation()
    _install_fake_nv_sdk(monkeypatch, fake)
    # The mock's {nonce, gpu_uuid} shape carries no arch/certificate/evidence.
    with pytest.raises(att.AttestationError, match="no usable NVIDIA GPU evidence"):
        att._verify_nvidia_gpu([{"nonce": "00" * 32, "gpu_uuid": "x"}], "00" * 32)


def test_dcap_accepts_ok_status(monkeypatch):
    _install_fake_dcap(monkeypatch, status="OK")
    quote = base64.b64encode(b"q" * 640).decode()
    assert att._verify_tdx_quote(quote) == "OK"


def test_dcap_rejects_bad_tcb_status(monkeypatch):
    _install_fake_dcap(monkeypatch, status="OUT_OF_DATE")
    quote = base64.b64encode(b"q" * 640).decode()
    with pytest.raises(att.AttestationError, match="OUT_OF_DATE"):
        att._verify_tdx_quote(quote)


def test_dcap_reports_collateral_failure_distinctly(monkeypatch):
    _install_fake_dcap(monkeypatch, error=RuntimeError("no route to PCCS"))
    quote = base64.b64encode(b"q" * 640).decode()
    with pytest.raises(att.AttestationError, match="collateral fetch failed"):
        att._verify_tdx_quote(quote)


def test_verify_chute_end_to_end(mock_server):
    verified = att.verify_chute(
        "cpk_att_ok",
        MODEL_ID,
        api_base=mock_server.base_url,
        models_base=mock_server.base_url,
        force=True,
    )
    assert INSTANCE_ID in verified


def test_verify_chute_report_surfaces_instances(mock_server):
    report = att.verify_chute_report(
        "cpk_report",
        MODEL_ID,
        api_base=mock_server.base_url,
        models_base=mock_server.base_url,
    )
    assert report.ok
    assert INSTANCE_ID in report.verified
    assert report.results and report.results[0].verified
    assert report.results[0].checks["key_possession"] is True


def test_verify_chute_report_marks_failures(mock_server):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    try:
        report = att.verify_chute_report(
            "cpk_report_bad",
            MODEL_ID,
            api_base=mock_server.base_url,
            models_base=mock_server.base_url,
        )
    finally:
        mock_server.state.evidence_ok = True
    assert not report.ok
    assert report.failures
    assert INSTANCE_ID in report.failures[0]


def test_verify_chute_fails_closed_when_tampered(mock_server):
    mock_server.state.evidence_ok = False
    try:
        with pytest.raises(att.AttestationError, match="attestation failed"):
            att.verify_chute(
                "cpk_att_bad",
                MODEL_ID,
                api_base=mock_server.base_url,
                models_base=mock_server.base_url,
                force=True,
            )
    finally:
        mock_server.state.evidence_ok = True


def test_transport_gate_fails_closed(mock_server):
    """With CHUTES_VERIFY_ATTESTATION=true a request is refused when evidence is bad."""
    import litellm

    ensure_installed()
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    try:
        async def good():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_gate_ok",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        resp = asyncio.run(good())
        assert resp.choices[0].message.content == "pong from mock"
        assert any("evidence" in p for p in mock_server.state.seen_paths)

        # Now serve unbound evidence and watch the gate block the request.
        att._verify_cache.clear()  # the 'good' call above cached this chute
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
            # Fail closed: no encrypted inference request may have gone out, and
            # the evidence endpoint must have been re-checked.
            assert len(mock_server.state.invokes) == invokes_before, (
                "request was sent to /e2e/invoke despite failed attestation"
            )
            assert any("evidence" in p for p in mock_server.state.seen_paths)
            # The surfaced error must explain *why* (not an opaque 500 traceback).
            message = str(excinfo.value)
            assert "attestation failed" in message
            assert "request not sent" in message
            assert INSTANCE_ID in message
            assert "signature=ok" in message
            assert "tdx_binding=FAIL" in message or "gpu_binding=FAIL" in message
        finally:
            mock_server.state.evidence_ok = True
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)


# ---------------------------------------------------------------------------
# Error reporting: the failure must say *why*, per instance
# ---------------------------------------------------------------------------


def test_format_checks_rendering():
    rendered = att._format_checks(
        {"key_possession": True, "quote_report_data": False, "gpu_nonce": None}
    )
    assert rendered == "signature=ok, tdx_binding=FAIL, gpu_binding=n/a"


def test_verify_chute_failure_message_is_actionable(mock_server):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    try:
        with pytest.raises(att.AttestationError) as excinfo:
            att.verify_chute(
                "cpk_msg_bad",
                MODEL_ID,
                api_base=mock_server.base_url,
                models_base=mock_server.base_url,
                force=True,
            )
    finally:
        mock_server.state.evidence_ok = True

    message = str(excinfo.value)
    assert "attestation failed" in message
    assert "request refused" in message
    assert MODEL_ID in message
    assert INSTANCE_ID in message
    assert "per-instance reason(s):" in message
    # signature is valid even for tampered binding; the binding checks are the
    # ones that must be reported as failures.
    assert "signature=ok" in message
    assert "tdx_binding=FAIL" in message or "gpu_binding=FAIL" in message
    # the counts + endpoints make it debuggable from a log alone
    assert "evidence blobs:" in message
    assert "api_base=" in message


def test_verify_chute_reports_missing_evidence(mock_server, monkeypatch):
    monkeypatch.setattr(att, "fetch_chute_evidence", lambda *a, **k: [])
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute(
            "cpk_no_evidence",
            MODEL_ID,
            api_base=mock_server.base_url,
            models_base=mock_server.base_url,
            force=True,
        )
    assert "no evidence blobs" in str(excinfo.value)


def test_verify_chute_reports_no_instances(mock_server, monkeypatch):
    monkeypatch.setattr(att, "fetch_instances", lambda *a, **k: [])
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute(
            "cpk_no_instances",
            MODEL_ID,
            api_base=mock_server.base_url,
            models_base=mock_server.base_url,
            force=True,
        )
    message = str(excinfo.value)
    assert "no E2EE instances" in message
    assert MODEL_ID in message


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
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(set(seen)) == 1, "concurrent callers created multiple clients"
    finally:
        att.close_shared_http()


def test_fetch_helpers_use_shared_client(monkeypatch):
    """A default (http=None) fetch must go through the pooled client, not a
    freshly constructed one."""
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
    assert att.fetch_model_map("cpk_x", models_base="https://example.test") == {}
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
        att.verify_chute(
            "cpk_cache",
            MODEL_ID,
            api_base=mock_server.base_url,
            models_base=mock_server.base_url,
            force=True,
        )
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# Fail-closed on *any* unverified instance (the transport picks one of them)
# ---------------------------------------------------------------------------


def _evidence_for(state, instance_id: str, pubkey_b64: str, nonce: str, *, ok: bool = True) -> dict:
    """Build evidence bound to an arbitrary instance id/pubkey (mock signer)."""
    digest = att._digest_of(nonce, pubkey_b64)
    bind = digest if ok else "0" * 64
    quote = bytearray(QUOTE_LEN)
    quote[QUOTE_REPORT_DATA_OFFSET : QUOTE_REPORT_DATA_OFFSET + 64] = bytes.fromhex(bind).ljust(64, b"\x00")
    gpu = [{"nonce": bind, "gpu_uuid": "00000000-0000-0000-0000-000000000009"}]
    body = json.dumps(
        {
            "instance_id": instance_id,
            "digest": bind,
            "quote": base64.b64encode(bytes(quote)).decode(),
            "gpu_evidence": gpu,
        }
    ).encode()
    return {
        "instance_id": instance_id,
        "quote": base64.b64encode(bytes(quote)).decode(),
        "gpu_evidence": gpu,
        "certificate": base64.b64encode(state.cert_der()).decode(),
        "signature": base64.b64encode(state.sign(body)).decode(),
        "attested_body": base64.b64encode(body).decode(),
    }


_ID_1 = "11111111-1111-4111-8111-111111111111"
_ID_2 = "22222222-2222-4222-8222-222222222222"


def _two_instance_setup(mock_server, monkeypatch, *, second_ok: bool):
    pk1, pk2 = "PUBKEY-ONE", "PUBKEY-TWO"
    monkeypatch.setattr(
        att,
        "fetch_instances",
        lambda *a, **k: [att.InstanceInfo(_ID_1, pk1), att.InstanceInfo(_ID_2, pk2)],
    )

    def fake_evidence(api_key, chute_id, nonce, **_k):
        return [
            _evidence_for(mock_server.state, _ID_1, pk1, nonce, ok=True),
            _evidence_for(mock_server.state, _ID_2, pk2, nonce, ok=second_ok),
        ]

    monkeypatch.setattr(att, "fetch_chute_evidence", fake_evidence)


def test_verify_chute_verifies_all_instances(mock_server, monkeypatch):
    _two_instance_setup(mock_server, monkeypatch, second_ok=True)
    verified = att.verify_chute(
        "cpk_all_ok", MODEL_ID, api_base=mock_server.base_url, models_base=mock_server.base_url, force=True
    )
    assert set(verified) == {_ID_1, _ID_2}


def test_verify_chute_fails_closed_if_any_instance_unverified(mock_server, monkeypatch):
    _two_instance_setup(mock_server, monkeypatch, second_ok=False)
    with pytest.raises(att.AttestationError) as excinfo:
        att.verify_chute(
            "cpk_partial", MODEL_ID, api_base=mock_server.base_url, models_base=mock_server.base_url, force=True
        )
    message = str(excinfo.value)
    assert "attestation failed" in message
    assert _ID_2 in message  # the bad instance is named
    assert _ID_1 not in message.split("per-instance reason(s):")[1]


def test_verify_chute_fails_closed_if_listed_instance_has_no_evidence(mock_server, monkeypatch):
    pk1, pk2 = "PUBKEY-ONE", "PUBKEY-TWO"
    monkeypatch.setattr(
        att,
        "fetch_instances",
        lambda *a, **k: [att.InstanceInfo(_ID_1, pk1), att.InstanceInfo(_ID_2, pk2)],
    )

    def only_one(api_key, chute_id, nonce, **_k):
        return [_evidence_for(mock_server.state, _ID_1, pk1, nonce, ok=True)]

    monkeypatch.setattr(att, "fetch_chute_evidence", only_one)
    with pytest.raises(att.AttestationError, match="no attestation evidence returned"):
        att.verify_chute(
            "cpk_missing", MODEL_ID, api_base=mock_server.base_url, models_base=mock_server.base_url, force=True
        )


# ---------------------------------------------------------------------------
# Cache correctness / performance
# ---------------------------------------------------------------------------


def test_cache_key_includes_verification_mode(mock_server, monkeypatch):
    """A cache hit without DCAP/GPU checks must not satisfy a request that needs
    them (and vice versa)."""
    calls = {"n": 0}
    real = att._do_verify_chute

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(att, "_do_verify_chute", counting)
    base = dict(api_base=mock_server.base_url, models_base=mock_server.base_url)

    att.verify_chute("cpk_mode", MODEL_ID, force=True, **base)
    att.verify_chute("cpk_mode", MODEL_ID, **base)  # same mode -> cached
    assert calls["n"] == 1

    att.verify_chute("cpk_mode", MODEL_ID, check_signature=False, **base)  # different mode
    assert calls["n"] == 2


def test_failed_attestation_is_negatively_cached(mock_server, monkeypatch):
    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    calls = {"n": 0}
    real = att.fetch_chute_evidence

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(att, "fetch_chute_evidence", counting)
    base = dict(api_base=mock_server.base_url, models_base=mock_server.base_url)
    try:
        with pytest.raises(att.AttestationError):
            att.verify_chute("cpk_neg", MODEL_ID, force=True, **base)
        with pytest.raises(att.AttestationError):
            att.verify_chute("cpk_neg", MODEL_ID, **base)  # served from the failure cache
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
        return [INSTANCE_ID]

    monkeypatch.setattr(att, "_do_verify_chute", slow)
    results: list[list[str]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(
                att.verify_chute(
                    "cpk_sf", MODEL_ID, api_base=mock_server.base_url, models_base=mock_server.base_url
                )
            )
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
    assert results and all(r == [INSTANCE_ID] for r in results)


def test_uuid_chute_with_force_verifies(mock_server):
    """``--chute <uuid> --force`` must work (force must not defeat uuid detection)."""
    att._verify_cache.clear()
    verified = att.verify_chute(
        "cpk_uuid",
        CHUTE_ID,
        api_base=mock_server.base_url,
        models_base=mock_server.base_url,
        force=True,
    )
    assert INSTANCE_ID in verified


# ---------------------------------------------------------------------------
# The transport must forward CHUTES_VERIFY_QUOTE/_GPU to the verifier
# ---------------------------------------------------------------------------


def test_transport_wires_quote_and_gpu_flags(mock_server, monkeypatch):
    import litellm

    from chutes_litellm import attestation as a
    from chutes_litellm import e2ee_litellm

    ensure_installed()
    e2ee_litellm._clients.clear()
    att._verify_cache.clear()
    captured: dict = {}
    real = a.verify_model

    def spy(api_key, model, **kwargs):
        captured.update(kwargs)
        return real(api_key, model, **kwargs)

    def fail_dcap(quote_b64):
        raise a.AttestationError("simulated DCAP failure (offline test)")

    monkeypatch.setattr(a, "verify_model", spy)
    # Keep the test hermetic whether or not dcap-qvl is installed: the point is
    # that the transport *passes* the flag, not that DCAP itself runs here.
    monkeypatch.setattr(a, "_verify_tdx_quote", fail_dcap)
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    os.environ["CHUTES_VERIFY_QUOTE"] = "true"
    os.environ.pop("CHUTES_VERIFY_GPU", None)
    try:
        async def call():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_wire",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        # dcap-qvl is not installed (or the mock quote is not a real DCAP quote),
        # so the request must be refused -- proving the flag reached the verifier.
        with pytest.raises(Exception):
            asyncio.run(call())
        assert captured.get("verify_quote") is True
        assert captured.get("verify_gpu") is False
        assert not mock_server.state.invokes
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)
        os.environ.pop("CHUTES_VERIFY_QUOTE", None)
        e2ee_litellm._clients.clear()


