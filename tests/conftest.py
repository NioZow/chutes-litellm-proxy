import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
# Avoid writing to the default log file during tests; logs still go to stderr.
os.environ.setdefault("CHUTES_LOG_FILE", "none")

from chutes_litellm import attestation, e2ee_litellm, endpoints  # noqa: E402
from mock_chutes_server import (  # noqa: E402
    MOCK_GPU_ARCH,
    MOCK_MRTD,
    MOCK_RTMR0,
    MOCK_RTMR1,
    MOCK_RTMR2,
    MOCK_RTMR3,
    MockChutesServer,
)


class TeeFakes:
    """Test-controlled DCAP report, measurement policy and NRAS verdict.

    The production verifier always calls real DCAP, the public measurement
    endpoint and NVIDIA NRAS; the offline suite swaps only those three network
    boundaries.
    """

    def __init__(self) -> None:
        self.dcap_status = "UpToDate"
        self.dcap_advisories: list[str] = []
        self.dcap_debug = False
        self.dcap_error: BaseException | None = None
        self.measurements = None
        self.measurements_error: BaseException | None = None
        self.mrtd = MOCK_MRTD
        self.rtmr0 = MOCK_RTMR0
        self.rtmr1 = MOCK_RTMR1
        self.rtmr2 = MOCK_RTMR2
        self.rtmr3 = MOCK_RTMR3
        self.gpu_count = 1
        self.expected_gpus = [MOCK_GPU_ARCH.lower()]
        self.nras_ok = True
        self.nras_calls: list[list[dict[str, str]]] = []


class _FakeDcapReport:
    def __init__(self, raw_quote: bytes, fakes: TeeFakes) -> None:
        self._raw = raw_quote
        self._fakes = fakes

    def to_json(self) -> str:
        import json

        from chutes_litellm.tee.quote import TdxQuote

        quote = TdxQuote.from_bytes(self._raw)
        td_attributes = quote.td_attributes
        if self._fakes.dcap_debug:
            td_attributes = "01" + "00" * 7
        return json.dumps(
            {
                "status": self._fakes.dcap_status,
                "advisory_ids": list(self._fakes.dcap_advisories),
                "report": {
                    "TD10": {
                        "mr_td": quote.mrtd.lower(),
                        "rt_mr0": quote.rtmr0.lower(),
                        "rt_mr1": quote.rtmr1.lower(),
                        "rt_mr2": quote.rtmr2.lower(),
                        "rt_mr3": quote.rtmr3.lower(),
                        "report_data": quote.report_data.lower(),
                        "td_attributes": td_attributes.lower(),
                    }
                },
            }
        )


class _FakeNvAttestation:
    """Minimal stand-in for the NVIDIA SDK's ``Attestation`` client."""

    def __init__(self, fakes: TeeFakes) -> None:
        self._fakes = fakes
        self._nonce: str | None = None

    def reset(self) -> None:
        self._nonce = None

    def add_verifier(self, *_args, **_kwargs) -> None:
        return None

    def set_nonce(self, nonce: str) -> None:
        self._nonce = nonce

    def attest(self, evidence: list[dict[str, str]]) -> bool:
        self._fakes.nras_calls.append(evidence)
        return self._fakes.nras_ok


@pytest.fixture()
def tee_fakes() -> TeeFakes:
    return TeeFakes()


@pytest.fixture(autouse=True)
def _fake_tee_boundaries(monkeypatch, tee_fakes):
    """Replace only DCAP, the measurement source and NRAS for offline tests."""
    import types

    from chutes_litellm.tee import quote as tee_quote
    from chutes_litellm.tee import verifier as tee_verifier
    from chutes_litellm.tee.measurements import TeeMeasurement

    async def fake_dcap(raw_quote, *, pccs_url=None):
        if tee_fakes.dcap_error is not None:
            raise tee_fakes.dcap_error
        return _FakeDcapReport(raw_quote, tee_fakes)

    def fake_measurements(*, force=False):
        if tee_fakes.measurements_error is not None:
            raise tee_fakes.measurements_error
        if tee_fakes.measurements is not None:
            if callable(tee_fakes.measurements):
                return tee_fakes.measurements(force=force)
            return list(tee_fakes.measurements)
        return [
            TeeMeasurement(
                version="test",
                name="test",
                mrtd=tee_fakes.mrtd,
                rtmr0=tee_fakes.rtmr0,
                rtmr1=tee_fakes.rtmr1,
                rtmr2=tee_fakes.rtmr2,
                runtime_rtmr3=tee_fakes.rtmr3,
                expected_gpus=list(tee_fakes.expected_gpus),
                gpu_count=tee_fakes.gpu_count,
            )
        ]

    nv_package = types.ModuleType("nv_attestation_sdk")
    nv_attestation = types.ModuleType("nv_attestation_sdk.attestation")
    nv_attestation.Attestation = lambda *_args, **_kwargs: _FakeNvAttestation(tee_fakes)
    nv_attestation.Devices = types.SimpleNamespace(GPU="GPU")
    nv_attestation.Environment = types.SimpleNamespace(REMOTE="REMOTE")
    nv_package.attestation = nv_attestation

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)
    monkeypatch.setattr(tee_verifier, "get_measurements", fake_measurements)
    monkeypatch.setitem(sys.modules, "nv_attestation_sdk", nv_package)
    monkeypatch.setitem(sys.modules, "nv_attestation_sdk.attestation", nv_attestation)
    yield


@pytest.fixture(scope="session", autouse=True)
def _point_env_at_mock():
    # Live runs (CHUTES_LIVE_TEE=1) must keep the user's real credentials and
    # production defaults; the mock pointers are only injected for offline runs.
    if os.environ.get("CHUTES_LIVE_TEE"):
        yield
        return
    os.environ["CHUTES_API_KEY"] = "cpk_test"
    yield


@pytest.fixture()
def mock_server(monkeypatch):
    """Point the fixed Chutes endpoints at the in-process mock server."""
    server = MockChutesServer()
    monkeypatch.setattr(endpoints, "API_BASE", server.base_url)
    monkeypatch.setattr(endpoints, "MODELS_BASE", server.base_url)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _reset_litellm_clients():
    # Isolate tests: drop per-key transport clients + the attestation cache so a
    # fresh (env-faithful) transport is built for the next test.  Attestation,
    # GPU verification and TLS verification are unconditional, so there are no
    # toggles to reset.
    e2ee_litellm._clients.clear()
    e2ee_litellm._aclients.clear()
    from chutes_litellm import custom_provider
    from chutes_litellm.tee import measurements as tee_measurements

    custom_provider.reset_clients()
    attestation._verify_cache.clear()
    attestation._model_map_cache.clear()
    attestation.close_shared_http()
    tee_measurements.reset_cache()
    # Reset the shared logger so each test gets a fresh handler set.
    import chutes_litellm
    import logging

    chutes_litellm._logger = None
    log = logging.getLogger("chutes_litellm")
    log.handlers.clear()
    log.setLevel(logging.NOTSET)
    yield


def ensure_installed():
    assert e2ee_litellm.install(), "install() returned False (CHUTES_API_KEY / chutes_e2ee missing)"
