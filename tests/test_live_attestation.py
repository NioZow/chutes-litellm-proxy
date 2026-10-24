"""Live verification against the real Chutes API (network + credentials).

Skipped unless ``CHUTES_LIVE_TEE=1`` and ``CHUTES_API_KEY`` are set.  These
tests exercise the real trust chain: Intel DCAP quote verification, matching
against the published Chutes measurement list, and NVIDIA NRAS GPU
verification.  All three are mandatory and have no disable switch.

``dcap-qvl`` and ``nv-attestation-sdk`` are base dependencies.  The hardware
tests keep a defensive skip in case a module is genuinely missing from the
environment.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from chutes_litellm import attestation as att
from chutes_litellm.tee import measurements as tee_measurements

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CHUTES_LIVE_TEE") and os.environ.get("CHUTES_API_KEY")),
    reason="set CHUTES_LIVE_TEE=1 and CHUTES_API_KEY to run live attestation checks",
)

_HAS_HARDWARE_SDKS = (
    importlib.util.find_spec("dcap_qvl") is not None and importlib.util.find_spec("nv_attestation_sdk") is not None
)

# Phrases that indicate a *verifier service* (Intel PCCS collateral, NVIDIA
# NRAS) or the measurement endpoint was unreachable rather than the evidence
# being invalid.  Those are environment problems and are skipped, not failed.
_UNREACHABLE = (
    "collateral",
    "nras",
    "pccs",
    "unavailable",
    "timed out",
    "timeout",
    "temporary failure",
    "connection",
    "network",
)


def _skip_if_unreachable(exc: BaseException, service: str) -> None:
    """Skip when *exc* looks like a network/verifier outage; otherwise re-raise."""
    message = str(exc).lower()
    if any(token in message for token in _UNREACHABLE):
        pytest.skip(f"{service} unreachable: {exc}")


@pytest.fixture(autouse=True)
def _fake_tee_boundaries():
    """Run live tests against the real DCAP verifier, measurements and NRAS.

    ``tests/conftest.py`` fakes those offline boundaries for the unit suite; a
    live run must not inherit those fakes.
    """
    yield


def _verify_live_model() -> dict[str, str]:
    """Verify a live ``-TEE`` model, skipping on verifier/network outages."""
    api_key = os.environ["CHUTES_API_KEY"]
    model = os.environ.get("CHUTES_LIVE_MODEL", "Qwen/Qwen3.5-397B-A17B-TEE")
    try:
        return att.verify_chute(api_key, model, force=True)
    except att.AttestationError as exc:
        _skip_if_unreachable(exc, "hardware verifier (PCCS/NRAS)")
        raise


def test_live_measurements_endpoint_is_non_empty():
    """Fetch the real published measurement list used by the mandatory match."""
    tee_measurements.reset_cache()
    try:
        published = tee_measurements.get_measurements(force=True)
    except att.AttestationError as exc:
        _skip_if_unreachable(exc, "TEE measurements endpoint")
        raise
    assert published, "published TEE measurement list is empty"


@pytest.mark.skipif(
    not _HAS_HARDWARE_SDKS,
    reason="dcap-qvl / nv-attestation-sdk are base dependencies; reinstall the package",
)
def test_live_verify_chute():
    """Live ``-TEE`` chute with mandatory DCAP + measurement + NVIDIA GPU."""
    verified = _verify_live_model()
    assert verified


@pytest.mark.skipif(
    not _HAS_HARDWARE_SDKS,
    reason="dcap-qvl / nv-attestation-sdk are base dependencies; reinstall the package",
)
def test_live_verify_chute_hardware_roots():
    """Real Intel DCAP + measurement + NVIDIA NRAS verification of a live chute.

    Needs outbound access to the PCCS and NRAS endpoints.  If those are
    unreachable the test skips; a genuine attestation failure still fails.
    """
    verified = _verify_live_model()
    assert verified
