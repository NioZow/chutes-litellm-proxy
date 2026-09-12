"""Live verification against the real Chutes API (network + credentials).

Skipped unless CHUTES_LIVE_TEE=1 and CHUTES_API_KEY are set.  Verifies binding +
key-possession evidence for a real -TEE model.  Full Intel DCAP quote / NVIDIA
attestation verification additionally requires ``dcap-qvl`` and
``nv-attestation-sdk`` and outbound access to Intel/NVIDIA verifiers.
"""

import importlib.util
import os

import pytest

from chutes_litellm import attestation as att

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CHUTES_LIVE_TEE") and os.environ.get("CHUTES_API_KEY")),
    reason="set CHUTES_LIVE_TEE=1 and CHUTES_API_KEY to run live attestation checks",
)

_HAS_HARDWARE_SDKS = (
    importlib.util.find_spec("dcap_qvl") is not None
    and importlib.util.find_spec("nv_attestation_sdk") is not None
)

# Phrases that indicate the *verifier service* was unreachable rather than the
# evidence being invalid; those are skipped, not failed (see PLAN Phase 6.4).
_UNREACHABLE = (
    "collateral fetch failed",
    "nras",
    "timed out",
    "timeout",
    "temporary failure",
    "connection",
)


def test_live_verify_chute():
    api_key = os.environ["CHUTES_API_KEY"]
    model = os.environ.get("CHUTES_LIVE_MODEL", "Qwen/Qwen3.5-397B-A17B-TEE")
    verified = att.verify_chute(
        api_key,
        model,
        api_base=att.DEFAULT_API_BASE,
        models_base=att.DEFAULT_MODELS_BASE,
        force=True,
    )
    assert verified


@pytest.mark.skipif(
    not _HAS_HARDWARE_SDKS,
    reason="install dcap-qvl + nv-attestation-sdk (uv sync --extra attestation) for hardware-root checks",
)
def test_live_verify_chute_hardware_roots():
    """Real Intel DCAP + NVIDIA NRAS verification of a live -TEE chute.

    Needs outbound access to the PCCS and NRAS endpoints.  If those are
    unreachable the test skips (it does not fail), but a *failed* attestation
    result is reported as a failure.
    """
    api_key = os.environ["CHUTES_API_KEY"]
    model = os.environ.get("CHUTES_LIVE_MODEL", "Qwen/Qwen3.5-397B-A17B-TEE")
    try:
        verified = att.verify_chute(
            api_key,
            model,
            api_base=att.DEFAULT_API_BASE,
            models_base=att.DEFAULT_MODELS_BASE,
            verify_quote=True,
            verify_gpu=True,
            force=True,
        )
    except att.AttestationError as exc:
        message = str(exc).lower()
        if any(token in message for token in _UNREACHABLE):
            pytest.skip(f"hardware verifier unreachable (PCCS/NRAS): {exc}")
        raise
    assert verified
