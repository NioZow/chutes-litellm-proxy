"""Model-mode classification, forced-TEE policy, and the negative cache (offline).

These tests exercise the real :class:`nearai_inference_sdk.ModelMetadata`
dataclass and the process-local registry; nothing touches the network.
"""

from __future__ import annotations

import threading

import pytest
from nearai_inference_sdk import ModelMetadata

from chutes_litellm.near.modes import (
    CODE_FORCED_TEE_MISMATCH,
    CODE_MODE_CHANGED,
    ForcedTeeMatcher,
    ModelMode,
    ModelModeRegistry,
    mode_from_metadata,
)
from chutes_litellm.near.policy import FORCED_TEE_ENV, FailureCache, NearPolicyError


def metadata(provider_type: str, attestation_supported: bool) -> ModelMetadata:
    return ModelMetadata(provider_type=provider_type, attestation_supported=attestation_supported)


TEE = metadata("vllm", True)
ROUTED_VLLM = metadata("vllm", False)
ROUTED_OTHER = metadata("openai", True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (TEE, "tee"),
        (ROUTED_VLLM, "routed"),
        (ROUTED_OTHER, "routed"),
        (metadata("openai", False), "routed"),
    ],
)
def test_mode_from_metadata(value, expected):
    assert mode_from_metadata(value) == expected


def test_mode_from_metadata_returns_literal():
    mode: ModelMode = mode_from_metadata(TEE)
    assert mode in {"tee", "routed"}


# ---------------------------------------------------------------------------
# Forced-TEE allowlist
# ---------------------------------------------------------------------------


def test_forcetee_matcher_exact_and_glob():
    matcher = ForcedTeeMatcher(["known-tee", "vendor/*", " exact-id "])
    assert matcher.matches("known-tee")
    assert matcher.matches("vendor/model-a")
    assert matcher.matches("exact-id")
    assert not matcher.matches("known-tee-2")
    assert not matcher.matches("vendor")
    assert matcher.patterns == ("known-tee", "vendor/*", "exact-id")


def test_forcetee_matcher_is_case_sensitive():
    matcher = ForcedTeeMatcher(["Known-TEE"])
    assert not matcher.matches("known-tee")


def test_forcetee_matcher_strips_user_facing_prefixes():
    matcher = ForcedTeeMatcher(["vendor/*"])
    assert matcher.matches("nearai/vendor/model-a")
    assert matcher.matches("nearai_e2ee/vendor/model-a")
    # A pasted user-facing pattern is normalized too.
    assert ForcedTeeMatcher(["nearai/vendor/*"]).matches("vendor/model-a")


def test_forcetee_matcher_empty_and_unset():
    assert not ForcedTeeMatcher()
    assert not ForcedTeeMatcher([])
    assert not ForcedTeeMatcher(["", "  "])
    assert not ForcedTeeMatcher.from_spec(None)
    assert not ForcedTeeMatcher.from_spec("")
    assert not ForcedTeeMatcher.from_env({})


def test_forcetee_matcher_reads_spec_and_env():
    assert ForcedTeeMatcher.from_spec("a, b/*").patterns == ("a", "b/*")
    assert ForcedTeeMatcher.from_env({FORCED_TEE_ENV: "x, y/*"}).patterns == ("x", "y/*")


def test_registry_reads_env_at_construction(monkeypatch):
    monkeypatch.setenv(FORCED_TEE_ENV, "forced-model")
    registry = ModelModeRegistry()
    assert registry.forced_tee.matches("forced-model")
    with pytest.raises(NearPolicyError) as excinfo:
        registry.observe("forced-model", ROUTED_VLLM)
    assert excinfo.value.code == CODE_FORCED_TEE_MISMATCH


def test_registry_reads_explicit_environ():
    registry = ModelModeRegistry(environ={FORCED_TEE_ENV: "sys/*"})
    with pytest.raises(NearPolicyError):
        registry.observe("sys/routed", ROUTED_OTHER)
    assert registry.observe("other/model", ROUTED_OTHER) == "routed"


# ---------------------------------------------------------------------------
# Mode freezing
# ---------------------------------------------------------------------------


def test_registry_freezes_first_observation():
    registry = ModelModeRegistry()
    assert not registry.is_frozen("m")
    assert registry.get("m") is None
    assert registry.observe("m", TEE) == "tee"
    assert registry.is_frozen("m")
    assert registry.get("m") == "tee"
    assert registry.observe("m", TEE) == "tee"


def test_frozen_tee_then_routed_is_a_downgrade_attempt():
    registry = ModelModeRegistry()
    registry.observe("m", TEE)
    with pytest.raises(NearPolicyError) as excinfo:
        registry.observe("m", ROUTED_VLLM)
    assert excinfo.value.code == CODE_MODE_CHANGED
    assert excinfo.value.details["previous_mode"] == "tee"
    assert excinfo.value.details["observed_mode"] == "routed"
    # The failed observation must not overwrite the frozen mode.
    assert registry.get("m") == "tee"


def test_frozen_routed_then_tee_fails_closed():
    registry = ModelModeRegistry()
    registry.observe("m", ROUTED_OTHER)
    with pytest.raises(NearPolicyError) as excinfo:
        registry.observe("m", TEE)
    assert excinfo.value.code == CODE_MODE_CHANGED
    assert registry.get("m") == "routed"


def test_forced_tee_with_routed_metadata_rejected_on_first_observation():
    registry = ModelModeRegistry(ForcedTeeMatcher(["forced"]))
    with pytest.raises(NearPolicyError) as excinfo:
        registry.observe("forced", ROUTED_OTHER)
    assert excinfo.value.code == CODE_FORCED_TEE_MISMATCH
    assert not registry.is_frozen("forced")


def test_forced_tee_glob_with_tee_metadata_is_accepted():
    registry = ModelModeRegistry(ForcedTeeMatcher(["vendor/*"]))
    assert registry.observe("vendor/tee", TEE) == "tee"


def test_registry_reset_clears_modes_but_keeps_forced_list():
    registry = ModelModeRegistry(ForcedTeeMatcher(["forced"]))
    registry.observe("m", TEE)
    registry.reset()
    assert registry.get("m") is None
    assert not registry.is_frozen("m")
    with pytest.raises(NearPolicyError):
        registry.observe("forced", ROUTED_OTHER)
    # Re-observing after reset is allowed and may pick a different mode only
    # because the process state was intentionally discarded (tests only).
    assert registry.observe("m", ROUTED_OTHER) == "routed"


def test_registries_are_isolated():
    first = ModelModeRegistry()
    second = ModelModeRegistry()
    first.observe("m", TEE)
    assert first.get("m") == "tee"
    assert second.get("m") is None
    second.observe("m", ROUTED_OTHER)
    assert second.get("m") == "routed"


def test_concurrent_observations_are_single_mode():
    registry = ModelModeRegistry()
    results: list[ModelMode] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(registry.observe("m", TEE))
        except BaseException as exc:  # noqa: BLE001 - collect for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results == ["tee"] * 32
    assert registry.get("m") == "tee"


# ---------------------------------------------------------------------------
# Negative cache
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_failure_cache_miss_returns_none():
    cache = FailureCache(clock=FakeClock())
    assert cache.get("unknown") is None


def test_failure_cache_hit_returns_stored_error():
    clock = FakeClock()
    cache = FailureCache(clock=clock)
    error = NearPolicyError("near.model.tcb_not_uptodate")
    cache.put("model-a", error)
    clock.advance(29.0)
    assert cache.get("model-a") is error


def test_failure_cache_default_ttl_is_30_seconds():
    clock = FakeClock()
    cache = FailureCache(clock=clock)
    error = NearPolicyError("near.model.none_verified")
    cache.put("model-a", error)
    clock.advance(29.999)
    assert cache.get("model-a") is error
    clock.advance(0.001)
    assert cache.get("model-a") is None


def test_failure_cache_entries_expire_and_are_removed():
    clock = FakeClock()
    cache = FailureCache(ttl=5.0, clock=clock)
    cache.put("model-a", NearPolicyError("x"))
    clock.advance(5.0)
    assert cache.get("model-a") is None
    clock.advance(0.0)
    assert cache.get("model-a") is None


def test_failure_cache_keys_are_isolated():
    clock = FakeClock()
    cache = FailureCache(clock=clock)
    error_a = NearPolicyError("a")
    cache.put("model-a", error_a)
    assert cache.get("model-b") is None
    cache.put("model-b", NearPolicyError("b"))
    assert cache.get("model-a") is error_a
    assert cache.get("model-b") is not None


def test_failure_cache_clear():
    cache = FailureCache()
    cache.put("model-a", NearPolicyError("a"))
    assert cache.get("model-a") is not None
    cache.clear()
    assert cache.get("model-a") is None
