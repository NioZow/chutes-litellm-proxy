"""Offline tests for the per-API-key NEAR AI runtime clients (Phase 2).

Fakes are injected at ``chutes_litellm.near.client`` so no network, TLS, DCAP,
NRAS, or real API key is involved.  Policy assertions still run against the
real SDK verified dataclasses, so the tests exercise the same shapes the SDK
returns.  Time is controlled with injected clocks and short hard timeouts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
from nearai_inference_sdk import (
    ApiError,
    ApiFailure,
    AttestationVerificationResult,
    GatewayTlsBinding,
    MeasuredDeployment,
    ModelMetadata,
    RuntimeMeasurements,
    SigningIdentity,
    VerificationError,
    VerificationFailure,
    VerifiedGatewayAttestation,
    VerifiedModelAttestation,
)

from chutes_litellm.near import client, endpoints, policy
from chutes_litellm.near.modes import CODE_FORCED_TEE_MISMATCH, CODE_MODE_CHANGED

TEE_METADATA = ModelMetadata(provider_type="vllm", attestation_supported=True)
ROUTED_METADATA = ModelMetadata(provider_type="openai", attestation_supported=False)


# ---------------------------------------------------------------------------
# Real SDK object factories (mirrors tests/test_near_policy.py)
# ---------------------------------------------------------------------------


def _signer() -> SigningIdentity:
    return SigningIdentity(signing_algo="ed25519", signing_address="0xsigner")


def _deployment() -> MeasuredDeployment:
    return MeasuredDeployment(app_compose="app-compose", runtime_measurements=RuntimeMeasurements())


def _tls_binding(kind: str = "attested", spki: str | None = "ab" * 32) -> GatewayTlsBinding:
    return GatewayTlsBinding(kind=kind, spki_fingerprint=spki)  # type: ignore[arg-type]


def gateway(**overrides: object) -> VerifiedGatewayAttestation:
    fields: dict[str, object] = {
        "signer": _signer(),
        "tcb_status": "UpToDate",
        "advisory_ids": (),
        "deployment": _deployment(),
        "deployment_provenance": "not_checked",
        "tls_binding": _tls_binding(),
    }
    fields.update(overrides)
    return VerifiedGatewayAttestation(**fields)  # type: ignore[arg-type]


def model(**overrides: object) -> VerifiedModelAttestation:
    fields: dict[str, object] = {
        "signer": _signer(),
        "tcb_status": "UpToDate",
        "advisory_ids": (),
        "deployment": _deployment(),
        "deployment_provenance": "not_checked",
        "gpu_evidence": "verified",
        "signing_public_key": "cd" * 32,
    }
    fields.update(overrides)
    return VerifiedModelAttestation(**fields)  # type: ignore[arg-type]


def attestation_result(*models: VerifiedModelAttestation, gateway_obj=None) -> AttestationVerificationResult:
    return AttestationVerificationResult(
        gateway=gateway_obj or gateway(),
        models=tuple(models),
        verified_at=1_700_000_000_000,
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FetchedGateway:
    """Stand-in for ``FetchedGatewayAttestation`` (opaque to the fake verifier)."""

    def __init__(self) -> None:
        self.attestation = object()
        self.client_binding = object()


class _FakeHttpClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeAttestationClient:
    def __init__(self, harness: Harness, api_key: str | None, kwargs: dict[str, Any]) -> None:
        self.harness = harness
        self.api_key = api_key
        self.kwargs = kwargs
        self.fetch_gateway_calls = 0
        self.metadata_calls: list[str] = []

    async def fetch_gateway_attestation(self, **kwargs: Any) -> _FetchedGateway:
        self.fetch_gateway_calls += 1
        return _FetchedGateway()

    async def fetch_model_metadata(self, model: str) -> ModelMetadata:
        self.metadata_calls.append(model)
        self.harness.metadata_calls.append(model)
        if self.harness.metadata_delay:
            await asyncio.sleep(self.harness.metadata_delay)
        value = self.harness.metadata.get(model, self.harness.default_metadata)
        if callable(value):
            value = value()
        if isinstance(value, Exception):
            raise value
        return value


class _FakeInferenceClient:
    def __init__(self, api_key: str | None, kwargs: dict[str, Any]) -> None:
        self.api_key = api_key
        self.kwargs = kwargs
        self.verify_calls: list[str] = []
        self.handler: Any = None
        self.close_calls = 0

    async def verify(self, model: str) -> AttestationVerificationResult:
        self.verify_calls.append(model)
        if self.handler is None:
            raise AssertionError(f"unexpected verify({model!r})")
        return await self.handler(model)

    async def aclose(self) -> None:
        self.close_calls += 1


class Harness:
    """Injects every SDK symbol the runtime imports."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.created_inference: list[_FakeInferenceClient] = []
        self.created_attestation: list[_FakeAttestationClient] = []
        self.pinned_clients: list[_FakeHttpClient] = []
        self.pinned_fingerprints: list[str] = []
        self.metadata_calls: list[str] = []
        self.metadata: dict[str, ModelMetadata | Exception | Any] = {}
        self.default_metadata: ModelMetadata = TEE_METADATA
        self.metadata_delay = 0.0
        self.gateway_result: VerifiedGatewayAttestation | BaseException = gateway()
        self.verify_gateway_calls = 0
        self.runtimes: list[client.NearRuntime] = []

        def inference_factory(api_key: str | None = None, **kwargs: Any) -> _FakeInferenceClient:
            fake = _FakeInferenceClient(api_key, kwargs)
            self.created_inference.append(fake)
            return fake

        def attestation_factory(api_key: str | None = None, **kwargs: Any) -> _FakeAttestationClient:
            fake = _FakeAttestationClient(self, api_key, kwargs)
            self.created_attestation.append(fake)
            return fake

        def pinned_factory(fingerprint: str) -> _FakeHttpClient:
            self.pinned_fingerprints.append(fingerprint)
            fake = _FakeHttpClient()
            self.pinned_clients.append(fake)
            return fake

        async def verify_gateway(
            attestation: object,
            client_binding: object,
            *,
            policy: object = None,
            verifiers: object = None,
        ) -> VerifiedGatewayAttestation:
            self.verify_gateway_calls += 1
            if isinstance(self.gateway_result, BaseException):
                raise self.gateway_result
            return self.gateway_result

        monkeypatch.setattr(client, "InferenceClient", inference_factory)
        monkeypatch.setattr(client, "AttestationClient", attestation_factory)
        monkeypatch.setattr(client, "create_pinned_tls_client", pinned_factory)
        monkeypatch.setattr(client, "verify_gateway_attestation", verify_gateway)

    def make_runtime(self, api_key: str = "test-key", **kwargs: Any) -> client.NearRuntime:
        kwargs.setdefault("environ", {})
        runtime = client.NearRuntime(api_key, **kwargs)
        self.runtimes.append(runtime)
        return runtime

    def close(self) -> None:
        for runtime in self.runtimes:
            asyncio.run(runtime.aclose())
        self.runtimes.clear()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(monkeypatch)


@pytest.fixture(autouse=True)
def _clean_process_runtimes() -> Any:
    client.reset_runtimes()
    yield
    client.reset_runtimes()


def _tee_handler(result: AttestationVerificationResult | None = None) -> Any:
    async def handler(_model: str) -> AttestationVerificationResult:
        return result or attestation_result(model())

    return handler


def _routed_handler() -> Any:
    async def handler(_model: str) -> AttestationVerificationResult:
        return attestation_result()

    return handler


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructs_two_strict_clients(harness: Harness) -> None:
    runtime = harness.make_runtime()

    assert len(harness.created_inference) == 2
    tee, routed = runtime.tee_client, runtime.routed_client

    assert tee.kwargs["e2ee"] is True
    assert tee.kwargs["model_verification"] is policy.TEE_MODEL_VERIFICATION_OPTIONS

    assert routed.kwargs["e2ee"] is False
    assert routed.kwargs["model_verification"] is None

    for sdk_client in (tee, routed):
        assert sdk_client.api_key == "test-key"
        assert sdk_client.kwargs["base_url"] == endpoints.API_BASE
        assert sdk_client.kwargs["signing_algo"] == "ed25519"
        assert sdk_client.kwargs["gateway_verification"] is policy.GATEWAY_VERIFICATION_OPTIONS
        assert sdk_client.kwargs["attestation_cache_time_to_live_ms"] == policy.ATTESTATION_TTL_MS
        assert sdk_client.kwargs["response_cache_time_to_live_ms"] == policy.RESPONSE_TTL_MS


def test_base_url_read_at_construction(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(endpoints, "API_BASE", "https://example.test/v1")
    runtime = harness.make_runtime()
    assert runtime.tee_client.kwargs["base_url"] == "https://example.test/v1"
    assert runtime.routed_client.kwargs["base_url"] == "https://example.test/v1"


def test_request_timeout_values(harness: Harness) -> None:
    runtime = harness.make_runtime()
    timeout = runtime.request_timeout()
    assert timeout.connect == policy.CONNECT_TIMEOUT_SECONDS
    assert timeout.read == policy.READ_TIMEOUT_SECONDS


def test_client_for_rejects_unknown_mode(harness: Harness) -> None:
    runtime = harness.make_runtime()
    assert runtime.client_for("tee") is runtime.tee_client
    assert runtime.client_for("routed") is runtime.routed_client
    with pytest.raises(policy.NearPolicyError) as info:
        runtime.client_for("bogus")  # type: ignore[arg-type]
    assert info.value.code == policy.CODE_UNKNOWN_MODE


# ---------------------------------------------------------------------------
# Mode detection, coalescing, and warm path
# ---------------------------------------------------------------------------


def test_concurrent_cold_requests_coalesce_metadata_and_verify(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler()
        harness.metadata_delay = 0.01

        modes = await asyncio.gather(*(runtime.verify_model("m") for _ in range(10)))

        assert modes == ["tee"] * 10
        assert harness.metadata_calls == ["m"]
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


def test_warm_path_makes_no_network_calls(harness: Harness) -> None:
    clock = _Clock()

    async def scenario() -> None:
        runtime = harness.make_runtime(clock=clock)
        runtime.tee_client.handler = _tee_handler()

        assert await runtime.verify_model("m") == "tee"
        assert harness.metadata_calls == ["m"]
        assert runtime.tee_client.verify_calls == ["m"]

        clock.advance(policy.ATTESTATION_TTL_SECONDS - 1)
        assert await runtime.verify_model("m") == "tee"

        assert harness.metadata_calls == ["m"]
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


def test_expired_verification_revalidates_metadata(harness: Harness) -> None:
    clock = _Clock()

    async def scenario() -> None:
        runtime = harness.make_runtime(clock=clock)
        runtime.tee_client.handler = _tee_handler()
        await runtime.verify_model("m")

        clock.advance(policy.ATTESTATION_TTL_SECONDS)
        assert await runtime.verify_model("m") == "tee"

        # Metadata is revalidated once against the frozen mode.
        assert harness.metadata_calls == ["m", "m"]
        assert runtime.tee_client.verify_calls == ["m", "m"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# TEE verification
# ---------------------------------------------------------------------------


def test_tee_verification_uses_tee_client(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler()

        assert await runtime.verify_model("m") == "tee"
        assert runtime.tee_client.verify_calls == ["m"]
        assert runtime.routed_client.verify_calls == []
        assert runtime.registry.get("m") == "tee"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"tcb_status": "OutOfDate"}, policy.CODE_MODEL_TCB),
        ({"advisory_ids": ("ADV-1",)}, policy.CODE_MODEL_ADVISORIES),
        ({"gpu_evidence": "not_provided"}, policy.CODE_MODEL_GPU),
        ({"signing_public_key": None}, policy.CODE_MODEL_KEY),
    ],
)
def test_tee_bad_report_fails_closed_and_is_negative_cached(
    harness: Harness, overrides: dict[str, object], code: str
) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler(attestation_result(model(**overrides)))

        with pytest.raises(policy.NearPolicyError) as first:
            await runtime.verify_model("m")
        assert first.value.code == code

        with pytest.raises(policy.NearPolicyError) as second:
            await runtime.verify_model("m")
        assert second.value.code == code

        # One metadata fetch and one verify; the retry came from the cache.
        assert harness.metadata_calls == ["m"]
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


def test_tee_missing_reports_fails_closed(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler(attestation_result())

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.verify_model("m")
        assert info.value.code == policy.CODE_MODEL_EMPTY

    asyncio.run(scenario())


def test_gateway_policy_is_enforced(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler(
            attestation_result(model(), gateway_obj=gateway(tls_binding=_tls_binding(kind="none", spki=None)))
        )

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.verify_model("m")
        assert info.value.code == policy.CODE_GATEWAY_TLS

    asyncio.run(scenario())


def test_sdk_verification_error_is_a_policy_error(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()

        async def failing(_model: str) -> AttestationVerificationResult:
            raise VerificationError(VerificationFailure(code="binding.nonce_mismatch"))

        runtime.tee_client.handler = failing

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.verify_model("m")
        assert info.value.code == client.CODE_ATTESTATION_FAILED

        # Negative-cached: a retry does not call verify again.
        with pytest.raises(policy.NearPolicyError):
            await runtime.verify_model("m")
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Routed verification
# ---------------------------------------------------------------------------


def test_routed_verification_uses_routed_client_and_accepts_gateway(harness: Harness) -> None:
    harness.default_metadata = ROUTED_METADATA

    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.routed_client.handler = _routed_handler()

        assert await runtime.verify_model("m") == "routed"
        assert runtime.routed_client.verify_calls == ["m"]
        assert runtime.tee_client.verify_calls == []

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Mode freezing
# ---------------------------------------------------------------------------


def test_forced_tee_model_with_routed_metadata_fails_closed(harness: Harness) -> None:
    harness.default_metadata = ROUTED_METADATA

    async def scenario() -> None:
        runtime = harness.make_runtime(environ={"NEARAI_TEE_MODELS": "m"})
        runtime.routed_client.handler = _routed_handler()

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.verify_model("m")
        assert info.value.code == CODE_FORCED_TEE_MISMATCH
        assert runtime.routed_client.verify_calls == []

        with pytest.raises(policy.NearPolicyError):
            await runtime.verify_model("m")
        assert harness.metadata_calls == ["m"]

    asyncio.run(scenario())


def test_tee_to_routed_metadata_change_fails_closed(harness: Harness) -> None:
    clock = _Clock()

    async def scenario() -> None:
        runtime = harness.make_runtime(clock=clock)
        runtime.tee_client.handler = _tee_handler()
        assert await runtime.verify_model("m") == "tee"

        # A cold attempt revalidates metadata; the registry rejects the change.
        clock.advance(policy.ATTESTATION_TTL_SECONDS)
        harness.metadata["m"] = ROUTED_METADATA
        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.verify_model("m")
        assert info.value.code == CODE_MODE_CHANGED
        assert runtime.registry.get("m") == "tee"
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


def test_observe_mode_rejects_downgrade_without_verifying(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler()
        assert await runtime.verify_model("m") == "tee"

        harness.metadata["m"] = ROUTED_METADATA
        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.observe_mode("m")
        assert info.value.code == CODE_MODE_CHANGED
        assert runtime.routed_client.verify_calls == []

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Negative cache and timeouts
# ---------------------------------------------------------------------------


def test_negative_cache_suppresses_failures_then_expires(harness: Harness) -> None:
    clock = _Clock()
    calls = {"n": 0}

    async def flaky(_model: str) -> AttestationVerificationResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApiError(ApiFailure(code="api.server_error", retryable=True))
        return attestation_result(model())

    async def scenario() -> None:
        runtime = harness.make_runtime(clock=clock)
        runtime.tee_client.handler = flaky

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.verify_model("m")
        assert info.value.code == client.CODE_UNAVAILABLE_ATTESTATION

        with pytest.raises(client.NearUnavailableError):
            await runtime.verify_model("m")
        assert runtime.tee_client.verify_calls == ["m"]
        assert harness.metadata_calls == ["m"]

        clock.advance(policy.FAILURE_TTL_SECONDS + 1)
        assert await runtime.verify_model("m") == "tee"
        assert runtime.tee_client.verify_calls == ["m", "m"]

    asyncio.run(scenario())


def test_hard_timeout_maps_to_clean_unavailable_and_caches(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "HARD_REQUEST_TIMEOUT_SECONDS", 0.02)

    async def slow(_model: str) -> AttestationVerificationResult:
        await asyncio.sleep(0.2)
        return attestation_result(model())

    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = slow

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.verify_model("m")
        assert info.value.code == client.CODE_UNAVAILABLE_TIMEOUT
        assert not isinstance(info.value, TimeoutError)

        with pytest.raises(client.NearUnavailableError) as second:
            await runtime.verify_model("m")
        assert second.value.code == client.CODE_UNAVAILABLE_TIMEOUT
        assert runtime.tee_client.verify_calls == ["m"]

    asyncio.run(scenario())


def test_hard_timeout_context_manager_translates(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = harness.make_runtime()
    monkeypatch.setattr(client, "HARD_REQUEST_TIMEOUT_SECONDS", 0.01)

    async def scenario() -> None:
        with pytest.raises(client.NearUnavailableError) as info:
            async with runtime.hard_timeout():
                await asyncio.sleep(0.1)
        assert info.value.code == client.CODE_UNAVAILABLE_TIMEOUT

        # An explicit bound also wins.
        async with runtime.hard_timeout(10):
            await asyncio.sleep(0)

        with pytest.raises(client.NearUnavailableError):
            async with runtime.hard_timeout(0.01):
                await asyncio.sleep(0.1)

    asyncio.run(scenario())


def test_metadata_failure_is_unavailable_not_guessed(harness: Harness) -> None:
    harness.default_metadata = ApiError(ApiFailure(code="api.unavailable", retryable=True))

    async def scenario() -> None:
        runtime = harness.make_runtime()

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.verify_model("m")
        assert info.value.code == client.CODE_UNAVAILABLE_MODE_UNKNOWN
        assert runtime.registry.get("m") is None
        assert runtime.tee_client.verify_calls == []

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Prewarm
# ---------------------------------------------------------------------------


def test_prewarm_is_bounded_and_isolates_failures(harness: Harness) -> None:
    state = {"active": 0, "max": 0}

    async def handler(model_id: str) -> AttestationVerificationResult:
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        try:
            await asyncio.sleep(0.01)
            if model_id == "bad":
                raise ApiError(ApiFailure(code="api.server_error", retryable=True))
            return attestation_result(model())
        finally:
            state["active"] -= 1

    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = handler

        results = await runtime.prewarm(["a", "b", "bad", "c", "a"], concurrency=2)

        assert set(results) == {"a", "b", "bad", "c"}
        assert results["a"] is None
        assert results["b"] is None
        assert results["c"] is None
        assert isinstance(results["bad"], client.NearUnavailableError)
        assert state["max"] <= 2

    asyncio.run(scenario())


def test_prewarm_empty_is_a_noop(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        assert await runtime.prewarm([]) == {}
        assert harness.metadata_calls == []
        assert harness.created_attestation == []

    asyncio.run(scenario())


def test_prewarm_total_budget_returns_timeout_entries_instead_of_hanging(harness: Harness) -> None:
    never = asyncio.Event()

    async def hanging(_model: str) -> AttestationVerificationResult:
        await never.wait()
        raise AssertionError("unreachable")

    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = hanging

        started = time.monotonic()
        results = await runtime.prewarm(["a", "b"], concurrency=2, timeout=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert set(results) == {"a", "b"}
        for error in results.values():
            assert isinstance(error, client.NearUnavailableError)
            assert error.code == client.CODE_UNAVAILABLE_TIMEOUT

    asyncio.run(scenario())


def test_prewarm_models_from_env() -> None:
    assert client.prewarm_models_from_env({}) == ()
    assert client.prewarm_models_from_env({client.PREWARM_MODELS_ENV: ""}) == ()
    assert client.prewarm_models_from_env({client.PREWARM_MODELS_ENV: " a, b ,a, , c "}) == (
        "a",
        "b",
        "c",
    )
    assert client.prewarm_models_from_env(
        {
            client.PREWARM_MODELS_ENV: (
                "nearai/vendor/a, nearai_e2ee/vendor/b, vendor/a, "
                "nearai_e2ee/z-ai/glm"
            )
        }
    ) == ("vendor/a", "vendor/b", "z-ai/glm")


def test_prewarm_from_env_uses_runtime_environ(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime(environ={client.PREWARM_MODELS_ENV: "a,b"})
        runtime.tee_client.handler = _tee_handler()
        results = await runtime.prewarm_from_env(concurrency=2)
        assert results == {"a": None, "b": None}

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Pinned metadata bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_pins_verified_gateway_then_reuses_client(harness: Harness) -> None:
    expected_fingerprint = harness.gateway_result.tls_binding.spki_fingerprint  # type: ignore[union-attr]

    async def scenario() -> None:
        runtime = harness.make_runtime()

        assert await runtime.observe_mode("m") == "tee"
        assert await runtime.observe_mode("m2") == "tee"

        # Exactly one bootstrap (unauthenticated) and one pinned client.
        assert len(harness.pinned_fingerprints) == 1
        assert harness.pinned_fingerprints == [expected_fingerprint]
        assert harness.verify_gateway_calls == 1
        assert len(harness.created_attestation) == 2

        bootstrap, metadata_client = harness.created_attestation
        assert bootstrap.api_key is None  # never send the key before verification
        assert bootstrap.kwargs.get("http_client") is None
        assert bootstrap.kwargs["base_url"] == endpoints.API_BASE
        assert bootstrap.fetch_gateway_calls == 1

        assert metadata_client.api_key == "test-key"
        assert metadata_client.kwargs["base_url"] == endpoints.API_BASE
        assert metadata_client.kwargs["http_client"] is harness.pinned_clients[0]
        assert harness.metadata_calls == ["m", "m2"]

    asyncio.run(scenario())


def test_bootstrap_awaits_gateway_verification(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the SDK verifier is a coroutine and must be awaited.

    A missing ``await`` leaves a coroutine object behind, so
    ``assert_gateway_verified`` raises ``AttributeError`` and bootstrap aborts
    before any SPKI is pinned.  The flag is only set if the verifier body
    actually ran, and the returned fingerprint proves the awaited result -- not
    the coroutine object -- drove the pinning.
    """
    fingerprint = "ef" * 32
    completed: list[str] = []

    async def async_verify(
        _attestation: object,
        _client_binding: object,
        *,
        policy: object = None,
        verifiers: object = None,
    ) -> VerifiedGatewayAttestation:
        await asyncio.sleep(0)
        completed.append("awaited")
        return gateway(tls_binding=_tls_binding(spki=fingerprint))

    monkeypatch.setattr(client, "verify_gateway_attestation", async_verify)

    async def scenario() -> None:
        runtime = harness.make_runtime()
        metadata_client = await runtime._ensure_metadata_client()

        assert completed == ["awaited"]
        assert harness.pinned_fingerprints == [fingerprint]
        assert metadata_client.api_key == "test-key"
        assert metadata_client.kwargs["http_client"] is harness.pinned_clients[0]

    asyncio.run(scenario())


def test_pinned_metadata_client_has_the_fixed_request_timeout(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        await runtime._ensure_metadata_client()

        timeout = harness.pinned_clients[0].timeout
        assert timeout.connect == policy.CONNECT_TIMEOUT_SECONDS
        assert timeout.read == policy.READ_TIMEOUT_SECONDS

    asyncio.run(scenario())


def test_bootstrap_coalesces_concurrent_models(harness: Harness) -> None:
    harness.metadata_delay = 0.01

    async def scenario() -> None:
        runtime = harness.make_runtime()
        await asyncio.gather(*(runtime.observe_mode(model_id) for model_id in ("a", "b", "c")))

        assert harness.verify_gateway_calls == 1
        assert len(harness.pinned_fingerprints) == 1
        assert len(harness.created_attestation) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("gateway_result", "expected_code"),
    [
        (gateway(tls_binding=_tls_binding(kind="none", spki=None)), policy.CODE_GATEWAY_TLS),
        (gateway(tcb_status="OutOfDate"), policy.CODE_GATEWAY_TCB),
        (gateway(advisory_ids=("ADV-1",)), policy.CODE_GATEWAY_ADVISORIES),
        (gateway(tls_binding=_tls_binding(spki=None)), policy.CODE_GATEWAY_SPKI),
        (
            VerificationError(VerificationFailure(code="binding.spki_fingerprint_mismatch")),
            client.CODE_ATTESTATION_FAILED,
        ),
        (
            VerificationError(VerificationFailure(code=policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH)),
            policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH,
        ),
    ],
)
def test_bootstrap_policy_failures_are_502_and_send_no_key(
    harness: Harness, gateway_result: object, expected_code: str
) -> None:
    harness.gateway_result = gateway_result  # type: ignore[assignment]

    async def scenario() -> None:
        runtime = harness.make_runtime()

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.observe_mode("m")
        assert info.value.code == expected_code

        assert harness.pinned_fingerprints == []
        # Only the unauthenticated bootstrap client exists.
        assert len(harness.created_attestation) == 1
        assert harness.created_attestation[0].api_key is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "gateway_result",
    [
        ApiError(ApiFailure(code="api.unavailable", retryable=True)),
        httpx.ConnectError("connection refused"),
        OSError("socket failure"),
    ],
)
def test_bootstrap_transport_failures_are_503_and_send_no_key(
    harness: Harness, gateway_result: object
) -> None:
    harness.gateway_result = gateway_result  # type: ignore[assignment]

    async def scenario() -> None:
        runtime = harness.make_runtime()

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.observe_mode("m")
        assert info.value.code in {
            client.CODE_UNAVAILABLE_ATTESTATION,
            client.CODE_UNAVAILABLE_TIMEOUT,
        }

        assert harness.pinned_fingerprints == []
        assert len(harness.created_attestation) == 1
        assert harness.created_attestation[0].api_key is None

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Stale pinned metadata client after Gateway SPKI/TLS rotation
# ---------------------------------------------------------------------------


def test_metadata_verification_error_rebootstraps_once_with_a_new_pin(harness: Harness) -> None:
    old_fingerprint = harness.gateway_result.tls_binding.spki_fingerprint  # type: ignore[union-attr]
    new_fingerprint = "ef" * 32
    calls = {"n": 0}

    def metadata_once() -> ModelMetadata:
        calls["n"] += 1
        if calls["n"] == 1:
            # The rotation is visible when the runtime re-bootstraps.
            harness.gateway_result = gateway(tls_binding=_tls_binding(spki=new_fingerprint))
            raise VerificationError(VerificationFailure(code="tls.spki_fingerprint_mismatch"))
        return TEE_METADATA

    harness.metadata["m"] = metadata_once

    async def scenario() -> None:
        runtime = harness.make_runtime()

        assert await runtime.observe_mode("m") == "tee"

        assert harness.verify_gateway_calls == 2
        assert harness.pinned_fingerprints == [old_fingerprint, new_fingerprint]
        assert harness.pinned_clients[0].close_calls == 1
        assert len(harness.created_attestation) == 4

        _, first_metadata, _, second_metadata = harness.created_attestation
        assert first_metadata.kwargs["http_client"] is harness.pinned_clients[0]
        assert second_metadata.kwargs["http_client"] is harness.pinned_clients[1]
        assert harness.metadata_calls == ["m", "m"]

    asyncio.run(scenario())


def test_second_metadata_verification_failure_is_503_without_looping(harness: Harness) -> None:
    harness.metadata["m"] = VerificationError(VerificationFailure(code="tls.spki_fingerprint_mismatch"))

    async def scenario() -> None:
        runtime = harness.make_runtime()

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.observe_mode("m")
        assert info.value.code == client.CODE_UNAVAILABLE_MODE_UNKNOWN

        # Exactly one re-bootstrap and one retry; no infinite loop.
        assert harness.verify_gateway_calls == 2
        assert harness.pinned_fingerprints == ["ab" * 32, "ab" * 32]
        assert harness.metadata_calls == ["m", "m"]
        assert len(harness.created_attestation) == 4
        assert harness.pinned_clients[0].close_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code",
    [policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH, policy.CODE_MEASUREMENTS_MODEL_MISMATCH],
)
def test_metadata_measurement_mismatch_is_502_without_rebootstrap(
    harness: Harness, code: str
) -> None:
    harness.metadata["m"] = VerificationError(VerificationFailure(code=code))

    async def scenario() -> None:
        runtime = harness.make_runtime()

        with pytest.raises(policy.NearPolicyError) as info:
            await runtime.observe_mode("m")
        assert info.value.code == code

        # A pin mismatch is a policy failure, not a stale TLS pin.
        assert harness.verify_gateway_calls == 1
        assert len(harness.pinned_fingerprints) == 1
        assert harness.metadata_calls == ["m"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Process-wide registry and lifecycle
# ---------------------------------------------------------------------------


def test_get_runtime_is_singleton_per_key(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(client.API_KEY_ENV, raising=False)

    first = client.get_runtime("key-a")
    assert client.get_runtime("key-a") is first
    assert client.get_runtime("key-a ") is first
    assert client.get_runtime("key-b") is not first
    assert len(harness.created_inference) == 4


def test_get_runtime_resolves_env_and_rejects_missing_key(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(client.API_KEY_ENV, raising=False)
    with pytest.raises(client.NearUnavailableError) as info:
        client.get_runtime()
    assert info.value.code == client.CODE_UNAVAILABLE_API_KEY
    assert "NEARAI_API_KEY" not in str(info.value)

    monkeypatch.setenv(client.API_KEY_ENV, "env-key")
    runtime = client.get_runtime()
    assert runtime is client.get_runtime("env-key")


def test_aclose_closes_everything_and_is_idempotent(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        await runtime.observe_mode("m")

        tee, routed = runtime.tee_client, runtime.routed_client
        pinned = harness.pinned_clients[0]

        await runtime.aclose()
        assert tee.close_calls == 1
        assert routed.close_calls == 1
        assert pinned.close_calls == 1

        await runtime.aclose()
        assert tee.close_calls == 1
        assert routed.close_calls == 1
        assert pinned.close_calls == 1

        with pytest.raises(client.NearUnavailableError) as info:
            await runtime.verify_model("m")
        assert info.value.code == client.CODE_UNAVAILABLE_CLOSED

    asyncio.run(scenario())


def test_close_runtimes_clears_registry(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = client.get_runtime("key-a")
        agent = runtime.tee_client
        await client.close_runtimes()
        assert agent.close_calls == 1
        assert client.get_runtime("key-a") is not runtime

    asyncio.run(scenario())


def test_reset_caches_reevaluates(harness: Harness) -> None:
    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = _tee_handler()
        await runtime.verify_model("m")

        runtime.reset_caches()
        assert await runtime.verify_model("m") == "tee"
        assert runtime.tee_client.verify_calls == ["m", "m"]

    asyncio.run(scenario())


def test_no_runtime_created_at_import() -> None:
    # The module is already imported by this test collection; a clean reset must
    # leave the map empty and get_runtime must not have run at import time.
    client.reset_runtimes()
    assert client._RUNTIMES == {}


def test_timeout_is_not_confused_with_httpx_timeout(harness: Harness) -> None:
    # ``request_timeout`` is a plain ``httpx.Timeout`` contract, distinct from
    # the hard wall-clock bound.
    runtime = harness.make_runtime()
    assert isinstance(runtime.request_timeout(), httpx.Timeout)
