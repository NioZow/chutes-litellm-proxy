"""Offline tests for the NEAR AI LiteLLM provider (``near/provider.py``).

The SDK and the runtime are faked at the provider seam, so no network, TLS,
DCAP, NRAS, or API key is involved.  Receipts use the real SDK dataclasses so
the provider's assertions run against the same shapes the SDK returns.

Covered: registration, payload construction, API-key resolution, TEE/routed
receipt policy, HTTP error mapping, safe policy/unavailable mapping, bounded
receipt retries that never resend inference, and verify-before-translate.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import litellm
import pytest
from litellm.types.utils import ModelResponse
from nearai_inference_sdk import (
    ApiError,
    ApiFailure,
    CompletionSignature,
    GatewayTlsBinding,
    MeasuredDeployment,
    RuntimeMeasurements,
    SigningIdentity,
    VerificationError,
    VerificationFailure,
    VerifiedGatewayAttestation,
    VerifiedGatewayCompletionResult,
    VerifiedModelAttestation,
    VerifiedModelCompletionResult,
)

from chutes_litellm.custom_provider import _SUPPORTED_EXTRA_PARAMS
from chutes_litellm.near import endpoints, policy, provider
from chutes_litellm.near.client import CODE_UNAVAILABLE_TRANSPORT, NearUnavailableError

MODEL = "test-model"
COMPLETION_ID = "cmpl-1"


# ---------------------------------------------------------------------------
# Real SDK object factories
# ---------------------------------------------------------------------------


def _signer() -> SigningIdentity:
    return SigningIdentity(signing_algo="ed25519", signing_address="0xsigner")


def _deployment() -> MeasuredDeployment:
    return MeasuredDeployment(app_compose="app-compose", runtime_measurements=RuntimeMeasurements())


def _gateway() -> VerifiedGatewayAttestation:
    return VerifiedGatewayAttestation(
        signer=_signer(),
        tcb_status="UpToDate",
        advisory_ids=(),
        deployment=_deployment(),
        deployment_provenance="not_checked",
        tls_binding=GatewayTlsBinding(kind="attested", spki_fingerprint="ab" * 32),
    )


def _model_report() -> VerifiedModelAttestation:
    return VerifiedModelAttestation(
        signer=_signer(),
        tcb_status="UpToDate",
        advisory_ids=(),
        deployment=_deployment(),
        deployment_provenance="not_checked",
        gpu_evidence="verified",
        signing_public_key="cd" * 32,
    )


def _signature(kind: str) -> CompletionSignature:
    return CompletionSignature(
        kind=kind,  # type: ignore[arg-type]
        signer=_signer(),
        signed_text="signed-text",
        signature="sig",
    )


def tee_receipt(completion_id: str = COMPLETION_ID) -> VerifiedModelCompletionResult:
    report = _model_report()
    return VerifiedModelCompletionResult(
        id=completion_id,
        signature=_signature("provider_tee"),
        attestations=(report,),
        attestation=report,
    )


def gateway_receipt(completion_id: str = COMPLETION_ID) -> VerifiedGatewayCompletionResult:
    return VerifiedGatewayCompletionResult(
        id=completion_id,
        signature=_signature("gateway"),
        attestation=_gateway(),
    )


def completion_body(
    *,
    content: str = "hello",
    reasoning: str | None = "thinking",
    tool_call: bool = True,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ]
    return {
        "id": COMPLETION_ID,
        "object": "chat.completion",
        "created": 123,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_call else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TrackingResponse(httpx.Response):
    """An ``httpx.Response`` that counts ``aclose`` calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


class FakeSDK:
    """Records sends/verifications and replays scripted receipts or errors."""

    def __init__(
        self,
        *,
        response: httpx.Response | BaseException,
        verify_results: list | None = None,
        verify_errors: list[BaseException] | None = None,
    ) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []
        self.verify_ids: list[str] = []
        self.calls: list[str] = []
        self._results = list(verify_results or [])
        self._errors = list(verify_errors or [])

    async def send(self, request: httpx.Request) -> httpx.Response:
        self.calls.append("send")
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    async def verify_response(self, completion_id: str):
        self.calls.append("verify")
        self.verify_ids.append(completion_id)
        if self._errors:
            raise self._errors.pop(0)
        if self._results:
            return self._results.pop(0)
        return tee_receipt(completion_id)


class FakeRuntime:
    """Minimal stand-in for :class:`~chutes_litellm.near.client.NearRuntime`."""

    def __init__(
        self,
        *,
        mode: str | BaseException = "tee",
        sdk: FakeSDK | None = None,
    ) -> None:
        self.mode = mode
        self.sdk = sdk if sdk is not None else FakeSDK(response=TrackingResponse(200, json=completion_body()))
        self.verified_models: list[str] = []
        self.hard_timeouts = 0

    async def verify_model(self, model: str) -> str:
        if isinstance(self.mode, BaseException):
            raise self.mode
        self.verified_models.append(model)
        return self.mode

    def client_for(self, mode: str) -> FakeSDK:
        assert mode == self.mode
        return self.sdk

    def request_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(600.0, connect=10.0)

    @asynccontextmanager
    async def hard_timeout(self, seconds: float | None = None):
        self.hard_timeouts += 1
        yield


def _call(p: provider.NearCustomProvider, **overrides):
    """Invoke ``acompletion`` with the LiteLLM CustomLLM call shape."""
    kwargs = {
        "model": f"{provider.PROVIDER}/{MODEL}",
        "messages": [{"role": "user", "content": "prompt-secret"}],
        "api_base": "",
        "custom_prompt_dict": {},
        "model_response": ModelResponse(),
        "print_verbose": None,
        "encoding": None,
        "api_key": "sk-near-secret",
        "logging_obj": None,
        "optional_params": {},
    }
    kwargs.update(overrides)
    return p.acompletion(**kwargs)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_provider_id():
    assert provider.PROVIDER == "nearai_e2ee"
    assert provider.CHAT_PATH == "/v1/chat/completions"


def test_registration_is_idempotent_and_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("NEAR_API_KEY", raising=False)
    provider.register()
    provider.register()

    entries = [entry for entry in litellm.custom_provider_map if entry.get("provider") == provider.PROVIDER]
    assert len(entries) == 1
    assert isinstance(entries[0]["custom_handler"], provider.NearCustomProvider)


def test_supported_params_wrapper_only_applies_to_our_provider():
    provider.register()

    ours = litellm.get_supported_openai_params(model=f"{provider.PROVIDER}/x", custom_llm_provider=provider.PROVIDER)
    assert set(_SUPPORTED_EXTRA_PARAMS) <= set(ours)

    other = litellm.get_supported_openai_params(model="gpt-4o", custom_llm_provider="openai")
    assert not set(_SUPPORTED_EXTRA_PARAMS) <= set(other)

    from litellm.litellm_core_utils import get_supported_openai_params as core

    first = core.get_supported_openai_params
    provider._install_supported_params()
    assert core.get_supported_openai_params is first
    assert getattr(first, "__nearai_supported_params__", False) is True


# ---------------------------------------------------------------------------
# Payload and request shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [provider.PROVIDER, "nearai"])
@pytest.mark.parametrize("suffix", [MODEL, f"org/{MODEL}"])
def test_model_prefix_is_stripped(monkeypatch, prefix, suffix):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()))
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider(), model=f"{prefix}/{suffix}")

    _run(run())

    assert runtime.verified_models == [suffix]
    body = json.loads(sdk.requests[0].content)
    assert body["model"] == suffix


def test_payload_forwards_only_whitelisted_params(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()))
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    optional_params = {
        "temperature": 0.2,
        "top_p": None,
        "max_tokens": 16,
        "extra_body": {"top_k": 7},
        # LiteLLM bookkeeping / secrets that must never reach the wire.
        "api_key": "sk-in-payload",
        "custom_llm_provider": provider.PROVIDER,
        "drop_params": True,
        "stream": True,
    }
    messages = [
        {
            "role": "user",
            "content": "prompt-secret",
            "reasoning_content": "stripped",
            "provider_specific_fields": {"x": 1},
        }
    ]

    async def run():
        return await _call(
            provider.NearCustomProvider(),
            messages=messages,
            optional_params=optional_params,
        )

    _run(run())

    request = sdk.requests[0]
    assert request.method == "POST"
    assert str(request.url) == endpoints.chat_completions_url()
    assert request.headers["content-type"] == "application/json"
    timeout = request.extensions["timeout"]
    assert isinstance(timeout, dict)
    assert {"connect", "read", "write", "pool"} <= set(timeout)

    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 16
    assert body["top_k"] == 7
    for forbidden in ("api_key", "custom_llm_provider", "drop_params", "top_p"):
        assert forbidden not in body
    assert body["messages"] == [{"role": "user", "content": "prompt-secret"}]


def test_request_timeout_extension_is_a_plain_dict():
    """httpcore expects a mapping; an ``httpx.Timeout`` instance is rejected."""
    runtime = FakeRuntime()
    payload = provider._build_payload(MODEL, [{"role": "user", "content": "x"}], {}, stream=False)
    request = provider._build_request(runtime, payload)

    timeout = request.extensions["timeout"]
    assert isinstance(timeout, dict)
    assert {"connect", "read", "write", "pool"} <= set(timeout)
    assert timeout["connect"] == policy.CONNECT_TIMEOUT_SECONDS
    assert timeout["read"] == policy.READ_TIMEOUT_SECONDS


def test_extra_body_cannot_override_reserved_fields(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()))
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    optional_params = {
        "extra_body": {
            "model": "evil-model",
            "messages": [{"role": "user", "content": "injected"}],
            "stream": True,
            "top_k": 7,
        }
    }

    async def run():
        return await _call(provider.NearCustomProvider(), optional_params=optional_params)

    _run(run())

    assert runtime.verified_models == [MODEL]
    body = json.loads(sdk.requests[0].content)
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "prompt-secret"}]
    assert body["stream"] is False
    assert body["top_k"] == 7


def test_build_payload_reasserts_reserved_fields_after_extra_body():
    messages = [{"role": "user", "content": "safe"}]
    payload = provider._build_payload(
        MODEL,
        messages,
        {"extra_body": {"model": "evil", "messages": [{"role": "user", "content": "no"}], "stream": True}},
        stream=False,
    )
    assert payload["model"] == MODEL
    assert payload["messages"] == messages
    assert payload["stream"] is False


def test_enable_thinking_is_forwarded():
    payload = provider._build_payload(
        MODEL,
        [{"role": "user", "content": "x"}],
        {"enable_thinking": True},
        stream=False,
    )
    assert payload["enable_thinking"] is True


def test_api_key_resolution(monkeypatch):
    monkeypatch.setenv("NEAR_API_KEY", "sk-from-env")
    # The canonical environment key wins over an explicit/body key so callers
    # cannot grow the process-wide runtime map.
    assert provider._resolve_api_key("sk-explicit", MODEL) == "sk-from-env"
    assert provider._resolve_api_key("os.environ/NEAR_API_KEY", MODEL) == "sk-from-env"
    assert provider._resolve_api_key(None, MODEL) == "sk-from-env"

    monkeypatch.delenv("NEAR_API_KEY", raising=False)
    assert provider._resolve_api_key("sk-explicit", MODEL) == "sk-explicit"

    with pytest.raises(litellm.AuthenticationError) as excinfo:
        provider._resolve_api_key(None, MODEL)
    assert "NEAR_API_KEY" in str(excinfo.value)
    assert "sk-" not in str(excinfo.value)


def test_env_api_key_wins_over_a_body_key(monkeypatch):
    monkeypatch.setenv("NEAR_API_KEY", "sk-from-env")
    runtime = FakeRuntime(mode="tee", sdk=FakeSDK(response=TrackingResponse(200, json=completion_body())))
    seen: list[str] = []

    def fake_get_runtime(key: str) -> FakeRuntime:
        seen.append(key)
        return runtime

    monkeypatch.setattr(provider, "get_runtime", fake_get_runtime)

    async def run():
        return await _call(provider.NearCustomProvider(), api_key="sk-body")

    _run(run())
    assert seen == ["sk-from-env"]


def test_missing_api_key_uses_authentication_error(monkeypatch):
    monkeypatch.delenv("NEAR_API_KEY", raising=False)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: pytest.fail("runtime must not be built"))

    async def run():
        await _call(provider.NearCustomProvider(), api_key=None)

    with pytest.raises(litellm.AuthenticationError):
        _run(run())


# ---------------------------------------------------------------------------
# Non-streaming receipt policy and translation
# ---------------------------------------------------------------------------


def test_tee_non_streaming_verifies_before_translation(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()), verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    order: list[str] = []
    real_translate = provider._to_model_response

    def spy(*args, **kwargs):
        order.append("translate")
        return real_translate(*args, **kwargs)

    monkeypatch.setattr(provider, "_to_model_response", spy)

    async def run():
        return await _call(provider.NearCustomProvider())

    response = _run(run())

    assert sdk.calls == ["send", "verify"]
    assert order == ["translate"]
    assert sdk.verify_ids == [COMPLETION_ID]

    message = response.choices[0].message
    assert message.content == "hello"
    assert message.reasoning_content == "thinking"
    assert message.tool_calls[0]["id"] == "call_1"
    assert message.tool_calls[0]["function"]["name"] == "get_weather"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.usage.total_tokens == 3
    assert response.id == COMPLETION_ID


def test_tee_gateway_receipt_is_rejected(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()), verify_results=[gateway_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError) as excinfo:
        _run(run())

    assert excinfo.value.status_code == 502
    assert policy.CODE_RECEIPT_KIND in str(excinfo.value)


def test_routed_accepts_gateway_receipt(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()), verify_results=[gateway_receipt()])
    runtime = FakeRuntime(mode="routed", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    response = _run(run())
    assert response.choices[0].message.content == "hello"
    assert sdk.verify_ids == [COMPLETION_ID]


def test_routed_provider_tee_receipt_is_rejected(monkeypatch):
    sdk = FakeSDK(response=TrackingResponse(200, json=completion_body()), verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="routed", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError) as excinfo:
        _run(run())
    assert excinfo.value.status_code == 502
    assert policy.CODE_RECEIPT_KIND in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error_cls"),
    [
        (400, litellm.BadRequestError),
        (401, litellm.AuthenticationError),
        (408, litellm.Timeout),
        (429, litellm.RateLimitError),
        (503, litellm.ServiceUnavailableError),
    ],
)
def test_http_error_status_mapping(monkeypatch, status, error_cls):
    response = TrackingResponse(status, json={"error": {"message": "upstream refused"}})
    sdk = FakeSDK(response=response)
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(error_cls) as excinfo:
        _run(run())
    message = str(excinfo.value)
    assert f"NEAR AI upstream returned HTTP {status}" in message
    assert "upstream refused" not in message
    assert "sk-near-secret" not in message
    assert response.close_calls >= 1
    assert sdk.calls == ["send"]  # no verification for an HTTP error


def test_http_error_forwards_only_an_allowlisted_machine_code(monkeypatch):
    response = TrackingResponse(
        429,
        json={
            "error": {
                "message": "free-form upstream detail with prompt-secret",
                "code": "rate_limit_exceeded",
                "type": "ignored_second_identifier",
            }
        },
    )
    sdk = FakeSDK(response=response)
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.RateLimitError) as excinfo:
        _run(run())
    message = str(excinfo.value)
    assert "NEAR AI upstream returned HTTP 429 (rate_limit_exceeded)" in message
    assert "free-form upstream detail" not in message
    assert "prompt-secret" not in message
    assert "ignored_second_identifier" not in message


def test_http_error_drops_a_non_allowlisted_upstream_code(monkeypatch):
    response = TrackingResponse(
        500,
        json={"error": {"message": "prompt-secret", "code": "not a machine code: <html>"}},
    )
    sdk = FakeSDK(response=response)
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.InternalServerError) as excinfo:
        _run(run())
    message = str(excinfo.value)
    assert "NEAR AI upstream returned HTTP 500" in message
    assert "not a machine code" not in message
    assert "prompt-secret" not in message


# ---------------------------------------------------------------------------
# Safe error mapping
# ---------------------------------------------------------------------------


def test_policy_error_maps_to_502_with_only_the_code(monkeypatch):
    runtime = FakeRuntime(mode=policy.NearPolicyError(policy.CODE_RECEIPT_KIND))
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError) as excinfo:
        _run(run())

    message = str(excinfo.value)
    assert excinfo.value.status_code == 502
    assert policy.CODE_RECEIPT_KIND in message
    assert "sk-near-secret" not in message
    assert "prompt-secret" not in message


def test_unavailable_error_maps_to_503(monkeypatch):
    runtime = FakeRuntime(mode=NearUnavailableError(CODE_UNAVAILABLE_TRANSPORT))
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.ServiceUnavailableError) as excinfo:
        _run(run())
    assert excinfo.value.status_code == 503
    assert CODE_UNAVAILABLE_TRANSPORT in str(excinfo.value)
    assert "sk-near-secret" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Bounded receipt retry
# ---------------------------------------------------------------------------


def test_retryable_receipt_error_retries_without_resending(monkeypatch):
    errors = [
        ApiError(ApiFailure(code="api.transport_failed", retryable=True)),
        ApiError(ApiFailure(code="api.transport_failed", retryable=True)),
    ]
    sdk = FakeSDK(
        response=TrackingResponse(200, json=completion_body()),
        verify_results=[tee_receipt()],
        verify_errors=errors,
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider.asyncio, "sleep", fake_sleep)

    async def run():
        return await _call(provider.NearCustomProvider())

    response = _run(run())

    assert len(sdk.verify_ids) == 3
    assert sdk.calls == ["send", "verify", "verify", "verify"]
    assert len(sdk.requests) == 1  # inference was never resent
    assert sleeps == [0.1, 0.2]
    assert response.choices[0].message.content == "hello"


def test_non_retryable_api_error_is_not_retried(monkeypatch):
    sdk = FakeSDK(
        response=TrackingResponse(200, json=completion_body()),
        verify_errors=[ApiError(ApiFailure(code="api.completion_not_found", retryable=False))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError) as excinfo:
        _run(run())
    assert excinfo.value.status_code == 502
    assert len(sdk.verify_ids) == 1
    assert len(sdk.requests) == 1


def test_verification_error_is_not_retried(monkeypatch):
    sdk = FakeSDK(
        response=TrackingResponse(200, json=completion_body()),
        verify_errors=[VerificationError(VerificationFailure(code="signature.invalid"))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    async def run():
        return await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError) as excinfo:
        _run(run())
    message = str(excinfo.value)
    assert excinfo.value.status_code == 502
    assert provider.CODE_RECEIPT_VERIFICATION_FAILED in message
    assert len(sdk.verify_ids) == 1
    assert len(sdk.requests) == 1


def test_non_streaming_verification_failure_returns_no_content(monkeypatch):
    sdk = FakeSDK(
        response=TrackingResponse(200, json=completion_body()),
        verify_errors=[VerificationError(VerificationFailure(code="signature.payload_mismatch"))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    returned = None

    async def run():
        nonlocal returned
        returned = await _call(provider.NearCustomProvider())

    with pytest.raises(litellm.APIError):
        _run(run())
    assert returned is None
