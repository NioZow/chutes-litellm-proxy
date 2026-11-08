"""Offline streaming tests for the NEAR AI provider (``near/provider.py``).

Both streaming modes are exercised against fake SDK streams:

* ``verified`` buffers every chunk, consumes the SSE stream to ``[DONE]``/EOF,
  verifies the receipt, and only then replays the chunks.
* ``tail`` yields content live but withholds the terminal ``finish_reason``
  chunk until verification succeeds.

In both modes a receipt/policy failure raises a 502 and the terminal chunk is
never emitted.  The SDK stream is always closed, and no network is involved.
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

from chutes_litellm.near import provider

MODEL = "test-model"
COMPLETION_ID = "cmpl-stream-1"


# ---------------------------------------------------------------------------
# Receipt factories (real SDK dataclasses)
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


# ---------------------------------------------------------------------------
# SSE and fakes
# ---------------------------------------------------------------------------


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    finish: str | None = None,
    usage: dict | None = None,
    choices: bool = True,
    completion_id: str = COMPLETION_ID,
) -> dict:
    event: dict = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
    }
    if choices:
        delta: dict = {}
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
        event["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    else:
        event["choices"] = []
    if usage is not None:
        event["usage"] = usage
    return event


class RecordingStream(httpx.AsyncByteStream):
    """Yield SSE bytes in small pieces and record whether EOF was reached."""

    def __init__(self, payload: bytes, *, chunk_size: int = 9) -> None:
        self._chunks = [payload[index : index + chunk_size] for index in range(0, len(payload), chunk_size)]
        self.fully_consumed = False

    async def __aiter__(self):
        for piece in self._chunks:
            yield piece
        self.fully_consumed = True


class TrackingResponse(httpx.Response):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


def sse_response(*parts: str) -> tuple[TrackingResponse, RecordingStream]:
    payload = "".join(parts).encode()
    stream = RecordingStream(payload)
    response = TrackingResponse(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )
    return response, stream


class FakeSDK:
    def __init__(
        self,
        *,
        response: httpx.Response,
        verify_results: list | None = None,
        verify_errors: list[BaseException] | None = None,
    ) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []
        self.verify_ids: list[str] = []
        self.calls: list[str] = []
        self.verified = False
        self._results = list(verify_results or [])
        self._errors = list(verify_errors or [])

    async def send(self, request: httpx.Request) -> httpx.Response:
        self.calls.append("send")
        self.requests.append(request)
        return self.response

    async def verify_response(self, completion_id: str):
        self.calls.append("verify")
        self.verify_ids.append(completion_id)
        if self._errors:
            raise self._errors.pop(0)
        self.verified = True
        if self._results:
            return self._results.pop(0)
        return tee_receipt(completion_id)


class FakeRuntime:
    def __init__(self, *, mode: str = "tee", sdk: FakeSDK) -> None:
        self.mode = mode
        self.sdk = sdk
        self.verified_models: list[str] = []
        self.hard_timeouts = 0

    async def verify_model(self, model: str) -> str:
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


def _astream(p: provider.NearCustomProvider, **overrides):
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
    return p.astreaming(**kwargs)


def _collect_with_state(agen, sdk: FakeSDK) -> list[tuple[object, bool]]:
    """Collect chunks paired with ``sdk.verified`` at yield time."""

    async def run():
        seen: list[tuple[object, bool]] = []
        async for chunk in agen:
            seen.append((chunk, sdk.verified))
        return seen

    return asyncio.run(run())


def _contents(chunks: list) -> str:
    text = ""
    for item in chunks:
        c = item[0] if isinstance(item, tuple) else item
        if c.choices and c.choices[0].delta is not None and c.choices[0].delta.content:
            text += c.choices[0].delta.content
    return text


def _finish_reason(item) -> str | None:
    c = item[0] if isinstance(item, tuple) else item
    return c.choices[0].finish_reason if c.choices else None


# ---------------------------------------------------------------------------
# verified mode
# ---------------------------------------------------------------------------


def test_verified_yields_nothing_before_verification(monkeypatch):
    response, stream = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(reasoning="why")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4})),
        sse_event(chunk(choices=False, usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4})),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)

    # Every chunk was released only after verification, in order.
    assert all(verified for _, verified in seen)
    assert _contents(seen) == "ab"
    assert _finish_reason(seen[-1]) == "stop"
    assert sdk.calls == ["send", "verify"]
    assert sdk.verify_ids == [COMPLETION_ID]
    assert stream.fully_consumed is True  # consumed through [DONE] to EOF
    assert response.close_calls >= 1


def test_verified_failure_yields_zero_chunks_and_raises_502(monkeypatch):
    response, stream = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(
        response=response,
        verify_errors=[VerificationError(VerificationFailure(code="signature.invalid"))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.status_code == 502
    assert provider.CODE_RECEIPT_VERIFICATION_FAILED in str(excinfo.value)
    assert collected == []
    assert sdk.verify_ids == [COMPLETION_ID]
    assert len(sdk.requests) == 1  # inference was not resent
    assert stream.fully_consumed is True
    assert response.close_calls >= 1


def test_verified_accepts_routed_gateway_receipt(monkeypatch):
    response, _ = sse_response(
        sse_event(chunk(content="routed")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[gateway_receipt()])
    runtime = FakeRuntime(mode="routed", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)
    assert _contents(seen) == "routed"
    assert _finish_reason(seen[-1]) == "stop"


def test_verified_rejects_wrong_receipt_kind(monkeypatch):
    response, _ = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[gateway_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())
    assert excinfo.value.status_code == 502
    assert collected == []


# ---------------------------------------------------------------------------
# tail mode
# ---------------------------------------------------------------------------


@pytest.fixture()
def tail_mode(monkeypatch):
    monkeypatch.setenv(provider.STREAM_MODE_ENV, "tail")


def test_tail_streams_content_before_verification_but_holds_terminal(monkeypatch, tail_mode):
    response, stream = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)

    assert _contents(seen) == "ab"
    # Content was released before verification; the terminal chunk only after.
    assert [verified for _, verified in seen] == [False, False, True]
    assert _finish_reason(seen[-1]) == "stop"
    assert sdk.calls == ["send", "verify"]
    assert stream.fully_consumed is True  # [DONE]/EOF reached before verify
    assert response.close_calls >= 1


def test_tail_multiple_terminal_chunks_hold_all_until_verification(monkeypatch, tail_mode):
    response, _ = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(finish="stop")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="length")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)

    # Live content stays live; no terminal chunk is released before verification.
    assert _contents(seen) == "ab"
    terminals = [item for item in seen if _finish_reason(item) is not None]
    assert terminals and all(item[1] for item in terminals)

    # Only the last terminal chunk is emitted, never the superseded first one.
    assert len(terminals) == 1
    assert _finish_reason(terminals[0]) == "length"
    assert _finish_reason(seen[-1]) == "length"
    assert sdk.calls == ["send", "verify"]


def test_tail_multiple_terminal_chunks_failure_withholds_all(monkeypatch, tail_mode):
    response, stream = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(finish="stop")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="length")),
        sse_done(),
    )
    sdk = FakeSDK(
        response=response,
        verify_errors=[VerificationError(VerificationFailure(code="signature.payload_mismatch"))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.status_code == 502
    assert provider.CODE_RECEIPT_VERIFICATION_FAILED in str(excinfo.value)
    assert _contents(collected) == "ab"  # live content may already be out
    assert all(_finish_reason(item) is None for item in collected)  # every terminal withheld
    assert stream.fully_consumed is True
    assert len(sdk.requests) == 1


def test_tail_failure_keeps_content_but_never_emits_terminal(monkeypatch, tail_mode):
    response, stream = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(content="b")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(
        response=response,
        verify_errors=[VerificationError(VerificationFailure(code="signature.payload_mismatch"))],
    )
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.status_code == 502
    assert _contents(collected) == "ab"  # live content may already be out
    assert all(_finish_reason(item) is None for item in collected)  # terminal withheld
    assert stream.fully_consumed is True
    assert response.close_calls >= 1
    assert len(sdk.requests) == 1


def test_tail_accepts_routed_gateway_receipt(monkeypatch, tail_mode):
    response, _ = sse_response(
        sse_event(chunk(content="routed")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[gateway_receipt()])
    runtime = FakeRuntime(mode="routed", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)
    assert _contents(seen) == "routed"
    assert _finish_reason(seen[-1]) == "stop"


def test_tail_retryable_receipt_errors_retry_without_resending(monkeypatch, tail_mode):
    response, _ = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    errors = [ApiError(ApiFailure(code="api.completion_signature_unavailable", retryable=True)) for _ in range(3)]
    sdk = FakeSDK(response=response, verify_errors=errors)
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider.asyncio, "sleep", fake_sleep)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.status_code == 502
    assert len(sdk.verify_ids) == 3  # bounded retries
    assert [call for call in sdk.calls if call == "send"] == ["send"]  # never resent
    assert sleeps == [0.1, 0.2]
    assert _contents(collected) == "a"
    assert all(_finish_reason(item) is None for item in collected)  # terminal withheld


def test_tail_rejects_wrong_receipt_kind(monkeypatch, tail_mode):
    response, _ = sse_response(
        sse_event(chunk(content="a")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="routed", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())
    assert excinfo.value.status_code == 502
    assert _contents(collected) == "a"
    assert all(_finish_reason(item) is None for item in collected)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_malformed_and_unknown_sse_lines_are_ignored(monkeypatch):
    response, stream = sse_response(
        ": keep-alive comment\n\n",
        "event: message\n\n",
        "data: not-json\n\n",
        'data: {"broken": \n\n',
        sse_event(chunk(content="ok")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)
    assert _contents(seen) == "ok"
    assert _finish_reason(seen[-1]) == "stop"
    assert stream.fully_consumed is True


def test_empty_choice_usage_chunk_is_handled(monkeypatch):
    response, _ = sse_response(
        sse_event(chunk(choices=False, usage={"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2})),
        sse_event(chunk(content="x")),
        sse_event(chunk(finish="stop")),
        sse_done(),
    )
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    seen = _collect_with_state(_astream(provider.NearCustomProvider()), sdk)
    assert _contents(seen) == "x"


def test_unknown_stream_mode_fails_closed(monkeypatch):
    monkeypatch.setenv(provider.STREAM_MODE_ENV, "unsafe")
    response, _ = sse_response(sse_event(chunk(content="a")), sse_event(chunk(finish="stop")), sse_done())
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.status_code == 502
    assert collected == []
    assert sdk.calls == []  # no inference, no verification


def test_missing_completion_id_fails_closed(monkeypatch):
    no_id_content = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}],
    }
    no_id_finish = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    response, _ = sse_response(sse_event(no_id_content), sse_event(no_id_finish), sse_done())
    sdk = FakeSDK(response=response, verify_results=[tee_receipt()])
    runtime = FakeRuntime(mode="tee", sdk=sdk)
    monkeypatch.setattr(provider, "get_runtime", lambda _key: runtime)

    collected: list = []

    async def run():
        async for item in _astream(provider.NearCustomProvider()):
            collected.append(item)

    with pytest.raises(litellm.APIError) as excinfo:
        asyncio.run(run())
    assert excinfo.value.status_code == 502
    assert collected == []
