"""LiteLLM ``CustomLLM`` provider for NEAR AI Cloud with mandatory receipts.

This is the Phase 3 provider described in ``PLAN.md`` §4-§5, §9-§10.  It is a
thin protocol adapter around the audited ``nearai-inference-sdk`` and the
process-wide :class:`~chutes_litellm.near.client.NearRuntime`:

* the model's mode is resolved by the runtime before every request;
* a true TEE model sends through the E2EE client and must return a
  ``provider_tee`` receipt;
* a routed model sends through the non-E2EE client and must return a
  ``gateway`` receipt;
* every response is verified before any content is returned to LiteLLM
  (``verified`` streaming buffers everything; ``tail`` withholds the terminal
  chunk until verification succeeds);
* a signature/verification failure never resends inference, only the receipt
  fetch is retried (bounded, retryable ``ApiError`` only).

The wire model id is the raw NEAR model id: a leading ``nearai_e2ee/`` or
``nearai/`` prefix is stripped.  The endpoint is fixed by
:mod:`chutes_litellm.near.endpoints`; ``api_base`` is ignored.  The API key
comes from ``NEARAI_API_KEY`` when set and non-empty (it is canonical and wins
over a request-level key so callers cannot create extra runtimes), otherwise
from the ``api_key`` argument (including the literal
``os.environ/NEARAI_API_KEY``).  It is never logged.

Environment:

* ``NEARAI_API_KEY`` -- API key (not required at registration time).
* ``NEARAI_STREAM_MODE`` -- ``verified`` (default) or ``tail``.

The proxy uses the async entry points.  The sync ones are a documented
convenience fallback: they run on a private event loop with a private runtime
so SDK clients are never shared across loops, and ``streaming`` buffers and
verifies everything before yielding (it never silently degrades to ``tail``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, ModelResponseStream
from nearai_inference_sdk import ApiError, VerificationError

from chutes_litellm.custom_provider import (
    _OPENAI_PARAMS,
    _SUPPORTED_EXTRA_PARAMS,
    _clean_messages,
    _event_to_stream_chunk,
    _to_model_response,
)
from chutes_litellm.near import endpoints, policy
from chutes_litellm.near.client import (
    API_KEY_ENV,
    CODE_UNAVAILABLE_TRANSPORT,
    NearRuntime,
    NearUnavailableError,
    get_runtime,
)
from chutes_litellm.near.modes import ModelMode
from chutes_litellm.near.policy import NearPolicyError

PROVIDER = "nearai_e2ee"

#: Chat-completions path (documentation/reference only): the absolute URL is
#: always taken from ``endpoints.chat_completions_url()`` at call time.
CHAT_PATH = "/v1/chat/completions"

#: Prefixes accepted on the model name; the wire model is the raw id.
_MODEL_PREFIXES = (PROVIDER, "nearai")

#: Parameters this provider forwards upstream.  ``_OPENAI_PARAMS`` is shared
#: with Chutes; ``enable_thinking`` is advertised through
#: :data:`_SUPPORTED_EXTRA_PARAMS` but is not in the shared whitelist, so NEAR
#: adds it locally without changing the Chutes payload builder.
_NEAR_OPENAI_PARAMS = _OPENAI_PARAMS | {"enable_thinking"}

#: ``extra_body`` is client-controlled.  These keys are always set by this
#: provider before flattening and re-asserted afterwards so a client cannot
#: substitute another model, messages, or streaming mode.
_RESERVED_PAYLOAD_KEYS = frozenset({"model", "messages", "stream"})

#: Allowlisted upstream error identifiers: short machine codes only, never
#: free text from the upstream body.
_UPSTREAM_ERROR_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

#: ``os.environ/NAME`` indirection LiteLLM uses for env-backed credentials.
_ENV_KEY_PREFIX = "os.environ/"

#: Stable, machine-readable codes for failures raised by this module.  They are
#: safe to log and never contain prompts, responses, evidence, or keys.
CODE_RECEIPT_VERIFICATION_FAILED = "near.receipt.verification_failed"
CODE_RESPONSE_INVALID = "near.response.invalid"
CODE_RESPONSE_STATUS = "near.response.unexpected_status"
CODE_COMPLETION_ID_MISSING = "near.receipt.id_missing"
CODE_COMPLETION_ID_MISMATCH = "near.receipt.id_mismatch"

#: Environment variable selecting the streaming mode (fail closed on unknown).
STREAM_MODE_ENV = "NEARAI_STREAM_MODE"

#: Bounded receipt retry: 3 total attempts, small jitter-free backoff.
_RECEIPT_ATTEMPTS = 3
_RECEIPT_BACKOFF_SECONDS = (0.1, 0.2)

#: Error families translated to LiteLLM errors at the entry points.
_MAPPED_ERRORS = (
    NearPolicyError,
    NearUnavailableError,
    ApiError,
    VerificationError,
    httpx.HTTPError,
    TimeoutError,
)


# ---------------------------------------------------------------------------
# Credentials and model ids
# ---------------------------------------------------------------------------


def _clean_model(model: str) -> str:
    """Strip a leading ``nearai_e2ee/`` or ``nearai/`` prefix."""
    if "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in _MODEL_PREFIXES and rest:
            return rest
    return model


def _resolve_api_key(api_key: str | None, model: str) -> str:
    """Resolve the API key without ever logging or echoing it.

    ``NEARAI_API_KEY`` is canonical: when it is set and non-empty it wins over
    an explicit ``api_key`` (including a request-body key LiteLLM forwards),
    so caller-supplied keys cannot create arbitrary process-wide runtimes.
    Otherwise the explicit argument is used, including the literal
    ``os.environ/NEARAI_API_KEY`` form LiteLLM emits.  Raises a LiteLLM
    authentication error with a stable message when no key is available.
    """
    candidate: str | None = api_key
    if isinstance(candidate, str) and candidate.startswith(_ENV_KEY_PREFIX):
        candidate = os.environ.get(candidate[len(_ENV_KEY_PREFIX) :])

    env_candidate = os.environ.get(API_KEY_ENV)
    key = ""
    for value in (env_candidate, candidate):
        if isinstance(value, str) and value.strip():
            key = value.strip()
            break
    if not key:
        raise litellm.AuthenticationError(
            message=f"{API_KEY_ENV} is not set",
            model=model,
            llm_provider=PROVIDER,
        )
    return key


# ---------------------------------------------------------------------------
# Payload and request construction
# ---------------------------------------------------------------------------


def _build_payload(
    model: str,
    messages: list,
    optional_params: dict,
    *,
    stream: bool,
) -> dict:
    """Build the OpenAI chat-completions body for NEAR AI.

    Mirrors :func:`chutes_litellm.custom_provider._build_payload`'s whitelist
    approach but deliberately omits the Chutes thinking/template rewrite: NEAR
    AI receives exactly what the caller asked for.  ``extra_body`` is flattened
    only for non-reserved keys; ``model``, ``messages``, and ``stream`` are
    always re-asserted so a client cannot override the verified model, the
    messages, or the streaming mode.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": _clean_messages(messages),
        "stream": stream,
    }
    for key in _NEAR_OPENAI_PARAMS:
        if key in optional_params and optional_params[key] is not None:
            payload[key] = optional_params[key]

    extra_body = optional_params.get("extra_body")
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            # Reserved fields are provider-controlled and re-asserted below.
            if key in _RESERVED_PAYLOAD_KEYS or value is None:
                continue
            payload[key] = value

    # ``extra_body`` is client-controlled: never let it substitute the model
    # (bypassing mode/TEE checks), the messages, or the streaming mode.
    payload["model"] = model
    payload["messages"] = _clean_messages(messages)
    payload["stream"] = stream
    return payload


def _build_request(runtime: NearRuntime, payload: dict) -> httpx.Request:
    """Build the exact request the SDK expects (POST to chat/completions)."""
    return httpx.Request(
        "POST",
        endpoints.chat_completions_url(),
        json=payload,
        headers={"content-type": "application/json"},
        # httpcore expects a plain mapping, not an ``httpx.Timeout`` instance.
        extensions={"timeout": runtime.request_timeout().as_dict()},
    )


# ---------------------------------------------------------------------------
# HTTP status mapping
# ---------------------------------------------------------------------------


def _upstream_error_identifier(body: dict[str, Any]) -> str | None:
    """Return an allowlisted upstream ``error.code``/``error.type`` or ``None``.

    Only short machine identifiers are accepted; anything free-form (a message,
    a prompt echo, an HTML body) is dropped.
    """
    for key in ("code", "type"):
        value = body.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if _UPSTREAM_ERROR_ID_RE.fullmatch(candidate):
                return candidate
    return None


async def _raise_for_error_async(response: httpx.Response, model: str) -> None:
    """Raise a typed LiteLLM error for a non-success upstream response.

    Only a stable status message and, when present, a short allowlisted
    ``error.code``/``error.type`` are surfaced.  Upstream free text and bodies
    are never forwarded.
    """
    status = response.status_code
    if 200 <= status < 300:
        return

    body: dict[str, Any] = {}
    raw = b""
    try:
        raw = await response.aread()
    except Exception:  # noqa: BLE001 - the status is the actionable signal
        raw = b""
    try:
        parsed = json.loads(raw or b"{}")
        if isinstance(parsed, dict):
            body = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
    except Exception:  # noqa: BLE001 - only the status and allowlisted id matter
        body = {}

    message = f"NEAR AI upstream returned HTTP {status}"
    identifier = _upstream_error_identifier(body)
    if identifier is not None:
        message = f"{message} ({identifier})"

    if status < 400:
        raise litellm.APIError(
            status_code=502,
            message=CODE_RESPONSE_STATUS,
            model=model,
            llm_provider=PROVIDER,
        )

    error_cls: type = litellm.InternalServerError
    if status == 400:
        error_cls = litellm.BadRequestError
    elif status == 401:
        error_cls = litellm.AuthenticationError
    elif status == 403:
        error_cls = litellm.PermissionDeniedError
    elif status == 404:
        error_cls = litellm.NotFoundError
    elif status == 408:
        error_cls = litellm.Timeout
    elif status == 422:
        error_cls = litellm.UnprocessableEntityError
    elif status == 429:
        error_cls = litellm.RateLimitError
    elif status == 503:
        error_cls = litellm.ServiceUnavailableError
    elif 500 <= status < 600:
        error_cls = litellm.InternalServerError

    # Some LiteLLM exception classes dereference ``response.request``; a
    # response created by a fake or a low-level transport may not carry one.
    kwargs: dict[str, Any] = {
        "message": message,
        "model": model,
        "llm_provider": PROVIDER,
    }
    try:
        has_request = response.request is not None
    except RuntimeError:
        has_request = False
    if has_request and error_cls is not litellm.Timeout:
        kwargs["response"] = response
    raise error_cls(**kwargs)


# ---------------------------------------------------------------------------
# Response parsing, receipt verification, and error translation
# ---------------------------------------------------------------------------


def _parse_completion_body(body: bytes) -> dict:
    """Parse a non-streaming completion, requiring a non-empty ``id``."""
    try:
        data = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise NearPolicyError(CODE_RESPONSE_INVALID) from exc
    if not isinstance(data, dict):
        raise NearPolicyError(CODE_RESPONSE_INVALID)
    completion_id = data.get("id")
    if not isinstance(completion_id, str) or not completion_id.strip():
        raise NearPolicyError(CODE_COMPLETION_ID_MISSING)
    return data


async def _verify_with_retry(sdk: Any, completion_id: str) -> Any:
    """Verify a retained response, retrying only retryable receipt failures.

    At most :data:`_RECEIPT_ATTEMPTS` calls to ``verify_response``.  The SDK
    caches the inference bytes, so this never resends the prompt.  A
    :class:`VerificationError` or non-retryable :class:`ApiError` propagates
    immediately.
    """
    for attempt in range(_RECEIPT_ATTEMPTS):
        try:
            return await sdk.verify_response(completion_id)
        except ApiError as exc:
            if not exc.retryable or attempt == _RECEIPT_ATTEMPTS - 1:
                raise
            await asyncio.sleep(_RECEIPT_BACKOFF_SECONDS[min(attempt, len(_RECEIPT_BACKOFF_SECONDS) - 1)])


async def _verify_receipt(sdk: Any, completion_id: str | None, mode: ModelMode) -> None:
    """Verify the receipt and enforce the mode's required signature kind."""
    if not isinstance(completion_id, str) or not completion_id.strip():
        raise NearPolicyError(CODE_COMPLETION_ID_MISSING)
    verified = await _verify_with_retry(sdk, completion_id)
    policy.assert_receipt_kind(verified, mode)


def _map_error(error: BaseException, model: str) -> BaseException:
    """Translate a NEAR AI/SDK failure into a safe LiteLLM exception.

    Policy and receipt failures are 502; availability/transport failures are
    503.  Messages are stable codes or SDK machine codes only -- never request
    messages, response bodies, evidence, or API keys.
    """
    if isinstance(error, NearPolicyError):
        return litellm.APIError(
            status_code=502,
            message=error.code,
            model=model,
            llm_provider=PROVIDER,
        )
    if isinstance(error, NearUnavailableError):
        return litellm.ServiceUnavailableError(
            message=error.code,
            model=model,
            llm_provider=PROVIDER,
        )
    if isinstance(error, VerificationError):
        return litellm.APIError(
            status_code=502,
            message=f"{CODE_RECEIPT_VERIFICATION_FAILED}: {error.failure.code}",
            model=model,
            llm_provider=PROVIDER,
        )
    if isinstance(error, ApiError):
        if error.failure.code == "api.transport_failed":
            return litellm.ServiceUnavailableError(
                message=CODE_UNAVAILABLE_TRANSPORT,
                model=model,
                llm_provider=PROVIDER,
            )
        return litellm.APIError(
            status_code=502,
            message=f"{CODE_RECEIPT_VERIFICATION_FAILED}: {error.failure.code}",
            model=model,
            llm_provider=PROVIDER,
        )
    if isinstance(error, httpx.HTTPError):
        return litellm.ServiceUnavailableError(
            message=CODE_UNAVAILABLE_TRANSPORT,
            model=model,
            llm_provider=PROVIDER,
        )
    # ``TimeoutError`` / ``asyncio.TimeoutError`` (the same class on 3.11+).
    return litellm.ServiceUnavailableError(
        message="near.unavailable.timeout",
        model=model,
        llm_provider=PROVIDER,
    )


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def _completion_id_of(event: dict) -> str | None:
    value = event.get("id")
    return value if isinstance(value, str) and value.strip() else None


def _merge_completion_id(current: str | None, event: dict) -> str | None:
    """Track the completion id and reject a mid-stream change."""
    incoming = _completion_id_of(event)
    if incoming is None:
        return current
    if current is not None and incoming != current:
        raise NearPolicyError(CODE_COMPLETION_ID_MISMATCH)
    return incoming


async def _aiter_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield well-formed ``data:`` JSON events, consuming the stream to EOF."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _is_terminal(chunk: ModelResponseStream) -> bool:
    return bool(chunk.choices and chunk.choices[0].finish_reason)


# ---------------------------------------------------------------------------
# Streaming mode resolution
# ---------------------------------------------------------------------------


def _streaming_mode() -> policy.StreamingMode:
    """Resolve ``NEARAI_STREAM_MODE`` at call time (unknown values fail closed)."""
    return policy.resolve_streaming_mode(os.environ.get(STREAM_MODE_ENV))


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


class NearCustomProvider(CustomLLM):
    """NEAR AI Cloud chat provider with preflight and mandatory receipts.

    The async entry points are what the proxy uses.  Sync entry points run the
    same implementation on a private loop (see :func:`_run_sync`).
    """

    # -- shared async implementation --------------------------------------

    async def _acompletion(
        self,
        runtime: NearRuntime,
        *,
        model: str,
        messages: list,
        optional_params: dict,
        model_response: ModelResponse,
    ) -> ModelResponse:
        wire_model = _clean_model(model)
        async with runtime.hard_timeout():
            mode = await runtime.verify_model(wire_model)
            payload = _build_payload(wire_model, messages, optional_params, stream=False)
            sdk = runtime.client_for(mode)
            response = await sdk.send(_build_request(runtime, payload))
            try:
                await _raise_for_error_async(response, model)
                body = await response.aread()
            finally:
                await response.aclose()

            # Verification happens before translation or return.
            data = _parse_completion_body(body)
            await _verify_receipt(sdk, data["id"], mode)
            return _to_model_response(model, data, model_response)

    async def _astream(
        self,
        runtime: NearRuntime,
        *,
        model: str,
        messages: list,
        optional_params: dict,
    ) -> AsyncIterator[ModelResponseStream]:
        wire_model = _clean_model(model)
        streaming_mode = _streaming_mode()
        if streaming_mode == "verified":
            buffered: list[ModelResponseStream]
            async with runtime.hard_timeout():
                mode = await runtime.verify_model(wire_model)
                payload = _build_payload(wire_model, messages, optional_params, stream=True)
                sdk = runtime.client_for(mode)
                response = await sdk.send(_build_request(runtime, payload))
                try:
                    await _raise_for_error_async(response, model)
                    buffered = []
                    completion_id: str | None = None
                    async for event in _aiter_events(response):
                        completion_id = _merge_completion_id(completion_id, event)
                        chunk = _event_to_stream_chunk(event)
                        if chunk is not None:
                            buffered.append(chunk)
                    await _verify_receipt(sdk, completion_id, mode)
                finally:
                    await response.aclose()
            # Nothing is released until the whole stream was verified.
            for chunk in buffered:
                yield chunk
            return

        # ``tail``: live content, terminal chunk withheld until verification.
        async with runtime.hard_timeout():
            mode = await runtime.verify_model(wire_model)
            payload = _build_payload(wire_model, messages, optional_params, stream=True)
            sdk = runtime.client_for(mode)
            response = await sdk.send(_build_request(runtime, payload))
            try:
                await _raise_for_error_async(response, model)
                completion_id = None
                held: ModelResponseStream | None = None
                async for event in _aiter_events(response):
                    completion_id = _merge_completion_id(completion_id, event)
                    chunk = _event_to_stream_chunk(event)
                    if chunk is None:
                        continue
                    if _is_terminal(chunk):
                        # A well-formed stream has one terminal chunk.  If the
                        # provider sends several, keep only the last one and
                        # never release an earlier terminal chunk: every
                        # terminal chunk is withheld until verification.
                        held = chunk
                    else:
                        yield chunk
                await _verify_receipt(sdk, completion_id, mode)
                if held is not None:
                    yield held
            finally:
                await response.aclose()

    # -- async entry points ------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key: str | None,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        try:
            runtime = get_runtime(_resolve_api_key(api_key, model))
            return await self._acompletion(
                runtime,
                model=model,
                messages=messages,
                optional_params=optional_params or {},
                model_response=model_response,
            )
        except _MAPPED_ERRORS as exc:
            raise _map_error(exc, model) from exc

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key: str | None,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> AsyncIterator[ModelResponseStream]:
        try:
            runtime = get_runtime(_resolve_api_key(api_key, model))
            async for chunk in self._astream(
                runtime,
                model=model,
                messages=messages,
                optional_params=optional_params or {},
            ):
                yield chunk
        except _MAPPED_ERRORS as exc:
            raise _map_error(exc, model) from exc

    # -- sync fallback -----------------------------------------------------

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key: str | None,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        try:
            return _run_sync(
                self._sync_acompletion(
                    api_key,
                    model=model,
                    messages=messages,
                    optional_params=optional_params or {},
                    model_response=model_response,
                )
            )
        except _MAPPED_ERRORS as exc:
            raise _map_error(exc, model) from exc

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key: str | None,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> Iterator[ModelResponseStream]:
        try:
            chunks = _run_sync(
                self._sync_collect_stream(
                    api_key,
                    model=model,
                    messages=messages,
                    optional_params=optional_params or {},
                )
            )
        except _MAPPED_ERRORS as exc:
            raise _map_error(exc, model) from exc
        yield from chunks

    # -- sync helpers ------------------------------------------------------

    async def _sync_acompletion(
        self,
        api_key: str | None,
        *,
        model: str,
        messages: list,
        optional_params: dict,
        model_response: ModelResponse,
    ) -> ModelResponse:
        runtime = _sync_runtime(_resolve_api_key(api_key, model))
        return await self._acompletion(
            runtime,
            model=model,
            messages=messages,
            optional_params=optional_params,
            model_response=model_response,
        )

    async def _sync_collect_stream(
        self,
        api_key: str | None,
        *,
        model: str,
        messages: list,
        optional_params: dict,
    ) -> list[ModelResponseStream]:
        runtime = _sync_runtime(_resolve_api_key(api_key, model))
        return [
            chunk
            async for chunk in self._astream(
                runtime,
                model=model,
                messages=messages,
                optional_params=optional_params,
            )
        ]


# ---------------------------------------------------------------------------
# Private loop and runtime for the sync fallback
# ---------------------------------------------------------------------------

_sync_lock = threading.Lock()
_sync_loop: asyncio.AbstractEventLoop | None = None
_sync_thread: threading.Thread | None = None
_sync_runtimes: dict[str, NearRuntime] = {}


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """Return the private event loop used by the sync entry points."""
    global _sync_loop, _sync_thread
    with _sync_lock:
        if _sync_loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="nearai-sync",
                daemon=True,
            )
            thread.start()
            _sync_loop = loop
            _sync_thread = thread
        return _sync_loop


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` on the private loop, blocking the caller.

    Kept separate from the async path so the SDK's loop-bound ``httpx``
    clients are never shared between two loops.
    """
    loop = _get_sync_loop()
    if threading.current_thread() is _sync_thread:
        raise RuntimeError("NEAR AI sync fallback cannot run on its own loop thread")
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _sync_runtime(api_key: str) -> NearRuntime:
    """Return (and lazily create on the private loop) the sync runtime."""
    cached = _sync_runtimes.get(api_key)
    if cached is not None:
        return cached
    runtime = NearRuntime(api_key)
    _sync_runtimes[api_key] = runtime
    return runtime


def close_sync_runtimes() -> None:
    """Close the private sync runtimes (best effort; tests and shutdown)."""
    with _sync_lock:
        runtimes = list(_sync_runtimes.values())
        _sync_runtimes.clear()
        loop = _sync_loop
    if loop is None:
        return
    for runtime in runtimes:
        try:
            asyncio.run_coroutine_threadsafe(runtime.aclose(), loop).result(timeout=5)
        except Exception:  # noqa: BLE001, S110 - shutdown keeps releasing the rest
            pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _install_supported_params() -> None:
    """Teach LiteLLM that ``nearai_e2ee`` accepts the extra thinking params.

    Same idempotent resolver-wrapper pattern as
    :func:`chutes_litellm.custom_provider._install_supported_params`: LiteLLM
    resolves a custom provider's parameters through the generic OpenAI list,
    which omits ``reasoning_effort`` / ``thinking`` / ``chat_template_kwargs``
    and friends, so they would be rejected before dispatch.
    """
    import litellm.utils as litellm_utils
    from litellm.litellm_core_utils import get_supported_openai_params as core

    original = core.get_supported_openai_params
    if getattr(original, "__nearai_supported_params__", False):
        return

    def patched(model, custom_llm_provider=None, *args, **kwargs):
        params = original(model, custom_llm_provider, *args, **kwargs)
        resolved = custom_llm_provider
        if resolved is None:
            try:
                resolved = litellm.get_llm_provider(model=model)[1]
            except Exception:  # noqa: BLE001 - unknown models simply keep defaults
                resolved = None
        if isinstance(params, list) and resolved == PROVIDER:
            return list(dict.fromkeys([*params, *_SUPPORTED_EXTRA_PARAMS]))
        return params

    patched.__nearai_supported_params__ = True  # type: ignore[attr-defined]
    core.get_supported_openai_params = patched
    litellm.get_supported_openai_params = patched
    litellm_utils.get_supported_openai_params = patched


def register() -> None:
    """Register :class:`NearCustomProvider` as the ``nearai_e2ee`` provider.

    Idempotent, needs no API key (CLI registration is Phase 4), and does not
    touch the Chutes attestation hook.
    """
    _install_supported_params()
    for entry in litellm.custom_provider_map:
        if entry.get("provider") == PROVIDER:
            return
    litellm.custom_provider_map.append({"provider": PROVIDER, "custom_handler": NearCustomProvider()})
    from litellm.utils import custom_llm_setup

    custom_llm_setup()


__all__ = [
    "CHAT_PATH",
    "CODE_RECEIPT_VERIFICATION_FAILED",
    "PROVIDER",
    "STREAM_MODE_ENV",
    "NearCustomProvider",
    "close_sync_runtimes",
    "register",
]
