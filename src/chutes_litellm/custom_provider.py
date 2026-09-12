"""A self-contained LiteLLM ``CustomLLM`` provider for Chutes with E2EE transport.

Unlike :mod:`chutes_litellm.e2ee_litellm`, this provider does **not** patch any
LiteLLM request path.  It is registered through the documented extension point
``litellm.custom_provider_map`` (see :func:`register`), so the model prefix it
serves is a distinct, non-builtin provider id (``chutes_e2ee``).  Every request
is performed by this handler against an httpx transport from ``chutes_e2ee``,
which transparently encrypts the payload (ML-KEM-768 + ChaCha20-Poly1305)
before it is sent to Chutes.

LiteLLM resolves a *custom* provider's supported parameters from a generic
OpenAI list, which omits the Chutes thinking params (``reasoning_effort``,
``thinking``, ``chat_template_kwargs``, ...).  :func:`register` wraps that
resolver (see :func:`_install_supported_params`) so these are declared and
forwarded instead of being rejected/dropped; nothing else on the request path is
touched.  In the proxy, the equivalent declaration is
``litellm_params.allowed_openai_params`` on each deployment.

Because the transport decrypts the reply back into plain OpenAI wire format,
this handler only has to speak the OpenAI chat-completion protocol:

* request building from ``messages`` + ``optional_params``,
* non-streaming responses (content, ``reasoning_content``, tool calls, usage),
* streaming responses parsed straight from the decrypted SSE stream
  (content deltas, ``reasoning_content`` deltas, tool-call fragments, usage),
* HTTP error mapping onto LiteLLM exceptions.

A live ``reasoning_effort`` also enables thinking on servers that gate it behind
a chat-template flag (e.g. Gemma) by adding ``chat_template_kwargs`` — harmless
where the server honours ``reasoning_effort`` directly (DeepSeek).

The wire model id is the raw Chutes model id (the ``chutes_e2ee/`` prefix, if
present, is stripped) so the transport can resolve the model to its chute.

Environment overrides (shared with :mod:`chutes_litellm.e2ee_litellm`):

* ``CHUTES_E2EE_API_BASE``    - E2EE API base (default ``https://api.chutes.ai``)
* ``CHUTES_E2EE_MODELS_BASE`` - OpenAI-compatible model base (default ``https://llm.chutes.ai``)
* ``CHUTES_E2EE_VERIFY_SSL``  - TLS verification for the underlying transport (default ``true``)
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any, AsyncIterator, Iterator

import httpx
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse, ModelResponseStream, Usage

import litellm

PROVIDER = "chutes_e2ee"

DEFAULT_E2EE_API_BASE = "https://api.chutes.ai"
DEFAULT_MODELS_BASE = "https://llm.chutes.ai"
CHAT_PATH = "/v1/chat/completions"

# OpenAI chat.completions parameters forwarded as-is.  Everything else in
# optional_params is LiteLLM bookkeeping (retries, logging, drop_params, ...)
# and must not reach Chutes.
_OPENAI_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "seed",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "reasoning_effort",
        "thinking",
        "reasoning",
        "chat_template_kwargs",
        "top_k",
        "repetition_penalty",
        "min_p",
        "logprobs",
        "top_logprobs",
        "metadata",
        "service_tier",
        "modalities",
        "audio",
        "prediction",
        "store",
        "stream_options",
    }
)

# Extra OpenAI-ish parameters this provider forwards upstream.  Declared to
# LiteLLM (see ``_install_supported_params``) so a client can actually send
# them without ``UnsupportedParamsError`` -- e.g. ``reasoning_effort`` for the
# DeepSeek thinking levels, or ``thinking`` / ``reasoning`` /
# ``chat_template_kwargs`` to enable thinking on servers that gate it behind a
# template flag (Gemma and friends).
_SUPPORTED_EXTRA_PARAMS = (
    "reasoning_effort",
    "thinking",
    "reasoning",
    "chat_template_kwargs",
    "top_k",
    "repetition_penalty",
    "min_p",
    "enable_thinking",
)

# ``reasoning_effort`` values that mean "do not think".
_DISABLED_EFFORT = frozenset({"", "none", "off", "disabled", "false"})

# Message keys LiteLLM may attach that Chutes (an OpenAI-compatible upstream)
# does not understand.
_STRIP_MESSAGE_KEYS = frozenset(
    {
        "id",
        "reasoning_content",
        "thinking_blocks",
        "reasoning_items",
        "provider_specific_fields",
        "citations",
    }
)

_lock = threading.Lock()
# Cache key: (api_key, api_base, models_base, verify_ssl, attest, quote, gpu) so a
# transport built with a different attestation mode is never reused.
_sync_transports: dict[tuple[str, str, str, bool, bool, bool, bool], httpx.BaseTransport] = {}
_async_transports: dict[tuple[str, str, str, bool, bool, bool, bool], httpx.AsyncBaseTransport] = {}


def reset_clients() -> None:
    """Drop cached transports (mainly for tests).

    Sync transports are closed eagerly; async transports only lose their
    references (they may be bound to an event loop that is already gone).
    """
    with _lock:
        for transport in _sync_transports.values():
            transport.close()
        _sync_transports.clear()
        _async_transports.clear()


def _verify_setting() -> bool:
    return os.environ.get("CHUTES_E2EE_VERIFY_SSL", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _bases() -> tuple[str, str]:
    return (
        os.environ.get("CHUTES_E2EE_API_BASE") or DEFAULT_E2EE_API_BASE,
        os.environ.get("CHUTES_E2EE_MODELS_BASE") or DEFAULT_MODELS_BASE,
    )


def _api_key_of(api_key: str | None) -> str | None:
    return api_key or os.environ.get("CHUTES_API_KEY")


def _attestation_instance_filter(api_key: str, verify_quote: bool, verify_gpu: bool):
    """Build an E2EE-transport instance filter backed by the attestation gate.

    The transport calls it with the resolved ``chute_id``; only instances the
    gate verified in the same decision are returned, so the encrypted request
    can only be sent to attested hardware.  Raises a typed LiteLLM error to
    refuse the chute when verification fails.
    """

    def _filter(chute_id: str, instances: list) -> list:
        from .attestation import AttestationError, verify_chute
        from .e2ee_litellm import _attestation_failure

        try:
            verified = verify_chute(
                api_key,
                chute_id,
                verify_quote=verify_quote,
                verify_gpu=verify_gpu,
            )
        except AttestationError as exc:
            raise _attestation_failure(chute_id, exc) from exc
        allowed = set(verified)
        return [inst for inst in instances if inst.instance_id in allowed]

    return _filter


def _attestation_mode() -> tuple[bool, bool, bool]:
    return (
        _env_bool("CHUTES_VERIFY_ATTESTATION"),
        _env_bool("CHUTES_VERIFY_QUOTE"),
        _env_bool("CHUTES_VERIFY_GPU"),
    )


def _sync_transport(api_key: str, api_base: str, models_base: str) -> httpx.BaseTransport:
    attest, quote, gpu = _attestation_mode()
    key = (api_key, api_base, models_base, _verify_setting(), attest, quote, gpu)
    with _lock:
        transport = _sync_transports.get(key)
        if transport is None:
            from chutes_e2ee import ChutesE2EETransport

            kwargs: dict = {
                "api_key": api_key,
                "api_base": api_base,
                "models_base": models_base,
                "inner": httpx.HTTPTransport(verify=_verify_setting()),
            }
            if attest:
                # Fork-only kwarg; only pass it when the gate is on so a stock
                # chutes-e2ee keeps working with attestation off.
                kwargs["instance_filter"] = _attestation_instance_filter(api_key, quote, gpu)
            transport = ChutesE2EETransport(**kwargs)
            _sync_transports[key] = transport
        return transport


def _async_transport(api_key: str, api_base: str, models_base: str) -> httpx.AsyncBaseTransport:
    attest, quote, gpu = _attestation_mode()
    key = (api_key, api_base, models_base, _verify_setting(), attest, quote, gpu)
    with _lock:
        transport = _async_transports.get(key)
        if transport is None:
            from chutes_e2ee import AsyncChutesE2EETransport

            kwargs: dict = {
                "api_key": api_key,
                "api_base": api_base,
                "models_base": models_base,
                "inner": httpx.AsyncHTTPTransport(verify=_verify_setting()),
            }
            if attest:
                kwargs["instance_filter"] = _attestation_instance_filter(api_key, quote, gpu)
            transport = AsyncChutesE2EETransport(**kwargs)
            _async_transports[key] = transport
        return transport


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _clean_model(model: str) -> str:
    if "/" in model and model.split("/", 1)[0] in (PROVIDER, "chutes"):
        return model.split("/", 1)[1]
    return model


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_verify_attestation(api_key: str, model: str) -> None:
    """Fail-closed TEE/GPU attestation gate (mirrors :mod:`e2ee_litellm`).

    No-op unless ``CHUTES_VERIFY_ATTESTATION`` is set.  Raises a LiteLLM
    ``APIError`` carrying the per-instance reasons when verification fails,
    before any request is built or encrypted.
    """
    if not _env_bool("CHUTES_VERIFY_ATTESTATION"):
        return
    from .attestation import AttestationError, verify_model
    from .e2ee_litellm import _attestation_failure

    wire_model = _clean_model(model)
    try:
        verify_model(
            api_key,
            wire_model,
            verify_quote=_env_bool("CHUTES_VERIFY_QUOTE"),
            verify_gpu=_env_bool("CHUTES_VERIFY_GPU"),
        )
    except AttestationError as exc:
        raise _attestation_failure(wire_model, exc) from exc


def _clean_messages(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for message in messages or []:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        msg = {k: v for k, v in message.items() if k not in _STRIP_MESSAGE_KEYS}
        cleaned.append(msg)
    return cleaned


def _build_payload(
    model: str,
    messages: list[dict],
    optional_params: dict,
    *,
    stream: bool,
) -> dict:
    payload: dict[str, Any] = {
        "model": _clean_model(model),
        "messages": _clean_messages(messages),
        "stream": stream,
    }
    for key in _OPENAI_PARAMS:
        if key in optional_params and optional_params[key] is not None:
            payload[key] = optional_params[key]

    # ``extra_body`` is LiteLLM's escape hatch for arbitrary provider params;
    # flatten it into the top-level request body.
    extra_body = optional_params.get("extra_body")
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            if value is not None:
                payload[key] = value

    # Uniform thinking switch.  A live ``reasoning_effort`` turns thinking on;
    # servers that gate it behind a chat-template flag (Gemma) ignore
    # ``reasoning_effort`` and need ``chat_template_kwargs.enable_thinking``,
    # while servers that honor the effort (DeepSeek) ignore the template flag.
    # Sending both is therefore safe and lets one client setting work everywhere.
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip().lower() not in _DISABLED_EFFORT:
        template_kwargs = payload.get("chat_template_kwargs")
        if template_kwargs is None:
            template_kwargs = {}
        if isinstance(template_kwargs, dict):
            template_kwargs.setdefault("enable_thinking", True)
            payload["chat_template_kwargs"] = template_kwargs
    return payload


def _raise_for_error(response: httpx.Response, model: str, api_key: str | None) -> None:
    if response.status_code < 400:
        return
    body: dict[str, Any] = {}
    try:
        raw = response.read()
    except Exception:
        raw = b""
    try:
        parsed = json.loads(raw or b"{}")
        if isinstance(parsed, dict):
            body = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
    except Exception:
        pass
    message = str(body.get("message") or body.get("error") or raw.decode("utf-8", "replace") or "unknown error")[:1024]
    status = response.status_code
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
    elif 500 <= status < 600:
        error_cls = litellm.ServiceUnavailableError if status == 503 else litellm.InternalServerError
    raise error_cls(
        message=message,
        model=model,
        llm_provider=PROVIDER,
        response=response,
    )


# ---------------------------------------------------------------------------
# Non-streaming response translation
# ---------------------------------------------------------------------------


def _populate_message(message: Message, data: dict) -> None:
    message.role = data.get("role") or "assistant"
    message.content = data.get("content")
    if data.get("reasoning_content") is not None:
        message.reasoning_content = data["reasoning_content"]
    if data.get("tool_calls"):
        from litellm.types.utils import ChatCompletionMessageToolCall

        try:
            message.tool_calls = [ChatCompletionMessageToolCall.model_validate(tc) for tc in data["tool_calls"]]
        except Exception:
            message.tool_calls = data["tool_calls"]
    if data.get("function_call") is not None:
        message.function_call = data["function_call"]


def _populate_usage(model_response: ModelResponse, data: dict | None) -> None:
    if not data:
        return
    try:
        model_response.usage = Usage.model_validate(data)
    except Exception:
        model_response.usage = Usage(
            prompt_tokens=data.get("prompt_tokens") or 0,
            completion_tokens=data.get("completion_tokens") or 0,
            total_tokens=data.get("total_tokens") or 0,
        )


def _to_model_response(model: str, data: dict, model_response: ModelResponse) -> ModelResponse:
    model_response.id = data.get("id") or model_response.id
    model_response.created = data.get("created") or int(time.time())
    model_response.model = data.get("model") or model
    model_response.object = data.get("object") or "chat.completion"

    choices: list[Any] = data.get("choices") or []
    populated: list[Choices] = []
    for index, choice in enumerate(choices):
        message_data = choice.get("message") if isinstance(choice, dict) else None
        message = Message(role="assistant")
        _populate_message(message, message_data or {})
        populated.append(
            Choices(
                message=message,
                finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
                index=index,
            )
        )
    if populated:
        model_response.choices = populated
    _populate_usage(model_response, data.get("usage"))
    return model_response


# ---------------------------------------------------------------------------
# Streaming response translation
# ---------------------------------------------------------------------------
#
# Each decrypted SSE event is a standard OpenAI ``chat.completion.chunk``
# object.  We forward it to LiteLLM as a ``ModelResponseStream`` chunk; LiteLLM
# passes such chunks from custom providers through unchanged (its
# ``chunk_parser`` treats a ``ModelResponseStream`` produced by a provider in
# ``litellm._custom_providers`` as already-parsed), so ``reasoning_content``,
# tool-call fragments, content and finish_reason all reach the caller intact.


def _event_to_stream_chunk(event: dict) -> ModelResponseStream | None:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    try:
        return ModelResponseStream.model_validate(event)
    except Exception:
        return None


def _yield_stream_chunks(events: Iterator[dict]) -> Iterator[ModelResponseStream]:
    for event in events:
        chunk = _event_to_stream_chunk(event)
        if chunk is None:
            continue
        yield chunk
        # A chunk carrying finish_reason is the last thing worth forwarding:
        # LiteLLM synthesizes the terminal chunk from it, and anything trailing
        # it in the SSE stream (e.g. a separate usage event) has no chunk
        # surface of its own.
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
        if finish_reason:
            return


async def _ayield_stream_chunks(events: AsyncIterator[dict]) -> AsyncIterator[ModelResponseStream]:
    async for event in events:
        chunk = _event_to_stream_chunk(event)
        if chunk is None:
            continue
        yield chunk
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
        if finish_reason:
            return


def _iter_events_sync(response: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in response.iter_lines():
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


async def _iter_events_async(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


class ChutesCustomProvider(CustomLLM):
    """Featureful Chutes chat provider backed by the chutes-e2ee transport.

    Implementations are shared between the sync/async and streaming/non
    streaming entry points that LiteLLM's ``custom_provider_map`` dispatch
    calls (``completion``/``acompletion``/``streaming``/``astreaming``).
    """

    # -- sync non-streaming ------------------------------------------------

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
        api_key = _api_key_of(api_key)
        if not api_key:
            raise litellm.AuthenticationError(
                message="CHUTES_API_KEY is not set",
                model=model,
                llm_provider=PROVIDER,
            )
        _maybe_verify_attestation(api_key, model)
        _, models_base = _bases()
        payload = _build_payload(model, messages, optional_params or {}, stream=False)
        request = httpx.Request(
            "POST",
            f"{models_base}{CHAT_PATH}",
            json=payload,
            extensions={"timeout": timeout or httpx.Timeout(600.0, connect=10.0)},
        )
        e2ee_base, _ = _bases()
        response = _sync_transport(api_key, e2ee_base, models_base).handle_request(request)
        try:
            _raise_for_error(response, model, api_key)
            data = response.json()
        finally:
            response.close()
        if not isinstance(data, dict):
            raise litellm.APIError(
                status_code=500,
                message=f"unexpected non-JSON response: {data!r}",
                model=model,
                llm_provider=PROVIDER,
            )
        return _to_model_response(model, data, model_response)

    # -- async non-streaming ----------------------------------------------

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
        api_key = _api_key_of(api_key)
        if not api_key:
            raise litellm.AuthenticationError(
                message="CHUTES_API_KEY is not set",
                model=model,
                llm_provider=PROVIDER,
            )
        await asyncio.to_thread(_maybe_verify_attestation, api_key, model)
        api_base, models_base = _bases()
        payload = _build_payload(model, messages, optional_params or {}, stream=False)
        request = httpx.Request(
            "POST",
            f"{models_base}{CHAT_PATH}",
            json=payload,
            extensions={"timeout": timeout or httpx.Timeout(600.0, connect=10.0)},
        )
        transport = _async_transport(api_key, api_base, models_base)
        response = await transport.handle_async_request(request)
        try:
            _raise_for_error(response, model, api_key)
            data = response.json()
        finally:
            await response.aclose()
        if not isinstance(data, dict):
            raise litellm.APIError(
                status_code=500,
                message=f"unexpected non-JSON response: {data!r}",
                model=model,
                llm_provider=PROVIDER,
            )
        return _to_model_response(model, data, model_response)

    # -- sync streaming ----------------------------------------------------

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
        api_key = _api_key_of(api_key)
        if not api_key:
            raise litellm.AuthenticationError(
                message="CHUTES_API_KEY is not set",
                model=model,
                llm_provider=PROVIDER,
            )
        _maybe_verify_attestation(api_key, model)
        api_base, models_base = _bases()
        payload = _build_payload(model, messages, optional_params or {}, stream=True)
        request = httpx.Request(
            "POST",
            f"{models_base}{CHAT_PATH}",
            json=payload,
            extensions={"timeout": timeout or httpx.Timeout(600.0, connect=10.0)},
        )
        response = _sync_transport(api_key, api_base, models_base).handle_request(request)
        try:
            _raise_for_error(response, model, api_key)
            yield from _yield_stream_chunks(_iter_events_sync(response))
        finally:
            response.close()

    # -- async streaming ---------------------------------------------------

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
        api_key = _api_key_of(api_key)
        if not api_key:
            raise litellm.AuthenticationError(
                message="CHUTES_API_KEY is not set",
                model=model,
                llm_provider=PROVIDER,
            )
        await asyncio.to_thread(_maybe_verify_attestation, api_key, model)
        api_base, models_base = _bases()
        payload = _build_payload(model, messages, optional_params or {}, stream=True)
        request = httpx.Request(
            "POST",
            f"{models_base}{CHAT_PATH}",
            json=payload,
            extensions={"timeout": timeout or httpx.Timeout(600.0, connect=10.0)},
        )
        transport = _async_transport(api_key, api_base, models_base)
        response = await transport.handle_async_request(request)
        try:
            _raise_for_error(response, model, api_key)
            async for chunk in _ayield_stream_chunks(_iter_events_async(response)):
                yield chunk
        finally:
            await response.aclose()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _install_supported_params() -> None:
    """Teach LiteLLM that ``chutes_e2ee`` accepts the extra thinking params.

    LiteLLM resolves a custom provider's supported OpenAI parameters through
    the generic ``OpenAIConfig`` list, which omits ``reasoning_effort`` /
    ``thinking`` / ``chat_template_kwargs`` and therefore rejects them with
    ``UnsupportedParamsError`` before dispatch.  There is no per-custom-provider
    hook in this LiteLLM version, so we wrap the resolver and append our params
    only when the provider is ours.  Idempotent.
    """
    from litellm.litellm_core_utils import get_supported_openai_params as core

    import litellm.utils as litellm_utils

    original = core.get_supported_openai_params
    if getattr(original, "__chutes_supported_params__", False):
        return

    def patched(model, custom_llm_provider=None, *args, **kwargs):
        params = original(model=model, custom_llm_provider=custom_llm_provider, *args, **kwargs)
        resolved = custom_llm_provider
        if resolved is None:
            try:
                resolved = litellm.get_llm_provider(model=model)[1]
            except Exception:
                resolved = None
        if isinstance(params, list) and resolved == PROVIDER:
            return list(dict.fromkeys([*params, *_SUPPORTED_EXTRA_PARAMS]))
        return params

    patched.__chutes_supported_params__ = True  # type: ignore[attr-defined]
    core.get_supported_openai_params = patched
    litellm.get_supported_openai_params = patched
    litellm_utils.get_supported_openai_params = patched


def register() -> None:
    """Register :class:`ChutesCustomProvider` as the ``chutes_e2ee`` provider.

    Idempotent.  Models served by the proxy must use the ``chutes_e2ee/``
    prefix (e.g. ``chutes_e2ee/Test/Model-TEE``) so LiteLLM dispatches them to
    this handler instead of its builtin ``chutes`` openai-compatible provider.
    """
    _install_supported_params()
    for entry in litellm.custom_provider_map:
        if entry.get("provider") == PROVIDER:
            return
    litellm.custom_provider_map.append({"provider": PROVIDER, "custom_handler": ChutesCustomProvider()})
    from litellm.utils import custom_llm_setup

    custom_llm_setup()
