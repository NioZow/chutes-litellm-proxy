"""Option A: native ``chutes`` provider + Chutes E2EE transport.

LiteLLM's ``chutes`` provider is *declarative* (a ``providers.json`` entry for
``https://llm.chutes.ai/v1``) — every request it makes is performed by the
generic OpenAI-SDK code path (``OpenAIChatCompletion``).  That code path builds
its sync/async OpenAI clients from two class-level factories,
``BaseOpenAILLM._get_sync_http_client()`` and ``_get_async_http_client()``.

Instead of shipping a bespoke LiteLLM provider (the old ``chutes_provider.py``
approach), this module *swaps those two factories* for shared ``httpx`` clients
whose transport is the Chutes E2EE transport (ML-KEM-768 + ChaCha20-Poly1305)
for requests addressed to Chutes hosts, and a plain passthrough transport for
every other host.  The result:

  * native ``chutes`` provider semantics (param validation, transforms,
    streaming, usage, retries — all maintained upstream);
  * end-to-end encryption only where it belongs (``llm.chutes.ai`` /
    ``api.chutes.ai``);
  * works through the Router / proxy YAML config: no client objects need to be
    embedded in ``model_list``.

Optional: with ``CHUTES_VERIFY_ATTESTATION=true`` every request additionally
runs the :mod:`chutes_litellm.attestation` gate, which refuses to encrypt to an
instance whose TDX/GPU attestation evidence does not verify (fail closed).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import httpx

try:
    import chutes_e2ee as _chutes_e2ee
except Exception as e:  # pragma: no cover - import guard
    _chutes_e2ee = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _transport_classes() -> tuple[type, type]:
    if _chutes_e2ee is None:  # pragma: no cover
        raise RuntimeError(f"chutes_e2ee is not importable: {_IMPORT_ERROR}")
    return _chutes_e2ee.ChutesE2EETransport, _chutes_e2ee.AsyncChutesE2EETransport

DEFAULT_LLM_HOST = "llm.chutes.ai"
DEFAULT_API_HOST = "api.chutes.ai"
DEFAULT_E2EE_API_BASE = f"https://{DEFAULT_API_HOST}"
DEFAULT_MODELS_BASE = f"https://{DEFAULT_LLM_HOST}"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _e2ee_hosts() -> set[str]:
    raw = os.environ.get("CHUTES_E2EE_HOSTS", f"{DEFAULT_LLM_HOST},{DEFAULT_API_HOST}")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _json_model(body: bytes) -> str | None:
    """Return the ``model`` field of an OpenAI JSON request body, if any."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data.get("model")


def _attestation_failure(model: str, exc: Exception) -> Exception:
    """Wrap a failed attestation in a typed LiteLLM error.

    The check runs inside the httpx transport, before the request leaves the
    process.  Raising ``litellm.APIError`` makes the proxy emit a structured
    ``{"error": {"message": ...}}`` body carrying the per-instance reasons
    instead of an opaque 500 traceback, so the *why* is visible to the caller.
    """
    message = f"Chutes attestation failed for {model!r}; request not sent.\n{exc}"
    try:
        import litellm

        return litellm.APIError(
            status_code=503,
            message=message,
            llm_provider="chutes",
            model=model,
        )
    except Exception:  # pragma: no cover - litellm import guard
        return RuntimeError(message)


class _ChutesScopedTransport(httpx.BaseTransport):
    """Sync transport: E2EE for Chutes hosts, passthrough otherwise."""

    def __init__(
        self,
        api_key: str,
        *,
        hosts: set[str],
        verify: bool = True,
        api_key_provider: Any = None,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(f"chutes_e2ee is not importable: {_IMPORT_ERROR}")
        _SyncCls, _ = _transport_classes()
        self._api_key = api_key
        self._hosts = hosts
        self._verify = verify
        self._e2e_api_base = os.environ.get("CHUTES_E2EE_API_BASE") or DEFAULT_E2EE_API_BASE
        self._models_base = os.environ.get("CHUTES_E2EE_MODELS_BASE") or DEFAULT_MODELS_BASE
        self._verify_attestation = _env_bool("CHUTES_VERIFY_ATTESTATION")
        self._verify_quote = _env_bool("CHUTES_VERIFY_QUOTE")
        self._verify_gpu = _env_bool("CHUTES_VERIFY_GPU")
        self._api_key_provider = api_key_provider
        self._inner = httpx.HTTPTransport(verify=verify)
        transport_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_base": self._e2e_api_base,
            "models_base": self._models_base,
            "inner": self._inner,
        }
        if self._verify_attestation:
            # `instance_filter` is a fork-only kwarg: only pass it when the gate
            # is on, so a stock `chutes-e2ee` still works with attestation off.
            # With the gate on, it restricts the transport's own instance pool to
            # the instances the gate verified, so selection and verification
            # cannot diverge.
            transport_kwargs["instance_filter"] = self._instance_filter
        self._transport = _SyncCls(**transport_kwargs)

    def _instance_filter(self, chute_id: str, instances: list[Any]) -> list[Any]:
        """Attestation-backed instance filter for the E2EE transport.

        Called by ``DiscoveryManager`` with the same ``chute_id`` the transport
        resolved, and cached by the attestation gate, so after the pre-send
        ``_maybe_verify`` this is a cache hit.  Raises a typed LiteLLM error to
        refuse the chute when verification fails.
        """
        from .attestation import AttestationError, verify_chute

        api_key = (self._api_key_provider() if self._api_key_provider else None) or self._api_key
        try:
            verified = verify_chute(
                api_key,
                chute_id,
                verify_quote=self._verify_quote,
                verify_gpu=self._verify_gpu,
            )
        except AttestationError as exc:
            raise _attestation_failure(chute_id, exc) from exc
        allowed = set(verified)
        return [inst for inst in instances if inst.instance_id in allowed]

    def _maybe_verify(self, request: httpx.Request) -> None:
        if not self._verify_attestation:
            return
        from .attestation import AttestationError, verify_model

        model = _json_model(request.content)
        if model is None:
            return
        try:
            verify_model(
                self._api_key_provider(),
                model,
                verify_quote=self._verify_quote,
                verify_gpu=self._verify_gpu,
            )
        except AttestationError as exc:
            raise _attestation_failure(model, exc) from exc

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() in self._hosts:
            self._maybe_verify(request)
            return self._transport.handle_request(request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._transport.close()
        self._inner.close()


class _ChutesScopedAsyncTransport(httpx.AsyncBaseTransport):
    """Async transport: E2EE for Chutes hosts, passthrough otherwise."""

    def __init__(
        self,
        api_key: str,
        *,
        hosts: set[str],
        verify: bool = True,
        api_key_provider: Any = None,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(f"chutes_e2ee is not importable: {_IMPORT_ERROR}")
        _, _AsyncCls = _transport_classes()
        self._api_key = api_key
        self._hosts = hosts
        self._verify = verify
        self._e2e_api_base = os.environ.get("CHUTES_E2EE_API_BASE") or DEFAULT_E2EE_API_BASE
        self._models_base = os.environ.get("CHUTES_E2EE_MODELS_BASE") or DEFAULT_MODELS_BASE
        self._verify_attestation = _env_bool("CHUTES_VERIFY_ATTESTATION")
        self._verify_quote = _env_bool("CHUTES_VERIFY_QUOTE")
        self._verify_gpu = _env_bool("CHUTES_VERIFY_GPU")
        self._api_key_provider = api_key_provider
        self._inner = httpx.AsyncHTTPTransport(verify=verify)
        transport_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_base": self._e2e_api_base,
            "models_base": self._models_base,
            "inner": self._inner,
        }
        if self._verify_attestation:
            # Fork-only kwarg; see the sync transport above.
            transport_kwargs["instance_filter"] = self._instance_filter
        self._transport = _AsyncCls(**transport_kwargs)

    def _instance_filter(self, chute_id: str, instances: list[Any]) -> list[Any]:
        """Attestation-backed instance filter (see the sync transport).

        ``_maybe_verify`` runs first on the event loop's executor and warms the
        gate's cache, so this call is a cache hit and does not block on network.
        """
        from .attestation import AttestationError, verify_chute

        api_key = (self._api_key_provider() if self._api_key_provider else None) or self._api_key
        try:
            verified = verify_chute(
                api_key,
                chute_id,
                verify_quote=self._verify_quote,
                verify_gpu=self._verify_gpu,
            )
        except AttestationError as exc:
            raise _attestation_failure(chute_id, exc) from exc
        allowed = set(verified)
        return [inst for inst in instances if inst.instance_id in allowed]

    async def _maybe_verify(self, request: httpx.Request) -> None:
        if not self._verify_attestation:
            return
        import asyncio

        from .attestation import AttestationError, verify_model

        model = _json_model(request.content)
        if model is None:
            return
        try:
            await asyncio.to_thread(
                verify_model,
                self._api_key_provider(),
                model,
                verify_quote=self._verify_quote,
                verify_gpu=self._verify_gpu,
            )
        except AttestationError as exc:
            raise _attestation_failure(model, exc) from exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() in self._hosts:
            await self._maybe_verify(request)
            return await self._transport.handle_async_request(request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Shared clients + patch
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_clients: dict[str, httpx.Client] = {}
_aclients: dict[str, httpx.AsyncClient] = {}
_patched = False
_originals: dict[str, Any] = {}


def _api_key() -> str | None:
    return os.environ.get("CHUTES_API_KEY")


def _verify_setting() -> bool:
    return bool(os.environ.get("CHUTES_E2EE_VERIFY_SSL", "true").strip().lower() in {"1", "true", "yes", "on"})


def _scoped_client() -> httpx.Client | None:
    key = _api_key()
    if not key:
        return None
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = httpx.Client(
                transport=_ChutesScopedTransport(
                    key,
                    hosts=_e2ee_hosts(),
                    verify=_verify_setting(),
                    api_key_provider=_api_key,
                ),
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
            _clients[key] = client
        return client


def _scoped_aclient() -> httpx.AsyncClient | None:
    key = _api_key()
    if not key:
        return None
    with _lock:
        client = _aclients.get(key)
        if client is None:
            client = httpx.AsyncClient(
                transport=_ChutesScopedAsyncTransport(
                    key,
                    hosts=_e2ee_hosts(),
                    verify=_verify_setting(),
                    api_key_provider=_api_key,
                ),
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
            _aclients[key] = client
        return client


def _sync_factory() -> httpx.Client | None:
    return _scoped_client()


def _async_factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient | None:
    return _scoped_aclient()


def _find_base_class():
    """Locate the OpenAI-SDK base class exposing the http-client factories."""
    try:
        from litellm.llms.openai.openai import BaseOpenAILLM
    except ImportError:  # pragma: no cover - version resilience
        BaseOpenAILLM = None
    if BaseOpenAILLM is not None and hasattr(BaseOpenAILLM, "_get_sync_http_client"):
        return BaseOpenAILLM
    import litellm.llms.openai.openai as _o

    for name in dir(_o):
        cls = getattr(_o, name)
        if isinstance(cls, type) and hasattr(cls, "_get_sync_http_client") and hasattr(cls, "_get_async_http_client"):
            return cls
    raise RuntimeError("Could not locate litellm's OpenAI http-client factory class")


def install() -> bool:
    """Install the scoped E2EE http clients into litellm's OpenAI-SDK path.

    No-op (returns ``False``) when no ``CHUTES_API_KEY`` is set or when the
    ``chutes_e2ee`` package is unavailable, so the proxy keeps working for the
    other providers.  Idempotent.
    """
    global _patched
    if _patched:
        return True
    if _IMPORT_ERROR is not None or not _api_key():
        return False

    cls = _find_base_class()
    with _lock:
        if _patched:
            return True
        _originals["sync"] = getattr(cls, "_get_sync_http_client", None)
        _originals["async"] = getattr(cls, "_get_async_http_client", None)
        setattr(cls, "_get_sync_http_client", staticmethod(_sync_factory))
        setattr(cls, "_get_async_http_client", staticmethod(_async_factory))
        _patched = True
    return True


def uninstall() -> None:
    """Restore the original factories (mainly for tests)."""
    global _patched
    if not _patched:
        return
    cls = _find_base_class()
    with _lock:
        for name in ("sync", "async"):
            original = _originals.get(name)
            if original is not None:
                setattr(cls, f"_get_{'sync' if name == 'sync' else 'async'}_http_client", original)
        _patched = False


if __name__ == "__main__":  # `python -m chutes_litellm.e2ee_litellm` => install then continue
    install()
