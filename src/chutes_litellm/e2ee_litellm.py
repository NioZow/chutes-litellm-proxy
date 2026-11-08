"""Option A: native ``chutes`` provider + Chutes E2EE transport.

LiteLLM's ``chutes`` provider is *declarative* (a ``providers.json`` entry for
``https://llm.chutes.ai/v1``) — every request it makes is performed by the
generic OpenAI-SDK code path (``OpenAIChatCompletion``).  That code path builds
its sync/async OpenAI clients from two class-level factories,
``BaseOpenAILLM._get_sync_http_client()`` and ``_get_async_http_client()``.

instead of shipping a bespoke LiteLLM provider (the old ``chutes_provider.py``
approach), this module *swaps those two factories* for shared ``httpx`` clients
whose transport is the Chutes E2EE transport (ML-KEM-768 + ChaCha20-Poly1305)
for TEE models addressed to Chutes hosts, and a plain passthrough transport for
plain Chutes deployments and every other host.  The result:

  * native ``chutes`` provider semantics (param validation, transforms,
    streaming, usage, retries — all maintained upstream);
  * end-to-end encryption only where it belongs (TEE models on the fixed
    ``https://llm.chutes.ai`` / ``https://api.chutes.ai`` hosts);
  * works through the Router / proxy YAML config: no client objects need to be
    embedded in ``model_list``.

Requests for TEE models run the :mod:`chutes_litellm.attestation` gate, which
refuses to encrypt to an instance whose TDX/GPU attestation evidence does not
verify (fail closed).  Attestation, NVIDIA GPU verification and TLS
verification are always on and cannot be disabled.  Plain Chutes models are not
confidential-compute deployments: they have no E2EE instance key, so they are
forwarded unmodified (the model id is classified at startup from the catalog's
``confidential_compute`` flag).
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import endpoints

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


_ATTESTATION_HOOK = "instance_filter"


def _require_attestation_hook() -> None:
    """Fail fast unless both transports expose the ``instance_filter`` kwarg.

    The hook pins each request to the instance whose attestation quote passed.
    An older chutes-e2ee cannot accept it, so constructing the transport then
    raises at request time with an opaque 500.  Requiring the explicit
    parameter here turns that into a clear startup failure instead of serving
    Chutes without attested-instance pinning.
    """
    missing: list[str] = []
    for transport_cls in _transport_classes():
        params = inspect.signature(transport_cls.__init__).parameters
        param = params.get(_ATTESTATION_HOOK)
        if param is None or param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            missing.append(transport_cls.__name__)
    if missing:
        raise RuntimeError(
            "chutes_e2ee is missing the mandatory "
            f"{_ATTESTATION_HOOK!r} hook on {', '.join(missing)}; refusing to "
            "run without attested-instance pinning. Install the pinned "
            "chutes-e2ee revision from uv.lock / flake.nix."
        )


DEFAULT_LLM_HOST = "llm.chutes.ai"
DEFAULT_API_HOST = "api.chutes.ai"


def _host_of(base: str) -> str:
    """Return the lowercase hostname of *base*, or an empty string."""
    return (urlsplit(base).hostname or "").strip().lower()


def _e2ee_hosts() -> set[str]:
    """Hosts whose traffic is end-to-end encrypted and attested.

    The hosts are derived at call time from the fixed ``endpoints.API_BASE`` /
    ``endpoints.MODELS_BASE``, and the canonical Chutes hosts are always
    included.  There is no environment override: a Chutes host can never be
    opted out of the E2EE/attestation path.
    """
    hosts = {DEFAULT_LLM_HOST, DEFAULT_API_HOST}
    for base in (endpoints.API_BASE, endpoints.MODELS_BASE):
        host = _host_of(base)
        if host:
            hosts.add(host)
    return hosts


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


def _passthrough_transport() -> httpx.BaseTransport:
    """Build the plain transport for non-E2EE requests.

    A bare ``httpx.HTTPTransport()`` never consults ``HTTP(S)_PROXY``: httpx
    only applies environment proxies when a *client* owns its transport, and a
    client built with an explicit transport opts out (``allow_env_proxies`` is
    false).  Hardened deployments give the proxy guest no direct egress, so
    plain Chutes models resolved to a DNS timeout and surfaced as
    ``ChutesException - Connection error``.  Reuse ``chutes-e2ee``'s default
    inner transport, which already honours ``HTTP(S)_PROXY`` and ``NO_PROXY``.
    """
    from chutes_e2ee.transport import _default_transport

    return _default_transport()


def _passthrough_async_transport() -> httpx.AsyncBaseTransport:
    """Async counterpart of :func:`_passthrough_transport`."""
    from chutes_e2ee.transport import _default_async_transport

    return _default_async_transport()


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
        api_key_provider: Any = None,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(f"chutes_e2ee is not importable: {_IMPORT_ERROR}")
        _require_attestation_hook()
        _SyncCls, _ = _transport_classes()
        self._api_key = api_key
        self._hosts = hosts
        self._e2e_api_base = endpoints.API_BASE
        self._models_base = endpoints.MODELS_BASE
        self._api_key_provider = api_key_provider
        # Both paths must honour HTTP(S)_PROXY / NO_PROXY: the deployment has
        # no direct egress.  See `_passthrough_transport`.
        self._inner = _passthrough_transport()
        self._transport = _SyncCls(
            api_key=api_key,
            api_base=self._e2e_api_base,
            models_base=self._models_base,
            # `instance_filter` restricts the transport's own instance pool to
            # the instances the gate verified, so selection and verification
            # cannot diverge.
            instance_filter=self._instance_filter,
        )

    def _instance_filter(self, chute_id: str, instances: list[Any]) -> list[Any]:
        """Attestation-backed instance filter for the E2EE transport.

        Called by ``DiscoveryManager`` with the same ``chute_id`` the transport
        resolved, and cached by the attestation gate, so after the pre-send
        ``_verify_request`` this is a cache hit.  Raises a typed LiteLLM error to
        refuse the chute when verification fails.
        """
        from .attestation import AttestationError, verify_chute

        api_key = (self._api_key_provider() if self._api_key_provider else None) or self._api_key
        try:
            verified = verify_chute(api_key, chute_id)
        except AttestationError as exc:
            raise _attestation_failure(chute_id, exc) from exc
        # Pin the key the gate verified: a racing/malicious discovery response
        # must not be able to substitute a different key for the same id.
        return [inst for inst in instances if verified.get(inst.instance_id) == getattr(inst, "e2e_pubkey", None)]

    def _verify_request(self, request: httpx.Request, model: str | None = None) -> None:
        from .attestation import AttestationError, verify_model

        if model is None:
            model = _json_model(request.content)
        if model is None:
            return
        from chutes_litellm import get_logger

        logger = get_logger("chutes_litellm.e2ee")
        logger.info("_verify_request: model=%s", model)
        try:
            verify_model(self._api_key_provider(), model)
        except AttestationError as exc:
            logger.warning("_verify_request: model=%s FAILED %s", model, exc)
            raise _attestation_failure(model, exc) from exc

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() in self._hosts:
            model = _json_model(request.content)
            # Only TEE models are encrypted; plain Chutes deployments share
            # the host but have no E2EE instance/key and must pass through.
            if is_e2ee_model(model):
                self._verify_request(request)
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
        api_key_provider: Any = None,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(f"chutes_e2ee is not importable: {_IMPORT_ERROR}")
        _require_attestation_hook()
        _, _AsyncCls = _transport_classes()
        self._api_key = api_key
        self._hosts = hosts
        self._e2e_api_base = endpoints.API_BASE
        self._models_base = endpoints.MODELS_BASE
        self._api_key_provider = api_key_provider
        # As above: both paths must honour the deployment's proxy.
        self._inner = _passthrough_async_transport()
        self._transport = _AsyncCls(
            api_key=api_key,
            api_base=self._e2e_api_base,
            models_base=self._models_base,
            # Same mandatory attested-instance pinning as the sync transport.
            instance_filter=self._instance_filter,
        )

    def _instance_filter(self, chute_id: str, instances: list[Any]) -> list[Any]:
        """Attestation-backed instance filter (see the sync transport).

        ``_verify_request`` runs first on the event loop's executor and warms the
        gate's cache, so this call is a cache hit and does not block on network.
        """
        from .attestation import AttestationError, verify_chute

        api_key = (self._api_key_provider() if self._api_key_provider else None) or self._api_key
        try:
            verified = verify_chute(api_key, chute_id)
        except AttestationError as exc:
            raise _attestation_failure(chute_id, exc) from exc
        # Pin the key the gate verified: a racing/malicious discovery response
        # must not be able to substitute a different key for the same id.
        return [inst for inst in instances if verified.get(inst.instance_id) == getattr(inst, "e2e_pubkey", None)]

    async def _verify_request(self, request: httpx.Request, model: str | None = None) -> None:
        import asyncio

        from .attestation import AttestationError, verify_model

        if model is None:
            model = _json_model(request.content)
        if model is None:
            return
        from chutes_litellm import get_logger

        logger = get_logger("chutes_litellm.e2ee")
        logger.info("_verify_request(async): model=%s", model)
        try:
            await asyncio.to_thread(verify_model, self._api_key_provider(), model)
        except AttestationError as exc:
            logger.warning("_verify_request(async): model=%s FAILED %s", model, exc)
            raise _attestation_failure(model, exc) from exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() in self._hosts:
            model = _json_model(request.content)
            if is_e2ee_model(model):
                await self._verify_request(request)
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

# Exact TEE/non-TEE classification from the generated config (Chutes publishes
# `confidential_compute`).  The `-TEE` suffix is the fallback for request paths
# that bypass config generation (tests, ad-hoc clients).
_tee_models: frozenset[str] = frozenset()
_non_tee_models: frozenset[str] = frozenset()


def configure_tee_models(
    tee: Iterable[str] = (), non_tee: Iterable[str] = ()
) -> None:
    """Record which bare Chutes model ids require end-to-end encryption.

    Non-TEE models (``confidential_compute: false``) must be sent in plaintext;
    wrapping them in the E2EE transport fails because no instance advertises an
    encryption key for them.  The sets are built at startup from the generated
    config; without them the ``-TEE`` id suffix decides.
    """
    global _tee_models, _non_tee_models
    _tee_models = frozenset(item for item in tee if item)
    _non_tee_models = frozenset(item for item in non_tee if item)


def _bare_chutes_model(model: str | None) -> str | None:
    """Strip a ``chutes/`` or ``chutes_e2ee/`` provider prefix from *model*."""
    if not isinstance(model, str) or not model.strip():
        return None
    cleaned = model.strip()
    for prefix in ("chutes_e2ee/", "chutes/"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):]
    return cleaned


def is_e2ee_model(model: str | None) -> bool:
    """Return whether *model* must go through the E2EE transport.

    Explicit non-TEE classification wins over the suffix so a future TEE
    deployment without ``-TEE`` in its id still gets encrypted when the catalog
    marks it ``confidential_compute``.
    """
    bare = _bare_chutes_model(model)
    if not bare:
        return False
    if bare in _non_tee_models:
        return False
    if bare in _tee_models:
        return True
    return bare.upper().endswith("-TEE")


def _api_key() -> str | None:
    return os.environ.get("CHUTES_API_KEY")


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

    _require_attestation_hook()
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
