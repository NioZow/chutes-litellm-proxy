"""LiteLLM worker startup hook that prewarms NEAR AI models.

The CLI exports ``LITELLM_WORKER_STARTUP_HOOKS=chutes_litellm.near.startup:prewarm``
when a NEAR key and at least one ``nearai/`` model are configured.  LiteLLM
imports the hook and, because :func:`prewarm` is a coroutine function, awaits it
inside ``proxy_startup_event`` -- i.e. in the serving event loop.  That matters:
the process-wide :class:`~chutes_litellm.near.client.NearRuntime` is bound to
the loop that created its SDK clients, so prewarming from a different loop (or
at import time) would be unsound.

The hook is deliberately best-effort: a temporary NEAR AI outage must not keep
the proxy from starting for its other providers.  Failures are logged with
model ids and stable codes only -- never keys, prompts, evidence or upstream
messages.
"""

from __future__ import annotations

import os

from chutes_litellm import get_logger
from chutes_litellm.near.client import (
    API_KEY_ENV,
    get_runtime,
    prewarm_models_from_env,
)

_LOGGER_NAME = "chutes_litellm.near.startup"


def _safe_error_code(error: BaseException) -> str:
    """Return a stable, log-safe code for ``error`` (never its message)."""
    code = getattr(error, "code", None)
    return code if isinstance(code, str) else type(error).__name__


async def prewarm() -> None:
    """Verify the configured NEAR AI models in the serving event loop.

    No-op when ``NEARAI_API_KEY`` or ``NEARAI_PREWARM_MODELS`` is empty.  The
    batch uses the runtime's default 60 s total prewarm budget, so it can never
    hold LiteLLM readiness for the 900 s per-model hard timeout times the
    catalog size.  Any failure -- including runtime construction -- is caught
    and logged so the LiteLLM startup hook never raises and never blocks the
    proxy from serving.
    """
    logger = get_logger(_LOGGER_NAME)
    try:
        api_key = (os.environ.get(API_KEY_ENV) or "").strip()
        models = prewarm_models_from_env()
        if not api_key or not models:
            return

        runtime = get_runtime(api_key)
        results = await runtime.prewarm(models)
        failures = {model: error for model, error in results.items() if error is not None}
        logger.info(
            "NEAR AI startup prewarm: %d/%d models verified",
            len(results) - len(failures),
            len(results),
        )
        for model, error in failures.items():
            logger.warning("NEAR AI startup prewarm failed for model %s: %s", model, _safe_error_code(error))
    except Exception as exc:  # noqa: BLE001 - a startup hook must never break the proxy
        logger.warning("NEAR AI startup prewarm skipped: %s", _safe_error_code(exc))


__all__ = ["prewarm"]
