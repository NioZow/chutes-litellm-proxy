"""Fixed NEAR AI Cloud API endpoints.

These are module-level attributes on purpose: the NEAR AI Cloud API host is not
configurable and there is no environment or CLI override surface.  Production
modules import this module and read ``endpoints.API_BASE`` at call time (never
binding it into local constants or default arguments at import time) so a test
can monkeypatch the module attribute.

The base URL already includes the ``/v1`` suffix; the path helpers below return
URLs relative to it.  This mirrors the upstream SDK convention: the SDK joins
``chat/completions``, ``model/{model}``, ``signature/{id}`` and
``attestation/report`` onto ``DEFAULT_NEAR_AI_CLOUD_BASE_URL``.

Response signatures and attestations are fetched by the SDK's
``AttestationClient``; :func:`completion_signature_path` exists only so callers
can log or test the expected URL shape.  It does not duplicate the SDK's
fetching or verification logic.
"""

from __future__ import annotations

from urllib.parse import quote

from nearai_inference_sdk import DEFAULT_NEAR_AI_CLOUD_BASE_URL

API_BASE = "https://cloud-api.near.ai/v1"

# Guard against drifting away from the audited SDK if upstream changes its
# default endpoint.  Fixing the literal here is deliberate: this provider must
# never be redirected by configuration.
if API_BASE != DEFAULT_NEAR_AI_CLOUD_BASE_URL:  # pragma: no cover - import guard
    raise RuntimeError(
        "NEAR AI base URL diverged from nearai-inference-sdk: "
        f"local={API_BASE!r} sdk={DEFAULT_NEAR_AI_CLOUD_BASE_URL!r}"
    )

CHAT_COMPLETIONS_PATH = "chat/completions"
MODELS_PATH = "models"
_MODEL_PATH = "model"
_SIGNATURE_PATH = "signature"


def _url(path: str) -> str:
    """Join ``path`` onto the fixed base URL read at call time."""
    return f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"


def chat_completions_url() -> str:
    """Return the chat-completions URL (``/v1/chat/completions``)."""
    return _url(CHAT_COMPLETIONS_PATH)


def models_url() -> str:
    """Return the model-discovery URL (``/v1/models``)."""
    return _url(MODELS_PATH)


def model_metadata_path(model: str) -> str:
    """Return the relative metadata path for ``model`` (``model/{quoted}``)."""
    return f"{_MODEL_PATH}/{quote(model, safe='')}"


def model_metadata_url(model: str) -> str:
    """Return the absolute model-metadata URL for ``model``."""
    return _url(model_metadata_path(model))


def completion_signature_path(completion_id: str) -> str:
    """Return the relative completion-signature path for ``completion_id``.

    The quoting matches the SDK: slashes are escaped and dots are percent
    encoded because HTTP clients normalize bare ``.``/``..`` path segments.
    """
    escaped = quote(completion_id, safe="").replace(".", "%2E")
    return f"{_SIGNATURE_PATH}/{escaped}"


__all__ = [
    "API_BASE",
    "CHAT_COMPLETIONS_PATH",
    "MODELS_PATH",
    "chat_completions_url",
    "completion_signature_path",
    "model_metadata_path",
    "model_metadata_url",
    "models_url",
]
