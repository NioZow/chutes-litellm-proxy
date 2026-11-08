"""NEAR AI ``/v1/models`` discovery and chat-capability normalisation.

This is the Phase 4 catalog layer from ``PLAN.md`` §7.  It only talks to the
fixed Cloud API endpoint in :mod:`chutes_litellm.near.endpoints`; the URL is
read at call time so tests can observe it, and there is no base-URL override.

The upstream catalog is OpenAI-shaped but richer than the bare OpenAI schema:
entries carry ``context_length`` / ``max_output_length``, ``input_modalities``,
``output_modalities`` and ``supported_features`` (``tools``, ``reasoning``,
...).  :func:`is_chat_model` deliberately tolerates unknown schemas -- it only
rejects catalog entries for which it has positive evidence that they are not
chat models -- so a future field addition does not silently drop chat models.

No verification, TLS, attestation or mode policy lives here: this module is
only used to generate LiteLLM config entries.  Requests are still served by the
verified provider in :mod:`chutes_litellm.near.provider`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from chutes_litellm.near import endpoints

#: Discovery timeout, matching the other config generators.
DEFAULT_TIMEOUT_SECONDS = 15.0

#: Explicit ``mode`` / ``type`` / ``model_type`` values must look like a chat
#: model.  Unknown values are treated as non-chat; absent values stay unknown.
_DECLARED_MODE_MARKERS = ("chat", "complet", "text", "language", "llm")

#: Id substrings that identify a non-chat model even when the catalog does not
#: advertise a mode.  Chat models accepting images are *not* excluded here:
#: ``vision`` is a capability, not a task type.
_EXCLUDED_ID_TOKENS = (
    "embed",
    "embedding",
    "rerank",
    "audio",
    "tts",
    "speech",
    "whisper",
    "transcribe",
    "moderation",
    "dall-e",
    "stable-diffusion",
    "sdxl",
    "flux",
    "image-gen",
)

_CONTEXT_LENGTH_KEYS = ("context_length", "max_context_length", "context_window", "max_input_tokens")
_OUTPUT_LENGTH_KEYS = ("max_output_tokens", "max_output_length", "max_completion_tokens")

_TOOL_FEATURE_KEYS = ("tools", "tool_calling", "function_calling", "supports_tools", "supports_function_calling")
_VISION_FEATURE_KEYS = ("vision", "supports_vision")
_REASONING_FEATURE_KEYS = ("reasoning", "thinking", "supports_reasoning")


def _json_list(value: object) -> list[str]:
    """Return the string members of a JSON list, or ``[]`` for anything else."""
    if not isinstance(value, (list, tuple, set)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _feature_names(model: dict) -> set[str]:
    """Collect lower-cased capability names from the known feature containers."""
    names: set[str] = set()
    for key in ("supported_features", "capabilities", "features"):
        names.update(item.lower() for item in _json_list(model.get(key)))
    return names


def _has_boolean(model: dict, keys: tuple[str, ...]) -> bool:
    """Return whether any of ``keys`` is explicitly ``True`` on ``model``."""
    return any(model.get(key) is True for key in keys)


def _enabled(model: dict, feature_keys: tuple[str, ...]) -> bool:
    """Return whether a capability is advertised as a feature or a boolean."""
    return bool(_feature_names(model) & set(feature_keys)) or _has_boolean(model, feature_keys)


def _input_modalities(model: dict) -> list[str]:
    """Return declared input modalities (top-level or nested ``architecture``)."""
    modalities = _json_list(model.get("input_modalities"))
    if modalities:
        return modalities
    architecture = model.get("architecture")
    if isinstance(architecture, dict):
        return _json_list(architecture.get("inputModalities") or architecture.get("input_modalities"))
    return []


def _output_modalities(model: dict) -> list[str]:
    """Return declared output modalities (top-level or nested ``architecture``)."""
    modalities = _json_list(model.get("output_modalities"))
    if modalities:
        return modalities
    architecture = model.get("architecture")
    if isinstance(architecture, dict):
        return _json_list(architecture.get("outputModalities") or architecture.get("output_modalities"))
    return []


def _positive_int(value: object) -> int | None:
    """Return ``value`` as a positive ``int``, or ``None`` when unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def _first_int(model: dict, keys: tuple[str, ...]) -> int | None:
    """Return the first positive integer among ``keys`` in ``model``."""
    for key in keys:
        value = _positive_int(model.get(key))
        if value is not None:
            return value
    return None


def _declared_chat_mode(model: dict) -> bool:
    """Return whether an explicit mode/type field is chat-compatible.

    An absent, empty or non-string value is treated as unknown (``True``).
    """
    for key in ("mode", "type", "model_type"):
        value = model.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip().lower()
        if not any(marker in normalized for marker in _DECLARED_MODE_MARKERS):
            return False
    return True


def fetch_models(api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[dict]:
    """Fetch the raw NEAR AI model catalog from the fixed ``/v1/models`` URL.

    Args:
        api_key: Bearer token sent in the ``Authorization`` header.  Never logged.
        timeout: Per-request timeout in seconds.

    Returns:
        Normalised model dicts: every upstream field is preserved and ``id`` is
        a stripped, non-empty string.  Junk entries (non-dicts, missing or
        empty ids) are dropped rather than failing the whole catalog.

    Raises:
        httpx.HTTPError: On transport or HTTP status failures (the caller --
            :func:`chutes_litellm.generate_config.expand_wildcard` -- logs and
            skips).
        TypeError: When the payload is neither an object with a list ``data``
            nor a bare list.
    """
    response = httpx.get(
        endpoints.models_url(),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, dict):
        raw = payload.get("data")
    else:
        raw = payload
    if not isinstance(raw, list):
        raise TypeError("NEAR AI /v1/models payload has no model list")

    models: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        if not isinstance(raw_id, str):
            continue
        model_id = raw_id.strip()
        if not model_id:
            continue
        normalized = dict(entry)
        normalized["id"] = model_id
        models.append(normalized)
    return models


def is_chat_model(model: dict) -> bool:
    """Return whether ``model`` is a chat/completions model.

    The filter is intentionally conservative about rejecting: it only returns
    ``False`` on positive evidence (an explicit non-chat mode, a non-chat id
    token, or input modalities without ``text``).  Vision chat models are kept.
    """
    if not isinstance(model, dict):
        return False
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return False

    lowered = model_id.strip().lower()
    if any(token in lowered for token in _EXCLUDED_ID_TOKENS):
        return False

    if not _declared_chat_mode(model):
        return False

    modalities = _input_modalities(model)
    if modalities:
        return "text" in {item.lower() for item in modalities}

    return True


def discover_models(api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[dict]:
    """Fetch and filter the catalog to chat models.

    Errors from :func:`fetch_models` are deliberately **not** swallowed: the
    config generator catches them, logs, and skips the provider.
    """
    return [model for model in fetch_models(api_key, timeout=timeout) if is_chat_model(model)]


def _supports_vision(model: dict) -> bool:
    """Return whether the model accepts image input (modalities or feature)."""
    if _enabled(model, _VISION_FEATURE_KEYS):
        return True
    return "image" in {item.lower() for item in _input_modalities(model)}


def normalized_model_info(model: dict) -> dict:
    """Build a LiteLLM ``model_info`` block from a NEAR AI catalog entry.

    Maps the upstream capability/limit metadata to the fields opencode reads
    from ``/v1/model/info``.  Output length is included only when the catalog
    advertises one (unlike Chutes, NEAR reports a meaningful value).
    """
    # Local import: ``generate_config`` imports this module lazily to keep the
    # NEAR SDK out of non-NEAR startup paths; importing it at module load here
    # would create a cycle and heavy side effects.
    from chutes_litellm.generate_config import _REASONING_EFFORTS

    info: dict[str, Any] = {
        "mode": "chat",
        "supports_function_calling": _enabled(model, _TOOL_FEATURE_KEYS),
        "supports_vision": _supports_vision(model),
        "supports_reasoning": _enabled(model, _REASONING_FEATURE_KEYS),
    }

    input_tokens = _first_int(model, _CONTEXT_LENGTH_KEYS)
    if input_tokens is not None:
        info["max_input_tokens"] = input_tokens

    output_tokens = _first_int(model, _OUTPUT_LENGTH_KEYS)
    if output_tokens is not None:
        info["max_output_tokens"] = output_tokens

    input_modalities = _input_modalities(model)
    if input_modalities:
        info["input_modalities"] = sorted({item.lower() for item in input_modalities})
    output_modalities = _output_modalities(model)
    if output_modalities:
        info["output_modalities"] = sorted({item.lower() for item in output_modalities})

    created = model.get("created")
    if isinstance(created, (int, float)) and not isinstance(created, bool) and created > 0:
        info["release_date"] = time.strftime("%Y-%m-%d", time.gmtime(created))

    if info["supports_reasoning"]:
        info["reasoning_effort_levels"] = list(_REASONING_EFFORTS)
        for effort in _REASONING_EFFORTS:
            info[f"supports_{effort}_reasoning_effort"] = True

    return info


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "discover_models",
    "fetch_models",
    "is_chat_model",
    "normalized_model_info",
]
