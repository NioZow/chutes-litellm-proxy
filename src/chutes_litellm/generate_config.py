"""
Reads config.template.yml, expands wildcard provider entries into concrete
model entries by querying each provider's models API, then writes
config.generated.yml. Runs once at startup (Docker entrypoint, Nix wrapper,
`litellm-proxy` console script) or standalone (scripts/generate_config.py).
"""

import os
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

# Paths are overridable through the environment (the Nix wrapper and the NixOS
# module set LITELLM_TEMPLATE/LITELLM_OUTPUT).  Defaults resolve to the template
# shipped as package data next to this module, and a generated config in the
# current working directory (the Docker image sets WORKDIR /app).
CONFIG_DIR = Path(__file__).resolve().parent / "config"
TEMPLATE = os.environ.get("LITELLM_TEMPLATE") or str(CONFIG_DIR / "config.template.yml")
OUTPUT = os.environ.get("LITELLM_OUTPUT") or str(Path.cwd() / "config.generated.yml")
TIMEOUT = 15
CHUTES_API_BASE = "https://llm.chutes.ai"


def log(msg: str) -> None:
    print(f"[generate_config] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Resolve *_API_KEY_PATH env variables so API keys can be declared as file
# paths in systemd service files (e.g. ANTHROPIC_API_KEY_PATH=/run/secrets/...)
# ---------------------------------------------------------------------------

def _resolve_api_key_paths() -> None:
    """For any env var ending in `_API_KEY_PATH`, read the file and inject the
    raw value as the corresponding `_API_KEY` variable into ``os.environ``."""
    for key, path in os.environ.items():
        if not key.endswith("_API_KEY_PATH"):
            continue
        base = key[:-5]  # strip `_PATH`
        if base in os.environ:
            continue  # explicit key wins
        try:
            with open(path, "r") as f:
                os.environ[base] = f.read().strip()
            log(f"resolved {key} -> {base}")
        except OSError as exc:
            log(f"failed to read {key} ({exc})")


# Run once at module load so callers don't need to remember it.
_resolve_api_key_paths()


# Per-provider fetch functions — return list of bare model IDs


# Blacklists — models matching these prefixes are dropped.
SKIP_OPENAI = {
    "text-embedding",
    "text-moderation",
    "omni-moderation",
    "babbage",
    "davinci",
}

SKIP_GEMINI = {
    "embedding",
}

SKIP_MISTRAL = {
    "mistral-embed",
    "codestral-embed",
    "mistral-moderation",
}


def fetch_anthropic(api_key: str) -> list[str]:
    r = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def fetch_openai(api_key: str) -> list[str]:
    r = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    all_ids = [m["id"] for m in r.json()["data"]]
    return sorted(m for m in all_ids if not any(m.startswith(p) for p in SKIP_OPENAI))


def fetch_gemini(api_key: str) -> list[str]:
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    all_ids = [
        m["name"].removeprefix("models/")
        for m in r.json().get("models", [])
    ]
    return sorted(m for m in all_ids if not any(m.startswith(p) for p in SKIP_GEMINI))


#: Id substrings that identify a non-chat Chutes deployment.  Chutes serves
#: both TEE (``-TEE``) and plain models through the same catalog; only chat
#: models belong in the proxy's model list.
_CHUTES_NON_CHAT_TOKENS = (
    "embed",
    "embedding",
    "rerank",
    "audio",
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


def _is_chat_chutes_model(model: dict) -> bool:
    """Return whether a Chutes catalog entry is a chat/completions model.

    Chutes does not publish a ``mode`` field, so the id is the only signal;
    unrecognised ids are kept (conservative: a new chat model must not be
    dropped).  Embedding/rerank/audio/image deployments are excluded.
    """
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    lowered = model_id.lower()
    return not any(token in lowered for token in _CHUTES_NON_CHAT_TOKENS)


def fetch_chutes(api_key: str) -> list[dict]:
    r = httpx.get(
        f"{CHUTES_API_BASE}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    # Keep TEE *and* plain chat models, with full metadata so we can advertise
    # capabilities (reasoning, tools, vision), limits, and the TEE flag.
    return [m for m in r.json()["data"] if isinstance(m, dict) and _is_chat_chutes_model(m)]


def fetch_openai_compat(api_key: str, base_url: str) -> list[str]:
    r = httpx.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def fetch_mistral(api_key: str) -> list[str]:
    r = httpx.get(
        "https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    all_ids = [m["id"] for m in r.json()["data"]]
    return sorted(m for m in all_ids if not any(m.startswith(p) for p in SKIP_MISTRAL))


def fetch_nearai(api_key: str) -> list[dict]:
    """Fetch NEAR AI chat models from the fixed Cloud API ``/v1/models`` URL.

    The import is local so a deployment without ``NEAR_API_KEY`` never pulls
    the NEAR AI SDK into the config-generation path.  Chat filtering and
    capability normalisation live in :mod:`chutes_litellm.near.models`.
    """
    from chutes_litellm.near.models import discover_models

    return discover_models(api_key, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "fetch": fetch_anthropic,
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "fetch": fetch_openai,
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "fetch": fetch_gemini,
    },
    "perplexity": {
        "env": "PERPLEXITY_API_KEY",
        "fetch": lambda key: fetch_openai_compat(key, "https://api.perplexity.ai/v1"),
    },
    "xai": {
        "env": "XAI_API_KEY",
        "fetch": lambda key: fetch_openai_compat(key, "https://api.x.ai/v1"),
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "fetch": lambda key: fetch_openai_compat(key, "https://api.deepseek.com"),
    },
    "meta": {
        "env": "META_API_KEY",
        "fetch": lambda key: fetch_openai_compat(key, "https://api.meta.ai/v1"),
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "fetch": fetch_mistral,
    },
    "chutes": {
        "env": "CHUTES_API_KEY",
        "fetch": fetch_chutes,
    },
    "nearai": {
        "env": "NEAR_API_KEY",
        "fetch": fetch_nearai,
    },
}


# ---------------------------------------------------------------------------
# Display names
#
# Every model is exposed under the single `litellm` provider, so clients that
# only look at the model id cannot tell a Chutes TEE deployment from the same
# model served directly by its vendor.  Each entry therefore carries
# `model_info.display_name = "<Pretty Name> (<Source>)"`, which survives
# LiteLLM's `/v1/model/info` passthrough.  The name is computed here, at proxy
# startup, from the live provider catalogs -- clients never need a static model
# list.
# ---------------------------------------------------------------------------

#: Human-readable label for the first segment of a model name.
SOURCE_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Google",
    "perplexity": "Perplexity",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "meta": "Meta",
    "mistral": "Mistral",
    "chutes": "Chutes",
    "nearai": "NEAR",
}

#: TEE models are labelled with their host, e.g. ``(Chutes TEE)``.  A model
#: served through NEAR AI's gateway only gets ``(NEAR)``: the TEE badge means
#: the model runs on the provider's own attested hardware.
TEE_LABELS: dict[str, str] = {
    "chutes": "Chutes TEE",
    "nearai": "NEAR TEE",
}

#: Lower-cased tokens rendered in upper case instead of title case.
_ACRONYMS = {
    "ai",
    "api",
    "asr",
    "cli",
    "e2ee",
    "fim",
    "ggml",
    "gguf",
    "glm",
    "gpt",
    "hd",
    "it",
    "llm",
    "mlx",
    "ocr",
    "oss",
    "sdk",
    "stt",
    "tee",
    "tts",
    "vl",
}

#: Mixed-case brand names that generic title casing would mangle.
_BRAND_NAMES = {
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "minimaxai": "MiniMax",
    "qwen": "Qwen",
}


_TEE_SUFFIX = re.compile(r"[-_ ]?TEE$", re.IGNORECASE)


def _strip_tee_suffix(model_id: str) -> str:
    """Remove a trailing ``-TEE`` marker from a model id (badge carries it)."""
    return _TEE_SUFFIX.sub("", model_id).strip()


def _join_version_pair(tokens: list[str]) -> list[str]:
    """Collapse a trailing ``4-5`` pair into ``4.5`` (Anthropic-style ids).

    Only the first mergeable pair scanning from the right is collapsed, and a
    four-digit year immediately before the pair blocks the merge so dashed
    dates (``gpt-4.1-2025-04-14``) keep their dashes.
    """
    for index in range(len(tokens) - 1, 0, -1):
        left, right = tokens[index - 1], tokens[index]
        if not (re.fullmatch(r"[vkm]?\d{1,2}", left) and re.fullmatch(r"\d{1,2}", right)):
            continue
        previous = tokens[index - 2] if index >= 2 else None
        if previous is not None and re.fullmatch(r"(19|20)\d{2}", previous):
            continue
        if previous is not None and re.fullmatch(r"\d{1,2}", previous):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            following is not None
            and re.fullmatch(r"\d+", following)
            and not re.fullmatch(r"\d{6,8}", following)
        ):
            continue
        return [*tokens[: index - 1], f"{left}.{right}", *tokens[index + 1 :]]
    return tokens


def humanize_model_id(model_id: str) -> str:
    """Turn the last path segment of a model id into a readable name.

    Only the last segment is used so ``nearai/anthropic/claude-sonnet-5``
    becomes ``Claude Sonnet 5`` (the source label already carries NEAR).
    Dashes and underscores separate words; dots stay inside version tokens
    (``deepseek-v4.1-flash`` -> ``DeepSeek V4.1 Flash``) and dash-separated
    version pairs collapse (``claude-opus-4-5`` -> ``Claude Opus 4.5``).
    """
    raw = model_id.rsplit("/", 1)[-1]
    tokens = [token for token in re.split(r"[-_\s]+", raw) if token]
    words: list[str] = []
    for token in _join_version_pair(tokens):
        lower = token.lower()
        if lower in _BRAND_NAMES:
            words.append(_BRAND_NAMES[lower])
        elif lower in _ACRONYMS:
            words.append(token.upper())
        elif re.fullmatch(r"\d+[bkmg]", lower) or re.fullmatch(r"q\d+", lower):
            # Sizes (27b, 4k) and quantization levels (q4, q8).
            words.append(token.upper())
        elif re.fullmatch(r"[a-z]\d+[a-z]?", lower):
            # Shapes like a4b, a12b, e4b.
            words.append(token.upper())
        elif re.search(r"\d", token):
            # Version-like tokens keep their internal punctuation/casing.
            words.append(token[0].upper() + token[1:])
        else:
            words.append(token[0].upper() + token[1:].lower())
    return " ".join(words) or raw


def _catalog_display_name(model_id: str, name: object) -> str:
    """Prefer a provider-supplied display name when it is a real name.

    NEAR AI's catalog carries ``name`` (``"Claude Opus 5"``, ``"Qwen 3.8 27B"``)
    which is better than anything derived from the id.  Some entries simply
    repeat the slug (``"deepseek-v3.2"``); those fall back to the humanizer.
    """
    if isinstance(name, str):
        candidate = name.strip()
        if candidate and (" " in candidate or candidate[:1].isupper() or candidate[:1].isdigit()):
            return candidate
    return humanize_model_id(model_id)


def source_label(provider: str) -> str:
    """Return the human-readable source label for a provider prefix."""
    if provider in SOURCE_LABELS:
        return SOURCE_LABELS[provider]
    return provider.replace("-", " ").replace("_", " ").title()


def build_display_name(
    model_name: str,
    *,
    tee: bool | None = None,
    name: object = None,
) -> str:
    """Build ``<Pretty Model Name> (<Source>[ TEE])`` from a LiteLLM model name.

    Args:
        model_name: LiteLLM model name (``chutes/Qwen/Qwen3.8-27B-TEE``).
        tee: Whether the backend runs on attested hardware.  ``None`` infers it
            from a ``-TEE`` id suffix (Chutes).
        name: Optional provider-supplied display name (NEAR AI's catalog).
    """
    provider, separator, model_id = model_name.partition("/")
    if not separator:
        return humanize_model_id(model_name)

    if provider == "chutes":
        is_tee = model_id.upper().endswith("-TEE") if tee is None else tee
        label = TEE_LABELS["chutes"] if is_tee else source_label("chutes")
        pretty = _catalog_display_name(_strip_tee_suffix(model_id), name)
    elif provider == "nearai":
        is_tee = bool(tee)
        label = TEE_LABELS["nearai"] if is_tee else source_label("nearai")
        pretty = _catalog_display_name(model_id, name)
    else:
        pretty = _catalog_display_name(model_id, name)
        label = source_label(provider)
    return f"{pretty} ({label})"


#: Limits and capabilities for direct-provider models that LiteLLM's bundled
#: model map does not know yet.  Chutes and NEAR AI entries carry their own
#: metadata, so this table is only consulted for the plain OpenAI-compatible
#: providers.  Keys are matched as model-id prefixes, longest match first;
#: values mirror what the vendor/models.dev publishes.  Unknown models keep
#: their previous behaviour (no fabricated limits).
DIRECT_MODEL_METADATA: dict[str, dict[str, dict]] = {
    "deepseek": {
        "deepseek-flash": {
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "max_output_tokens": 393_216,
            "supports_function_calling": True,
            "supports_reasoning": True,
            "supports_vision": True,
            "release_date": "2026-10-17",
            "reasoning_effort_levels": ["low", "high", "max"],
        },
        "deepseek-v4-pro": {
            "mode": "chat",
            "max_input_tokens": 1_048_576,
            "max_output_tokens": 393_216,
            "supports_function_calling": True,
            "supports_reasoning": True,
            "supports_vision": False,
            "release_date": "2026-08-13",
            "reasoning_effort_levels": ["low", "high", "max"],
        },
    },
}


def _direct_model_metadata(provider: str, model_id: str) -> dict:
    """Return a copy of the metadata entry for a direct-provider model."""
    table = DIRECT_MODEL_METADATA.get(provider, {})
    match = table.get(model_id)
    if match is None:
        for prefix in sorted(table, key=len, reverse=True):
            if model_id.startswith(prefix):
                match = table[prefix]
                break
    return dict(match) if match else {}


def _with_display_name(
    model_name: str,
    info: dict | None,
    *,
    tee: bool | None = None,
    name: object = None,
) -> dict:
    """Merge generated metadata under explicit/upstream model info.

    Upstream fields win over the static table, and an explicit
    ``display_name`` (for example one written by the user in a fixed entry)
    is never overwritten.
    """
    provider, _, model_id = model_name.partition("/")
    merged = _direct_model_metadata(provider, model_id)
    merged.update(info or {})
    merged.setdefault(
        "display_name", build_display_name(model_name, tee=tee, name=name)
    )
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Reasoning-effort levels supported by DeepSeek-family thinking models. These
# surface as opencode variants via `supports_<effort>_reasoning_effort: true`
# flags in model_info (see opencode-plugin-litellm, which scans model_info for
# keys matching `supports_([a-z]+)_reasoning_effort`).
_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

# OpenAI-style parameters Chutes and NEAR AI models accept that LiteLLM would
# otherwise strip (its custom-provider param list is a plain OpenAI list).
# These are declared per deployment so a client can actually send
# `reasoning_effort` (DeepSeek thinking levels) or `chat_template_kwargs` /
# `thinking` (servers that gate thinking behind a template flag, e.g. Gemma).
_ALLOWED_OPENAI_PARAMS = (
    "reasoning_effort",
    "thinking",
    "reasoning",
    "chat_template_kwargs",
    "top_k",
    "repetition_penalty",
    "min_p",
    "enable_thinking",
)


def _build_chutes_model_info(model: dict) -> dict:
    """Build a LiteLLM `model_info` block from Chutes model metadata.

    Advertises capabilities opencode reads from `/v1/model/info`:
      - supports_reasoning / reasoning-effort variants
      - supports_function_calling / supports_vision
      - max_input_tokens / max_output_tokens
      - mode: "chat"

    Output limits are deliberately reported as ``0`` for every Chutes model,
    regardless of the upstream ``max_output_length`` metadata.  This lets each
    client fall back to its own default output limit (opencode uses 32000)
    instead of inheriting a Chutes-specific value.  Input limits still come
    from Chutes ``context_length``.
    """
    features = set(model.get("supported_features") or [])
    modalities = set(model.get("input_modalities") or [])

    info: dict = {
        "mode": "chat",
        "supports_function_calling": "tools" in features,
        "supports_vision": "image" in modalities,
        "supports_reasoning": "reasoning" in features,
        # Chutes marks confidential-compute (TEE) deployments explicitly.
        # Non-TEE models are served in plaintext and must bypass the E2EE
        # transport, so the flag is mirrored for the startup wiring.
        "confidential_compute": bool(model.get("confidential_compute")),
    }

    if model.get("context_length"):
        info["max_input_tokens"] = model["context_length"]
    # Always 0 so clients apply their own default output limit.  Chutes
    # upstream reports ``max_output_length``, but it must not be propagated.
    info["max_output_tokens"] = 0

    if modalities:
        info["input_modalities"] = sorted(modalities)
    output_modalities = set(model.get("output_modalities") or [])
    if output_modalities:
        info["output_modalities"] = sorted(output_modalities)

    sampling = model.get("supported_sampling_parameters") or []
    if sampling:
        info["supported_sampling_parameters"] = sorted(sampling)

    created = model.get("created")
    if isinstance(created, (int, float)) and created > 0:
        info["release_date"] = time.strftime("%Y-%m-%d", time.gmtime(created))

    # Any model that supports reasoning gets the effort variants.  On Chutes a
    # live ``reasoning_effort`` enables thinking for the whole family: DeepSeek
    # honors the effort directly, while sglang/vLLM models (Gemma/V3.x) are
    # switched on through the template flag the custom provider derives from it.
    if "reasoning" in features:
        info["reasoning_effort_levels"] = list(_REASONING_EFFORTS)
        for effort in _REASONING_EFFORTS:
            info[f"supports_{effort}_reasoning_effort"] = True

    return info


def expand_wildcard(provider: str, template_entry: dict) -> list[dict]:
    """Fetch models for a provider and return concrete config entries."""
    cfg = PROVIDERS.get(provider)
    if not cfg:
        log(f"unknown provider '{provider}', keeping wildcard as-is")
        return [template_entry]

    api_key = os.environ.get(cfg["env"])
    if not api_key:
        log(f"{cfg['env']} not set — skipping {provider}")
        return []

    try:
        models = cfg["fetch"](api_key)
        log(f"{provider}: {len(models)} models fetched")
        entries: list[dict] = []
        for model in models:
            if isinstance(model, dict):
                model_id = model.get("id", "")
                # NEAR AI is served through its CustomLLM provider; the
                # user-facing name stays `nearai/<id>`.
                wire_provider = "nearai_e2ee" if provider == "nearai" else provider
                litellm_params: dict = {
                    "model": f"{wire_provider}/{model_id}",
                    "api_key": f"os.environ/{cfg['env']}",
                }
                # Let clients send thinking params LiteLLM would otherwise drop
                # for a custom/OpenAI-compatible provider.
                if provider in {"chutes", "nearai"}:
                    litellm_params["allowed_openai_params"] = list(_ALLOWED_OPENAI_PARAMS)
                entry: dict = {
                    "model_name": f"{provider}/{model_id}",
                    "litellm_params": litellm_params,
                }
                if provider == "nearai":
                    from chutes_litellm.near.models import (
                        is_tee_model,
                        normalized_model_info,
                    )

                    info = normalized_model_info(model)
                    # NEAR publishes true TEE models as `owned_by: nearai`
                    # (providerType vllm + attestation).  Models it merely
                    # proxies (`owned_by: attested 3p`, vendor names) are not
                    # NEAR TEE.
                    tee = is_tee_model(model)
                    display_name = model.get("name")
                else:
                    info = _build_chutes_model_info(model)
                    tee = bool(model.get("confidential_compute")) or str(
                        model_id
                    ).upper().endswith("-TEE")
                    display_name = None
                info = _with_display_name(
                    entry["model_name"], info, tee=tee, name=display_name
                )
                if info:
                    entry["model_info"] = info
                entries.append(entry)
            else:
                model_name = f"{provider}/{model}"
                info = _with_display_name(model_name, None)
                entry = {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": f"{provider}/{model}",
                        "api_key": f"os.environ/{cfg['env']}",
                    },
                }
                if info:
                    entry["model_info"] = info
                entries.append(entry)
        return entries
    except Exception as exc:  # noqa: BLE001 - one provider must never break config generation
        log(f"failed to fetch {provider} models ({exc}) — skipping")
        return []


def _annotate_fixed_entry(entry: dict) -> dict:
    """Add a display name to a non-wildcard template entry.

    User-provided ``model_info`` (including an explicit ``display_name``) is
    preserved; only a missing display name is filled in.
    """
    info = entry.get("model_info")
    if info is None:
        info = {}
    if not isinstance(info, dict) or info.get("display_name"):
        return entry
    info["display_name"] = build_display_name(entry["model_name"])
    entry["model_info"] = info
    return entry


def generate(template: str | None = None, output: str | None = None) -> str:
    """Expand the model template into a generated config file.

    ``template`` / ``output`` fall back to the LITELLM_TEMPLATE / LITELLM_OUTPUT
    env vars, then to the defaults computed at import time.  Returns the path of
    the generated file.
    """
    template = template or TEMPLATE
    output = output or OUTPUT

    with open(template) as f:
        config = yaml.safe_load(f)

    fixed: list[dict] = []
    expanded: list[dict] = []

    for entry in config.get("model_list", []):
        name: str = entry["model_name"]
        if name.endswith("/*"):
            provider = name[:-2]  # "anthropic/*" -> "anthropic"
            expanded.extend(expand_wildcard(provider, entry))
        else:
            fixed.append(_annotate_fixed_entry(entry))

    config["model_list"] = fixed + expanded
    total = len(config["model_list"])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    log(f"wrote {output} with {total} model(s)")
    return output


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
