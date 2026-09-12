"""
Reads config.template.yml, expands wildcard provider entries into concrete
model entries by querying each provider's models API, then writes
config.generated.yml. Runs once at startup (Docker entrypoint, Nix wrapper,
`litellm-proxy` console script) or standalone (scripts/generate_config.py).
"""

import os
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


def fetch_chutes(api_key: str) -> list[dict]:
    r = httpx.get(
        f"{CHUTES_API_BASE}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    # Keep only TEE (end-to-end encrypted) models, with full metadata so we can
    # advertise capabilities (reasoning, tools, vision) and limits to opencode.
    return [m for m in r.json()["data"] if m.get("id", "").endswith("-TEE")]


def fetch_openai_compat(api_key: str, base_url: str) -> list[str]:
    r = httpx.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


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
    "chutes": {
        "env": "CHUTES_API_KEY",
        "fetch": fetch_chutes,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Reasoning-effort levels supported by DeepSeek-family thinking models. These
# surface as opencode variants via `supports_<effort>_reasoning_effort: true`
# flags in model_info (see opencode-plugin-litellm, which scans model_info for
# keys matching `supports_([a-z]+)_reasoning_effort`).
_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

# OpenAI-style parameters Chutes models accept that LiteLLM would otherwise
# strip (its custom-provider param list is a plain OpenAI list).  These are
# declared per deployment so a client can actually send `reasoning_effort`
# (DeepSeek thinking levels) or `chat_template_kwargs` / `thinking` (servers
# that gate thinking behind a template flag, e.g. Gemma).
_CHUTES_ALLOWED_OPENAI_PARAMS = (
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
    """
    features = set(model.get("supported_features") or [])
    modalities = set(model.get("input_modalities") or [])

    info: dict = {
        "mode": "chat",
        "supports_function_calling": "tools" in features,
        "supports_vision": "image" in modalities,
        "supports_reasoning": "reasoning" in features,
    }

    if model.get("context_length"):
        info["max_input_tokens"] = model["context_length"]
    if model.get("max_output_length"):
        info["max_output_tokens"] = model["max_output_length"]

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
                litellm_params: dict = {
                    "model": f"{provider}/{model_id}",
                    "api_key": f"os.environ/{cfg['env']}",
                }
                # Let clients send thinking params LiteLLM would otherwise drop
                # for a custom/OpenAI-compatible provider.
                if provider == "chutes":
                    litellm_params["allowed_openai_params"] = list(_CHUTES_ALLOWED_OPENAI_PARAMS)
                entry: dict = {
                    "model_name": f"{provider}/{model_id}",
                    "litellm_params": litellm_params,
                }
                info = _build_chutes_model_info(model)
                if info:
                    entry["model_info"] = info
                entries.append(entry)
            else:
                entries.append(
                    {
                        "model_name": f"{provider}/{model}",
                        "litellm_params": {
                            "model": f"{provider}/{model}",
                            "api_key": f"os.environ/{cfg['env']}",
                        },
                    }
                )
        return entries
    except Exception as exc:
        log(f"failed to fetch {provider} models ({exc}) — skipping")
        return []


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
            fixed.append(entry)

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
