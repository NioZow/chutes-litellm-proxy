#!/usr/bin/env python3
"""
Reads config.template.yml, expands wildcard provider entries into concrete
model entries by querying each provider's models API, then writes
config.generated.yml. Runs once at container startup.
"""

import os
import sys

import httpx
import yaml

# Paths are overridable so the same script works both in the container
# (defaults, see Dockerfile) and in the Nix wrapper (which sets the env vars).
TEMPLATE = os.environ.get("LITELLM_TEMPLATE", "/app/config.template.yml")
OUTPUT = os.environ.get("LITELLM_OUTPUT", "/app/config.generated.yml")
TIMEOUT = 15
CHUTES_API_BASE = "https://llm.chutes.ai"


def log(msg: str) -> None:
    print(f"[generate_config] {msg}", file=sys.stderr)


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
    "chutes-e2ee": {
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
_DEEPSEEK_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _build_chutes_model_info(model: dict) -> dict:
    """Build a LiteLLM `model_info` block from Chutes model metadata.

    Advertises capabilities opencode reads from `/v1/model/info`:
      - supports_reasoning / reasoning-effort variants (for DeepSeek thinking models)
      - supports_function_calling / supports_vision
      - max_input_tokens / max_output_tokens
      - mode: "chat"
    """
    features = set(model.get("supported_features") or [])
    modalities = set(model.get("input_modalities") or [])
    model_id = model.get("id", "")

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

    # DeepSeek-family thinking models advertise reasoning-effort variants.
    if "deepseek" in model_id.lower() and "reasoning" in features:
        for effort in _DEEPSEEK_REASONING_EFFORTS:
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
                litellm_params = {
                    "model": f"{provider}/{model_id}",
                    "api_key": f"os.environ/{cfg['env']}",
                }
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


def main() -> None:
    with open(TEMPLATE) as f:
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

    with open(OUTPUT, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    log(f"wrote {OUTPUT} with {total} model(s)")


if __name__ == "__main__":
    main()
