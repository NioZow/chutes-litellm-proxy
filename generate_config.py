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

TEMPLATE = "/app/config.template.yml"
OUTPUT = "/app/config.generated.yml"
TIMEOUT = 15
CHUTES_API_BASE = "https://llm.chutes.ai"


def log(msg: str) -> None:
    print(f"[generate_config] {msg}", file=sys.stderr)


# Per-provider fetch functions — return list of bare model IDs


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
    # Filter to chat/reasoning models; exclude embeddings, tts, dall-e, whisper, etc.
    keep = ("gpt-4", "gpt-3.5-turbo", "o1", "o3", "o4", "chatgpt-4o")
    return sorted(m for m in all_ids if any(m.startswith(p) for p in keep))


def fetch_gemini(api_key: str) -> list[str]:
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    models = []
    for m in r.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            # "models/gemini-2.0-flash" -> "gemini-2.0-flash"
            models.append(m["name"].removeprefix("models/"))
    return sorted(models)


def fetch_chutes(api_key: str) -> list[str]:
    r = httpx.get(
        f"{CHUTES_API_BASE}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    all_ids = [m["id"] for m in r.json()["data"]]
    # Keep only TEE (end-to-end encrypted) models
    return [m for m in all_ids if m.endswith("-TEE")]


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
        "fetch": lambda key: fetch_openai_compat(key, "https://api.perplexity.ai"),
    },
    "xai": {
        "env": "XAI_API_KEY",
        "fetch": lambda key: fetch_openai_compat(key, "https://api.x.ai/v1"),
    },
    "chutes-e2ee": {
        "env": "CHUTES_API_KEY",
        "fetch": fetch_chutes,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        return [
            {
                "model_name": f"{provider}/{model_id}",
                "litellm_params": {
                    "model": f"{provider}/{model_id}",
                    "api_key": f"os.environ/{cfg['env']}",
                },
            }
            for model_id in models
        ]
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
