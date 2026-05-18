#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///
"""
Generate or update the models block in opencode.jsonc from LiteLLM's /model/info API.

Strategy:
  - All boolean capability flags (tool_call, temperature, attachment, reasoning) are derived
    exclusively from LiteLLM's model_info fields — no keyword guessing, no hardcoded lists.
  - For models with no LiteLLM metadata (custom routes like chutes-e2ee), all capabilities
    default to True (conservative: assume the model supports everything until proven otherwise).
  - Limits fall back in order: LiteLLM API → Chutes API → models.dev → existing config → hardcoded default.
  - When --apply is used, the script rewrites only the provider.litellm.models block in
    opencode.jsonc, preserving all other config untouched (agent temps, permissions, etc.)
    and preserving any manually set "name" values.

Usage:
  ./gen_opencode_models.py                          # print JSON models block, all available models
  ./gen_opencode_models.py --filter-current         # only models already in opencode.jsonc
  ./gen_opencode_models.py --apply                  # update opencode.jsonc in-place
  ./gen_opencode_models.py --output ~/opencode.json # update a plain JSON file in-place
  ./gen_opencode_models.py --debug                  # show why each model was skipped
  ./gen_opencode_models.py --url http://...         # custom LiteLLM base URL
  ./gen_opencode_models.py --chutes-api-key cpk_...# Chutes API key for TEE model limits
                                                    # (also reads CHUTES_API_KEY env var)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

LITELLM_BASE = "http://litellm-proxy:4000"
CHUTES_BASE = "https://llm.chutes.ai"
OPENCODE_CONFIG = Path.home() / ".config/opencode/opencode.jsonc"

FALLBACK_CONTEXT = 128000
FALLBACK_OUTPUT = 8192

# Cache for models.dev data
_MODELS_DEV_CACHE = None


def get_models_dev_data() -> dict:
    """Fetch and cache models.dev API data."""
    global _MODELS_DEV_CACHE
    if _MODELS_DEV_CACHE is None:
        try:
            resp = requests.get("https://models.dev/api.json", timeout=10)
            resp.raise_for_status()
            _MODELS_DEV_CACHE = resp.json()
        except Exception as e:
            print(f"Warning: Failed to fetch models.dev data: {e}", file=sys.stderr)
            _MODELS_DEV_CACHE = {}
    return _MODELS_DEV_CACHE


def find_models_dev_limits(model_id: str) -> tuple[int | None, int | None]:
    """Try to find context and output limits from models.dev for a given model ID."""
    data = get_models_dev_data()
    if not data:
        return None, None

    # model_id is usually "provider/model-name"
    parts = model_id.split("/", 1)
    if len(parts) == 2:
        provider, model_name = parts

        # Mapping LiteLLM providers to models.dev providers
        provider_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "gemini": "google",
            "xai": "xai",
            "perplexity": "perplexity",
        }

        dev_provider = provider_map.get(provider)
        if dev_provider and dev_provider in data:
            provider_models = data[dev_provider].get("models", {})

            # Try exact match first
            if model_name in provider_models:
                limits = provider_models[model_name].get("limit", {})
                return limits.get("context"), limits.get("output")

            # Try fuzzy match (e.g., if LiteLLM has a date suffix)
            for dev_model_id, dev_model_info in provider_models.items():
                if dev_model_id in model_name or model_name in dev_model_id:
                    limits = dev_model_info.get("limit", {})
                    return limits.get("context"), limits.get("output")

    return None, None


def fetch_chutes_model_info(api_key: str) -> dict[str, dict]:
    """
    Fetch model metadata from the Chutes /v1/models endpoint.
    Returns a dict keyed by Chutes model ID (e.g. "moonshotai/Kimi-K2.6-TEE").
    Each value contains at least context_length and max_output_length.
    """
    try:
        resp = requests.get(
            f"{CHUTES_BASE}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return {m["id"]: m for m in resp.json().get("data", [])}
    except Exception as e:
        print(f"Warning: Failed to fetch Chutes model info: {e}", file=sys.stderr)
        return {}


def find_chutes_limits(model_id: str, chutes_map: dict[str, dict]) -> tuple[int | None, int | None]:
    """Look up context/output limits for a chutes-e2ee/* model from Chutes /v1/models data."""
    if not chutes_map or not model_id.startswith("chutes-e2ee/"):
        return None, None
    chutes_id = model_id[len("chutes-e2ee/"):]  # e.g. "moonshotai/Kimi-K2.6-TEE"
    info = chutes_map.get(chutes_id, {})
    return info.get("context_length"), info.get("max_output_length")


# Modes that indicate the model is a chat/completion model usable by opencode.
# Any other non-null mode (e.g. "embedding", "image_generation") is skipped.
CHAT_MODES = {"chat", "completion", "responses"}


def strip_jsonc_comments(text: str) -> str:
    text = re.sub(r"(?<!:)//[^\n]*", "", text)
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    return text


def load_opencode_config(path: Path | None = None) -> tuple[dict, dict[str, str]]:
    """
    Parse an opencode config file (.json or .jsonc) and return
    (full_config_dict, model_id -> name overrides).
    Falls back to the default OPENCODE_CONFIG path when none is given.
    Returns ({}, {}) if the file doesn't exist or can't be parsed.
    """
    target = path or OPENCODE_CONFIG
    if not target.exists():
        return {}, {}
    raw = target.read_text()
    try:
        cfg = json.loads(strip_jsonc_comments(raw) if target.suffix == ".jsonc" else raw)
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse {target}: {e}", file=sys.stderr)
        return {}, {}
    name_overrides: dict[str, str] = {}
    for provider_cfg in cfg.get("provider", {}).values():
        for mid, mdef in provider_cfg.get("models", {}).items():
            if "name" in mdef:
                name_overrides[mid] = mdef["name"]
    return cfg, name_overrides


def load_current_model_ids(path: Path | None = None) -> set[str]:
    cfg, _ = load_opencode_config(path)
    ids: set[str] = set()
    for p in cfg.get("provider", {}).values():
        ids.update(p.get("models", {}).keys())
    return ids


def fetch_model_info(base_url: str) -> dict[str, dict]:
    """Return model_name -> model_info dict from /model/info."""
    resp = requests.get(f"{base_url}/model/info", timeout=10)
    resp.raise_for_status()
    return {
        entry["model_name"]: entry.get("model_info", {})
        for entry in resp.json().get("data", [])
    }


def fetch_all_model_ids(base_url: str) -> list[str]:
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def guess_display_name(model_id: str) -> str:
    """Derive a readable name from the last path component of a model ID."""
    raw = model_id.split("/")[-1]
    name = raw.replace("-", " ").replace("_", " ")
    return re.sub(r" +", " ", name).strip()


def build_entry(
    model_id: str,
    info: dict,
    existing: dict | None = None,
    chutes_map: dict[str, dict] | None = None,
) -> dict:
    """
    Build an opencode model entry from LiteLLM model_info.

    Capability rules (all sourced from the API, no keyword matching):
      tool_call   — info["supports_function_calling"]:  True/None→True, False→False
      temperature — "temperature" in info["supported_openai_params"]: present→True, absent→False,
                    no params field (unknown) → True
      attachment  — info["supports_vision"]: True→True, False/None→False
      reasoning   — info["supports_reasoning"]: True→True, False/None→False

    Limits resolution order:
      LiteLLM API → existing config → Chutes API → models.dev → hardcoded fallback
    """
    # tool_call: False only when explicitly stated
    raw_tool = info.get("supports_function_calling")
    tool_call = False if raw_tool is False else True

    # temperature: check supported_openai_params if available; default True when unknown
    params = info.get("supported_openai_params")
    if params is not None:
        temperature = "temperature" in params
    else:
        temperature = True  # unknown → assume supported

    # attachment (vision): conservative — only True when explicitly stated
    attachment = info.get("supports_vision") is True

    # reasoning: only True when explicitly stated
    reasoning = info.get("supports_reasoning") is True

    # limits: LiteLLM API → existing config → Chutes API → models.dev → hardcoded fallback
    existing_limit = (existing or {}).get("limit", {})

    context = info.get("max_input_tokens") or existing_limit.get("context")
    output = info.get("max_output_tokens") or existing_limit.get("output")

    # Try Chutes API for chutes-e2ee/* models
    if (context is None or output is None) and chutes_map:
        chutes_context, chutes_output = find_chutes_limits(model_id, chutes_map)
        if context is None and chutes_context is not None:
            context = chutes_context
        if output is None and chutes_output is not None:
            output = chutes_output

    # If limits are still missing, try models.dev
    if context is None or output is None:
        dev_context, dev_output = find_models_dev_limits(model_id)
        if context is None and dev_context is not None:
            context = dev_context
        if output is None and dev_output is not None:
            output = dev_output

    # Final fallback with warning
    if context is None:
        print(
            f"Warning: No context limit found for {model_id}, falling back to {FALLBACK_CONTEXT}",
            file=sys.stderr,
        )
        context = FALLBACK_CONTEXT

    if output is None:
        print(
            f"Warning: No output limit found for {model_id}, falling back to {FALLBACK_OUTPUT}",
            file=sys.stderr,
        )
        output = FALLBACK_OUTPUT

    entry: dict = {
        "name": guess_display_name(model_id),
        "tool_call": tool_call,
        "temperature": temperature,
    }
    if reasoning:
        entry["reasoning"] = True
    if attachment:
        entry["attachment"] = True
    entry["limit"] = {"context": context, "output": output}
    return entry


def is_skippable(info: dict, in_current: bool) -> bool | str:
    """
    Return False if the model should be included, or a reason string if it should be skipped.
    Two hard filters: explicitly no tool-call support, or an explicitly non-chat mode.
    Models in the current config are never skipped.
    """
    if in_current:
        return False

    if info.get("supports_function_calling") is False:
        return "supports_function_calling=False"

    mode = info.get("mode")
    if mode and mode not in CHAT_MODES:
        return f"mode={mode!r} not in CHAT_MODES"

    return False


def build_models_block(
    model_info_map: dict[str, dict],
    all_model_ids: list[str],
    current_ids: set[str],
    name_overrides: dict[str, str],
    existing_models: dict[str, dict],
    filter_current: bool,
    debug: bool = False,
    chutes_map: dict[str, dict] | None = None,
) -> dict[str, dict]:
    # Build ordered deduplicated list: metadata-known models first, then extras
    seen: set[str] = set()
    ordered: list[str] = []
    for mid in model_info_map:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    for mid in all_model_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    for mid in current_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    result: dict[str, dict] = {}
    skipped = 0

    for mid in ordered:
        if filter_current and mid not in current_ids:
            continue

        info = model_info_map.get(mid, {})
        in_current = mid in current_ids

        skip_reason = is_skippable(info, in_current)
        if skip_reason:
            skipped += 1
            if debug:
                print(f"  skip {mid}: {skip_reason}", file=sys.stderr)
            continue

        entry = build_entry(mid, info, existing=existing_models.get(mid), chutes_map=chutes_map)
        # Restore manually set name if present
        if mid in name_overrides:
            entry["name"] = name_overrides[mid]
        result[mid] = entry

    if skipped:
        print(f"Skipped {skipped} non-chat/no-tool-call models", file=sys.stderr)
    return result


def apply_to_config(models_block: dict[str, dict]) -> None:
    """
    Rewrite only the provider.litellm.models section of opencode.jsonc.
    All other config (agent, permissions, etc.) is preserved verbatim.
    The raw file text is rebuilt by replacing just the models block so that
    comments and formatting outside that section are kept intact.
    """
    raw = OPENCODE_CONFIG.read_text()

    # Serialise the models block with consistent indentation (8 spaces = 2 levels of 4)
    models_json = json.dumps(models_block, indent=8)
    # Shift to match the file's 2-space indent style (collapse 8→4→2... not needed,
    # just use 2-space indent relative to the "models" key which is at 6 spaces)
    # We'll embed it verbatim — the leading/trailing braces are part of the replacement.

    # Pattern: "models": { ... } inside the litellm provider block
    # We match from `"models": {` to the matching closing `}` by counting braces.
    # Since regex can't count braces, we do it manually.
    marker = '"models":'
    start = raw.find(marker)
    if start == -1:
        print("Error: could not find models block in config", file=sys.stderr)
        sys.exit(1)

    # Find the opening brace
    brace_start = raw.index("{", start)
    depth = 0
    i = brace_start
    while i < len(raw):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break
        i += 1
    else:
        print("Error: unbalanced braces in models block", file=sys.stderr)
        sys.exit(1)

    # Build the indented models JSON (6 spaces for "models" key level → content at 8+)
    lines = models_json.splitlines()
    indented_lines = []
    for idx, line in enumerate(lines):
        if idx == 0:
            indented_lines.append("      " + line)  # opening { at 6-space indent
        else:
            # Replace the 8-space base indent from json.dumps with the file's style
            stripped = line.lstrip()
            leading = len(line) - len(stripped)
            # Map json.dumps indent levels (multiples of 8) to file's 2-space style at 6+base
            depth_level = leading // 8
            new_indent = "      " + "  " * depth_level  # 6 base + 2 per level
            indented_lines.append(new_indent + stripped)
    new_models_json = "\n".join(indented_lines)

    new_raw = raw[:brace_start] + new_models_json + raw[brace_end + 1 :]
    OPENCODE_CONFIG.write_text(new_raw)
    print(f"Updated {OPENCODE_CONFIG}", file=sys.stderr)


def apply_to_json(path: Path, models_block: dict[str, dict]) -> None:
    """
    Update provider.litellm.models in a plain JSON opencode config file.
    All other keys (model, agent, permission, etc.) are preserved exactly.
    """
    cfg = json.loads(path.read_text())
    cfg.setdefault("provider", {}).setdefault("litellm", {})["models"] = models_block
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Updated {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate opencode models block from LiteLLM"
    )
    parser.add_argument("--url", default=LITELLM_BASE, help="LiteLLM base URL")
    parser.add_argument(
        "--filter-current",
        action="store_true",
        help="Only include models already in opencode.jsonc",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the result directly into opencode.jsonc (preserves all other config)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Path to an opencode.json file; update its provider.litellm.models in-place "
        "(only models already in the file are regenerated)",
    )
    parser.add_argument(
        "--chutes-api-key",
        metavar="KEY",
        default=os.environ.get("CHUTES_API_KEY", ""),
        help="Chutes API key for fetching TEE model limits (also reads CHUTES_API_KEY env var)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each skipped model and the reason it was excluded",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    print(f"Fetching model info from {args.url}...", file=sys.stderr)
    try:
        model_info_map = fetch_model_info(args.url)
        all_model_ids = fetch_all_model_ids(args.url)
    except requests.RequestException as e:
        print(f"Error connecting to LiteLLM: {e}", file=sys.stderr)
        sys.exit(1)

    chutes_map: dict[str, dict] = {}
    if args.chutes_api_key:
        print("Fetching model limits from Chutes API...", file=sys.stderr)
        chutes_map = fetch_chutes_model_info(args.chutes_api_key)
        print(f"  Got {len(chutes_map)} Chutes models", file=sys.stderr)

    config_path = output_path if output_path else None
    cfg, name_overrides = load_opencode_config(config_path)
    current_ids: set[str] = set()
    existing_models: dict[str, dict] = {}
    for p in cfg.get("provider", {}).values():
        current_ids.update(p.get("models", {}).keys())
        existing_models.update(p.get("models", {}))

    filter_current = args.filter_current or args.apply or bool(output_path)
    if filter_current:
        print(
            f"Using {len(current_ids)} models from current opencode config",
            file=sys.stderr,
        )

    models_block = build_models_block(
        model_info_map,
        all_model_ids,
        current_ids,
        name_overrides,
        existing_models,
        filter_current=filter_current,
        debug=args.debug,
        chutes_map=chutes_map,
    )
    print(f"Generated {len(models_block)} model entries", file=sys.stderr)

    if output_path:
        apply_to_json(output_path, models_block)
    elif args.apply:
        apply_to_config(models_block)
    else:
        print(json.dumps(models_block, indent=2))


if __name__ == "__main__":
    main()
