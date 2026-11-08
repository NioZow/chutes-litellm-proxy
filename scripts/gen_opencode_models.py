#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///
"""
Generate or update the models block in an opencode config from LiteLLM's
/model/info API.

Strategy:
  - Display names come from the proxy's `model_info.display_name` (written by
    generate_config.py as "<Pretty Name> (<Source>)"), falling back to the id.
  - All boolean capability flags (tool_call, temperature, attachment, reasoning) are derived
    exclusively from LiteLLM's model_info fields — no keyword guessing, no hardcoded lists.
  - For models with no LiteLLM metadata (custom routes like chutes), all capabilities
    default to True (conservative: assume the model supports everything until proven otherwise).
  - Limits fall back in order: LiteLLM API → Chutes API → models.dev → existing config → hardcoded default.
  - Both the V2 `providers.<id>.models` shape and the deprecated V1
    `provider.<id>.models` shape are supported.
  - When --apply is used, the script rewrites only the litellm models block in
    the config, preserving all other config untouched (agent temps, permissions, etc.)
    and preserving any manually set "name" values.

Usage:
  ./gen_opencode_models.py                          # print JSON models block, all available models
  ./gen_opencode_models.py --filter-current         # only models already in the config
  ./gen_opencode_models.py --apply                  # update the default config in-place
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
import time
from pathlib import Path

import requests

LITELLM_BASE = "http://127.0.0.1:27740"
CHUTES_BASE = "https://llm.chutes.ai"
OPENCODE_CONFIG_DIR = Path.home() / ".config/opencode"
# V2 configs use `providers`, V1 configs use the deprecated `provider` key.
# Prefer a plain .json file when both exist; --output overrides this.
OPENCODE_CONFIG_CANDIDATES = (
    OPENCODE_CONFIG_DIR / "opencode.json",
    OPENCODE_CONFIG_DIR / "opencode.jsonc",
)
PROVIDER_ID = "litellm"

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
            "deepseek": "deepseek",
            "mistral": "mistral",
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


# Prefixes under which a Chutes model may be exposed by LiteLLM.  The builtin
# provider uses ``chutes/``; the E2EE providers/custom routes use
# ``chutes-e2ee/`` or ``chutes_e2ee/``.
CHUTES_PREFIXES = ("chutes/", "chutes-e2ee/", "chutes_e2ee/")

# Thinking levels advertised as opencode `variants` for reasoning models.
REASONING_EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")


def _chutes_model_id(model_id: str) -> str | None:
    """Return the bare Chutes model id for any known provider prefix."""
    for prefix in CHUTES_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return None


def find_chutes_limits(
    model_id: str, chutes_map: dict[str, dict]
) -> tuple[int | None, int | None]:
    """Look up context/output limits for a Chutes model from /v1/models data."""
    if not chutes_map:
        return None, None
    chutes_id = _chutes_model_id(model_id)
    if chutes_id is None:
        return None, None
    info = chutes_map.get(chutes_id, {})
    return info.get("context_length"), info.get("max_output_length")


def find_chutes_info(model_id: str, chutes_map: dict[str, dict]) -> dict:
    """Return the raw Chutes /v1/models metadata entry for a model, if any."""
    if not chutes_map:
        return {}
    chutes_id = _chutes_model_id(model_id)
    if chutes_id is None:
        return {}
    return chutes_map.get(chutes_id, {})


def find_chutes_cost(model_id: str, chutes_map: dict[str, dict]) -> dict | None:
    """Map Chutes pricing (USD per 1M tokens) to opencode's `cost` block."""
    if not chutes_map:
        return None
    chutes_id = _chutes_model_id(model_id)
    if chutes_id is None:
        return None
    info = chutes_map.get(chutes_id, {})
    price = info.get("price") or {}
    if not price:
        return None

    def usd(entry: object) -> float | None:
        if isinstance(entry, dict):
            return entry.get("usd")
        return None

    cost = {
        "input": usd(price.get("input")),
        "output": usd(price.get("output")),
        "cache_read": usd(price.get("input_cache_read")),
        "cache_write": 0,
    }
    if cost["input"] is None or cost["output"] is None:
        return None
    return cost


def reasoning_variants(info: dict) -> dict | None:
    """Build opencode `variants` that map a thinking level to a request.

    The proxy forwards ``chat_template_kwargs``/``reasoning_effort`` to Chutes
    (see generate_config's allowed_openai_params and the custom provider), so
    this mirrors the pi agent's ``thinkingLevelMap`` in opencode's variant form.
    """
    if info.get("supports_reasoning") is not True:
        return None
    levels = info.get("reasoning_effort_levels") or [
        level for level in REASONING_EFFORTS if info.get(f"supports_{level}_reasoning_effort")
    ]
    if not levels:
        levels = ["low", "medium", "high"]
    variants: dict[str, dict] = {
        "off": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    for level in levels:
        if level in {"off", "none"}:
            # "off" already disables thinking; "none" is the same request.
            continue
        variants[level] = {"chat_template_kwargs": {"reasoning_effort": level}}
    return variants



# Modes that indicate the model is a chat/completion model usable by opencode.
# Any other non-null mode (e.g. "embedding", "image_generation") is skipped.
CHAT_MODES = {"chat", "completion", "responses"}


def strip_jsonc_comments(text: str) -> str:
    text = re.sub(r"(?<!:)//[^\n]*", "", text)
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    return text


def default_config_path() -> Path | None:
    """Return the first existing default opencode config, if any."""
    for candidate in OPENCODE_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def provider_maps(cfg: dict):
    """Yield ``(shape, providers)`` for the V2 and deprecated V1 config keys."""
    for shape in ("providers", "provider"):
        providers = cfg.get(shape)
        if isinstance(providers, dict):
            yield shape, providers


def config_version(cfg: dict) -> int:
    """Return 2 for a `providers` config, 1 for a `provider` config."""
    return 2 if isinstance(cfg.get("providers"), dict) else 1


def load_opencode_config(path: Path | None = None) -> tuple[dict, dict[str, str]]:
    """
    Parse an opencode config file (.json or .jsonc) and return
    (full_config_dict, model_id -> name overrides).
    Understands both the V2 `providers` shape and the deprecated `provider`
    shape. Falls back to the first default config path when none is given.
    Returns ({}, {}) if no file exists or it can't be parsed.
    """
    target = path or default_config_path()
    if target is None or not target.exists():
        return {}, {}
    raw = target.read_text()
    try:
        cfg = json.loads(
            strip_jsonc_comments(raw) if target.suffix == ".jsonc" else raw
        )
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse {target}: {e}", file=sys.stderr)
        return {}, {}
    name_overrides: dict[str, str] = {}
    for _shape, providers in provider_maps(cfg):
        provider_cfg = providers.get(PROVIDER_ID) or {}
        for mid, mdef in (provider_cfg.get("models") or {}).items():
            if "name" in mdef:
                name_overrides[mid] = mdef["name"]
    return cfg, name_overrides


def load_current_model_ids(path: Path | None = None) -> set[str]:
    cfg, _ = load_opencode_config(path)
    ids: set[str] = set()
    for _shape, providers in provider_maps(cfg):
        provider_cfg = providers.get(PROVIDER_ID) or {}
        ids.update((provider_cfg.get("models") or {}).keys())
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

    # temperature: check supported params if available; default True when unknown
    params = info.get("supported_openai_params") or info.get("supported_sampling_parameters")
    if params is not None:
        temperature = "temperature" in params
    else:
        temperature = True  # unknown → assume supported

    # Special case: GPT-5 models do not support temperature
    if "gpt-5" in model_id.lower():
        temperature = False

    # attachment (vision): conservative — only True when explicitly stated
    attachment = info.get("supports_vision") is True

    # Modalities from the Chutes listing, falling back to LiteLLM model_info.
    chutes_info = find_chutes_info(model_id, chutes_map)
    in_modalities = chutes_info.get("input_modalities") or info.get("input_modalities") or []
    out_modalities = chutes_info.get("output_modalities") or info.get("output_modalities") or []
    if not attachment:
        attachment = bool({"image", "video", "pdf"} & set(in_modalities))

    # reasoning: only True when explicitly stated
    reasoning = info.get("supports_reasoning") is True

    # limits: LiteLLM API → existing config → Chutes API → models.dev → hardcoded fallback
    existing_limit = (existing or {}).get("limit", {})

    context = info.get("max_input_tokens") or existing_limit.get("context")
    output = info.get("max_output_tokens") or existing_limit.get("output")

    # Try Chutes API for chutes/* models
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
        # The proxy publishes `<Pretty Name> (<Source>)` in model_info so the
        # picker still shows which backend serves a model under the single
        # `litellm` provider. Fall back to the id for older proxies.
        "name": info.get("display_name") or guess_display_name(model_id),
        "tool_call": tool_call,
        "temperature": temperature,
    }
    if reasoning:
        entry["reasoning"] = True
        # This provider surfaces thinking in the standard `reasoning_content`.
        entry["interleaved"] = "reasoning_content"
        variants = reasoning_variants(info)
        if variants:
            entry["variants"] = variants
    if attachment:
        entry["attachment"] = True
    if in_modalities or out_modalities:
        entry["modalities"] = {}
        if in_modalities:
            entry["modalities"]["input"] = list(in_modalities)
        if out_modalities:
            entry["modalities"]["output"] = list(out_modalities)

    # release_date: Chutes `created` (unix) or the value carried in model_info.
    release_date = info.get("release_date")
    created = chutes_info.get("created")
    if not release_date and isinstance(created, (int, float)) and created > 0:
        release_date = time.strftime("%Y-%m-%d", time.gmtime(created))
    if release_date:
        entry["release_date"] = release_date

    if chutes_map:
        cost = find_chutes_cost(model_id, chutes_map)
        if cost is not None:
            entry["cost"] = cost
    entry["limit"] = {"context": context, "output": output}
    entry["status"] = "active"
    return entry


def to_v2_entry(entry: dict) -> dict:
    """Convert a legacy-shaped model entry to the V2 Config.ModelEncoded shape.

    V2 (`providers.<id>.models`) nests capability flags under `capabilities`,
    uses camelCase `releaseDate`, models variants as an array, and expresses
    the reasoning field through `compatibility.reasoningField`.
    """
    v2: dict = {"name": entry["name"]}

    if entry.get("release_date"):
        v2["releaseDate"] = entry["release_date"]

    capabilities: dict = {}
    if "tool_call" in entry:
        capabilities["tools"] = entry["tool_call"]
    modalities = entry.get("modalities") or {}
    if modalities.get("input"):
        capabilities["input"] = modalities["input"]
    if modalities.get("output"):
        capabilities["output"] = modalities["output"]
    if entry.get("temperature") is not None:
        capabilities["temperature"] = entry["temperature"]
    if entry.get("reasoning"):
        capabilities["reasoning"] = {"supported": True}
    if capabilities:
        v2["capabilities"] = capabilities

    if entry.get("limit"):
        v2["limit"] = {
            "context": entry["limit"]["context"],
            "output": entry["limit"]["output"],
        }

    cost = entry.get("cost")
    if cost:
        converted_cost: dict = {
            "input": cost.get("input") or 0,
            "output": cost.get("output") or 0,
        }
        cache = {
            "read": cost.get("cache_read") or 0,
            "write": cost.get("cache_write") or 0,
        }
        if cache["read"] or cache["write"]:
            converted_cost["cache"] = cache
        v2["cost"] = converted_cost

    variants = entry.get("variants")
    if variants:
        v2["variants"] = [
            {"id": level, "body": body} for level, body in variants.items()
        ]

    interleaved = entry.get("interleaved")
    if interleaved:
        field = interleaved if isinstance(interleaved, str) else interleaved.get("field")
        if field:
            v2["compatibility"] = {"reasoningField": field}

    return v2


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
    v2: bool = False,
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

        entry = build_entry(
            mid, info, existing=existing_models.get(mid), chutes_map=chutes_map
        )
        # Restore manually set name if present
        if mid in name_overrides:
            entry["name"] = name_overrides[mid]
        result[mid] = to_v2_entry(entry) if v2 else entry

    if skipped:
        print(f"Skipped {skipped} non-chat/no-tool-call models", file=sys.stderr)
    return result


def _find_object_span(text: str, start: int) -> tuple[int, int]:
    """Return the (open, close) brace indices of the object at/after start."""
    open_idx = text.index("{", start)
    depth = 0
    in_string = False
    escaped = False
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return open_idx, idx
    raise ValueError("unbalanced braces")


def _line_indent(text: str, index: int) -> str:
    """Return the leading whitespace of the line containing ``index``."""
    line_start = text.rfind("\n", 0, index) + 1
    return re.match(r"[ \t]*", text[line_start:]).group(0)


def _indent_continuation(value: str, indent: str) -> str:
    """Indent every line but the first, which is already positioned in the file."""
    lines = value.splitlines()
    if len(lines) < 2:
        return value
    return "\n".join(
        [lines[0]]
        + [indent + line if line.strip() else line for line in lines[1:]]
    )


def _replace_models_block(
    raw: str, shape: str, models_block: dict[str, dict]
) -> str | None:
    """Surgically replace/insert a models block, keeping the rest of the text."""
    shape_match = re.search(rf'"{re.escape(shape)}"\s*:', raw)
    if not shape_match:
        return None
    shape_open, shape_close = _find_object_span(raw, shape_match.end())
    provider_match = re.search(
        rf'"{re.escape(PROVIDER_ID)}"\s*:', raw[shape_open:shape_close]
    )
    if not provider_match:
        return None
    provider_key_index = shape_open + provider_match.start()
    provider_open, provider_close = _find_object_span(raw, shape_open + provider_match.end())

    models_json = json.dumps(models_block, indent=2)
    models_match = re.search(r'"models"\s*:', raw[provider_open:provider_close])
    if models_match:
        key_index = provider_open + models_match.start()
        indent = _line_indent(raw, key_index)
        obj_open, obj_close = _find_object_span(raw, provider_open + models_match.end())
        return (
            raw[:obj_open]
            + _indent_continuation(models_json, indent)
            + raw[obj_close + 1 :]
        )

    # No models key yet: insert one right after the provider's opening brace.
    # Adding the member first avoids having to place a comma before a possible
    # trailing `// comment` on the previous last member.
    key_indent = _line_indent(raw, provider_key_index) + "  "
    has_members = bool(raw[provider_open + 1 : provider_close].strip())
    insertion = (
        "\n"
        + key_indent
        + '"models": '
        + _indent_continuation(models_json, key_indent)
        + ("," if has_members else "")
    )
    return raw[: provider_open + 1] + insertion + raw[provider_open + 1 :]


def apply_to_config(path: Path, models_block: dict[str, dict]) -> None:
    """Update ``providers.litellm.models`` (V2) or ``provider.litellm.models``.

    Plain JSON files are parsed and rewritten, which preserves every key but
    normalises formatting. Files that are not valid JSON (JSONC, or JSON with
    comments) are edited surgically so comments and formatting outside the
    models block survive.
    """
    raw = path.read_text()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        try:
            cfg = json.loads(strip_jsonc_comments(raw))
        except json.JSONDecodeError as exc:
            print(f"Error: could not parse {path}: {exc}", file=sys.stderr)
            sys.exit(1)
        shape = "providers" if isinstance(cfg.get("providers"), dict) else "provider"
        new_raw = _replace_models_block(raw, shape, models_block)
        if new_raw is None:
            print(
                f"Error: could not find {shape}.{PROVIDER_ID} in {path}; "
                "add that provider first",
                file=sys.stderr,
            )
            sys.exit(1)
        path.write_text(new_raw)
    else:
        shape = "providers" if isinstance(cfg.get("providers"), dict) else "provider"
        cfg.setdefault(shape, {}).setdefault(PROVIDER_ID, {})["models"] = models_block
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
        help="Only include models already in the opencode config",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the result into the default opencode config (opencode.json or "
        "opencode.jsonc), creating a models block for the litellm provider",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Path to an opencode config; update its litellm models in-place "
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
    target_path = output_path
    if target_path is None and args.apply:
        target_path = default_config_path()
        if target_path is None:
            print(
                "Error: no opencode config found at "
                + " or ".join(str(p) for p in OPENCODE_CONFIG_CANDIDATES),
                file=sys.stderr,
            )
            sys.exit(1)

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

    cfg, name_overrides = load_opencode_config(target_path)
    v2 = config_version(cfg) == 2
    current_ids: set[str] = set()
    existing_models: dict[str, dict] = {}
    for _shape, providers in provider_maps(cfg):
        provider_cfg = providers.get(PROVIDER_ID) or {}
        models = provider_cfg.get("models") or {}
        current_ids.update(models.keys())
        existing_models.update(models)

    # --apply regenerates every discovered model so a config with no litellm
    # models yet (the usual V2 case) can be bootstrapped. --output keeps the
    # historical behaviour of only refreshing models already in the file.
    filter_current = args.filter_current or bool(output_path)
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
        v2=v2,
    )
    print(
        f"Generated {len(models_block)} {'V2' if v2 else 'V1'} model entries",
        file=sys.stderr,
    )

    if target_path:
        apply_to_config(target_path, models_block)
    else:
        print(json.dumps(models_block, indent=2))


if __name__ == "__main__":
    main()
