"""Unit tests for the Chutes model_info builder and wildcard expansion."""

import pytest

from chutes_litellm import generate_config


def _chutes_model(**overrides) -> dict:
    """Return a minimal Chutes ``/v1/models`` entry with optional overrides."""
    model = {
        "id": "deepseek-ai/DeepSeek-V3.2-TEE",
        "context_length": 131072,
        "max_output_length": 8192,
        "supported_features": ["tools", "reasoning"],
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "supported_sampling_parameters": ["temperature", "top_p"],
        "created": 1700000000,
    }
    model.update(overrides)
    return model


def test_output_limit_is_always_zero_and_input_limit_comes_from_context_length():
    """Upstream max_output_length is ignored; context_length still sets input."""
    info = generate_config._build_chutes_model_info(
        _chutes_model(max_output_length=8192, context_length=131072)
    )

    assert info["max_output_tokens"] == 0
    assert info["max_input_tokens"] == 131072


def test_output_limit_is_zero_without_upstream_max_output_length():
    """Models lacking max_output_length still advertise an output limit of 0."""
    model = _chutes_model()
    del model["max_output_length"]

    info = generate_config._build_chutes_model_info(model)

    assert info["max_output_tokens"] == 0


@pytest.mark.parametrize("effort", generate_config._REASONING_EFFORTS)
def test_reasoning_model_gets_reasoning_and_effort_flags(effort: str):
    """Reasoning models keep supports_reasoning and per-effort variant flags."""
    info = generate_config._build_chutes_model_info(_chutes_model())

    assert info["supports_reasoning"] is True
    assert info[f"supports_{effort}_reasoning_effort"] is True


def test_expand_wildcard_generated_entry_reports_zero_output_limit(monkeypatch):
    """A generated Chutes entry has model_info.max_output_tokens == 0."""
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["chutes"],
        "fetch",
        lambda _api_key: [_chutes_model()],
    )

    entries = generate_config.expand_wildcard("chutes", {"model_name": "chutes/*"})

    assert len(entries) == 1
    entry = entries[0]
    assert entry["model_name"] == "chutes/deepseek-ai/DeepSeek-V3.2-TEE"
    assert entry["model_info"]["max_output_tokens"] == 0
    assert entry["model_info"]["max_input_tokens"] == 131072


# ---------------------------------------------------------------------------
# NEAR AI
# ---------------------------------------------------------------------------


def _near_model(**overrides) -> dict:
    """Return a minimal NEAR AI ``/v1/models`` chat entry."""
    model = {
        "id": "Qwen/Qwen3.8-27B",
        "context_length": 262144,
        "max_output_length": 65536,
        "supported_features": ["tools", "reasoning"],
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "created": 1700000000,
    }
    model.update(overrides)
    return model


def test_nearai_wildcard_emits_user_and_wire_model_names(monkeypatch):
    """NEAR entries expose `nearai/<id>` but route through `nearai_e2ee/<id>`."""
    monkeypatch.setenv("NEARAI_API_KEY", "nk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: [_near_model()],
    )

    entries = generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"})

    assert len(entries) == 1
    entry = entries[0]
    assert entry["model_name"] == "nearai/Qwen/Qwen3.8-27B"
    assert entry["litellm_params"]["model"] == "nearai_e2ee/Qwen/Qwen3.8-27B"
    assert entry["litellm_params"]["api_key"] == "os.environ/NEARAI_API_KEY"


def test_nearai_wildcard_allows_reasoning_params(monkeypatch):
    """Clients can send the same extra OpenAI params as for Chutes."""
    monkeypatch.setenv("NEARAI_API_KEY", "nk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: [_near_model()],
    )

    entry = generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"})[0]
    allowed = entry["litellm_params"]["allowed_openai_params"]

    assert allowed == list(generate_config._ALLOWED_OPENAI_PARAMS)
    for param in ("reasoning_effort", "thinking", "reasoning", "chat_template_kwargs", "top_k", "min_p"):
        assert param in allowed


def test_nearai_wildcard_emits_normalized_model_info(monkeypatch):
    """Limits and capabilities are mapped from the NEAR catalog entry."""
    monkeypatch.setenv("NEARAI_API_KEY", "nk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: [_near_model()],
    )

    info = generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"})[0]["model_info"]

    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 262144
    assert info["max_output_tokens"] == 65536
    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is True
    assert info["input_modalities"] == ["image", "text"]
    for effort in generate_config._REASONING_EFFORTS:
        assert info[f"supports_{effort}_reasoning_effort"] is True


def test_nearai_wildcard_skips_when_key_is_absent(monkeypatch):
    """Without NEARAI_API_KEY the wildcard expands to nothing, like Chutes."""
    monkeypatch.delenv("NEARAI_API_KEY", raising=False)
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: pytest.fail("fetch must not run without a key"),
    )

    assert generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"}) == []


def test_nearai_wildcard_skips_on_discovery_error(monkeypatch):
    """Discovery errors are logged and skipped, not raised."""
    monkeypatch.setenv("NEARAI_API_KEY", "nk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: (_ for _ in ()).throw(RuntimeError("catalog down")),
    )

    assert generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"}) == []


# ---------------------------------------------------------------------------
# Packaged template
# ---------------------------------------------------------------------------


def test_packaged_template_disables_message_logging():
    """The shipped template must keep LiteLLM from persisting prompts/replies."""
    import yaml

    with open(generate_config.TEMPLATE) as handle:
        config = yaml.safe_load(handle)

    assert (config.get("litellm_settings") or {}).get("turn_off_message_logging") is True
