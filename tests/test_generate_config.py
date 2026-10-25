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
