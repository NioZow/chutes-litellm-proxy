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
        "confidential_compute": True,
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
# Display names and direct-provider metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("chutes/mistralai/Mistral-Large-4-TEE", "Mistral Large 4 (Chutes TEE)"),
        ("chutes/deepseek-ai/DeepSeek-V3.2-TEE", "DeepSeek V3.2 (Chutes TEE)"),
        ("chutes/Qwen/Qwen3.8-27B", "Qwen3.8 27B (Chutes)"),
        ("chutes/deepseek-ai/DeepSeek-V4.1-Flash-1017", "DeepSeek V4.1 Flash 1017 (Chutes)"),
        ("chutes/Qwen/Qwen3.5-397B-A17B-TEE", "Qwen3.5 397B A17B (Chutes TEE)"),
        ("chutes/google/gemma-4-31B-turbo-TEE", "Gemma 4 31B Turbo (Chutes TEE)"),
        ("nearai/anthropic/claude-sonnet-5", "Claude Sonnet 5 (NEAR)"),
        ("nearai/Qwen/Qwen3.8-27B", "Qwen3.8 27B (NEAR)"),
        ("deepseek/deepseek-flash", "DeepSeek Flash (DeepSeek)"),
        ("mistral/mistral-large-4", "Mistral Large 4 (Mistral)"),
        ("gemini/gemini-3.1-pro", "Gemini 3.1 Pro (Google)"),
        ("openai/gpt-5.2", "GPT 5.2 (OpenAI)"),
        ("xai/grok-4", "Grok 4 (xAI)"),
        ("anthropic/claude-opus-4-5", "Claude Opus 4.5 (Anthropic)"),
        ("anthropic/claude-opus-4-5-20251101", "Claude Opus 4.5 20251101 (Anthropic)"),
        ("openai/gpt-4.1-2025-04-14", "GPT 4.1 2025 04 14 (OpenAI)"),
    ],
)
def test_build_display_name_labels_the_upstream_source(model_name: str, expected: str):
    """Display names carry the source even though all models share one provider."""
    assert generate_config.build_display_name(model_name) == expected


def test_near_tee_and_catalog_names_are_used():
    """NEAR's own TEE models get the badge and the catalog's display name."""
    assert (
        generate_config.build_display_name(
            "nearai/Qwen/Qwen3.8-27B", tee=True, name="Qwen 3.8 27B"
        )
        == "Qwen 3.8 27B (NEAR TEE)"
    )
    # Proxied models keep no TEE badge even though they are encrypted upstream.
    assert (
        generate_config.build_display_name(
            "nearai/deepseek/deepseek-v3.2", tee=False, name="deepseek-v3.2"
        )
        == "DeepSeek V3.2 (NEAR)"
    )


def test_generated_entries_include_a_display_name(monkeypatch):
    """Wildcard expansion annotates the model_info LiteLLM passes through."""
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["chutes"],
        "fetch",
        lambda _api_key: [_chutes_model()],
    )

    info = generate_config.expand_wildcard("chutes", {"model_name": "chutes/*"})[0]["model_info"]

    assert info["display_name"] == "DeepSeek V3.2 (Chutes TEE)"
    assert info["confidential_compute"] is True


def test_chutes_non_tee_models_are_generated_without_the_badge(monkeypatch):
    """Plain Chutes deployments are exposed too, labelled just `(Chutes)`."""
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["chutes"],
        "fetch",
        lambda _api_key: [
            _chutes_model(
                id="deepseek-ai/DeepSeek-V4.1-Flash-1017", confidential_compute=False
            )
        ],
    )

    entry = generate_config.expand_wildcard("chutes", {"model_name": "chutes/*"})[0]

    assert entry["model_name"] == "chutes/deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert entry["model_info"]["display_name"] == "DeepSeek V4.1 Flash 1017 (Chutes)"
    assert entry["model_info"]["confidential_compute"] is False


def test_chutes_discovery_keeps_non_tee_chat_models():
    """Almost every Chutes model is non-TEE now, so discovery must keep them."""
    model = _chutes_model(confidential_compute=False)

    assert generate_config._is_chat_chutes_model(model) is True

    for non_chat in ("Qwen/Qwen3-Embedding-8B", "BAAI/bge-reranker-v2", "openai/whisper"):
        assert generate_config._is_chat_chutes_model({"id": non_chat}) is False


def test_nearai_wildcard_marks_only_own_tee_models(monkeypatch):
    """`owned_by: nearai` is TEE; proxied models only get `(NEAR)`."""
    monkeypatch.setenv("NEARAI_API_KEY", "nk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["nearai"],
        "fetch",
        lambda _api_key: [
            _near_model(owned_by="nearai", name="Qwen 3.8 27B"),
            _near_model(
                id="deepseek/deepseek-v3.2",
                owned_by="attested 3p",
                name="deepseek-v3.2",
            ),
        ],
    )

    entries = generate_config.expand_wildcard("nearai", {"model_name": "nearai/*"})

    assert entries[0]["model_info"]["display_name"] == "Qwen 3.8 27B (NEAR TEE)"
    assert entries[1]["model_info"]["display_name"] == "DeepSeek V3.2 (NEAR)"


def test_direct_provider_gets_known_limits_and_display_name(monkeypatch):
    """DeepSeek models newer than LiteLLM's model map get their real 1M window."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["deepseek"],
        "fetch",
        lambda _api_key: ["deepseek-flash", "deepseek-v4-pro"],
    )

    entries = generate_config.expand_wildcard("deepseek", {"model_name": "deepseek/*"})

    assert [entry["model_name"] for entry in entries] == [
        "deepseek/deepseek-flash",
        "deepseek/deepseek-v4-pro",
    ]
    for entry in entries:
        info = entry["model_info"]
        assert info["max_input_tokens"] == 1_048_576
        assert info["max_output_tokens"] == 393_216
        assert info["supports_reasoning"] is True
        assert info["mode"] == "chat"
    assert entries[0]["model_info"]["display_name"] == "DeepSeek Flash (DeepSeek)"
    assert entries[1]["model_info"]["supports_vision"] is False


def test_unknown_direct_model_gets_only_a_display_name(monkeypatch):
    """Models without a metadata entry are not given invented limits."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk_test")
    monkeypatch.setitem(
        generate_config.PROVIDERS["deepseek"],
        "fetch",
        lambda _api_key: ["deepseek-unreleased"],
    )

    info = generate_config.expand_wildcard("deepseek", {"model_name": "deepseek/*"})[0]["model_info"]

    assert info == {"display_name": "DeepSeek Unreleased (DeepSeek)"}


def test_generate_annotates_fixed_entries_and_preserves_explicit_names(tmp_path, monkeypatch):
    """Fixed prompt entries get a name too, unless the user set one already."""
    template = tmp_path / "template.yml"
    template.write_text(
        """
model_list:
  - model_name: "anthropic/claude-sonnet-5"
    litellm_params:
      model: "anthropic/claude-sonnet-5"
      api_key: "os.environ/ANTHROPIC_API_KEY"
  - model_name: "mistral/mistral-large-4"
    litellm_params:
      model: "mistral/mistral-large-4"
      api_key: "os.environ/MISTRAL_API_KEY"
    model_info:
      display_name: "My Mistral"
"""
    )
    output = tmp_path / "config.generated.yml"

    generate_config.generate(template=str(template), output=str(output))

    import yaml

    config = yaml.safe_load(output.read_text())
    assert config["model_list"][0]["model_info"]["display_name"] == "Claude Sonnet 5 (Anthropic)"
    assert config["model_list"][1]["model_info"]["display_name"] == "My Mistral"


# ---------------------------------------------------------------------------
# Packaged template
# ---------------------------------------------------------------------------


def test_packaged_template_disables_message_logging():
    """The shipped template must keep LiteLLM from persisting prompts/replies."""
    import yaml

    with open(generate_config.TEMPLATE) as handle:
        config = yaml.safe_load(handle)

    assert (config.get("litellm_settings") or {}).get("turn_off_message_logging") is True
