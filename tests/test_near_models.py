"""Offline tests for NEAR AI ``/v1/models`` discovery and normalisation.

No network is involved: ``httpx.get`` is replaced at the module boundary and
the fake response mirrors the Cloud API's OpenAI-shaped catalog.
"""

from __future__ import annotations

import time

import httpx
import pytest

from chutes_litellm import generate_config
from chutes_litellm.near import endpoints, models

FIXED_URL = "https://cloud-api.near.ai/v1/models"


class _FakeResponse:
    """Minimal ``httpx.Response`` stand-in used by ``fetch_models``."""

    def __init__(self, payload: object, *, status_error: Exception | None = None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> object:
        return self._payload


def _patch_get(monkeypatch: pytest.MonkeyPatch, payload: object = None, *, error: Exception | None = None) -> dict:
    """Replace ``models.httpx.get`` and capture the request arguments."""
    calls: dict = {}

    def fake_get(url: str, *, headers: dict, timeout: float) -> _FakeResponse:
        calls["url"] = url
        calls["headers"] = headers
        calls["timeout"] = timeout
        if error is not None:
            raise error
        return _FakeResponse(payload)

    monkeypatch.setattr(models.httpx, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_models_parses_data_envelope_and_tolerates_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "Qwen/Qwen3.8-27B", "context_length": 262144, "extra": True},
            {"id": "  padded/model  "},
            {"id": ""},
            {"id": "   "},
            {"id": 42},
            {"no": "id"},
            "not-a-dict",
        ],
    }
    _patch_get(monkeypatch, payload)

    result = models.fetch_models("nk_test")

    assert result == [
        {"id": "Qwen/Qwen3.8-27B", "context_length": 262144, "extra": True},
        {"id": "padded/model"},
    ]


def test_fetch_models_accepts_a_bare_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, [{"id": "bare/model"}])

    assert models.fetch_models("nk_test") == [{"id": "bare/model"}]


def test_fetch_models_uses_the_fixed_endpoint_and_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_get(monkeypatch, {"data": []})

    models.fetch_models("nk_test")

    assert calls["url"] == FIXED_URL
    assert calls["url"] == endpoints.models_url()
    assert calls["headers"] == {"Authorization": "Bearer nk_test"}
    assert calls["timeout"] == models.DEFAULT_TIMEOUT_SECONDS


def test_fetch_models_forwards_a_custom_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_get(monkeypatch, {"data": []})

    models.fetch_models("nk_test", timeout=2.5)

    assert calls["timeout"] == 2.5


def test_fetch_models_raises_on_http_status_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", FIXED_URL)
    status_error = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, request=request),
    )
    _patch_get(monkeypatch, error=status_error)

    with pytest.raises(httpx.HTTPStatusError):
        models.fetch_models("bad-key")


def test_fetch_models_raises_on_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, error=httpx.ConnectError("unreachable"))

    with pytest.raises(httpx.ConnectError):
        models.fetch_models("nk_test")


def test_fetch_models_rejects_a_payload_without_a_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, {"data": "not-a-list"})

    with pytest.raises(TypeError):
        models.fetch_models("nk_test")


# ---------------------------------------------------------------------------
# Chat filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        {"id": "Qwen/Qwen3.8-27B", "input_modalities": ["text", "image"]},
        {"id": "Qwen/Qwen3-VL-30B-A3B-Instruct", "input_modalities": ["text", "image"]},
        {"id": "deepseek/deepseek-v4.1-flash", "supported_features": ["tools", "reasoning"]},
        {"id": "plain/model"},
        {"id": "unknown/schema", "some_future_field": True},
        {"id": "declared/chat", "mode": "chat"},
        {"id": "declared/completion", "type": "completion"},
        {"id": "declared/text-generation", "model_type": "text-generation"},
    ],
)
def test_is_chat_model_keeps_chat_models_including_vision(model: dict) -> None:
    assert models.is_chat_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        {"id": "Qwen/Qwen3-Embedding-0.6B", "input_modalities": ["text"]},
        {"id": "embedding/model"},
        {"id": "Qwen/Qwen3-Reranker-0.6B"},
        {"id": "audio/model", "input_modalities": ["text"]},
        {"id": "tts/model"},
        {"id": "speech/model"},
        {"id": "openai/whisper-large-v3", "input_modalities": ["audio"]},
        {"id": "transcribe/model"},
        {"id": "moderation/model"},
        {"id": "dall-e-3"},
        {"id": "stable-diffusion-xl"},
        {"id": "sdxl-base"},
        {"id": "flux-schnell"},
        {"id": "image-gen/model"},
        {"id": "video/only", "input_modalities": ["video"]},
        {"id": "declared/embedding", "mode": "embedding"},
        {"id": "declared/rerank", "type": "reranker"},
        {"id": "declared/tts", "model_type": "text-to-speech"},
    ],
)
def test_is_chat_model_excludes_non_chat_models(model: dict) -> None:
    assert models.is_chat_model(model) is False


def test_is_chat_model_rejects_missing_or_non_string_ids() -> None:
    assert models.is_chat_model({}) is False
    assert models.is_chat_model({"id": ""}) is False
    assert models.is_chat_model({"id": 7}) is False  # type: ignore[dict-item]


def test_discover_models_fetches_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(
        monkeypatch,
        {
            "data": [
                {"id": "Qwen/Qwen3.8-27B", "input_modalities": ["text", "image"]},
                {"id": "Qwen/Qwen3-Embedding-0.6B", "input_modalities": ["text"]},
                {"id": "openai/whisper-large-v3", "input_modalities": ["audio"]},
            ]
        },
    )

    result = models.discover_models("nk_test")

    assert [model["id"] for model in result] == ["Qwen/Qwen3.8-27B"]


def test_discover_models_propagates_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, error=httpx.ConnectError("unreachable"))

    with pytest.raises(httpx.ConnectError):
        models.discover_models("nk_test")


# ---------------------------------------------------------------------------
# model_info normalisation
# ---------------------------------------------------------------------------


def test_normalized_model_info_maps_limits_capabilities_and_limits_omitted() -> None:
    info = models.normalized_model_info(
        {
            "id": "Qwen/Qwen3.8-27B",
            "context_length": 262144,
            "max_output_length": 65536,
            "supported_features": ["tools", "reasoning"],
            "input_modalities": ["image", "text"],
            "output_modalities": ["text"],
            "created": 1700000000,
        }
    )

    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 262144
    assert info["max_output_tokens"] == 65536
    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is True
    assert info["input_modalities"] == ["image", "text"]
    assert info["output_modalities"] == ["text"]
    assert info["release_date"] == time.strftime("%Y-%m-%d", time.gmtime(1700000000))
    assert info["reasoning_effort_levels"] == list(generate_config._REASONING_EFFORTS)
    for effort in generate_config._REASONING_EFFORTS:
        assert info[f"supports_{effort}_reasoning_effort"] is True


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({"no": "limits"}, None),
        ({"max_completion_tokens": 8192}, 8192),
        ({"max_output_tokens": 0, "max_completion_tokens": 4096}, 4096),
    ],
)
def test_normalized_model_info_omits_unknown_output_limit(model: dict, expected: int | None) -> None:
    info = models.normalized_model_info(model)

    assert info.get("max_output_tokens") == expected
    assert ("max_output_tokens" in info) is (expected is not None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_length", 1000),
        ("max_context_length", 2000),
        ("context_window", 3000),
        ("max_input_tokens", 4000),
    ],
)
def test_normalized_model_info_reads_each_context_field(field: str, value: int) -> None:
    assert models.normalized_model_info({field: value})["max_input_tokens"] == value


def test_normalized_model_info_reads_booleans_and_capabilities() -> None:
    info = models.normalized_model_info(
        {
            "capabilities": ["tool_calling", "vision", "thinking"],
            "supports_tools": False,
        }
    )

    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is True

    boolean_info = models.normalized_model_info({"function_calling": True, "vision": True, "reasoning": True})
    assert boolean_info["supports_function_calling"] is True
    assert boolean_info["supports_vision"] is True
    assert boolean_info["supports_reasoning"] is True


def test_normalized_model_info_defaults_capabilities_to_false() -> None:
    info = models.normalized_model_info({"id": "plain/model"})

    assert info["supports_function_calling"] is False
    assert info["supports_vision"] is False
    assert info["supports_reasoning"] is False
    assert "reasoning_effort_levels" not in info
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info
