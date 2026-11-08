"""Offline tests for the NEAR AI LiteLLM worker startup prewarm hook."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chutes_litellm.near import startup


class _CaptureLogger:
    """Records log calls without opening the on-disk log file."""

    def __init__(self) -> None:
        self.infos: list[tuple[Any, ...]] = []
        self.warnings: list[tuple[Any, ...]] = []

    def info(self, *args: object, **_kwargs: object) -> None:
        self.infos.append(args)

    def warning(self, *args: object, **_kwargs: object) -> None:
        self.warnings.append(args)


class _FakeRuntime:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.prewarm_calls: list[tuple[str, ...]] = []
        self._error = error

    async def prewarm(self, models: Any) -> dict[str, BaseException | None]:
        self.prewarm_calls.append(tuple(models))
        if self._error is not None:
            raise self._error
        return {model: None for model in models}


def test_prewarm_is_a_noop_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(startup.API_KEY_ENV, raising=False)
    monkeypatch.setenv("NEARAI_PREWARM_MODELS", "model/a")
    monkeypatch.setattr(
        startup,
        "get_runtime",
        lambda *_args, **_kwargs: pytest.fail("runtime must not be created without a key"),
    )

    assert asyncio.run(startup.prewarm()) is None


def test_prewarm_is_a_noop_without_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(startup.API_KEY_ENV, "nk_test")
    monkeypatch.delenv("NEARAI_PREWARM_MODELS", raising=False)
    monkeypatch.setattr(
        startup,
        "get_runtime",
        lambda *_args, **_kwargs: pytest.fail("runtime must not be created without models"),
    )

    assert asyncio.run(startup.prewarm()) is None


def test_prewarm_calls_the_runtime_with_parsed_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setenv(startup.API_KEY_ENV, "nk_test")
    monkeypatch.setenv("NEARAI_PREWARM_MODELS", " Qwen/Qwen3.8-27B , deepseek/deepseek-v4.1-flash ,,")
    monkeypatch.setattr(startup, "get_runtime", lambda api_key: runtime)
    monkeypatch.setattr(startup, "get_logger", lambda _name=None: _CaptureLogger())

    asyncio.run(startup.prewarm())

    assert runtime.prewarm_calls == [("Qwen/Qwen3.8-27B", "deepseek/deepseek-v4.1-flash")]


def test_prewarm_swallows_runtime_failures_and_logs_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _CaptureLogger()
    secret = "secret-key-material-123"
    monkeypatch.setenv(startup.API_KEY_ENV, "nk_test")
    monkeypatch.setenv("NEARAI_PREWARM_MODELS", "model/a")
    monkeypatch.setattr(
        startup,
        "get_runtime",
        lambda _api_key: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(startup, "get_logger", lambda _name=None: logger)

    assert asyncio.run(startup.prewarm()) is None
    assert logger.warnings
    rendered = " ".join(str(call) for call in logger.warnings)
    assert "RuntimeError" in rendered
    assert secret not in rendered


def test_prewarm_logs_per_model_failures_with_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _CaptureLogger()

    class _PartiallyFailingRuntime(_FakeRuntime):
        async def prewarm(self, models: Any) -> dict[str, BaseException | None]:
            self.prewarm_calls.append(tuple(models))
            return {model: ValueError("upstream body") if model == "model/b" else None for model in models}

    runtime = _PartiallyFailingRuntime()
    monkeypatch.setenv(startup.API_KEY_ENV, "nk_test")
    monkeypatch.setenv("NEARAI_PREWARM_MODELS", "model/a,model/b")
    monkeypatch.setattr(startup, "get_runtime", lambda _api_key: runtime)
    monkeypatch.setattr(startup, "get_logger", lambda _name=None: logger)

    asyncio.run(startup.prewarm())

    assert logger.infos
    assert logger.warnings
    rendered = " ".join(str(call) for call in logger.warnings)
    assert "model/b" in rendered
    assert "ValueError" in rendered
    assert "upstream body" not in rendered
