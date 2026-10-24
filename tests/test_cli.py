"""Tests for the ``chutes-litellm-proxy`` argparse CLI (cli.py).

No test starts LiteLLM, uvicorn or the network: ``generate``, the E2EE install
hook, the ``_serve`` boundary and LiteLLM's click command are all replaced with
recorders/fakes.
"""

from __future__ import annotations

import inspect
import sys

import click
import litellm
import pytest

from chutes_litellm import cli

GENERATED_PATH = "/tmp/config.generated.yml"


class _NullLogger:
    """Minimal logger stub so tests never open the on-disk log file."""

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None


class _Recorder:
    """Records the calls ``litellm_proxy`` makes instead of serving LiteLLM."""

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.serve_calls: list[dict[str, object]] = []
        self.install_calls = 0
        self.order: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_generate(*, template: str | None = None, output: str | None = None) -> str:
            self.generate_calls.append({"template": template, "output": output})
            return GENERATED_PATH

        def fake_install() -> bool:
            self.install_calls += 1
            self.order.append("install")
            return True

        def fake_serve(config_path: str, host: str, port: int) -> int:
            self.serve_calls.append({"config_path": config_path, "host": host, "port": port})
            self.order.append("serve")
            return 0

        def forbidden_exec(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("process replacement must not be used")

        monkeypatch.setattr(cli, "generate", fake_generate)
        monkeypatch.setattr(cli.e2ee_litellm, "install", fake_install)
        monkeypatch.setattr(cli, "_serve", fake_serve)
        monkeypatch.setattr(cli, "get_logger", lambda _name="chutes_litellm": _NullLogger())
        monkeypatch.setattr(cli.os, "execvp", forbidden_exec)


@pytest.fixture(autouse=True)
def _clean_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient overrides so precedence assertions are deterministic."""
    for name in (
        "LITELLM_TEMPLATE",
        "LITELLM_HOST",
        "LITELLM_PORT",
        "CHUTES_CUSTOM_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_help_documents_options_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.litellm_proxy(["--help"])

    assert excinfo.value.code == 0
    stdout = capsys.readouterr().out
    assert "--config" in stdout
    assert "--host" in stdout
    assert "--port" in stdout
    assert "always enforced" in stdout


def test_cli_options_override_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    config_file = tmp_path / "config.yml"
    config_file.write_text("model_list: []\n", encoding="utf-8")

    monkeypatch.setenv("LITELLM_TEMPLATE", str(tmp_path / "ignored-env-template.yml"))
    monkeypatch.setenv("LITELLM_HOST", "10.0.0.1")
    monkeypatch.setenv("LITELLM_PORT", "9999")

    cli.litellm_proxy(["--config", str(config_file), "--host", "127.0.0.1", "--port", "8123"])

    assert recorder.generate_calls == [{"template": str(config_file), "output": None}]
    assert recorder.install_calls == 1
    assert recorder.serve_calls == [{"config_path": GENERATED_PATH, "host": "127.0.0.1", "port": 8123}]


def test_environment_fallbacks_are_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    template = tmp_path / "env-template.yml"
    monkeypatch.setenv("LITELLM_TEMPLATE", str(template))
    monkeypatch.setenv("LITELLM_HOST", "192.0.2.10")
    monkeypatch.setenv("LITELLM_PORT", "8181")

    cli.litellm_proxy([])

    assert recorder.generate_calls == [{"template": str(template), "output": None}]
    assert recorder.serve_calls == [{"config_path": GENERATED_PATH, "host": "192.0.2.10", "port": 8181}]


def test_no_options_uses_historical_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    cli.litellm_proxy([])

    # ``generate`` itself resolves the packaged template when passed None.
    assert recorder.generate_calls == [{"template": None, "output": None}]
    assert recorder.serve_calls == [{"config_path": GENERATED_PATH, "host": "0.0.0.0", "port": 4000}]


def test_missing_config_file_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    missing = tmp_path / "does-not-exist.yml"
    with pytest.raises(SystemExit) as excinfo:
        cli.litellm_proxy(["--config", str(missing)])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "not found" in stderr
    assert str(missing) in stderr
    assert recorder.generate_calls == []
    assert recorder.serve_calls == []


def test_invalid_port_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.litellm_proxy(["--port", "not-a-number"])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "--port" in stderr
    assert recorder.generate_calls == []
    assert recorder.serve_calls == []


def test_invalid_environment_port_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    monkeypatch.setenv("LITELLM_PORT", "not-a-number")

    with pytest.raises(SystemExit) as excinfo:
        cli.litellm_proxy([])

    assert excinfo.value.code == 2
    assert "LITELLM_PORT" in capsys.readouterr().err
    assert recorder.generate_calls == []
    assert recorder.serve_calls == []


def test_e2ee_install_precedes_in_process_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the E2EE patch must exist before the server starts serving.

    A process replacement (or fresh workers) would drop the in-memory client
    swap and send plaintext to the default ``chutes/`` provider.
    """
    recorder = _Recorder()
    recorder.install(monkeypatch)

    cli.litellm_proxy([])

    assert recorder.order == ["install", "serve"]
    assert recorder.install_calls == 1
    source = inspect.getsource(cli)
    assert "os.execvp" not in source
    assert "os.exec" not in source


def test_serve_forwards_args_and_pins_single_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_main(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(litellm.run_server, "main", fake_main)

    assert cli._serve("/tmp/config.yml", "127.0.0.1", 4321) == 0
    args = calls["args"]
    assert isinstance(args, list)
    assert args[:6] == ["--config", "/tmp/config.yml", "--host", "127.0.0.1", "--port", "4321"]
    assert args[args.index("--num_workers") + 1] == "1"
    assert calls["prog_name"] == "litellm"
    assert calls["standalone_mode"] is False


def test_serve_maps_usage_error_to_exit_two_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_main(**_kwargs: object) -> None:
        raise click.UsageError("bad option")

    monkeypatch.setattr(litellm.run_server, "main", fake_main)

    assert cli._serve("/tmp/config.yml", "127.0.0.1", 4321) == 2
    stderr = capsys.readouterr().err
    assert "bad option" in stderr
    assert "Traceback" not in stderr


def test_serve_maps_click_error_to_its_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_main(**_kwargs: object) -> None:
        raise click.ClickException("boom")

    monkeypatch.setattr(litellm.run_server, "main", fake_main)

    assert cli._serve("/tmp/config.yml", "127.0.0.1", 4321) == 1


def test_serve_missing_litellm_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``None`` in sys.modules makes ``from litellm import ...`` raise ImportError.
    monkeypatch.setitem(sys.modules, "litellm", None)

    assert cli._serve("/tmp/config.yml", "127.0.0.1", 4321) == 1
    stderr = capsys.readouterr().err
    assert "litellm" in stderr
    assert "uv sync" in stderr
    assert "Traceback" not in stderr
