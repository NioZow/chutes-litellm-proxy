"""Live end-to-end checks against the real Chutes API.

Every test here is skipped unless ``CHUTES_LIVE_TEE=1`` and ``CHUTES_API_KEY``
are set.  Run the module on its own (the offline suite points at a mock and
would clobber the real credentials):

    CHUTES_LIVE_TEE=1 CHUTES_API_KEY=<key> \\
        .venv/bin/python -m pytest tests/test_live_proxy.py -v -s

Layers covered:

1. direct custom-provider invoke (non-streaming) over the real E2EE transport,
2. direct custom-provider invoke (streaming),
3. a real in-process LiteLLM proxy server reached over HTTP
   (``scripts/live_server.py``) serving a ``chutes_e2ee/<model>`` completion.

The target model is ``CHUTES_LIVE_MODEL`` if set, otherwise it is auto-selected
from the live ``/v1/models`` listing (preferring a DeepSeek ``-TEE`` model).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
import yaml

from chutes_litellm import attestation

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CHUTES_LIVE_TEE") and os.environ.get("CHUTES_API_KEY")),
    reason="set CHUTES_LIVE_TEE=1 and CHUTES_API_KEY to run live checks",
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "live_server.py"

PROD_API_BASE = "https://api.chutes.ai"
PROD_MODELS_BASE = "https://llm.chutes.ai"

_ENV_KEYS = ("CHUTES_API_KEY", "CHUTES_E2EE_API_BASE", "CHUTES_E2EE_MODELS_BASE", "CHUTES_E2EE_HOSTS", "CHUTES_E2EE_VERIFY_SSL")

_PREFERRED = ("flash-0731", "0731", "v4", "deepseek")

# Params LiteLLM would otherwise strip for a custom provider unless declared.
_ALLOWED_PARAMS = [
    "reasoning_effort",
    "thinking",
    "reasoning",
    "chat_template_kwargs",
    "top_k",
    "repetition_penalty",
    "min_p",
    "enable_thinking",
]

# Models whose thinking we assert is visible, with the request that enables it.
_REASONING_CASES = [
    ("moonshotai/Kimi-K2.6-TEE", {}),
    ("google/gemma-4-31B-turbo-TEE", {"reasoning_effort": "high"}),
    ("deepseek-ai/DeepSeek-V4-Flash-0731-TEE", {"reasoning_effort": "high"}),
]

_THINK_PROMPT = "Think step by step: what is 23*17? Show your reasoning, then the final answer."

_REASONING_PARAMS = dict(_REASONING_CASES)


def _live_api_key() -> str:
    return os.environ["CHUTES_API_KEY"]


def _resolve_live_model(api_key: str) -> str:
    override = os.environ.get("CHUTES_LIVE_MODEL", "").strip()
    if override:
        return override
    model_map = attestation.fetch_model_map(api_key, models_base=PROD_MODELS_BASE)
    tee = [m for m in model_map if m.endswith("-TEE")]
    if not tee:
        raise pytest.skip(f"no -TEE models available under this key ({sorted(model_map)[:5]!r})")
    for fragment in _PREFERRED:
        matches = [m for m in tee if fragment in m.lower()]
        if matches:
            return sorted(matches)[0]
    return sorted(tee)[0]


@pytest.fixture(scope="module")
def live_env() -> None:
    """Production-faithful transport defaults for the duration of this module.

    The offline suite (conftest) injects mock pointers; live runs must keep the
    real bases and enforce real TLS verification.
    """
    snapshot = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        os.environ["CHUTES_E2EE_API_BASE"] = PROD_API_BASE
        os.environ["CHUTES_E2EE_MODELS_BASE"] = PROD_MODELS_BASE
        os.environ.pop("CHUTES_E2EE_HOSTS", None)
        os.environ["CHUTES_E2EE_VERIFY_SSL"] = "true"
        yield
    finally:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def api_key() -> str:
    return _live_api_key()


@pytest.fixture(scope="module")
def model(live_env, api_key) -> str:
    resolved = _resolve_live_model(api_key)
    print(f"\nlive model: {resolved}")
    return resolved


@pytest.fixture(scope="module")
def registered_provider(live_env):
    from chutes_litellm.custom_provider import register

    register()


async def _direct_completion(model: str, api_key: str, stream: bool, prompt: str | None = None, **params):
    import litellm

    return await litellm.acompletion(
        model=f"chutes_e2ee/{model}",
        messages=[{"role": "user", "content": prompt or "Reply with exactly the word: pong"}],
        api_key=api_key,
        stream=stream,
        timeout=240,
        allowed_openai_params=_ALLOWED_PARAMS,
        **params,
    )


def _with_retries(fn, attempts: int = 3, delay: float = 20.0):
    """Chutes -TEE instances may cold-start; retry transient failures."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surfaced below
            last = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    raise last


def test_live_custom_provider_invoke(live_env, registered_provider, model, api_key):
    import asyncio

    import litellm

    resp = _with_retries(lambda: asyncio.run(_direct_completion(model, api_key, stream=False)))
    assert isinstance(resp, litellm.ModelResponse)
    content = resp.choices[0].message.content or ""
    assert "pong" in content, f"unexpected content: {content!r}"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens > 0


async def _collect_stream(wrapper):
    content = ""
    reasoning = ""
    finish_reason = None
    async for chunk in wrapper:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is not None:
            if getattr(delta, "content", None):
                content += delta.content
            if getattr(delta, "reasoning_content", None):
                reasoning += delta.reasoning_content
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
    return content, reasoning, finish_reason


def test_live_custom_provider_stream(live_env, registered_provider, model, api_key):
    import asyncio

    content, _reasoning, finish_reason = _with_retries(
        lambda: asyncio.run(_stream_live(model, api_key)),
        attempts=2,
        delay=15.0,
    )
    assert "pong" in content, f"unexpected streamed content: {content!r}"
    assert finish_reason == "stop"


async def _stream_live(model: str, api_key: str):
    wrapper = await _direct_completion(model, api_key, stream=True)
    return await _collect_stream(wrapper)


def test_live_thinking_visibility(live_env, registered_provider, api_key):
    """Thinking must be readable (separate ``reasoning_content``) for each model.

    Covers non-stream and stream.  Models absent from this account's listing are
    skipped rather than failed.
    """
    import asyncio

    available = set(attestation.fetch_model_map(api_key, models_base=PROD_MODELS_BASE))
    ran = 0
    for model_id, params in _REASONING_CASES:
        if model_id not in available:
            print(f"skip (not in live listing): {model_id}")
            continue
        ran += 1
        resp = _with_retries(
            lambda m=model_id, p=params: asyncio.run(
                _direct_completion(m, api_key, stream=False, prompt=_THINK_PROMPT, **p)
            )
        )
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        assert reasoning.strip(), f"{model_id}: no reasoning_content in non-stream response"

        _content, stream_reasoning, _finish = _with_retries(
            lambda m=model_id, p=params: asyncio.run(_stream_thinking(m, api_key, p)),
            attempts=2,
            delay=15.0,
        )
        assert stream_reasoning.strip(), f"{model_id}: no reasoning_content in stream"

    if ran == 0:
        pytest.skip("none of the target reasoning models are available under this key")


async def _stream_thinking(model: str, api_key: str, params: dict):
    wrapper = await _direct_completion(model, api_key, stream=True, prompt=_THINK_PROMPT, **params)
    return await _collect_stream(wrapper)


def _proxy_config(model: str) -> dict:
    return {
        "model_list": [
            {
                "model_name": f"chutes_e2ee/{model}",
                "litellm_params": {
                    "model": f"chutes_e2ee/{model}",
                    "api_key": "os.environ/CHUTES_API_KEY",
                    "allowed_openai_params": list(_ALLOWED_PARAMS),
                },
            }
        ],
        "general_settings": {"master_key": None},
    }


def test_live_proxy_server_reachable(live_env, registered_provider, model, api_key):
    """Boot the real in-process proxy (scripts/live_server.py) and hit it over HTTP."""
    proc = None
    tmpdir = Path(tempfile.mkdtemp(prefix="chutes-live-"))
    try:
        config_path = tmpdir / "config.live.yml"
        config_path.write_text(yaml.safe_dump(_proxy_config(model)))

        with _free_port_socket() as sock:
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env["CONFIG_FILE_PATH"] = str(config_path)
        proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), "--config", str(config_path), "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        base = f"http://127.0.0.1:{port}"
        _wait_for_health(base, proc)
        _chat_completion_over_http(base, f"chutes_e2ee/{model}")
        _stream_completion_over_http(base, f"chutes_e2ee/{model}")
        reasoning_params = _REASONING_PARAMS.get(model)
        if reasoning_params is not None:
            _reasoning_over_http(base, f"chutes_e2ee/{model}", reasoning_params)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_live_gated_request_uses_verified_instance(live_env, registered_provider, api_key, monkeypatch):
    """With the attestation gate on, the invoked X-Instance-Id must be verified.

    Runs a real encrypted completion through the custom provider while recording
    the instance the transport selected, and asserts it is a member of the set
    ``verify_chute`` approved.
    """
    import asyncio

    import chutes_e2ee.transport as transport_mod

    from chutes_litellm import custom_provider as cp

    model = os.environ.get("CHUTES_LIVE_MODEL", "").strip() or _resolve_live_model(api_key)
    attestation._verify_cache.clear()
    cp.reset_clients()
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    try:
        verified = set(attestation.verify_chute(api_key, model, force=True))
        assert verified, "live chute produced no verified instances"

        seen: dict[str, str] = {}
        original = transport_mod._build_invoke_headers

        def spy(api_key_, chute_id, instance_id, nonce, stream, e2e_path):
            seen["instance_id"] = instance_id
            return original(api_key_, chute_id, instance_id, nonce, stream, e2e_path)

        monkeypatch.setattr(transport_mod, "_build_invoke_headers", spy)
        _with_retries(lambda: asyncio.run(_direct_completion(model, api_key, stream=False)))
        assert seen.get("instance_id") in verified, (
            f"invoked instance {seen.get('instance_id')!r} is not in verified set {sorted(verified)!r}"
        )
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)
        cp.reset_clients()


def _free_port_socket():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock


def _wait_for_health(base: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + 90
    with httpx.Client(timeout=10) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"live server exited early ({proc.returncode}): {stderr}")
            try:
                resp = client.get(f"{base}/health/liveliness")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise RuntimeError("live server did not become healthy within 90s")


def _chat_completion_over_http(base: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
        "stream": False,
    }

    def _post():
        with httpx.Client(timeout=240) as client:
            resp = client.post(f"{base}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp

    resp = _with_retries(_post)
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert "pong" in content, f"unexpected proxied content: {content!r}"
    assert body["choices"][0]["finish_reason"] == "stop"


def _stream_completion_over_http(base: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
        "stream": True,
    }

    def _post():
        content = ""
        saw_done = False
        with httpx.Client(timeout=240) as client:
            with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content += delta["content"]
        return content, saw_done

    content, saw_done = _with_retries(_post)
    assert "pong" in content, f"unexpected streamed proxy content: {content!r}"
    assert saw_done, "stream did not terminate with [DONE]"


def _reasoning_over_http(base: str, model: str, params: dict) -> None:
    """The proxy must forward the thinking params and return reasoning_content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _THINK_PROMPT}],
        "stream": False,
        **params,
    }

    def _post():
        with httpx.Client(timeout=240) as client:
            resp = client.post(f"{base}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp

    body = _with_retries(_post).json()
    message = body["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    assert reasoning.strip(), f"proxy did not surface reasoning_content (params={params})"
