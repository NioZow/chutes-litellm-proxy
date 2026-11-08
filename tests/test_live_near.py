"""Live end-to-end checks against the real NEAR AI Cloud API.

Every test is skipped unless ``NEARAI_LIVE=1`` and ``NEARAI_API_KEY`` are set.
A TEE model and a routed model are selected explicitly:

    NEARAI_LIVE=1 NEARAI_API_KEY=near-... \\
    NEARAI_TEE_MODEL=Qwen/Qwen3.8-27B \\
    NEARAI_ROUTED_MODEL=z-ai/glm-5.3-flash \\
        uv run pytest tests/test_live_near.py -v -s

The TEE checks assert the complete chain: Gateway TLS/TCB/advisories, model TCB,
GPU evidence and signing key, E2EE ``provider_tee`` receipts, and both streaming
modes.  The routed check asserts Gateway attestation, no model reports, and a
``gateway`` receipt.  The proxy check boots ``scripts/live_server.py`` and drives
``/v1/chat/completions`` over HTTP.

No verification is disabled or relaxed: a genuine failure fails the test.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
import yaml

from chutes_litellm.near import endpoints
from chutes_litellm.near.client import NearRuntime

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEARAI_LIVE"),
    reason="set NEARAI_LIVE=1, NEARAI_API_KEY, NEARAI_TEE_MODEL and NEARAI_ROUTED_MODEL to run live NEAR checks",
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "live_server.py"

_PROMPT = "Reply with exactly the word: pong"


def _require(env_name: str) -> str:
    value = (os.environ.get(env_name) or "").strip()
    if not value:
        pytest.skip(f"{env_name} is not set")
    return value


def _credentials() -> tuple[str, str, str]:
    key = _require("NEARAI_API_KEY")
    tee = _require("NEARAI_TEE_MODEL")
    routed = _require("NEARAI_ROUTED_MODEL")
    return key, tee, routed


def _run(coro, timeout: float = 900):
    async def wrapped():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(wrapped())


async def _verify_receipt_with_retry(sdk, completion_id: str):
    """Fetch and verify one receipt, retrying only retryable ``ApiError``.

    The completion signature endpoint can briefly return 404 while the Gateway
    indexes a fresh completion; the SDK marks that status retryable.  The
    provider retries it too, so the live test mirrors that behavior instead of
    failing on the first transient miss.  Inference is never resent.
    """
    from nearai_inference_sdk import ApiError

    last: ApiError | None = None
    for attempt in range(3):
        try:
            return await sdk.verify_response(completion_id)
        except ApiError as error:
            if not error.retryable:
                raise
            last = error
            await asyncio.sleep(0.2 * (attempt + 1))
    assert last is not None
    raise last


async def _completion_receipt(runtime: NearRuntime, model_id: str, mode: str):
    """Send one non-streaming completion and return ``(receipt, content)``."""
    sdk = runtime.client_for(mode)  # type: ignore[arg-type]
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": _PROMPT}],
        "stream": False,
        # Reasoning models spend tokens on ``reasoning_content`` before the
        # visible answer; a tiny cap truncates the completion and yields empty
        # content or ``finish_reason == "length"``.
        "max_tokens": 256,
        "temperature": 0,
    }
    request = httpx.Request(
        "POST",
        endpoints.chat_completions_url(),
        json=payload,
        headers={"content-type": "application/json"},
        extensions={"timeout": runtime.request_timeout().as_dict()},
    )
    response = await sdk.send(request)
    try:
        body = await response.aread()
    finally:
        await response.aclose()
    data = json.loads(body)
    receipt = await _verify_receipt_with_retry(sdk, data["id"])
    content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    return receipt, content


# ---------------------------------------------------------------------------
# TEE model
# ---------------------------------------------------------------------------


def test_live_tee_model_attestation() -> None:
    key, tee, _ = _credentials()

    async def scenario():
        runtime = NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(tee)
            assert mode == "tee"

            result = await runtime.tee_client.verify(tee)
            gateway = result.gateway
            assert gateway.tls_binding.kind == "attested"
            assert gateway.tls_binding.spki_fingerprint
            assert gateway.tcb_status == "UpToDate"
            assert not gateway.advisory_ids

            assert result.models, "TEE model returned no model attestation"
            for report in result.models:
                assert report.tcb_status == "UpToDate"
                assert not report.advisory_ids
                assert report.gpu_evidence == "verified"
                assert report.signing_public_key
        finally:
            await runtime.aclose()

    _run(scenario())


def test_live_tee_completion_receipt() -> None:
    key, tee, _ = _credentials()

    async def scenario():
        runtime = NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(tee)
            assert mode == "tee"
            receipt, content = await _completion_receipt(runtime, tee, mode)
            assert receipt.signature_kind == "provider_tee"
            assert content.strip()
            return content
        finally:
            await runtime.aclose()

    content = _run(scenario())
    assert "pong" in content.lower(), f"unexpected content: {content!r}"


# ---------------------------------------------------------------------------
# Routed model
# ---------------------------------------------------------------------------


def test_live_routed_model_attestation_and_receipt() -> None:
    key, _, routed = _credentials()

    async def scenario():
        runtime = NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(routed)
            assert mode == "routed"

            result = await runtime.routed_client.verify(routed)
            assert result.gateway.tls_binding.kind == "attested"
            assert result.gateway.tcb_status == "UpToDate"
            assert not result.gateway.advisory_ids
            assert result.models == ()

            receipt, content = await _completion_receipt(runtime, routed, mode)
            assert receipt.signature_kind == "gateway"
            assert content.strip()
        finally:
            await runtime.aclose()

    _run(scenario())


# ---------------------------------------------------------------------------
# Streaming (both modes)
# ---------------------------------------------------------------------------


def test_live_streaming_verified_and_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    key, tee, routed = _credentials()

    from chutes_litellm.near.provider import register

    register()
    import litellm

    messages = [{"role": "user", "content": _PROMPT}]

    async def collect(model_id: str, stream_mode: str) -> tuple[str, str | None]:
        monkeypatch.setenv("NEARAI_STREAM_MODE", stream_mode)
        wrapper = await litellm.acompletion(
            model=f"nearai_e2ee/{model_id}",
            messages=messages,
            api_key=key,
            stream=True,
            timeout=600,
            # Reasoning models spend tokens before the visible answer; a tiny
            # cap can end the stream with ``finish_reason == "length"``.
            max_tokens=256,
            temperature=0,
        )
        content = ""
        finish_reason = None
        async for chunk in wrapper:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is not None and getattr(delta, "content", None):
                content += delta.content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        return content, finish_reason

    async def scenario():
        results = {}
        for label, model_id in (("tee", tee), ("routed", routed)):
            for stream_mode in ("verified", "tail"):
                results[(label, stream_mode)] = await collect(model_id, stream_mode)
        return results

    results = _run(scenario())
    for (label, stream_mode), (content, finish_reason) in results.items():
        assert "pong" in content.lower(), f"{label}/{stream_mode}: unexpected content {content!r}"
        # The terminal chunk must arrive authenticated.  A reasoning model may
        # still hit the token cap, so ``length`` is a valid terminal reason.
        assert finish_reason in {"stop", "length"}, f"{label}/{stream_mode}: finish_reason={finish_reason!r}"


# ---------------------------------------------------------------------------
# Proxy-level
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base: str, proc: subprocess.Popen, deadline_seconds: float = 120) -> None:
    deadline = time.time() + deadline_seconds
    with httpx.Client(timeout=10) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"live server exited early ({proc.returncode}): {stderr}")
            try:
                response = client.get(f"{base}/health/liveliness")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise RuntimeError(f"live server did not become healthy within {deadline_seconds:.0f}s")


def _proxy_config(tee: str) -> dict:
    return {
        "model_list": [
            {
                "model_name": f"nearai/{tee}",
                "litellm_params": {
                    "model": f"nearai_e2ee/{tee}",
                    "api_key": "os.environ/NEARAI_API_KEY",
                },
            }
        ],
        "litellm_settings": {"turn_off_message_logging": True},
        "general_settings": {"master_key": None},
    }


def test_live_proxy_server() -> None:
    _, tee, _ = _credentials()
    proc = None
    tmpdir = Path(tempfile.mkdtemp(prefix="nearai-live-"))
    try:
        config_path = tmpdir / "config.live.yml"
        config_path.write_text(yaml.safe_dump(_proxy_config(tee)))
        port = _free_port()

        env = os.environ.copy()
        env["CONFIG_FILE_PATH"] = str(config_path)
        env.setdefault("NEARAI_STREAM_MODE", "verified")
        proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), "--config", str(config_path), "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        base = f"http://127.0.0.1:{port}"
        _wait_for_health(base, proc)

        payload = {
            "model": f"nearai/{tee}",
            "messages": [{"role": "user", "content": _PROMPT}],
            # Reasoning models need room for ``reasoning_content`` before the
            # visible answer; a tiny cap can produce empty content.
            "max_tokens": 256,
            "temperature": 0,
        }
        with httpx.Client(timeout=600) as client:
            response = client.post(f"{base}/v1/chat/completions", json=payload)
        assert response.status_code == 200, response.text[:500]
        body = response.json()
        content = body["choices"][0]["message"]["content"] or ""
        assert "pong" in content.lower(), f"unexpected proxied content: {content!r}"
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)
