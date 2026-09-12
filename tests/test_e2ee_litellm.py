"""Proof that the native litellm ``chutes`` provider really encrypts on the wire.

The mock server holds the instance ML-KEM-768 *private* key and can therefore
only answer a request that was genuinely encrypted with its public key — so a
round-trip succeeding is cryptographic proof of end-to-end encryption.  We also
assert that the plaintext OpenAI payload never appears on the wire and that no
plaintext request ever reaches the ``/v1/chat/completions`` path.
"""

import asyncio
import json
import os

import httpx
import pytest

from chutes_e2ee import AsyncChutesE2EETransport, ChutesE2EETransport
from conftest import ensure_installed
from mock_chutes_server import MODEL_ID


def _native_completion(mock_server, api_key: str, prompt: str = "ping", stream: bool = False):
    """Call the real litellm native chutes provider path against the mock."""
    import litellm

    ensure_installed()
    return litellm.acompletion(
        model=f"chutes/{MODEL_ID}",
        api_base=f"{mock_server.base_url}/v1",
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        stream=stream,
        timeout=60,
    )


def test_native_chutes_request_is_e2e_encrypted(mock_server):
    resp = asyncio.run(_native_completion(mock_server, "cpk_e2e_1", prompt="ping"))

    assert resp.choices[0].message.content == "pong from mock"

    state = mock_server.state
    assert state.invokes, "mock never received an /e2e/invoke"

    invoke = state.last_invoke()
    # The server decrypted the request with its private key: proves the client
    # (litellm + native chutes provider) really encrypted it.
    assert invoke.plaintext["model"] == MODEL_ID
    assert invoke.plaintext["messages"][0]["content"] == "ping"

    # E2EE headers the transport must set on the invoke call.
    assert invoke.headers.get("X-E2E-Nonce")
    assert invoke.headers.get("X-Chute-Id") == mock_server.state.chute_id
    assert invoke.headers.get("X-Instance-Id") == mock_server.state.instance_id
    assert invoke.headers.get("Authorization", "").startswith("Bearer ")

    # The ciphertext on the wire must not be (or contain) the plaintext JSON.
    raw = invoke.blob
    assert b"ping" not in raw
    assert "application/json" not in invoke.headers.get("content-type", "")
    with pytest.raises(ValueError):
        json.loads(raw)

    # The plaintext OpenAI endpoint must never have been contacted.
    plaintext_paths = [p for p in state.seen_paths if "chat/completions" in p]
    assert plaintext_paths == [], f"plaintext leaked to: {plaintext_paths}"


def test_sync_openai_client_also_encrypts(mock_server):
    from openai import OpenAI

    client = OpenAI(
        api_key="cpk_e2e_sync",
        base_url=f"{mock_server.base_url}/v1",
        http_client=httpx.Client(
            transport=ChutesE2EETransport(
                api_key="cpk_e2e_sync",
                api_base=mock_server.base_url,
                models_base=mock_server.base_url,
            )
        ),
    )
    resp = client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": "sync ping"}], max_tokens=8
    )
    assert resp.choices[0].message.content == "pong from mock"
    assert mock_server.state.last_invoke().plaintext["messages"][0]["content"] == "sync ping"


def test_async_streaming_is_e2e_encrypted(mock_server):
    """Streaming over the real async E2EE transport: decrypted SSE, not plaintext."""
    from openai import AsyncOpenAI

    async def _run():
        client = AsyncOpenAI(
            api_key="cpk_e2e_stream",
            base_url=f"{mock_server.base_url}/v1",
            http_client=httpx.AsyncClient(
                transport=AsyncChutesE2EETransport(
                    api_key="cpk_e2e_stream",
                    api_base=mock_server.base_url,
                    models_base=mock_server.base_url,
                )
            ),
        )
        stream = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "stream ping"}],
            max_tokens=8,
            stream=True,
        )
        content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        await client.close()
        return content

    content = asyncio.run(_run())
    assert "pong" in content and "mock" in content

    invoke = mock_server.state.last_invoke()
    assert invoke.stream is True
    # Nothing plaintext on the wire for the streaming call either.
    assert b"stream ping" not in invoke.blob


def test_native_chutes_streaming_through_litellm(mock_server):
    """The whole proxy-style path: litellm acompletion(stream=True) -> E2EE."""

    async def _run():
        wrapper = await _native_completion(mock_server, "cpk_e2e_litellm_stream", prompt="hello", stream=True)
        content = ""
        async for chunk in wrapper:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return content

    content = asyncio.run(_run())
    assert "pong" in content and "mock" in content
    assert mock_server.state.last_invoke().stream is True


def test_roundtrip_plaintext_matches_request_body(mock_server):
    """The decrypted request equals what litellm wanted to send (modulo the
    transport-added e2e_response_pk)."""
    asyncio.run(_native_completion(mock_server, "cpk_e2e_sem", prompt="semantic check"))
    plain = mock_server.state.last_invoke().plaintext
    assert plain["messages"] == [{"role": "user", "content": "semantic check"}]
    assert plain.get("stream") in (None, False)
    assert "e2e_response_pk" in plain  # transport-injected reply key


def test_router_proxy_path_is_e2e_encrypted(mock_server):
    """Mirror of the real proxy: a Router built from YAML-style deployments (no
    client objects) must still send chutes traffic through the E2EE transport
    thanks to the module-level http-client swap."""
    from litellm import Router

    ensure_installed()
    router = Router(
        model_list=[
            {
                "model_name": f"chutes/{MODEL_ID}",
                "litellm_params": {
                    "model": f"chutes/{MODEL_ID}",
                    "api_key": "cpk_router",
                    "api_base": f"{mock_server.base_url}/v1",
                },
            }
        ],
        num_retries=0,
    )

    async def _run():
        resp = await router.acompletion(
            model=f"chutes/{MODEL_ID}",
            messages=[{"role": "user", "content": "router ping"}],
            max_tokens=8,
        )
        return resp

    resp = asyncio.run(_run())
    assert resp.choices[0].message.content == "pong from mock"
    invoke = mock_server.state.last_invoke()
    assert invoke.plaintext["messages"][0]["content"] == "router ping"
    assert "chat/completions" not in " ".join(mock_server.state.seen_paths)


# ---------------------------------------------------------------------------
# Attested-instance binding (the transport's selection filter)
# ---------------------------------------------------------------------------


def test_instance_filter_keeps_only_verified_instances(monkeypatch):
    """The filter intersects the transport's discovery with the verified set."""
    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el

    a = att.InstanceInfo("inst-a", "pk-a", [])
    b = att.InstanceInfo("inst-b", "pk-b", [])
    monkeypatch.setattr(att, "verify_chute", lambda *args, **kwargs: ["inst-a"])

    transport = object.__new__(el._ChutesScopedTransport)
    transport._api_key = "cpk_filter"
    transport._api_key_provider = None
    transport._verify_quote = False
    transport._verify_gpu = False

    kept = transport._instance_filter("some-chute", [a, b])
    assert [i.instance_id for i in kept] == ["inst-a"]


def _record_transport_ctor(monkeypatch):
    """Capture the kwargs a transport constructor receives."""
    from chutes_litellm import e2ee_litellm as el

    captured: dict = {}

    class _FakeTransport:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    class _FakeAsyncTransport:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def aclose(self):
            pass

    monkeypatch.setattr(el, "_transport_classes", lambda: (_FakeTransport, _FakeAsyncTransport))
    return captured


def test_stock_transport_compatible_when_attestation_off(monkeypatch):
    """With the gate off we must not pass the fork-only `instance_filter` kwarg,
    so a stock `chutes-e2ee` (no hook) keeps working."""
    from chutes_litellm import e2ee_litellm as el

    captured = _record_transport_ctor(monkeypatch)
    os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)
    el._ChutesScopedTransport("cpk", hosts={"h"}, verify=False)
    assert "instance_filter" not in captured

    captured.clear()
    el._ChutesScopedAsyncTransport("cpk", hosts={"h"}, verify=False)
    assert "instance_filter" not in captured


def test_filter_is_passed_when_attestation_on(monkeypatch):
    from chutes_litellm import e2ee_litellm as el

    captured = _record_transport_ctor(monkeypatch)
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    try:
        el._ChutesScopedTransport("cpk", hosts={"h"}, verify=False)
        assert callable(captured.get("instance_filter"))
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)


def test_instance_filter_raises_typed_error_on_failure(monkeypatch):
    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el

    def boom(*args, **kwargs):
        raise att.AttestationError("binding failed")

    monkeypatch.setattr(att, "verify_chute", boom)
    transport = object.__new__(el._ChutesScopedTransport)
    transport._api_key = "cpk_filter"
    transport._api_key_provider = None
    transport._verify_quote = False
    transport._verify_gpu = False

    with pytest.raises(Exception, match="attestation failed"):
        transport._instance_filter("some-chute", [])


def test_transport_filter_refuses_unverified_instance(mock_server, monkeypatch):
    """With only the selection filter active, bad evidence must still refuse the
    request before anything is encrypted/sent (no /e2e/invoke)."""
    import litellm

    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el

    ensure_installed()
    el._clients.clear()
    el._aclients.clear()
    att._verify_cache.clear()
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    # Silence the explicit pre-send check so only the transport's instance
    # filter can refuse the chute.
    monkeypatch.setattr(
        el._ChutesScopedAsyncTransport,
        "_maybe_verify",
        lambda self, request: asyncio.sleep(0),
    )
    mock_server.state.evidence_ok = False
    try:

        async def call():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_filter_e2e",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        with pytest.raises(Exception) as excinfo:
            asyncio.run(call())
        assert "attestation failed" in str(excinfo.value)
        assert mock_server.state.invokes == [], "request reached /e2e/invoke unverified"
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)
        mock_server.state.evidence_ok = True
        el._clients.clear()
        el._aclients.clear()
