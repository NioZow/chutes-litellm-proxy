"""Proof that the native litellm ``chutes`` provider really encrypts on the wire.

The mock server holds the instance ML-KEM-768 *private* key and can therefore
only answer a request that was genuinely encrypted with its public key — so a
round-trip succeeding is cryptographic proof of end-to-end encryption.  We also
assert that the plaintext OpenAI payload never appears on the wire and that no
plaintext request ever reaches the ``/v1/chat/completions`` path.
"""

import asyncio
import inspect
import json

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
    """The filter pins both the verified id *and* its e2e_pubkey."""
    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el

    a = att.InstanceInfo("inst-a", "pk-a", [])
    b = att.InstanceInfo("inst-b", "pk-b", [])
    monkeypatch.setattr(att, "verify_chute", lambda *args, **kwargs: {"inst-a": "pk-a"})

    transport = object.__new__(el._ChutesScopedTransport)
    transport._api_key = "cpk_filter"
    transport._api_key_provider = None

    kept = transport._instance_filter("some-chute", [a, b])
    assert [i.instance_id for i in kept] == ["inst-a"]

    # Same id, different pubkey: the substituted key must be dropped...
    substituted = att.InstanceInfo("inst-a", "pk-evil", [])
    assert transport._instance_filter("some-chute", [substituted]) == []

    # ...and a different id carrying the verified key is not accepted either.
    wrong_id = att.InstanceInfo("inst-c", "pk-a", [])
    assert transport._instance_filter("some-chute", [wrong_id]) == []


def _record_transport_ctor(monkeypatch):
    """Capture the kwargs a transport constructor receives."""
    from chutes_litellm import e2ee_litellm as el

    captured: dict = {}

    class _FakeTransport:
        def __init__(self, api_key, *, instance_filter, **kwargs):
            captured.update(api_key=api_key, instance_filter=instance_filter, **kwargs)

        def close(self):
            pass

    class _FakeAsyncTransport:
        def __init__(self, api_key, *, instance_filter, **kwargs):
            captured.update(api_key=api_key, instance_filter=instance_filter, **kwargs)

        async def aclose(self):
            pass

    monkeypatch.setattr(el, "_transport_classes", lambda: (_FakeTransport, _FakeAsyncTransport))
    return captured


def test_filter_is_always_installed(monkeypatch):
    """The attestation-backed instance filter has no opt-out."""
    from chutes_litellm import e2ee_litellm as el

    captured = _record_transport_ctor(monkeypatch)
    el._ChutesScopedTransport("cpk", hosts={"h"})
    assert callable(captured.get("instance_filter"))
    # Supplying `inner` would bypass chutes-e2ee's env-proxy-aware default.
    assert "inner" not in captured

    captured.clear()
    el._ChutesScopedAsyncTransport("cpk", hosts={"h"})
    assert callable(captured.get("instance_filter"))
    assert "inner" not in captured


def test_installed_transports_expose_instance_filter_hook():
    """Regression: a stale chutes-e2ee pin must fail here, not at request time.

    The Nix package previously pinned an older fork revision.  It built fine,
    then every Chutes request returned 500 with
    ``AsyncChutesE2EETransport.__init__() got an unexpected keyword argument
    'instance_filter'``.  Assert against the *installed* API, not a fake.
    """
    from chutes_e2ee import AsyncChutesE2EETransport, ChutesE2EETransport

    for transport_cls in (ChutesE2EETransport, AsyncChutesE2EETransport):
        param = inspect.signature(transport_cls.__init__).parameters.get("instance_filter")
        assert param is not None, f"{transport_cls.__name__} lacks instance_filter"
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ), f"{transport_cls.__name__}.instance_filter is not an explicit keyword"


def test_scoped_transports_construct_against_installed_chutes_e2ee():
    """No fakes: both scoped wrappers must accept the real package's API."""
    from chutes_litellm import e2ee_litellm as el

    sync_transport = el._ChutesScopedTransport("cpk_contract", hosts={"h"})
    async_transport = el._ChutesScopedAsyncTransport("cpk_contract", hosts={"h"})
    sync_transport.close()
    asyncio.run(async_transport.aclose())


def test_chutes_transport_uses_library_env_proxy_default(monkeypatch):
    """Regression: the Chutes transport must not be given a bare `inner`.

    `chutes-e2ee` only applies HTTP(S)_PROXY / NO_PROXY in its default inner.
    Overriding it with a plain httpx transport bypassed Squid in the deployed
    guest and made discovery time out after 15s.
    """
    import chutes_e2ee.transport as ct

    from chutes_litellm import e2ee_litellm as el

    sync_sentinel = httpx.HTTPTransport()
    async_sentinel = httpx.AsyncHTTPTransport()
    monkeypatch.setattr(ct, "_default_transport", lambda: sync_sentinel)
    monkeypatch.setattr(ct, "_default_async_transport", lambda: async_sentinel)

    sync_transport = el._ChutesScopedTransport("cpk_proxy", hosts={"h"})
    async_transport = el._ChutesScopedAsyncTransport("cpk_proxy", hosts={"h"})
    assert sync_transport._transport._inner is sync_sentinel
    assert async_transport._transport._inner is async_sentinel
    sync_transport.close()
    asyncio.run(async_transport.aclose())


def test_preflight_rejects_transport_without_instance_filter(monkeypatch):
    """The hook is mandatory, so a transport lacking it must be refused."""
    from chutes_litellm import e2ee_litellm as el

    class _OldTransport:
        def __init__(self, api_key):
            pass

    monkeypatch.setattr(el, "_transport_classes", lambda: (_OldTransport, _OldTransport))

    with pytest.raises(RuntimeError, match="instance_filter"):
        el._require_attestation_hook()


def test_install_fails_fast_when_instance_filter_hook_is_missing(monkeypatch):
    """A bad package must abort startup, not survive until the first request."""
    from chutes_litellm import e2ee_litellm as el

    class _OldTransport:
        def __init__(self, api_key):
            pass

    monkeypatch.setenv("CHUTES_API_KEY", "cpk_contract")
    monkeypatch.setattr(el, "_IMPORT_ERROR", None)
    monkeypatch.setattr(el, "_patched", False)
    monkeypatch.setattr(el, "_transport_classes", lambda: (_OldTransport, _OldTransport))

    with pytest.raises(RuntimeError, match="instance_filter"):
        el.install()
    assert el._patched is False


def test_e2ee_hosts_are_derived_and_env_overrides_are_ignored(monkeypatch):
    """Hosts follow the fixed module constants; every host env var is inert."""
    from chutes_litellm import e2ee_litellm as el
    from chutes_litellm import endpoints

    monkeypatch.setattr(endpoints, "API_BASE", "https://api.fixed.test")
    monkeypatch.setattr(endpoints, "MODELS_BASE", "https://llm.fixed.test")
    monkeypatch.setenv("CHUTES_E2EE_HOSTS", "evil.example")
    monkeypatch.setenv("CHUTES_E2EE_API_BASE", "https://api.private.test")
    monkeypatch.setenv("CHUTES_E2EE_MODELS_BASE", "https://llm.private.test")

    hosts = el._e2ee_hosts()
    # derived from the fixed endpoints plus the canonical Chutes hosts
    assert {"api.fixed.test", "llm.fixed.test"} <= hosts
    assert {"llm.chutes.ai", "api.chutes.ai"} <= hosts
    # neither the env host list nor the removed base overrides are honoured
    assert "evil.example" not in hosts
    assert "api.private.test" not in hosts
    assert "llm.private.test" not in hosts


def test_instance_filter_raises_typed_error_on_failure(monkeypatch):
    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el

    def boom(*args, **kwargs):
        raise att.AttestationError("binding failed")

    monkeypatch.setattr(att, "verify_chute", boom)
    transport = object.__new__(el._ChutesScopedTransport)
    transport._api_key = "cpk_filter"
    transport._api_key_provider = None

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

    # Silence the explicit pre-send check so only the transport's instance
    # filter can refuse the chute.
    async def _skip_verify_request(self, request):
        return None

    monkeypatch.setattr(
        el._ChutesScopedAsyncTransport,
        "_verify_request",
        _skip_verify_request,
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
        mock_server.state.evidence_ok = True
        el._clients.clear()
        el._aclients.clear()


def test_transport_key_substitution_is_refused_before_invoke(mock_server, monkeypatch):
    """TOCTOU regression: a substituted pubkey for a verified id must not be used.

    The gate is fed the *real* instance pubkey (monkeypatched ``fetch_instances``),
    while the transport's own ``/e2e/instances`` discovery receives a different
    key for the same instance id.  Pinning only the id would let the request go
    out encrypted to the attacker's key; pinning id+pubkey refuses it before any
    ``POST /e2e/invoke``.
    """
    import base64

    import litellm
    from pqcrypto.kem.ml_kem_768 import generate_keypair

    from chutes_litellm import attestation as att
    from chutes_litellm import e2ee_litellm as el
    from mock_chutes_server import INSTANCE_ID

    ensure_installed()
    el._clients.clear()
    el._aclients.clear()
    att._verify_cache.clear()

    real_pubkey = mock_server.state.pubkey_b64
    foreign_pk, _ = generate_keypair()
    mock_server.state.discovery_pubkey_b64 = base64.b64encode(foreign_pk).decode()

    # The attestation GATE sees the real key (and the evidence is signed for it).
    monkeypatch.setattr(
        att,
        "fetch_instances",
        lambda *a, **k: [att.InstanceInfo(INSTANCE_ID, real_pubkey, [])],
    )

    try:

        async def call():
            return await litellm.acompletion(
                model=f"chutes/{MODEL_ID}",
                api_base=f"{mock_server.base_url}/v1",
                api_key="cpk_toctou",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                timeout=60,
            )

        invokes_before = len(mock_server.state.invokes)
        with pytest.raises(Exception) as excinfo:
            asyncio.run(call())

        # The SDK wraps transport failures; the root cause must be the
        # attestation filter refusing the substituted key.
        chain = []
        error: BaseException | None = excinfo.value
        while error is not None:
            chain.append(str(error))
            error = error.__cause__ or error.__context__
        message = "\n".join(chain)
        assert "instance filter rejected every instance" in message.lower()
        assert len(mock_server.state.invokes) == invokes_before, "request reached /e2e/invoke with a substituted pubkey"
        assert any("instances" in p for p in mock_server.state.seen_paths)

        # Prove the gate itself had verified the real id+key: the refusal above
        # came from the key pin, not from a failed attestation.
        assert att.verify_chute("cpk_toctou", MODEL_ID, force=True) == {INSTANCE_ID: real_pubkey}
    finally:
        mock_server.state.discovery_pubkey_b64 = None
        el._clients.clear()
        el._aclients.clear()
