"""Tests for the self-contained Chutes ``CustomLLM`` provider (custom_provider.py).

Unlike test_e2ee_litellm.py (which exercises LiteLLM's builtin ``chutes``
provider plus a monkeypatch), these tests exercise ``ChutesCustomProvider``:
a provider registered through ``litellm.custom_provider_map`` that owns its own
chutes-e2ee transport.  The mock only answers requests it could decrypt, so a
passing round-trip is cryptographic proof that the handler really encrypted the
request end to end.
"""

import asyncio
import json

import pytest

from chutes_litellm.custom_provider import PROVIDER, register
from mock_chutes_server import MODEL_ID

register()


def _call(mock_server, api_key: str = "cpk_custom", prompt: str = "ping", stream: bool = False, **kwargs):
    import litellm

    return litellm.acompletion(
        model=f"{PROVIDER}/{MODEL_ID}",
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        stream=stream,
        timeout=60,
        **kwargs,
    )


async def _collect(wrapper):
    content = ""
    reasoning = ""
    tool_calls = []
    finish_reason = None
    usage = None
    async for chunk in wrapper:
        if chunk.choices and chunk.choices[0].delta is not None:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if getattr(delta, "reasoning_content", None):
                reasoning += delta.reasoning_content
            for tc in delta.tool_calls or []:
                tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments})
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        usage = getattr(chunk, "usage", None) or usage
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls, "finish_reason": finish_reason, "usage": usage}


def test_registration_and_encryption(mock_server):
    resp = asyncio.run(_call(mock_server, prompt="ping"))

    assert resp.choices[0].message.content == "pong from mock"

    state = mock_server.state
    invoke = state.last_invoke()
    assert invoke.plaintext["model"] == MODEL_ID
    assert invoke.plaintext["messages"][0]["content"] == "ping"
    assert invoke.headers.get("X-E2E-Nonce")
    assert invoke.headers.get("Authorization", "").startswith("Bearer ")
    assert b"ping" not in invoke.blob
    assert not any("chat/completions" in p for p in state.seen_paths)


def test_model_prefix_is_stripped_for_the_wire(mock_server):
    asyncio.run(_call(mock_server, prompt="wire model"))
    assert mock_server.state.last_invoke().plaintext["model"] == MODEL_ID


def test_non_streaming_usage(mock_server):
    resp = asyncio.run(_call(mock_server))
    assert resp.usage is not None
    assert resp.usage.total_tokens == 7


def test_tool_call_non_streaming(mock_server):
    mock_server.state.scenario = "tools"
    resp = asyncio.run(_call(mock_server, prompt="what is the weather"))

    message = resp.choices[0].message
    assert message.content is None
    assert resp.choices[0].finish_reason == "tool_calls"
    tool = message.tool_calls[0]
    assert tool["id"] == "call_abc"
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert json.loads(tool["function"]["arguments"]) == {"city": "Paris"}


def test_reasoning_non_streaming(mock_server):
    mock_server.state.scenario = "reasoning"
    resp = asyncio.run(_call(mock_server, prompt="think"))

    message = resp.choices[0].message
    assert message.content == "final answer"
    assert message.reasoning_content == "let me think carefully"
    assert resp.usage.completion_tokens_details.reasoning_tokens == 5


def test_streaming_content(mock_server):
    async def _run():
        wrapper = await _call(mock_server, prompt="hello", stream=True)
        return await _collect(wrapper)

    result = asyncio.run(_run())
    assert "pong" in result["content"] and "mock" in result["content"]
    assert result["finish_reason"] == "stop"
    assert mock_server.state.last_invoke().stream is True


def test_streaming_tool_call_fragments_are_reassembled(mock_server):
    mock_server.state.scenario = "tools"

    async def _run():
        wrapper = await _call(mock_server, prompt="weather", stream=True)
        result = await _collect(wrapper)
        args = "".join(tc["arguments"] or "" for tc in result["tool_calls"])
        name = next((tc["name"] for tc in result["tool_calls"] if tc["name"]), None)
        return args, name, result["finish_reason"]

    args, name, finish_reason = asyncio.run(_run())
    assert name == "get_weather"
    assert json.loads(args) == {"city": "Paris"}
    assert finish_reason == "tool_calls"


def test_streaming_reasoning_and_content(mock_server):
    mock_server.state.scenario = "reasoning"

    async def _run():
        wrapper = await _call(mock_server, prompt="think", stream=True)
        return await _collect(wrapper)

    result = asyncio.run(_run())
    # ModelResponseStream chunks carry reasoning_content through LiteLLM's
    # custom-provider streaming path untouched.
    assert result["reasoning"].startswith("step one")
    assert "step two" in result["reasoning"]
    assert result["content"] == "final answer"
    assert result["finish_reason"] == "stop"


def test_error_mapping(mock_server):
    mock_server.state.scenario = "rate_limit"
    import litellm

    with pytest.raises(litellm.RateLimitError):
        asyncio.run(_call(mock_server, prompt="please fail"))


def test_missing_api_key_raises(monkeypatch):
    import litellm

    monkeypatch.delenv("CHUTES_API_KEY", raising=False)

    async def _run():
        return await litellm.acompletion(
            model=f"{PROVIDER}/{MODEL_ID}",
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
        )

    with pytest.raises(litellm.AuthenticationError):
        asyncio.run(_run())


def test_attestation_gate_fails_closed(mock_server):
    """With CHUTES_VERIFY_ATTESTATION=true the custom provider must refuse to
    send when the instance evidence does not bind, and must not invoke."""
    import os

    from chutes_litellm import attestation as att

    att._verify_cache.clear()
    mock_server.state.evidence_ok = False
    os.environ["CHUTES_VERIFY_ATTESTATION"] = "true"
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(_call(mock_server, api_key="cpk_custom_att"))
        assert "attestation failed" in str(excinfo.value)
        assert mock_server.state.invokes == []
    finally:
        os.environ.pop("CHUTES_VERIFY_ATTESTATION", None)
        mock_server.state.evidence_ok = True

