#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai>=1.0.0",
# ]
# ///
"""
Smoke test against the local LiteLLM proxy.

Usage:
    ./proxy_test.py                          # chat + concurrent-stream tests on default models
    ./proxy_test.py model1 model2            # same, on the given models
    ./proxy_test.py --embed  <model>         # embedding test
    ./proxy_test.py --image  <model>         # image-generation test
"""

import sys
import threading
import time

from openai import OpenAI

PROXY_URL = "http://localhost:4000"
DEFAULT_MODELS = [
    "gemini/gemini-2.5-flash",
    "chutes-e2ee/Qwen/Qwen3-32B-TEE",
]

client = OpenAI(api_key="no-auth", base_url=f"{PROXY_URL}/v1")

# ── helpers ──────────────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def ok(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {_GREEN}OK{_RESET}    {label}{suffix}")


def fail(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {_RED}FAIL{_RESET}  {label}{suffix}")


# ── individual test functions ─────────────────────────────────────────────────


def test_models_endpoint() -> None:
    """
    GET /v1/models — the same endpoint the systemd watchdog health-checks.
    Verifies the proxy is up and the model list is non-empty.
    """
    print("=== /v1/models (watchdog endpoint) ===")
    try:
        models = list(client.models.list())
        ok("listed models", f"{len(models)} available")
    except Exception as e:
        fail("listed models", str(e))
    print()


def test_chat(model: str) -> None:
    """Non-streaming, streaming, and tool-calling chat completions."""
    print(f"=== chat: {model} ===")

    # Non-streaming
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        ok("non-streaming", repr(r.choices[0].message.content))
    except Exception as e:
        fail("non-streaming", str(e))

    # Streaming
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            stream=True,
        )
        parts = [
            c.choices[0].delta.content
            for c in stream
            if c.choices and c.choices[0].delta.content
        ]
        ok("streaming", repr("".join(parts)))
    except Exception as e:
        fail("streaming", str(e))

    # Tool calling — some models may not support it; exception is a soft fail
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}
            ],
            tools=tools,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            ok(
                "tool calling (non-streaming)",
                f"called {msg.tool_calls[0].function.name!r}",
            )
        else:
            fail("tool calling (non-streaming)", "no tool calls returned")
    except Exception as e:
        fail("tool calling (non-streaming)", str(e))

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}
            ],
            tools=tools,
            stream=True,
        )
        tc_chunks = [
            c.choices[0].delta.tool_calls[0]
            for c in stream
            if c.choices and c.choices[0].delta.tool_calls
        ]
        if tc_chunks and tc_chunks[0].id:
            ok(
                "tool calling (streaming)",
                f"{len(tc_chunks)} chunks, id={tc_chunks[0].id!r}",
            )
        elif tc_chunks:
            fail(
                "tool calling (streaming)", "chunks received but first chunk missing id"
            )
        else:
            fail("tool calling (streaming)", "no tool call chunks received")
    except Exception as e:
        fail("tool calling (streaming)", str(e))

    print()


def test_concurrent_streaming(model: str, n: int = 4) -> None:
    """
    Launch n streaming requests simultaneously using a threading.Barrier so
    they all start at once.

    This is the key regression test for the astreaming async-generator fix:
    the old sync-iterator approach triggered asyncio.to_thread(next, iter) once
    per chunk, exhausting the default ThreadPoolExecutor under concurrent load
    and causing the proxy to hang.  With the async-generator fix each stream
    uses exactly one background thread regardless of response length.
    """
    print(f"=== concurrent streaming ×{n}: {model} ===")
    results: list[str | Exception] = [Exception("not run")] * n
    barrier = threading.Barrier(n)

    def _stream(idx: int) -> None:
        barrier.wait()  # all threads fire at the same instant
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                stream=True,
            )
            parts = [
                c.choices[0].delta.content
                for c in stream
                if c.choices and c.choices[0].delta.content
            ]
            results[idx] = "".join(parts)
        except Exception as e:
            results[idx] = e

    threads = [
        threading.Thread(target=_stream, args=(i,), daemon=True) for i in range(n)
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        for e in errors:
            fail("concurrent streaming", str(e))
    else:
        ok("concurrent streaming", f"all {n} completed in {elapsed:.1f}s — {results}")
    print()


def test_embedding(model: str) -> None:
    """POST /v1/embeddings — exercises the new embedding() / aembedding() methods."""
    print(f"=== embedding: {model} ===")
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Embeddings enable semantic similarity search",
    ]
    try:
        r = client.embeddings.create(model=model, input=texts)
        dims = len(r.data[0].embedding)
        ok("create embeddings", f"{len(r.data)} vectors × {dims} dims")
    except Exception as e:
        fail("create embeddings", str(e))
    print()


def test_image_generation(model: str) -> None:
    """POST /v1/images/generations — exercises image_generation() / aimage_generation()."""
    print(f"=== image generation: {model} ===")
    try:
        r = client.images.generate(
            model=model,
            prompt="A simple red circle on a white background",
            n=1,
        )
        item = r.data[0]
        preview = (item.url or item.b64_json or "")[:72]
        ok("generate image", f"{preview}…")
    except Exception as e:
        fail("generate image", str(e))
    print()


# ── entry point ───────────────────────────────────────────────────────────────


def _pop_flag(args: list[str], flag: str) -> str | None:
    """Remove --flag <value> from args and return value, or None if absent."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            value = args[idx + 1]
            del args[idx : idx + 2]
            return value
        del args[idx]
    return None


args = list(sys.argv[1:])
embed_model = _pop_flag(args, "--embed")
image_model = _pop_flag(args, "--image")
chat_models = args if args else DEFAULT_MODELS

test_models_endpoint()

for model in chat_models:
    test_chat(model)
    test_concurrent_streaming(model)

if embed_model:
    test_embedding(embed_model)

if image_model:
    test_image_generation(image_model)
