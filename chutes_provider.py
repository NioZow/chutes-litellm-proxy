#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chutes-e2ee>=0.1.0",
#     "httpx>=0.28.1",
#     "litellm>=1.82.0",
# ]
# ///
#
# Sources used to understand LiteLLM's custom provider interface:
#
#   CustomLLM base class — canonical signatures for all overridable methods
#   (completion, streaming, acompletion, astreaming, embedding, aembedding,
#   image_generation, aimage_generation, image_edit, aimage_edit) and the
#   correct import paths for EmbeddingResponse / ImageResponse / ModelResponse:
#   https://github.com/BerriAI/litellm/blob/main/litellm/llms/custom_llm.py
#
#   CustomStreamWrapper.__anext__ — shows the two iteration paths:
#     Path A (is_async_iterable → __aiter__ present): `async for chunk in stream`
#       — truly non-blocking, zero thread-pool overhead per chunk.
#     Path B (sync iterator): `await asyncio.to_thread(next, stream)`
#       — one thread-pool slot per chunk; exhausts the default executor under
#       concurrent streaming load and hangs the proxy.
#   https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/streaming_handler.py
#
#   custom_chat_llm_router and how the proxy dispatches astreaming vs streaming
#   depending on the async_fn / stream flags:
#   https://github.com/BerriAI/litellm/blob/main/litellm/main.py
#
#   Anthropic provider — reference implementation of a well-behaved async
#   streaming provider: uses httpx.AsyncClient → aiter_lines() → ModelResponseIterator
#   (a native AsyncIterator), which CustomStreamWrapper iterates via Path A:
#   https://github.com/BerriAI/litellm/blob/main/litellm/llms/anthropic/chat/handler.py

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import litellm
from chutes_e2ee import ChutesE2EETransport
from litellm.llms.custom_llm import CustomLLM
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    EmbeddingResponse,
    GenericStreamingChunk,
    ImageResponse,
    ModelResponse,
)
from openai import OpenAI

CHUTES_API_BASE = "https://llm.chutes.ai"

# httpx read timeout for streaming.  Guards against the Chutes API stalling
# mid-stream: without this the background thread blocks forever.
_READ_TIMEOUT = 120.0

# LiteLLM injects these into optional_params but they are not OpenAI API params.
_LITELLM_INTERNAL_PARAMS = {
    "max_retries",
    "mock_response",
    "mock_timeout",
    "stream_timeout",
    "mcp",
}

# Cache OpenAI clients per API key to reuse E2EE transport state (nonce cache,
# model map cache) and avoid leaking metadata on every request.
_client_cache: dict[str, OpenAI] = {}
_client_lock = threading.Lock()


def _clean(params: dict, *, drop_stream: bool = False) -> dict:
    drop = _LITELLM_INTERNAL_PARAMS | ({"stream"} if drop_stream else set())
    return {k: v for k, v in params.items() if k not in drop}


def _client(api_key: str) -> OpenAI:
    with _client_lock:
        client = _client_cache.get(api_key)
        if client is None:
            client = OpenAI(
                api_key=api_key,
                base_url=f"{CHUTES_API_BASE}/v1",
                http_client=httpx.Client(
                    transport=ChutesE2EETransport(api_key=api_key),
                    timeout=httpx.Timeout(
                        connect=10.0, read=_READ_TIMEOUT, write=30.0, pool=10.0
                    ),
                ),
            )
            _client_cache[api_key] = client
        return client


def _to_chunk(chunk: Any) -> GenericStreamingChunk:
    choice = chunk.choices[0] if chunk.choices else None
    delta = choice.delta if choice else None

    tool_use: ChatCompletionToolCallChunk | None = None
    if delta and delta.tool_calls:
        tc = delta.tool_calls[0]
        # ChatCompletionToolCallFunctionChunk has total=False so all keys optional.
        func: ChatCompletionToolCallFunctionChunk = {}
        if tc.function:
            if tc.function.name is not None:
                func["name"] = tc.function.name
            if tc.function.arguments is not None:
                func["arguments"] = tc.function.arguments
        # ChatCompletionToolCallChunk has total=True; all four fields are required.
        # type is Literal["function"] — the only valid value for tool calls.
        tool_use = ChatCompletionToolCallChunk(
            id=tc.id,
            type="function",
            function=func,
            index=tc.index,
        )

    return GenericStreamingChunk(
        text=(delta.content or "") if delta else "",
        tool_use=tool_use,
        is_finished=bool(choice and choice.finish_reason),
        finish_reason=(choice.finish_reason or "") if choice else "stop",
        usage=chunk.usage.__dict__ if chunk.usage else None,
        index=0,
    )


class ChutesE2EEProvider(CustomLLM):
    # How LiteLLM proxy (async FastAPI) calls these methods:
    #
    #   acompletion(**kwargs)  — awaited directly; must be async def.
    #
    #   astreaming(**kwargs)   — called WITHOUT await to get the async generator
    #                            object, then iterated via `async for`.
    #                            Must be an async generator function (async def +
    #                            yield).  CustomStreamWrapper detects __aiter__ via
    #                            is_async_iterable() and uses the non-blocking
    #                            "Path A" — no per-chunk thread spawning.
    #
    #                            If it returned a sync Iterator instead, LiteLLM
    #                            would use "Path B": asyncio.to_thread(next, iter)
    #                            — one thread-pool slot per chunk.  Under concurrent
    #                            streaming load this exhausts the default executor
    #                            (~cpu_count+4 threads) and hangs the proxy.

    # ------------------------------------------------------------------ #
    #  Chat completions                                                    #
    # ------------------------------------------------------------------ #

    def completion(self, *_: Any, **kwargs: Any) -> ModelResponse:
        params = _clean(kwargs.get("optional_params", {}))
        client = _client(kwargs["api_key"])
        while True:
            try:
                response = client.chat.completions.create(
                    model=kwargs["model"],
                    messages=kwargs["messages"],
                    **params,
                )
                break
            except TypeError as e:
                msg = str(e)
                if "unexpected keyword argument" in msg:
                    bad = msg.split("'")[1]
                    params.pop(bad, None)
                else:
                    raise
        return litellm.ModelResponse(**response.model_dump())

    def streaming(self, *_: Any, **kwargs: Any) -> Iterator[GenericStreamingChunk]:
        params = _clean(kwargs.get("optional_params", {}), drop_stream=True)

        def _gen() -> Iterator[GenericStreamingChunk]:
            p = dict(params)
            client = _client(kwargs["api_key"])
            while True:
                try:
                    stream = client.chat.completions.create(
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        stream=True,
                        **p,
                    )
                    break
                except TypeError as e:
                    msg = str(e)
                    if "unexpected keyword argument" in msg:
                        bad = msg.split("'")[1]
                        p.pop(bad, None)
                    else:
                        raise
            for chunk in stream:
                yield _to_chunk(chunk)

        return _gen()

    async def acompletion(self, *_: Any, **kwargs: Any) -> ModelResponse:
        return await asyncio.to_thread(self.completion, **kwargs)

    async def astreaming(  # type: ignore[override]
        self, *_: Any, **kwargs: Any
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async generator: httpx I/O runs in one background thread; chunks are
        passed to the event loop via asyncio.Queue.  Each `async for` iteration
        is a non-blocking await — no thread-pool slot consumed per chunk."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=32)
        stop = threading.Event()
        _sentinel = object()

        def _fill() -> None:
            try:
                for chunk in self.streaming(**kwargs):
                    if stop.is_set():
                        return
                    # Schedule queue.put on the event loop; poll so that a
                    # cancelled generator (stop set) unblocks within ~0.5 s.
                    fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    while not stop.is_set():
                        try:
                            fut.result(timeout=0.5)
                            break
                        except TimeoutError:
                            continue
                        except Exception:
                            return
            except BaseException as exc:  # noqa: BLE001
                if not stop.is_set():
                    with contextlib.suppress(Exception):
                        asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result(
                            timeout=5
                        )
            finally:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(queue.put(_sentinel), loop).result(
                        timeout=5
                    )

        threading.Thread(target=_fill, daemon=True).start()

        try:
            while True:
                item = await queue.get()
                if item is _sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Signal fill thread to stop; it unblocks from fut.result(timeout=0.5)
            # within half a second.
            stop.set()

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embedding(self, *_: Any, **kwargs: Any) -> EmbeddingResponse:
        response = _client(kwargs["api_key"]).embeddings.create(
            model=kwargs["model"],
            input=kwargs["input"],
            **_clean(kwargs.get("optional_params", {})),
        )
        return litellm.EmbeddingResponse(**response.model_dump())

    async def aembedding(self, *_: Any, **kwargs: Any) -> EmbeddingResponse:
        return await asyncio.to_thread(self.embedding, **kwargs)

    # ------------------------------------------------------------------ #
    #  Image generation                                                    #
    # ------------------------------------------------------------------ #

    def image_generation(self, *_: Any, **kwargs: Any) -> ImageResponse:
        response = _client(kwargs["api_key"]).images.generate(
            model=kwargs["model"],
            prompt=kwargs["prompt"],
            **_clean(kwargs.get("optional_params", {})),
        )
        return litellm.ImageResponse(**response.model_dump())

    async def aimage_generation(self, *_: Any, **kwargs: Any) -> ImageResponse:
        return await asyncio.to_thread(self.image_generation, **kwargs)

    # ------------------------------------------------------------------ #
    #  Image editing                                                       #
    # ------------------------------------------------------------------ #

    def image_edit(self, *_: Any, **kwargs: Any) -> ImageResponse:
        response = _client(kwargs["api_key"]).images.edit(
            model=kwargs["model"],
            image=kwargs["image"],
            prompt=kwargs["prompt"],
            **_clean(kwargs.get("optional_params", {})),
        )
        return litellm.ImageResponse(**response.model_dump())

    async def aimage_edit(self, *_: Any, **kwargs: Any) -> ImageResponse:
        return await asyncio.to_thread(self.image_edit, **kwargs)


chutes_e2ee_provider = ChutesE2EEProvider()

litellm.custom_provider_map = [
    {"provider": "chutes-e2ee", "custom_handler": chutes_e2ee_provider}
]
