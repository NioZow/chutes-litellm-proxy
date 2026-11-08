"""Performance gates for the NEAR AI provider.

Two layers, both opt-in where they need the network:

* **Offline** (default suite) — stable regression gates for the warm
  ``verify_model`` path, the negative cache and prewarm concurrency.  These use
  the same fake SDK boundary as ``test_near_client.py`` and a fixed clock, so
  they never touch the network.
* **Live** (``NEARAI_LIVE=1``) — cold/warm attestation, receipt overhead,
  streaming TTFT and immediate response-byte release against the real API.

Live methodology, in short:

* *Cold attestation* is a fresh ``NearRuntime`` plus ``verify_model``.
* *Warm zero network* replaces the metadata client and both SDK ``verify``
  callables with objects that raise, then calls ``verify_model`` again.
* *Receipt overhead* is measured directly around ``verify_response`` after the
  model has already finished, so model latency is excluded by construction.
* *TTFT* proves the buffering guarantee: ``verified`` first-byte time equals
  its stream total (nothing is emitted before verification), while ``tail``
  reaches first byte earlier.  A full non-streaming call is measured as a
  sanity reference but is not asserted on, because model latency varies run to
  run.
* *Memory release* inspects the SDK's retained response record
  (``sdk._responses``, private but stable in 0.3.0rc1): after
  ``verify_response`` the request bytes are empty and the response future is
  ``None``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import pytest

# Reuse the offline fakes from the Phase 2 tests.  Pytest's default import mode
# puts ``tests/`` on ``sys.path`` (see ``tests/conftest.py``), so this is the
# same module the default suite collects.
from test_near_client import (
    Harness,
    _Clock,
    _tee_handler,
    attestation_result,
    model,
)

from chutes_litellm.near import client, endpoints, policy

# ---------------------------------------------------------------------------
# Offline gates
# ---------------------------------------------------------------------------

#: 2000 awaited coroutines is enough to see scheduler outliers without making
#: the offline suite slow.  The thresholds are deliberately generous: the warm
#: path is a dict lookup, so real overhead is microseconds.
_WARM_CALLS = 2000
_WARM_P50_MS = 1.0
_WARM_P95_MS = 5.0


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    h = Harness(monkeypatch)
    try:
        yield h
    finally:
        h.close()


@pytest.fixture(autouse=True)
def _clean_process_runtimes():
    client.reset_runtimes()
    yield
    client.reset_runtimes()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def test_warm_verify_model_makes_no_calls_and_stays_fast(harness: Harness) -> None:
    """After one verification, 2000 warm calls are memory lookups only."""
    fixed_now = 1_000.0

    def clock() -> float:
        return fixed_now

    async def scenario() -> None:
        runtime = harness.make_runtime(clock=clock)
        runtime.tee_client.handler = _tee_handler()

        assert await runtime.verify_model("m") == "tee"
        metadata_calls = list(harness.metadata_calls)
        verify_calls = list(runtime.tee_client.verify_calls)
        assert metadata_calls == ["m"]
        assert verify_calls == ["m"]

        samples: list[float] = []
        for _ in range(_WARM_CALLS):
            started = time.perf_counter_ns()
            assert await runtime.verify_model("m") == "tee"
            samples.append((time.perf_counter_ns() - started) / 1_000_000)

        # The warm path must not re-enter metadata or SDK verification.
        assert harness.metadata_calls == metadata_calls
        assert runtime.tee_client.verify_calls == verify_calls

        p50 = _percentile(samples, 0.50)
        p95 = _percentile(samples, 0.95)
        assert p50 < _WARM_P50_MS, f"warm p50 {p50:.3f} ms >= {_WARM_P50_MS} ms"
        assert p95 < _WARM_P95_MS, f"warm p95 {p95:.3f} ms >= {_WARM_P95_MS} ms"

    asyncio.run(scenario())


def test_negative_cache_lookup_is_constant_time_and_expires() -> None:
    """10k cached failures stay cheap to look up and disappear at the TTL."""
    clock = _Clock()
    cache = policy.FailureCache(policy.FAILURE_TTL_SECONDS, clock=clock)
    error = client.NearUnavailableError(client.CODE_UNAVAILABLE_TIMEOUT)

    for index in range(10_000):
        cache.put(f"m{index}", error)

    started = time.perf_counter_ns()
    for index in range(10_000):
        assert cache.get(f"m{index}") is error
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000

    # Dict lookups: this is ~1 ms locally; 200 ms leaves room for a slow runner.
    assert elapsed_ms < 200, f"10k negative-cache lookups took {elapsed_ms:.1f} ms"
    assert cache.get("absent") is None

    clock.advance(policy.FAILURE_TTL_SECONDS - 0.001)
    assert cache.get("m0") is error

    clock.advance(0.001)
    assert cache.get("m0") is None
    assert cache.get("m9999") is None


def test_prewarm_stays_within_the_default_concurrency(harness: Harness) -> None:
    """No more than PREWARM_CONCURRENCY verifications may be in flight."""
    state = {"active": 0, "max": 0}

    async def handler(model_id: str):
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        try:
            await asyncio.sleep(0.005)
            return attestation_result(model())
        finally:
            state["active"] -= 1

    async def scenario() -> None:
        runtime = harness.make_runtime()
        runtime.tee_client.handler = handler

        models = [f"m{index}" for index in range(12)]
        results = await runtime.prewarm(models)

        assert set(results) == set(models)
        assert all(error is None for error in results.values())
        assert policy.PREWARM_CONCURRENCY == 4
        assert state["max"] >= 2, "prewarm did not run concurrently"
        assert state["max"] <= policy.PREWARM_CONCURRENCY

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Live gates (opt-in)
# ---------------------------------------------------------------------------

_LIVE = os.environ.get("NEARAI_LIVE")
live = pytest.mark.skipif(
    not _LIVE,
    reason="set NEARAI_LIVE=1, NEARAI_API_KEY and NEARAI_TEE_MODEL to run live NEAR checks",
)


def _require_tee() -> tuple[str, str]:
    key = (os.environ.get("NEARAI_API_KEY") or "").strip()
    tee = (os.environ.get("NEARAI_TEE_MODEL") or "").strip()
    if not key:
        pytest.skip("NEARAI_API_KEY is not set")
    if not tee:
        pytest.skip("NEARAI_TEE_MODEL is not set")
    return key, tee


def _run(coro, timeout: float):
    async def wrapped():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(wrapped())


def _chat_request(model_id: str) -> httpx.Request:
    """Build a non-streaming chat request with the fixed per-request timeout."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
        "stream": False,
        "max_tokens": 16,
        "temperature": 0,
    }
    return httpx.Request(
        "POST",
        endpoints.chat_completions_url(),
        json=payload,
        headers={"content-type": "application/json"},
        extensions={
            "timeout": httpx.Timeout(
                policy.READ_TIMEOUT_SECONDS,
                connect=policy.CONNECT_TIMEOUT_SECONDS,
            ).as_dict()
        },
    )


async def _verify_receipt(sdk, completion_id: str):
    """Verify one completion, retrying only retryable ``ApiError``.

    The signature endpoint briefly 404s while a fresh completion is indexed;
    the SDK marks that status retryable and the provider retries it.  Live
    gates use the same behavior so a transient miss is not a product failure.
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


@live
def test_live_cold_attestation_under_ten_seconds() -> None:
    """A fresh runtime attests one model in under 10 s (prewarm removes this)."""
    key, tee = _require_tee()

    async def scenario():
        runtime = client.NearRuntime(api_key=key, environ={})
        try:
            started = time.perf_counter()
            mode = await runtime.verify_model(tee)
            return mode, time.perf_counter() - started
        finally:
            await runtime.aclose()

    mode, elapsed = _run(scenario(), timeout=120)
    assert mode == "tee"
    assert elapsed < 10.0, f"cold attestation took {elapsed:.2f}s"


@live
def test_live_warm_verification_makes_no_network_calls() -> None:
    """Once verified, ``verify_model`` must not touch metadata or the SDK."""
    key, tee = _require_tee()

    class _Poison:
        def __getattr__(self, name: str):
            raise AssertionError(f"warm path touched metadata client: {name}")

    def _poison_verify(*_args, **_kwargs):
        raise AssertionError("warm path called SDK verify()")

    async def scenario():
        runtime = client.NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(tee)
            runtime._metadata_client = _Poison()  # type: ignore[assignment]
            runtime.tee_client.verify = _poison_verify  # type: ignore[method-assign]
            runtime.routed_client.verify = _poison_verify  # type: ignore[method-assign]

            started = time.perf_counter()
            for _ in range(20):
                assert await runtime.verify_model(tee) == mode
            return (time.perf_counter() - started) / 20
        finally:
            await runtime.aclose()

    average = _run(scenario(), timeout=120)
    # A network round trip would be orders of magnitude slower; 50 ms is only a
    # sanity ceiling for extreme CI noise.
    assert average < 0.05, f"warm verification averaged {average * 1000:.1f} ms"


@live
def test_live_warm_receipt_overhead() -> None:
    """Receipt fetch + verify stays under the p50/p95 gates (model excluded)."""
    key, tee = _require_tee()

    async def one(sdk) -> float:
        response = await sdk.send(_chat_request(tee))
        try:
            data = json.loads(await response.aread())
        finally:
            await response.aclose()
        started = time.perf_counter()
        verified = await _verify_receipt(sdk, data["id"])
        elapsed = time.perf_counter() - started
        assert verified.signature_kind == "provider_tee"
        return elapsed

    async def scenario():
        runtime = client.NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(tee)
            sdk = runtime.client_for(mode)
            await one(sdk)  # warm the receipt path and TLS connection
            return [await one(sdk) for _ in range(5)]
        finally:
            await runtime.aclose()

    samples = _run(scenario(), timeout=900)
    p50 = _percentile(samples, 0.50)
    p95 = _percentile(samples, 0.95)
    assert p50 < 0.250, f"receipt p50 {p50 * 1000:.0f} ms >= 250 ms"
    assert p95 < 0.500, f"receipt p95 {p95 * 1000:.0f} ms >= 500 ms"


@live
def test_live_streaming_ttft_tradeoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """``verified`` waits for completion; ``tail`` reaches first byte first."""
    key, tee = _require_tee()

    from chutes_litellm.near.provider import register

    register()
    import litellm

    messages = [{"role": "user", "content": "Reply with exactly the word: pong"}]
    model_name = f"nearai_e2ee/{tee}"

    async def non_stream() -> float:
        started = time.perf_counter()
        await litellm.acompletion(
            model=model_name,
            messages=messages,
            api_key=key,
            timeout=600,
            max_tokens=16,
            temperature=0,
        )
        return time.perf_counter() - started

    async def stream(stream_mode: str) -> tuple[float, float]:
        monkeypatch.setenv("NEARAI_STREAM_MODE", stream_mode)
        started = time.perf_counter()
        wrapper = await litellm.acompletion(
            model=model_name,
            messages=messages,
            api_key=key,
            stream=True,
            timeout=600,
            max_tokens=16,
            temperature=0,
        )
        first_byte = None
        async for _chunk in wrapper:
            if first_byte is None:
                first_byte = time.perf_counter() - started
        total = time.perf_counter() - started
        assert first_byte is not None, "stream produced no chunks"
        return first_byte, total

    async def scenario():
        # Warm TLS and the model first: the very first request is cold and is
        # not comparable to the steady-state streaming measurements.
        await non_stream()
        full = await non_stream()
        tail_ttft, tail_total = await stream("tail")
        verified_ttft, verified_total = await stream("verified")
        return full, tail_ttft, tail_total, verified_ttft, verified_total

    full, tail_ttft, tail_total, verified_ttft, verified_total = _run(scenario(), timeout=900)

    # ``verified`` buffers the entire completion and verifies it before the
    # first chunk, so its TTFT equals its total time (within scheduler noise).
    # This deterministic invariant is stronger than comparing against a
    # separate non-streaming request, whose model latency varies run to run
    # (``full`` is kept as a sanity reference in the messages below).
    assert verified_ttft >= verified_total - 0.1, (
        f"verified TTFT {verified_ttft:.2f}s < stream total {verified_total:.2f}s; "
        "a verified stream must not emit before the completion verifies"
    )
    assert tail_ttft < verified_ttft, (
        f"tail TTFT {tail_ttft:.2f}s is not lower than verified {verified_ttft:.2f}s "
        f"(tail total {tail_total:.2f}s, non-streaming reference {full:.2f}s)"
    )
    assert tail_ttft <= tail_total


@live
def test_live_response_bytes_are_released_after_verification() -> None:
    """Captured request/response bytes are cleared as soon as verification ends."""
    key, tee = _require_tee()

    async def scenario():
        runtime = client.NearRuntime(api_key=key, environ={})
        try:
            mode = await runtime.verify_model(tee)
            sdk = runtime.client_for(mode)
            response = await sdk.send(_chat_request(tee))
            try:
                data = json.loads(await response.aread())
            finally:
                await response.aclose()

            verified = await _verify_receipt(sdk, data["id"])
            assert verified.signature_kind == "provider_tee"

            # The release callback runs on the event loop after the task
            # completes.
            for _ in range(3):
                await asyncio.sleep(0)

            record = sdk._responses.get(data["id"])  # type: ignore[attr-defined]
            assert record is not None, "response record was dropped before the TTL"
            assert record.request_body == b""
            assert record.response_body is None
        finally:
            await runtime.aclose()

    _run(scenario(), timeout=900)
