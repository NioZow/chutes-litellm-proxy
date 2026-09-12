# Testing

This document describes the test suite: what each layer proves, how to run it,
and how to add tests. Everything runs with `pytest`.

- **Offline suite** — fast, hermetic, no network or credentials. A
  protocol-faithful mock speaks the real Chutes E2EE wire protocol.
- **Live suite** — opt-in, hits the real Chutes API with a real key. Skipped
  unless `CHUTES_LIVE_TEE=1` and `CHUTES_API_KEY` are set.

## Quick reference

```sh
# Offline protocol / unit / integration tests (default)
uv run pytest tests/ -q

# Lint
uv run ruff check src tests scripts

# Live: attestation + real encrypted invokes + real HTTP proxy
CHUTES_LIVE_TEE=1 CHUTES_API_KEY=cpk_... \
  uv run pytest tests/test_live_attestation.py tests/test_live_proxy.py -v -s
```

## Layout

| Path | Kind | What it covers |
| ---- | ---- | -------------- |
| `tests/conftest.py` | fixture | Points the transport at the mock server; resets clients/caches between tests. Yields to real env when `CHUTES_LIVE_TEE=1`. |
| `tests/mock_chutes_server.py` | helper | Real HTTP server implementing the Chutes E2EE control plane (discovery, `/e2e/invoke`, encrypted SSE replies, evidence). Holds the instance private key. |
| `tests/test_custom_provider.py` | offline | The `chutes_e2ee` `CustomLLM` provider: encryption proof, wire model id, usage, tool calls, reasoning, error mapping. |
| `tests/test_attestation.py` | offline | Attestation evidence parsing/verification, tamper/fail-closed behaviour, strict all-instances gating, cache-mode/negative-cache/single-flight, transport wiring of `CHUTES_VERIFY_QUOTE`/`_GPU`, optional-hardware SDK gating, and the DCAP/NVIDIA verifiers with **fake SDKs** (evidence normalisation, TCB status, NRAS URL, fail-closed). |
| `tests/test_e2ee_litellm.py` | offline | The host-scoped transport swap (`e2ee_litellm.install()`) and the attested-instance selection filter (only verified instances survive; failures raise before any invoke). |
| `tests/test_live_attestation.py` | live | Real discovery + evidence binding/key-possession for a live `-TEE` chute; optional real Intel DCAP + NVIDIA NRAS hardware-root verification (skips when the SDKs are absent or PCCS/NRAS are unreachable). |
| `tests/test_live_proxy.py` | live | Real custom-provider invokes, thinking visibility, a real in-process proxy reached over HTTP, and a gated request whose invoked `X-Instance-Id` is asserted to be in the verified set. |

The fork adds hermetic hook tests at
`chutes-e2ee-transport/tests/test_discovery_filter.py` (run from that repo or
with `pytest chutes-e2ee-transport/tests/test_discovery_filter.py`).

## Offline suite

No API key needed. `tests/conftest.py` starts a local mock and points
`CHUTES_E2EE_API_BASE` / `CHUTES_E2EE_MODELS_BASE` at it. The mock performs
**real** ML-KEM-768 + HKDF + ChaCha20-Poly1305 crypto, so a passing test proves
the request was genuinely end-to-end encrypted (the mock can only read it
because it holds the instance private key).

```sh
uv run pytest tests/ -q                 # whole offline suite
uv run pytest tests/test_custom_provider.py -v
```

Optional: `CHUTES_E2EE_VERIFY_SSL=false` is forced for the mock (plain HTTP).
On NixOS, if importing `litellm` fails with a `libstdc++.so.6` error, run the
venv Python with the compiler's runtime on the path:

```sh
LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so)")" \
  .venv/bin/python -m pytest tests/ -q
```

## Live suite

### Requirements

- `CHUTES_API_KEY` — a real key.
- `CHUTES_LIVE_TEE=1` — unlocks the live tests.
- `CHUTES_LIVE_MODEL` — optional; the target model. Auto-selected from the live
  listing when unset (prefers a DeepSeek `-TEE` model, i.e. V4-Flash-0731).
- Outbound access to `api.chutes.ai` / `llm.chutes.ai`.

> Run the live modules on their **own**. The offline suite points the transport
> at a mock and would otherwise clobber your real credentials.

### What the live modules do

`test_live_attestation.py`

1. Resolves the model's chute, lists its E2EE instances, fetches fresh evidence
   for a random nonce, and verifies key-possession + key-binding.
2. `test_live_verify_chute_hardware_roots` — with `dcap-qvl` /
   `nv-attestation-sdk` installed (`uv sync --extra attestation`), runs the full
   Intel DCAP + NVIDIA NRAS verification.  It skips (not fails) when the SDKs
   are missing or the PCCS/NRAS endpoints are unreachable; a genuine
   attestation failure still fails the test.

`test_live_proxy.py`

1. `test_live_custom_provider_invoke` — one non-streaming encrypted completion.
2. `test_live_custom_provider_stream` — one streaming completion.
3. `test_live_thinking_visibility` — asserts that the model's thinking is
   readable as a separate `reasoning_content` field, both non-streaming and
   streaming, for Kimi K2.6, Gemma 4 and DeepSeek V4-Flash (models absent from
   the account's listing are skipped).
4. `test_live_proxy_server_reachable` — boots a real in-process LiteLLM proxy
   (`scripts/live_server.py`) and drives it over HTTP: health check, non-stream
   `/v1/chat/completions`, streamed completion, and a reasoning request that
   asserts `reasoning_content` survives the proxy.

> `-TEE` instances may cold-start; the live helpers retry transient failures
> with backoff.

### Thinking / reasoning parameters

Chutes exposes thinking differently per model family; the proxy normalises it:

| Family | How thinking is enabled | Notes |
| ------ | ----------------------- | ----- |
| DeepSeek (V3.x / V4-Flash) | `reasoning_effort: low\|medium\|high\|...` | `none` disables. |
| Gemma 4 | `reasoning_effort` **or** `chat_template_kwargs.enable_thinking` | A live `reasoning_effort` is translated to the template flag automatically. |
| Kimi K2.6 | reasons by default | `reasoning_content` present without params. |

These params are declared to LiteLLM so they are forwarded instead of being
dropped (`UnsupportedParamsError`). In the generated proxy config this is
`litellm_params.allowed_openai_params`; in direct SDK calls pass
`allowed_openai_params=[...]` or use `extra_body={...}`.

## Fixtures and isolation

- `mock_server` (function scope) starts a fresh mock and points the env at it.
- An autouse fixture clears cached E2EE transports and the attestation cache
  between tests so each test builds an env-faithful client.
- In live runs the autouse env-pointing fixture is disabled, so the real
  credentials and production bases are used.

## Adding a test

1. Offline: add to an existing `tests/test_*.py`; use the `mock_server` fixture
   and `mock_server.state.scenario` (`chat`, `tools`, `reasoning`, `rate_limit`)
   to shape the reply. The mock echoes what it decrypted at
   `mock_server.state.last_invoke()`, which is how encryption is asserted.
2. Live: add to `tests/test_live_proxy.py` and guard on the module-level
   `pytestmark` (already skips without `CHUTES_LIVE_TEE`/`CHUTES_API_KEY`).
3. Run `uv run ruff check src tests scripts` and the offline suite.

## Related tooling

| Script | Purpose |
| ------ | ------- |
| `scripts/proxy_test.py` | Ad-hoc smoke test against an already-running proxy. |
| `scripts/verify_attestation.py` | Standalone attestation CLI for one model/chute. |
| `scripts/live_server.py` | In-process proxy launcher used by the live server test (keeps registrations alive; the deployed CLI `exec`s `litellm`). |
