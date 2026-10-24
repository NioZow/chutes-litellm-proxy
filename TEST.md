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
| `tests/conftest.py` | fixture | Points the transport at the mock server; resets clients/caches between tests. Provides `TeeFakes`, which fakes only the three offline boundaries (DCAP, the measurement source and the NVIDIA SDK/NRAS call). Production calls all three for real; NRAS is faked at the SDK boundary, not bypassed. Yields to real env when `CHUTES_LIVE_TEE=1`. |
| `tests/mock_chutes_server.py` | helper | Real HTTP server implementing the Chutes E2EE control plane (discovery, `/e2e/invoke`, encrypted SSE replies, evidence). Holds the instance private key and builds production-shaped signed evidence with a valid TDX quote, fixed measurements and a bound certificate. |
| `tests/test_custom_provider.py` | offline | The `chutes_e2ee` `CustomLLM` provider: encryption proof, wire model id, usage, tool calls, reasoning, error mapping. |
| `tests/test_attestation.py` | offline | The always-on attestation pipeline: signed-body signature, exact `report_data[:32]`/`report_data[32:64]` bindings, certificate binding, mandatory DCAP, exact measurement matching, mandatory NRAS with GPU-count equality, unsigned/legacy rejection, tamper/fail-closed behaviour, strict all-instances gating, `chute_id` cache/negative-cache/single-flight, and the NVIDIA verifier with a fake SDK (evidence normalisation, GPU-count check, NRAS URL, fail-closed). Includes the regression test that the removed env toggles are inert. |
| `tests/test_tee_quote.py` | offline | TDX quote parser and DCAP wrapper: header/length validation, MRTD/RTMR/report_data extraction, parsed-vs-DCAP cross-check, TCB status/debug/advisory policy, and the TDX-module TCB fallback, with fake DCAP input. |
| `tests/test_tee_measurements.py` | offline | Measurement policy: public-endpoint and local-YAML parsing, TTL/last-known-good cache behaviour, exact MRTD + RTMR0-3 matching, ambiguity and unavailable-source fail-closed cases. |
| `tests/test_e2ee_litellm.py` | offline | The host-scoped transport swap (`e2ee_litellm.install()`), the always-installed attested-instance selection filter, and fixed host scoping (only the two fixed Chutes hosts, not overridable). Includes the TOCTOU regression: a same-id/different-`e2e_pubkey` substitution is refused before `/e2e/invoke`. |
| `tests/test_cli.py` | offline | The CLI entry point: argparse `--config`/`--host`/`--port`, CLI-over-environment precedence, missing-config and invalid-port exit codes, `_serve` argument forwarding with `--num_workers 1`, click error mapping, and the regression that `e2ee_litellm.install()` runs before in-process serving. No server, uvicorn or network is started. |
| `tests/test_live_attestation.py` | live | Real discovery + full evidence verification against a live `-TEE` chute: mandatory Intel DCAP, exact key/certificate bindings, a non-empty published measurement list, and mandatory NVIDIA NRAS GPU verification with GPU-count equality. Skips when the network or PCCS/NRAS is unreachable; there is no skip-by-disabling path. |
| `tests/test_live_proxy.py` | live | Real custom-provider invokes, thinking visibility, a real in-process proxy reached over HTTP, and a gated request whose invoked `X-Instance-Id` is asserted to be in the verified set. |

The fork adds hermetic hook tests at
`chutes-e2ee-transport/tests/test_discovery_filter.py` (run from that repo or
with `pytest chutes-e2ee-transport/tests/test_discovery_filter.py`).

## Offline suite

No API key needed. `tests/conftest.py` starts a local mock and monkeypatches the
`chutes_litellm.endpoints.API_BASE` / `MODELS_BASE` constants at it. The mock
performs **real** ML-KEM-768 + HKDF + ChaCha20-Poly1305 crypto, so a passing test
proves the request was genuinely end-to-end encrypted (the mock can only read it
because it holds the instance private key).

```sh
uv run pytest tests/ -q                 # whole offline suite
uv run pytest tests/test_custom_provider.py -v
```

The mock is plain HTTP on localhost. TLS certificate verification stays on for
every real HTTPS request and is not configurable, so the tests never disable it.
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

1. `test_live_measurements_endpoint_is_non_empty` — fetches the real published
   measurement list and asserts it is non-empty.
2. `test_live_verify_chute` — resolves the model's chute, lists its E2EE
   instances, fetches fresh evidence for a random nonce, and runs the full
   mandatory chain: signed-body signature, Intel DCAP, exact `report_data` key +
   certificate bindings, measurement matching, and NVIDIA NRAS GPU verification
   with GPU-count equality.
3. `test_live_verify_chute_hardware_roots` — re-runs the same live chute against
   the real DCAP verifier, the real published measurement list and real NRAS.
   The module skips (not fails) when the measurement endpoint, PCCS or NRAS is
   unreachable; a genuine attestation failure still fails the test. There is no
   disable switch and no GPU-only skip path: every live verification runs the
   whole chain.

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

- `mock_server` (function scope) starts a fresh mock and monkeypatches the
  endpoint constants at it.
- `tee_fakes` replaces only the three offline boundaries:
  `tee.quote._dcap_verify`, the measurement source, and the NVIDIA SDK's
  `Attestation` client. Production always calls real DCAP, the real measurement
  endpoint and real NRAS; the fake NRAS returns a controlled verdict and records
  the evidence passed to it.
- An autouse fixture clears cached E2EE transports and the attestation cache
  between tests so each test builds an env-faithful client. There are no
  verification toggle values to reset.
- `test_attestation.py::test_attestation_cannot_be_disabled_by_env` sets every
  removed toggle to a disabling value and asserts that bad evidence is still
  refused before `/e2e/invoke` — a regression guard against a backdoor.
- `test_e2ee_litellm.py::test_e2ee_hosts_are_derived_and_env_overrides_are_ignored`
  asserts that host scoping comes only from the two fixed Chutes endpoints; an
  injected endpoint/host variable is ignored.
- In live runs the autouse fixture is disabled, so the real credentials and the
  fixed production endpoints are used.

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
| `scripts/live_server.py` | In-process proxy launcher used by the live server test. The deployed entry point also serves in-process; this launcher remains for tests that serve a pre-expanded config and can pick a free port. |
