# NEAR AI Cloud

## Who this is for

Operators and integrators who serve NEAR AI Cloud models through this proxy and
need to know what is verified, which models are end-to-end encrypted, and what
the proxy deliberately does not protect.

## What you'll learn

- how a `nearai/<model>` request flows through LiteLLM and the
  `nearai_e2ee` provider to the upstream SDK,
- which cryptographic and hardware checks run, and in what order,
- which models must be TEE and which may be Gateway-routed,
- the fixed policy values and the environment variables that are safe to set,
- what the two streaming modes cost,
- the accepted limitations.

The Chutes deep dives ([attestation](attestation.md),
[E2EE](e2ee-encryption.md)) describe a different provider and do not apply to
NEAR AI.

## Architecture

```text
LiteLLM /v1
  └─ nearai/<id>                        user-facing model name
       └─ nearai_e2ee                  LiteLLM CustomLLM provider
            └─ NearRuntime             one per API key, process-wide
                 ├─ frozen mode map    TEE | routed
                 ├─ tee_client        InferenceClient(e2ee=True)
                 ├─ routed_client     InferenceClient(e2ee=False)
                 ├─ pinned metadata   AttestationClient over a pinned TLS (SPKI) client
                 ├─ negative cache    30 s
                 └─ prewarm           bounded concurrency (4), 60 s total budget
```

The provider is a protocol adapter, not a reimplementation:

- `src/chutes_litellm/near/provider.py` — LiteLLM `CustomLLM`, streaming,
  receipt verification, error mapping.
- `src/chutes_litellm/near/client.py` — the process-wide runtime, pinned
  metadata client, caches, timeouts and prewarming.
- `src/chutes_litellm/near/modes.py` — TEE/routed classification and the frozen
  map.
- `src/chutes_litellm/near/policy.py` — the fixed policy, post-verification
  assertions and the negative cache.
- `src/chutes_litellm/near/models.py` — `/v1/models` discovery for config
  generation.
- `src/chutes_litellm/near/endpoints.py` — the fixed HTTPS base URL.

The SDK is `nearai-inference-sdk==0.3.0rc1` and is used unmodified. No code in
this repository overrides an endpoint, disables a check, or patches the SDK.

### Model naming

| Layer | Name |
| ----- | ---- |
| Client → proxy | `nearai/<upstream id>` |
| Proxy → LiteLLM | `nearai_e2ee/<upstream id>` |
| Wire body | `<upstream id>` |

`generate_config.py` expands the `nearai/*` template entry into one
`model_name` per chat model, each pointing at `nearai_e2ee/<id>`. A request
using `nearai_e2ee/<id>` directly also works. There is no alias that maps a NEAR
model to another provider.

Bare upstream ids are canonical in configuration and environment lists:
`NEARAI_PREWARM_MODELS`, `NEARAI_TEE_MODELS`, and measurement pin model keys
accept a leading `nearai/` or `nearai_e2ee/` and strip it, so a user-facing name
can be pasted directly.

## Verification chain

All of the following run before content is returned. The SDK performs the
cryptography; this repository applies stricter defaults and re-asserts the
results afterwards.

1. **Gateway attestation.**
   `AttestationClient.fetch_gateway_attestation()` fetches a fresh challenge
   and evidence from `https://cloud-api.near.ai`, then
   `verify_gateway_attestation()` verifies the Intel DCAP TDX quote with
   `dcap-qvl` (including TCB status), binds the nonce and `report_data`, checks
   the signer, `mr_config_id` and `app_compose`, and replays dstack RTMR3. Only
   `UpToDate` is accepted and any advisory id is rejected.
2. **TLS pinning.** The Gateway TLS peer's SPKI is captured from the verified
   attestation and a pinned `httpx` client is created for metadata calls. An
   empty fingerprint is refused. For inference and mode-detection metadata, the
   API key is never sent to a Gateway whose attestation has not been verified.
   Catalog discovery (`/v1/models`) is different: it is TLS-only and sends the
   key before attestation (see [Model discovery](#model-discovery)).
3. **Model mode.** `AttestationClient.fetch_model_metadata()` classifies the
   model: TEE when `provider_type == "vllm"` **and** `attestation_supported`,
   otherwise routed. The result is frozen for the lifetime of the process. If
   the pinned metadata client fails with a `VerificationError` (for example
   after a Gateway SPKI/TLS rotation), the runtime closes it, repeats the full
   strict Gateway bootstrap once, and retries metadata once. A second failure
   maps to `503`; a measurement-pin mismatch is a `502` policy failure and is
   not retried.
4. **TEE model attestation.** For TEE models, `InferenceClient.verify()` checks
   the model's own TDX evidence with the same `UpToDate` and advisory rules,
   requires NVIDIA NRAS GPU evidence, and requires a model signing key. The
   post-check asserts every model report has `gpu_evidence == "verified"` and a
   non-empty `signing_public_key`.
5. **End-to-end encryption (TEE models only).** E2EE v3 derives a fresh
   Ed25519→X25519 shared secret per request, runs HKDF-SHA256, and encrypts
   request/response fields with XChaCha20-Poly1305 using AAD-bound fields. Fresh
   keys per request are required; they are not cached.
6. **Response signature.** After the completion, the proxy calls
   `verify_response(id)` against the same evidence session that was used to
   send. TEE models must return `provider_tee`; routed models must return
   `gateway`. A receipt of the wrong kind is a policy failure, never a fallback.
7. **Release.** Only after the receipt verifies does the provider translate the
   body for LiteLLM. In `verified` streaming this means the whole stream is
   buffered first; in `tail` the terminal chunk is withheld.

The `nearai-inference-sdk` release candidate is pinned and audited; re-audit
before moving off `0.3.0rc1`.

## Fixed policy

These values are constants in `src/chutes_litellm/near/policy.py`. They are not
environment variables, CLI flags or Nix options, because every one of them is
security-critical.

| Setting | Value |
| ------- | ----- |
| Base URL | `https://cloud-api.near.ai/v1` (fixed) |
| Gateway TCB | `UpToDate` only; `OutOfDate`/`TDRelaunchAdvised` rejected |
| Gateway advisories | any non-empty list rejected |
| Gateway TLS | `attested` required, SPKI pinned |
| Gateway signer | `ed25519` only |
| Model TCB | `UpToDate` only (TEE models) |
| Model advisories | any non-empty list rejected (TEE models) |
| GPU evidence | `required` (TEE models) |
| Model signing key | required (TEE models) |
| TEE receipt | `provider_tee`, no fallback |
| Routed receipt | `gateway` |
| Attestation TTL | 900 s (positive cache) |
| Failure TTL | 30 s (negative cache) |
| Response TTL | 600 s; captured bytes are cleared immediately after verification |
| Hard request timeout | 900 s, wrapping the whole send-plus-verify operation |
| Connect / read timeout | 10 s / 600 s per request (inference and pinned metadata) |
| Prewarm concurrency | 4 |
| Prewarm total budget | 60 s per startup batch (per-model hard timeout remains 900 s) |
| Streaming default | `verified` |
| Message logging | off (`litellm_settings.turn_off_message_logging: true`) |

There is no setting that disables attestation, TLS pinning, GPU evidence,
receipt verification, or E2EE for a TEE model.

## Mode policy

| Observed metadata | Mode | What is required |
| ----------------- | ---- | ---------------- |
| `provider_type == "vllm"` and `attestation_supported` | `tee` | E2EE, strict model attestation, `provider_tee` receipt |
| anything else | `routed` | Gateway attestation + TLS pinning, `gateway` receipt; no E2EE |

Rules:

- The first observed mode is frozen. A later metadata change in either
  direction fails closed; this includes TEE → routed (a downgrade attempt) and
  routed → TEE (an untrusted change).
- `NEARAI_TEE_MODELS` accepts exact ids or `fnmatch` globs. A match forces the
  model to verify as TEE: if its metadata reports routed, the request fails
  closed. It can only increase strictness.
- There is deliberately no "force routed" option and no automatic downgrade.
- Routed models cannot use NEAR E2EE. They rely on TLS, Gateway attestation and
  the Gateway signature.
- The model mode cannot be determined, or a Gateway/metadata bootstrap
  transport step fails: the request fails closed with `503`. Gateway policy or
  cryptographic failures (TLS binding, SPKI, TCB, advisories, signer algorithm,
  measurement pins) are `502`.

## Configuration and environment

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `NEARAI_API_KEY` | unset | Canonical API key. Required for `nearai/` models and preferred over a request-level key. |
| `NEARAI_API_KEY_PATH` | unset | File containing the key; read at startup. An explicit `NEARAI_API_KEY` wins. |
| `NEARAI_STREAM_MODE` | `verified` | `verified` or `tail`; any other value fails closed. |
| `NEARAI_PREWARM_MODELS` | unset | Comma-separated bare model ids to verify at startup. |
| `NEARAI_TEE_MODELS` | unset | Comma-separated ids/globs that must verify as TEE. |
| `NEARAI_MEASUREMENTS_FILE` | unset | Optional YAML/JSON deployment measurement pin file. |

TTLs and timeouts are intentionally absent from this table and have no
environment variables: 900 s attestation, 30 s failure and 600 s response
lifetimes, the 900 s hard request timeout, and the 60 s total prewarm budget are
fixed in `policy.py` and are not exposed as options.

The API key comes from `NEARAI_API_KEY` when it is set and non-empty; this is
canonical and wins over a request-level `api_key` (including one supplied in the
request body), so callers cannot create additional process-wide runtimes. When
`NEARAI_API_KEY` is unset or empty, the `api_key` LiteLLM resolves (including
`os.environ/NEARAI_API_KEY`) is used. A caller-supplied key is never logged,
never included in an error message, and is only sent after Gateway attestation
succeeds for inference and mode-detection metadata; discovery is TLS-only and
described below. `NEARAI_TEE_MODELS`, `NEARAI_PREWARM_MODELS`, and measurement
pin model keys use bare upstream ids, though a leading `nearai/` or
`nearai_e2ee/` is accepted and stripped.

### Model discovery

`fetch_models()` in `src/chutes_litellm/near/models.py` calls `GET /v1/models`
with the API key over ordinary TLS. It does **not** perform Gateway attestation
or TLS pinning, so the key is sent before attestation on this path. Discovery
output is used only to generate LiteLLM config entries; it is not a trust input
for inference. The runtime re-verifies Gateway attestation, TLS pinning, and the
model mode before any completion, so a compromised discovery response can only
mislabel entries in the generated config -- it cannot bypass the per-request
mode/receipt policy.

### Measurement pinning

By default the SDK verifies *what* a deployment proves but not *which build* it
runs: `deployment_provenance` is `not_checked`, so a valid but undesired image
could pass. `NEARAI_MEASUREMENTS_FILE` optionally pins the measured deployment
of the Gateway and of TEE model hosts.

The file is YAML or JSON parsed with `yaml.safe_load`. The schema is:

```yaml
gateway:
  os_image_hash: ["<hex>"]        # a string or a list is accepted
  compose_hash: ["<hex>"]
  app_compose_sha256: ["<hex>"]
models:
  "z-ai/glm-5.3-flash":
    os_image_hash: ["<hex>"]
  "*":
    compose_hash: ["<hex>"]
```

Rules:

- `gateway` pins apply to the Gateway deployment, including the metadata
  bootstrap. `models` pins apply to TEE model deployments.
- An exact model id always wins over `"*"`. A model with neither an exact entry
  nor `"*"` is unconstrained. Routed models have no model evidence, so only the
  Gateway pins apply to them.
- Values are normalized to lowercase hex with an optional `0x` prefix removed.
  Each value must be a non-empty, even-length hex string; an empty pin list or
  an omitted key means "no constraint" for that field.
- Three fields are checked independently: `runtime_measurements.os_image_hash`,
  `runtime_measurements.compose_hash`, and the SHA-256 of the exact
  `app_compose` string. A pinned field whose deployment value is missing is a
  mismatch.
- A mismatch fails the request with `near.measurements.gateway_mismatch` or
  `near.measurements.model_mismatch` (HTTP `502`). Error details contain only
  the field name and scope, never the expected or observed hashes.
- An empty mapping (`{}`) is valid but constrains nothing; the runtime logs one
  warning at construction. An unset variable also logs one warning and leaves
  provenance `not_checked`.
- A missing/unreadable file, unknown key, wrong type, or invalid hex fails
  closed with `near.measurements.invalid`. The error never includes file
  contents or pin values. There is no toggle that ignores a present file. The
  failure surfaces when the runtime is constructed -- on the first NEAR request,
  or during startup prewarm, where the hook logs it and lets the proxy continue.
  In both cases NEAR requests fail closed rather than running unpinned.

How to obtain the values with the pinned SDK:

```sh
uv run python - <<'PY'
import hashlib
import os

from chutes_litellm.near.client import NearRuntime

async def main() -> None:
    runtime = NearRuntime(api_key=os.environ["NEARAI_API_KEY"])
    model = "Qwen/Qwen3.8-27B"
    mode = await runtime.verify_model(model)
    result = await runtime.client_for(mode).verify(model)
    for label, deployment in [
        ("gateway", result.gateway.deployment),
        *(("model", report.deployment) for report in result.models),
    ]:
        measurements = deployment.runtime_measurements
        print(label, "os_image_hash:", measurements.os_image_hash)
        print(label, "compose_hash:", measurements.compose_hash)
        print(label, "app_compose_sha256:", hashlib.sha256(deployment.app_compose.encode()).hexdigest())
    await runtime.aclose()

import asyncio
asyncio.run(main())
PY
```

For a live inspection that also prints TCB status, advisories, TLS binding, GPU
evidence, provenance, and the chosen mode, use the "Verifying by hand" snippet
below. No separate `verify_cli` console script is installed in this repository
(the SDK is not forked or wrapped by a second CLI); the snippets call the same
`NearRuntime` the provider uses.

Pins are a deployment allowlist, not a trust-on-first-use cache: update them
deliberately after an expected image or compose change, review the diff, and
redeploy. Do not let a running process rewrite or reload the file; it is read
once per runtime construction. In a container, mount the file and set
`NEARAI_MEASUREMENTS_FILE` to the path *inside* the container; a path that is
not mounted is treated as missing and fails closed.

### Startup prewarm

When `NEARAI_API_KEY` and at least one `nearai/` model are configured, the CLI
sets `LITELLM_WORKER_STARTUP_HOOKS=chutes_litellm.near.startup:prewarm`. The
hook runs inside the serving event loop, so the process-wide runtime is bound to
the correct loop. It verifies each model with the same policy as a request,
bounded to four concurrent verifications and a **60 s total prewarm budget**.
Any model still unverified when the budget expires is cancelled and reported as
`near.unavailable.timeout`; each individual verification still has the 900 s
hard timeout. A failure is logged with the model id and a stable code; the hook
never blocks startup or other providers.

### Message logging

The bundled template sets:

```yaml
litellm_settings:
  turn_off_message_logging: true
```

Keep this on. The proxy itself never logs prompts, completions, evidence, or
keys, but LiteLLM can otherwise retain request and response bodies in its logs.
If you supply a custom template, copy this block.

## Streaming modes

| Mode | First byte | Authentication | Use when |
| ---- | ---------- | -------------- | -------- |
| `verified` (default) | after the full completion **and** receipt verification | every byte is authentic before it is released | correctness and auditability matter more than latency |
| `tail` | at model TTFT; content streams live | terminal chunk (`finish_reason`/`[DONE]`) is withheld until the receipt verifies | latency matters and a terminal failure can be tolerated |

`verified` therefore pays the full completion time plus one receipt round trip
before its first chunk. That is the security cost of never emitting
unauthenticated content. `tail` preserves model TTFT but only provides terminal
authentication: bytes already delivered before a verification failure are not
retroactively authenticated, and the stream ends without a terminal chunk on
failure.

`NEARAI_STREAM_MODE` is read at request time. Unknown values raise a policy
error (HTTP 502) rather than silently falling back to a weaker mode.

## Error mapping

| Failure | HTTP | Response `message` |
| ------- | ---- | ------------------ |
| Gateway/model policy violation, receipt kind mismatch, signature failure, measurement mismatch | `502` | stable `near.*` code (and the SDK failure code for signatures) |
| Timeout, undeterminable mode, Gateway/metadata bootstrap transport failure, upstream transport failure | `503` | stable `near.unavailable.*` code |

Upstream HTTP failures use a stable message, `NEAR AI upstream returned HTTP
<status>`, optionally followed by a short allowlisted `error.code` or
`error.type` (for example `NEAR AI upstream returned HTTP 429
(rate_limit_exceeded)`). Upstream free text, bodies, prompts, completions,
evidence, and keys are never included. Authentication failures from the
upstream API are mapped to LiteLLM's usual `401`/`403`.

## Accepted limitations

These are properties of the design, documented rather than treated as bugs:

- **E2EE is field-level, not whole-request.** Message content and selected
  response fields are encrypted; the model id, message roles, sampling
  parameters, `response_format`, `metadata`, `user`, and some request/response
  ids are visible to the Gateway.
- **Model weights and identity are not attested.** A verified TEE signer proves
  the deployment and signing key, not the exact model revision or weight digest.
- **GPU evidence is assembled by the Gateway.** NRAS verifies the GPU evidence
  and this proxy requires a `verified` verdict, but the SDK’s TDX/GPU binding
  relies on the Gateway’s assembly of the report.
- **Routed models are not E2EE.** They are protected by TLS, Gateway
  attestation, and the Gateway response signature only.
- **Discovery is TLS-only.** `/v1/models` sends the API key before Gateway
  attestation and is not a trust input for inference; the mode is re-verified at
  runtime (see [Model discovery](#model-discovery)).
- **SDK bootstrap does not honor HTTP proxies.** The SDK's
  `fetch_gateway_attestation` bootstrap uses an `aiohttp` session that ignores
  `HTTP_PROXY`/`HTTPS_PROXY`. Metadata and inference use `httpx` with
  `trust_env=True` and do honor them. In a proxy-only deployment the Gateway
  bootstrap can therefore fail with `503`; this repository does not reimplement
  the SDK bootstrap.
- **Deployment provenance is only as strong as the optional pins.** With no
  `NEARAI_MEASUREMENTS_FILE`, the SDK reports `not_checked`; a valid but
  undesired build could pass. Pin the Gateway and model measurements when you
  know the expected builds.

### Not implemented in this repository

The following Phase 6 items are deliberately out of scope and are not exposed
as options:

- **Image provenance (Sigstore).** The SDK ships helpers, but this provider has
  no known trusted NEAR repositories or build workflows to bind against.
  Accepting an arbitrary Sigstore identity would be weaker than the fingerprint
  pins above, so provenance is left to an explicit `NEARAI_MEASUREMENTS_FILE`.
- **OHTTP.** It is optional and adds latency; the default Gateway + TLS + E2EE
  path does not require it. There is no toggle.
- **Upstream hardening PRs** for lazy imports, `repr=False` on key material, and
  failure caching. They are upstream improvements, not requirements for this
  wrapper, and this repository does not fork or patch the SDK.

## Operational notes

- **Hung requests.** The SDK constructs its own clients with `timeout=None`, so
  every runtime operation is wrapped in `asyncio.timeout(900)`. Every inference
  request carries `httpx.Timeout(read=600, connect=10)`, and the pinned
  metadata client gets the same timeout after creation.
- **Negative cache.** A failed attestation is cached for 30 s, so an outage does
  not turn into a retry storm. The positive cache lives for 900 s, matching the
  SDK's attestation cache.
- **Response bytes.** The SDK retains captured request/response bytes for 600 s
  to allow receipt retries. The proxy verifies immediately after every
  completion; the SDK then clears the captured bytes and frames. Never defer
  verification: an unverified response should not be used at all.
- **No per-request clients.** One `NearRuntime` exists per API key, shared by
  every request, so TLS connections and attestation state are reused.
- **Receipt retries are bounded.** Only retryable `ApiError`s are retried, at
  most three attempts with a small backoff. A signature failure never resends
  inference.
- **Shutdown helper.** `close_runtimes()` closes the async runtimes and
  `close_sync_runtimes()` closes the private-loop sync fallback runtimes; both
  are idempotent. Nothing calls them automatically on process exit. They exist
  for embedders and tests that own the event loop; there is no `atexit` magic
  because closing loop-bound `httpx` clients from an arbitrary exit path is
  unreliable.

## Verifying by hand

There is no separate NEAR AI attestation CLI in this phase. To inspect a live
model, call the runtime directly from a checkout:

```sh
uv run python - <<'PY'
import asyncio
import os

from chutes_litellm.near.client import NearRuntime

async def main() -> None:
    runtime = NearRuntime(api_key=os.environ["NEARAI_API_KEY"])
    mode = await runtime.verify_model("Qwen/Qwen3.8-27B")
    print("mode:", mode)
    result = await runtime.client_for(mode).verify("Qwen/Qwen3.8-27B")
    gateway = result.gateway
    print(
        "gateway:",
        gateway.tcb_status,
        gateway.tls_binding.kind,
        gateway.deployment_provenance,
        gateway.advisory_ids,
    )
    for report in result.models:
        print(
            "model:",
            report.tcb_status,
            report.gpu_evidence,
            bool(report.signing_public_key),
            report.deployment_provenance,
            report.advisory_ids,
        )
        print("  os_image_hash:", report.deployment.runtime_measurements.os_image_hash)
        print("  compose_hash:", report.deployment.runtime_measurements.compose_hash)
    await runtime.aclose()

asyncio.run(main())
PY
```

Live integration tests are opt-in; see [TEST.md](../TEST.md).
