# NEAR AI provider — audit and integration plan

Status: proposed  
Target: `nearai/` provider for this LiteLLM proxy  
SDK: `nearai-inference-sdk` @ `bbca9aefc237b52850318c263679fc3100c787c9` (`0.3.0rc1`)  
Decision: **use the upstream SDK, do not fork it.** All required behaviour can be composed from its public API. The problems are defaults, so this provider sets strict defaults in its wrapper.

---

## 1. Decision summary

The SDK’s cryptographic verification is real:

- Intel DCAP TDX quote verification with TCB policy.
- Nonce, `report_data`, signer, and `mr_config_id`/`app_compose` bindings.
- dstack RTMR3 replay.
- NVIDIA NRAS GPU evidence with JWT/JWKS/nonce verification.
- TLS peer SPKI capture and pinning.
- Ed25519/ECDSA response signatures over exact request and response bytes.
- E2EE v3: Ed25519→X25519 ECDH, HKDF-SHA256, XChaCha20-Poly1305, fresh keys, AAD-bound fields.

The SDK is **not safe by default** for this repository’s fail-closed standard because its defaults are permissive and response verification is explicit. We fix that in the wrapper, not by forking.

### Receipt policy (important)

NEAR AI serves two kinds of models:

1. **True TEE models** — `providerType == "vllm"` and `attestationSupported == true`. These must use E2EE and must return a `provider_tee` receipt.
2. **Routed models** — the Gateway routes to an upstream provider. These cannot use NEAR’s model E2EE and normally return a `gateway` receipt. They are still protected by Gateway attestation, TLS pinning, and the Gateway signature.

The provider must **not** reject everything that is gateway-signed. It must:

- Require `provider_tee` for true TEE models.
- Accept `gateway` for routed models.
- Verify the Gateway attestation and TLS binding in both cases.
- Never silently downgrade a true TEE model to routed mode.

---

## 2. Audit result in one page

### Passed

- `790 passed, 3 subtests passed` in the SDK test suite.
- `ruff` clean.
- `bandit`: no high-severity findings; eight medium `B113` findings, all `httpx timeout=None`.
- No signature-bypass, key-exfiltration, or attestation-bypass defect found in the audited code.

### Remaining issues that matter to this provider

| Issue | Impact | Resolution |
|---|---|---|
| Default GPU evidence is `if-present` | A TEE model can pass without GPU evidence | Set `gpu_evidence="required"` |
| Default TCB accepts `OutOfDate`/`TDRelaunchAdvised` | A stale TCB can pass | Accept only `UpToDate` |
| Advisories not rejected by default | A deployment with known advisories can pass | Reject non-empty `advisory_ids` |
| Response verification is explicit | Content is unauthenticated if the caller forgets | Always call `verify_response()` |
| Gateway receipts accepted by default | Not proof of TEE execution | Require `provider_tee` for TEE models; accept `gateway` only for routed models |
| Internal clients use `timeout=None` | Hung requests can occupy tasks indefinitely | Wrap every operation in `asyncio.timeout` and set per-request `httpx.Timeout` |
| Failed attestations are not cached | Outage retry storms | Add a short negative cache |
| Response bytes retained 60 minutes | Memory growth if verification is skipped | Verify immediately; use a 10-minute response TTL |
| Deployment provenance is `not_checked` | A valid but undesired build can pass | Optional measurement pins (recommended) |
| Model weights/model ID not attested | Cannot claim exact model identity | Accepted limitation; document |
| E2EE is field-level, not whole-request | Some fields leak to the Gateway | Accepted limitation; document |

### Supply chain

- Pin `nearai-inference-sdk==0.3.0rc1`; re-audit on the final release.
- Bump `dcap-qvl` to `>=0.7.0` (SDK requirement).
- Upgrade `PyJWT` to `2.15.1`.
- Add `dnspython>=2.9.0` (CVE-2026-114605; currently pulled through Pydantic’s email extra).
- The SDK eagerly imports `sigstore`, `pyhpke`, `eth-account`, and `anthropic`; expect ~1 s import time and a larger Nix/Docker dependency set.

### Not a fork

No SDK change is required for the receipt policy. The wrapper can:

- Create two `InferenceClient` instances per API key: one with `e2ee=True` (TEE models), one with `e2ee=False` (routed models).
- Detect the model mode from `AttestationClient.fetch_model_metadata()`.
- Check the returned receipt kind itself.
- Enforce post-verification assertions on the SDK’s verified objects.

If upstream later changes defaults, our assertions still hold.

---

## 3. Good defaults

These are the defaults the provider will use. Security-critical values are not user-toggleable.

| Setting | Default | Notes |
|---|---|---|
| Endpoint | `https://cloud-api.near.ai/v1` | Fixed; no env/CLI override |
| `signing_algo` | `ed25519` | XChaCha20-Poly1305; required for OHTTP |
| E2EE | on for TEE models | Not available for routed models |
| OHTTP | off | Optional; adds latency |
| Gateway TCB | `UpToDate` only | Reject `OutOfDate`/`TDRelaunchAdvised` |
| Gateway advisories | reject any | Stricter than SDK default |
| Gateway TLS | require `attested` | SPKI captured and pinned |
| Model TCB | `UpToDate` only | TEE models only |
| Model advisories | reject any | TEE models only |
| GPU evidence | `required` | TEE models only |
| Model key | required | TEE models only |
| TEE receipt | `provider_tee` | TEE models only; no fallback |
| Routed receipt | `gateway` | Routed models only |
| Attestation TTL | 900 s | Positive cache |
| Failure TTL | 30 s | Negative cache |
| Response TTL | 600 s | Bytes cleared immediately after verification |
| Hard request timeout | 900 s | Wraps the whole operation |
| Connect timeout | 10 s | Per-request `httpx.Timeout` |
| Read timeout | 600 s | Per-request `httpx.Timeout` |
| Prewarm | on | Verify configured models at startup |
| Prewarm concurrency | 4 | Bounded |
| Streaming mode | `verified` | Buffer, verify, then replay |
| Forced-TEE allowlist | empty | `NEARAI_TEE_MODELS`; can only increase strictness |
| Measurement pins | unset | Optional; recommended for production |
| Message logging | off | `litellm.turn_off_message_logging=true` |

`NEARAI_TEE_MODELS` is an allowlist of model IDs or globs. If a model matches, it must verify as TEE; if metadata says routed, the request fails closed. There is deliberately no “force routed” option, because that would allow a downgrade.

---

## 4. Model-mode policy

### Mode detection

At startup, after Gateway verification and TLS pinning:

```python
metadata = await attestation_client.fetch_model_metadata(model)
is_tee = metadata.provider_type == "vllm" and metadata.attestation_supported
```

Store the mode in a process-local map. Once a model is observed as TEE, it cannot become routed for the lifetime of the process. If metadata later reports a different mode, fail closed and log a downgrade attempt. `NEARAI_TEE_MODELS` can force TEE, never the reverse.

### True TEE models

```python
client = InferenceClient(
    api_key,
    base_url=NEAR_BASE_URL,
    e2ee=True,
    signing_algo="ed25519",
    attestation_cache_time_to_live_ms=900_000,
    response_cache_time_to_live_ms=600_000,
    gateway_verification=GatewayVerificationOptions(
        include_spki_fingerprint=True,
        policy=AttestationPolicy(accepted_tcb_statuses=("UpToDate",)),
    ),
    model_verification=ModelVerificationOptions(
        policy=ModelAttestationPolicy(
            accepted_tcb_statuses=("UpToDate",),
            gpu_evidence="required",
        ),
    ),
)
```

Post-verification assertions:

- `result.gateway.tls_binding.kind == "attested"`
- `result.gateway.tcb_status == "UpToDate"`
- `not result.gateway.advisory_ids`
- `result.models` is non-empty
- every model report:
  - `tcb_status == "UpToDate"`
  - `not advisory_ids`
  - `gpu_evidence == "verified"`
  - `signing_public_key is not None`

Response receipt:

- `verify_response(id)` must return `signature_kind == "provider_tee"`.
- A `gateway` receipt for a model classified as TEE is rejected with `502`.

### Routed models

```python
client = InferenceClient(
    api_key,
    base_url=NEAR_BASE_URL,
    e2ee=False,
    signing_algo="ed25519",
    attestation_cache_time_to_live_ms=900_000,
    response_cache_time_to_live_ms=600_000,
    gateway_verification=GatewayVerificationOptions(
        include_spki_fingerprint=True,
        policy=AttestationPolicy(accepted_tcb_statuses=("UpToDate",)),
    ),
)
```

- Gateway attestation and TLS pinning are still mandatory.
- Model attestation is not expected; `result.models` is empty for routed models.
- Response receipt must be `gateway`; the SDK already rejects `provider_tee` when no model attestation was selected.
- No E2EE: the SDK does not support it for routed models. Requests are protected by TLS + Gateway attestation + Gateway signature.

### Request flow

```text
model -> mode map
  ├─ tee    -> tee_client.verify(model)
  │              send(E2EE)
  │              verify_response -> require provider_tee
  └─ routed -> routed_client.verify(model)
                 send(no E2EE)
                 verify_response -> require gateway
```

### Unsupported cases

- Model mode cannot be determined: fail closed with `503`.
- TEE model with no model evidence: fail closed.
- TEE model with a `gateway` receipt: fail closed.
- Routed model with a `provider_tee` receipt: fail closed (the SDK cannot verify it without a selected model attestation).
- Metadata changes from TEE to routed at runtime: fail closed.
- Responses/Messages endpoints: not supported by this provider; E2EE is not available there.

---

## 5. Architecture

```text
LiteLLM /v1
  └─ nearai/<model>                       (user-facing model name)
       └─ nearai_e2ee                     (LiteLLM CustomLLM provider)
            └─ NearRuntime                (process-wide, per API key)
                 ├─ mode map              (TEE | routed, frozen per process)
                 ├─ tee_client            (InferenceClient, e2ee=True)
                 ├─ routed_client         (InferenceClient, e2ee=False)
                 ├─ pinned attestation client (metadata + mode detection)
                 ├─ negative cache
                 └─ prewarm + hard timeouts
```

Both clients verify the Gateway independently and pin their own verified TLS peer. The metadata client is used only for mode detection and is created from a verified Gateway TLS binding.

---

## 6. Files

```text
src/chutes_litellm/near/
  __init__.py
  endpoints.py       # fixed HTTPS base; no override surface
  policy.py          # strict policies + post-verification assertions
  modes.py           # metadata -> TEE|routed, frozen map, NEARAI_TEE_MODELS
  client.py          # per-key clients, prewarm, negative cache, timeouts
  provider.py        # LiteLLM CustomLLM "nearai_e2ee"
  models.py          # /v1/models discovery + chat filtering
  verify_cli.py      # optional: print TCB, GPU, TLS, measurements, mode

tests/
  test_near_modes.py
  test_near_policy.py
  test_near_provider.py
  test_near_streaming.py
  test_near_models.py
  test_near_perf.py
  test_live_near.py

docs/
  near-ai.md
```

Configuration:

- `pyproject.toml`: add `nearai-inference-sdk==0.3.0rc1`; bump `dcap-qvl>=0.7.0`; upgrade `PyJWT`; add `dnspython>=2.9.0`.
- `config.template.yml`: add `nearai/*`.
- `generate_config.py`: add the `nearai` provider; user-facing `nearai/<id>`, internal `nearai_e2ee/<id>`.
- `cli.py`: register `nearai_e2ee` when `NEAR_API_KEY` is present; keep in-process single-worker serving.
- `options.nix`: TTLs, stream mode, prewarm, measurements file, forced-TEE list.
- `README.md`: provider table and security notes.
- `docs/near-ai.md`: verification chain, mode policy, field coverage, limitations.

---

## 7. Implementation phases

### Phase 0 — dependency spike

- Add the SDK and version pins.
- Bump `dcap-qvl` and re-run the Chutes tests.
- Validate `uv sync --frozen`, repo tests, SDK tests, Docker build.
- Nix: package the SDK wheel and missing pure-Python dependencies. If the dependency graph is too large, evaluate a minimal vendored subset, but only as a last resort; no fork is needed.

**Done when:** both `uv` and Nix builds import `InferenceClient` and the existing test suite passes.

### Phase 1 — modes and policy

- Implement `endpoints.py`, `policy.py`, `modes.py`.
- Fetch metadata through a pinned `AttestationClient`.
- Freeze mode per model.
- Implement `NEARAI_TEE_MODELS` (force TEE, never force routed).
- Implement strict Gateway/Model policies and the post-verification assertions.
- Add the negative cache.

**Done when:** unit tests cover TEE/routed classification, forced-TEE downgrade rejection, permissive-default rejection, and negative caching.

### Phase 2 — runtime clients

- Implement `client.py`.
- Two `InferenceClient` instances per API key.
- Prewarm both modes with bounded concurrency.
- Wrap every operation in `asyncio.timeout`.
- Set per-request `httpx.Timeout`.
- Verify response bytes are released after `verify_response`.

**Done when:** concurrent cold-start tests coalesce attestation; warm requests make no attestation calls; timeouts produce clean 503s.

### Phase 3 — LiteLLM provider

- Implement `provider.py` as `CustomLLM`.
- Register `nearai_e2ee`.
- Reuse the existing payload whitelist approach from `custom_provider.py`.
- Non-streaming: preflight, send, verify, translate.
- Streaming `verified`: buffer, verify, replay.
- Streaming `tail`: live output, terminal verify, withhold `[DONE]` on failure.
- Require `provider_tee` for TEE; accept `gateway` for routed.
- Bounded receipt retry only for retryable `ApiError`; never resend inference for a signature failure.
- Map SDK errors to LiteLLM errors without leaking prompts or keys.

**Done when:** provider tests cover both modes and both streaming modes.

### Phase 4 — discovery and config

- Add `nearai` to `generate_config.py`.
- Fetch `/v1/models`; filter to chat models (exclude embeddings, rerankers, audio, image).
- Emit `model_name: nearai/<id>` and `litellm_params.model: nearai_e2ee/<id>`.
- Add model info: context length, output length, tools, vision, reasoning.
- Add `nearai/*` to the template.
- Register the provider in `cli.py`.

**Done when:** a generated config contains both TEE and routed models, and startup prewarm classifies them.

### Phase 5 — ops, docs, tests

- Nix options and environment forwarding.
- Docker lock update.
- Docs and README.
- Offline and live tests.
- Performance gates.

### Phase 6 — optional hardening

- Measurement pinning (`NEARAI_MEASUREMENTS_FILE`) for Gateway and model `os_image_hash`/`compose_hash`/`app_compose_sha256`.
- Image provenance via the SDK’s Sigstore helpers, if trusted NEAR repositories/workflows are known.
- OHTTP for latency-tolerant clients.
- Upstream PRs for lazy imports, `repr=False` on key material, and failure caching (not required for our wrapper).

---

## 8. Test plan

### Unit / offline

- TEE mode requires `provider_tee`; gateway receipt is rejected.
- Routed mode accepts `gateway`; provider_tee is rejected.
- TEE `/v1/model/{id}` metadata but routed request attempt: fail closed.
- Forced-TEE model with routed metadata: fail closed.
- Missing GPU evidence: reject.
- `OutOfDate`/`TDRelaunchAdvised`: reject.
- Non-empty advisories: reject.
- Gateway TLS binding `none`: reject.
- Missing model signing key: reject.
- Negative cache suppresses repeated failures.
- Timeout maps to 503.
- Receipt retry does not resend inference.
- Streaming `verified` withholds all content until verification.
- Streaming `tail` withholds `[DONE]` on failure.
- Non-streaming verifies before translation.

### Live (opt-in)

- `client.verify(model)` for each TEE model; assert TLS attested, TCB `UpToDate`, GPU `verified`.
- `client.verify(model)` for a routed model; assert Gateway verified and `models == ()`.
- TEE chat non-streaming; assert `provider_tee`.
- Routed chat non-streaming; assert `gateway`.
- Streaming in both modes.
- Proxy-level test through `scripts/live_server.py`.

### Performance gates

- Cold attestation per model: < 10 s; prewarmed.
- Warm attestation: zero network calls.
- Warm non-streaming overhead excluding model latency: p50 < 250 ms, p95 < 500 ms.
- `verified` streaming: TTFT = full completion + receipt RTT; document as a security cost.
- `tail` streaming: TTFT = model TTFT; terminal verification.
- Response capture memory released immediately after verification; failure records expire at `NEARAI_RESPONSE_TTL`.

---

## 9. Performance design

1. **One runtime per API key**, shared by all requests.
2. **Prewarm** TEE and routed models at startup with a semaphore (4).
3. **Coalesce** concurrent verification through the SDK’s per-model pending map and our mode lock.
4. **Negative cache** failures for 30 s.
5. **Reuse pinned HTTPS connections**; never create a client per request.
6. **Hard timeout** every operation with `asyncio.timeout`; set per-request `httpx.Timeout`.
7. **Verify immediately** so captured request/response bytes are released.
8. **Non-streaming** pays one receipt RTT after the completion; unavoidable for authenticity.
9. **Streaming `verified`** pays the full completion before the first byte; this is the secure default.
10. **Streaming `tail`** preserves TTFT but only provides terminal authentication; opt-in.
11. **No prompt/response logging**; enable `litellm.turn_off_message_logging`.
12. **Keep E2EE crypto per request** (fresh keys are required); the overhead is small relative to model latency.

---

## 10. Accepted limitations (explicitly out of scope)

- E2EE is field-level, not whole-request. Model ID, roles, sampling parameters, `response_format`, `metadata`, `user`, and some IDs are visible to the Gateway.
- Model weights/model identity are not attested. A verified TEE signer is not proof of the exact model revision.
- GPU evidence is checked by NRAS but is not cryptographically bound to the CPU TDX quote by this SDK. “GPU verified” relies on the Gateway’s assembly of the report.
- Routed models are not E2EE; they rely on TLS + Gateway attestation + Gateway signature.
- Deployment provenance is only as strong as the optional measurement pins or image policies.

These are accepted for this provider and will be documented, not treated as blockers.

---

## 11. Open questions

1. **TEE allowlist:** provide the model IDs/globs that must always be TEE (recommended), or rely on live metadata.
2. **Measurement pins:** provide known-good `os_image_hash` / `compose_hash` values, or ship without pins and log a warning.
3. **Streaming default:** `verified` is proposed; confirm that buffering is acceptable, or choose `tail` as the default.
4. **Nix:** approve packaging the SDK’s dependency graph, or ask for a minimal vendored subset.
5. **Live validation:** provide a `NEAR_API_KEY` and one TEE model plus one routed model for pre-merge validation.

---

## 12. Bottom line

Do not fork the SDK. Compose it:

- `InferenceClient(e2ee=True)` + `provider_tee` for true TEE models.
- `InferenceClient(e2ee=False)` + `gateway` for routed models.
- Strict Gateway attestation, TLS pinning, TCB/advisory/GPU assertions for TEE models.
- Freeze model modes at startup and never downgrade.
- Verify every response before returning it.
- Use hard timeouts, prewarming, and negative caching for performance.

That gives a provider that is end-to-end encrypted where NEAR supports it, properly TEE-authenticated where it claims TEE, and still usable for Gateway-routed models.
