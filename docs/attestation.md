# TEE and GPU attestation

[End-to-end encryption](./e2ee-encryption.md) guarantees that only the holder of
an instance's ML-KEM private key can read your prompt. It says nothing about
**whose** key you encrypted to. This document explains how the proxy closes that
gap by binding the instance key to Intel TDX and NVIDIA GPU attestation
evidence, and by refusing to send when it cannot.

- [1. Why encryption is not enough](#1-why-encryption-is-not-enough)
- [2. The trust chain](#2-the-trust-chain)
- [3. Endpoints and the evidence envelope](#3-endpoints-and-the-evidence-envelope)
- [4. The checks, and what each one proves](#4-the-checks-and-what-each-one-proves)
- [5. TDX `report_data` extraction](#5-tdx-report_data-extraction)
- [6. NVIDIA GPU evidence](#6-nvidia-gpu-evidence)
- [7. Fail-closed semantics](#7-fail-closed-semantics)
- [8. Binding attested instances to selection](#8-binding-attested-instances-to-selection)
- [9. Caching and performance](#9-caching-and-performance)
- [10. Environment variables](#10-environment-variables)
- [11. Manual verification (CLI)](#11-manual-verification-cli)
- [12. Sequence diagram](#12-sequence-diagram)

---

## 1. Why encryption is not enough

A malicious or compromised Chutes host (or anyone positioned to answer the API)
could hand you **its own** ML-KEM public key. Your request would be perfectly
encrypted — to the attacker. The ciphertext would look identical and every
cryptographic check in the transport would pass.

The defence is to require that the key you are about to encrypt to is _provably
the key of an Intel TDX instance driving NVIDIA GPUs in Confidential-Compute
mode_. That proof is the attestation evidence, fetched from Chutes and
optionally verified against Intel's and NVIDIA's hardware roots.

Two levels exist, controlled by environment variables:

| Level                                              | Enabled by                                                                                     | Proves                                                                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Software binding** (default when the gate is on) | `CHUTES_VERIFY_ATTESTATION=true`                                                               | The envelope is signed by the presented certificate, and the key digest appears in the TDX `report_data` and the NVIDIA evidence. |
| **Hardware roots** (opt-in)                        | `+ CHUTES_VERIFY_QUOTE=true` and/or `CHUTES_VERIFY_GPU=true`, with the optional SDKs installed | The TDX quote chains to Intel's DCAP root with an acceptable TCB status, and NVIDIA NRAS attests the GPU evidence.                |

Without the hardware flags, the certificate is _self-asserted_: you have proven
key possession and internal consistency, not that the certificate belongs to
genuine Chutes TDX hardware. Hardware roots close that last mile.

---

## 2. The trust chain

```
                     sha256(nonce + e2e_pubkey) = digest
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
 RSA signature over          Intel TDX quote              NVIDIA GPU evidence
 attested_body               report_data[0:32]            nonce == digest
 (certificate in envelope)   (TD-REPORT offset 568)
        │                          │                           │
        │ (software)               │ (CHUTES_VERIFY_QUOTE)     │ (CHUTES_VERIFY_GPU)
        ▼                          ▼                           ▼
 key possession            dcap-qvl DCAP chain          nv-attestation-sdk
 by cert holder            to Intel root,               REMOTE -> NRAS
                           status OK/UpToDate           overall result == true
```

- `e2e_pubkey` is the instance ML-KEM-768 key the transport encrypts to.
- `nonce` is a fresh value from the instance's `/e2e/instances` pool.
- `digest` is the 32-byte `sha256(nonce + e2e_pubkey)` (hex, lower-case),
  computed by `chutes_litellm/attestation.py:_digest_of`.
- The same `digest` must appear in **three** places: inside the RSA-signed body,
  in the TDX quote's `report_data`, and as the NVIDIA evidence nonce. Any
  disagreement means the key is not the one the hardware measured.

---

## 3. Endpoints and the evidence envelope

The gate uses the same endpoints Chutes documents
(`chutes-api/docs/tee-verification.md`) and the transport's discovery uses:

| Call                                        | Returns                                                                                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `GET /v1/models`                            | `data[]` entries with `id` and `chute_id`                                                               |
| `GET /e2e/instances/{chute_id}`             | `instances[]` with `instance_id`, `e2e_pubkey` (base64 ML-KEM-768), `nonces[]`; plus `nonce_expires_in` |
| `GET /chutes/{chute_id}/evidence?nonce=...` | `{"evidence": [ ... ], "failed_instance_ids": [...]}`                                                   |

### Observed envelope (live capture)

A live `-TEE` chute produced one evidence blob per instance, keyed:

| Key             | Type          | Notes                                                        |
| --------------- | ------------- | ------------------------------------------------------------ |
| `instance_id`   | string        | Matches an entry from `/e2e/instances`                       |
| `quote`         | base64 string | Intel TDX quote; ~5247 bytes decoded                         |
| `gpu_evidence`  | list          | `[{ "arch", "certificate", "evidence" }, …]`                 |
| `certificate`   | base64 string | DER X.509 cert of the attestation proxy                      |
| `signature`     | base64 string | RSA signature over `attested_body`                           |
| `attested_body` | base64 string | Signed JSON; ~139 KB decoded, nests the quote + GPU evidence |

Real-listing snapshots from the reference capture: 14 models under the test key
(all `-TEE`); one large chute listed 5 instances but the evidence endpoint
returned 16 blobs, of which 11 had no matching `/e2e/instances` entry (stale
instances). The gate **ignores** unmatched evidence (it can only add risk) and
requires every _listed_ instance to have verifying evidence.

---

## 4. The checks, and what each one proves

`verify_evidence(evidence, e2e_pubkey, nonce=…)` returns a `VerificationResult`
with a `checks` map. `chutes_litellm/attestation.py` implements them:

| Check                  | Implementation                                                                                            | Proves                                                                                          | Does **not** prove                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `key_possession`       | `_signature_ok`: RSA PKCS#1 v1.5 / SHA-256 verify of `signature` over `attested_body` using `certificate` | The envelope was produced by the holder of the certificate's private key                        | That the certificate is genuine hardware                                 |
| `quote_report_data`    | `_quote_binds`: `_extract_report_data(quote)` contains the raw 32-byte digest                             | The TDX measurement (in the same quote) saw a report whose `report_data` binds this key + nonce | That the quote itself is valid — DCAP does that                          |
| `gpu_nonce`            | `_gpu_binds`: the digest (hex or raw bytes) appears in some GPU evidence blob                             | The GPU evidence was produced for the same freshness nonce / key                                | That NVIDIA validated the GPU                                            |
| `body_contains_digest` | substring search of the decoded body                                                                      | Diagnostic only; **never** a veto                                                               | —                                                                        |
| `pubkey_bound`         | true if **any** binding source passed; false if a source is present but fails; `None` if no source exists | Aggregate "the key is bound somewhere verifiable"                                               | Which source; detailed per-check flags are preserved                     |
| `dcap_quote`           | `_verify_tdx_quote` (only with `CHUTES_VERIFY_QUOTE=true`)                                                | Intel DCAP verified the quote and the TCB status is `OK`/`UpToDate`                             | —                                                                        |
| `nv_gpu`               | `_verify_nvidia_gpu` (only with `CHUTES_VERIFY_GPU=true`)                                                 | NVIDIA NRAS attests every GPU in the evidence                                                   | That the GPU is running the expected workload (that is the policy layer) |

### Fail-closed rules

From `verify_evidence`:

- If `key_possession` is present but **false** → fail.
- If a present binding source (`quote_report_data`, `gpu_nonce`) is **false** → fail.
- If **no** binding evidence exists at all → fail (a dishonest relay that
  forwards nothing must not be treated as "nothing to check").
- Hardware verifiers **raise** rather than return false; the caller records the
  per-instance reason.

`body_contains_digest` is informational. The signed `attested_body` is trusted
because of `key_possession`, so if the digest only appears there, the checks
still bind.

---

## 5. TDX `report_data` extraction

`_extract_report_data(tdx_quote)` reads the 64-byte `report_data` at a fixed
offset:

```
TD-REPORT starts at byte 48 of the DCAP quote.
report_data is the last 64 bytes of the 584-byte TD-REPORT:
    offset = 48 + 584 - 64 = 568
    set(range(0, 64)) bytes, quoted to the end.
```

This is a deliberate **heuristic** mirroring the public Chutes layout: the code
does not link an SGX/TDX parser for the software-binding path. If the blob is
too short (< 640 bytes) or all-zero, `_extract_report_data` returns `None`, and
`_quote_binds` returns `None` (not checked) rather than a false failure. If Intel
changes the binary format, the software check stops seeing the binding — and the
verdict fails closed only if there is no other binding source. The hardware path
(`dcap-qvl`) independently re-parses the quote when
`CHUTES_VERIFY_QUOTE=true`, so the fixed offset is not trusted for the root of
trust.

`_quote_binds` searches for the raw 32 digest bytes; because `sha256` output is
32 bytes and `report_data` is 64, the digest sits in `report_data[:64]`.

---

## 6. NVIDIA GPU evidence

Each GPU blob is `{ "arch", "certificate", "evidence" }`. The NVIDIA SDK's
remote verifier (`nv_attestation_sdk/gpu/attest_gpu_remote.py:build_payload`)
requires exactly these fields and a **single consistent `arch`** across the list,
so `_normalize_gpu_evidence`:

- flattens candidates discovered by `_collect_gpu` (top-level `gpu_evidence`,
  JSON strings, and the signed body's `evidence.gpu_evidence` /
  `nvtrust_evidence`),
- drops anything missing/empty `arch`, `certificate` or `evidence` (this is why
  the test mock's `{nonce, gpu_uuid}` shape yields "no usable GPU evidence" and
  fails closed),
- JSON-encodes non-string certificate/evidence values,
- de-duplicates identical entries, and
- rejects a mixed-architecture list.

`_verify_nvidia_gpu` then serialises access to the SDK's process-wide singleton
(`Attestation().reset()` → `add_verifier(Devices.GPU, Environment.REMOTE, url, "")`
→ `set_nonce(digest)` → `attest(normalized)`). `set_nonce(digest)` is what makes
NRAS check the freshness/anti-substitution binding. Every GPU must pass; the SDK
returns a single overall verdict, so failures report the arch/count and the NRAS
URL used.

---

## 7. Fail-closed semantics

`verify_chute` (the gate) is **strict**: it fetches `/e2e/instances`, then
`/chutes/{id}/evidence`, and requires **every listed instance to verify**. This
matters because the transport picks an instance itself; trusting a
partially-verified chute would let a request land on the unverified instance.
(The [instance filter](#8-binding-attested-instances-to-selection) removes even
that gap.)

It returns the list of verified `instance_id`s, or raises `AttestationError`
with an actionable multi-line message:

```
attestation failed for chute <id> (model '<model>') — request refused.
  instances listed: N, evidence blobs: M, failed: F, unmatched: U
  per-instance reason(s):
    - <instance>: failed checks: quote_report_data  [signature=ok, tdx_binding=FAIL, gpu_binding=ok, ...]
  api_base=…  models_base=…
```

In the proxy the error is wrapped in a `litellm.APIError` with HTTP **503**, so
the caller sees a structured body explaining _why_, not an opaque 500. Because
the check runs in the httpx transport before the request leaves the process, a
failed gate means **no encrypted request is ever sent** (the offline tests assert
`/e2e/invoke` was not called).

Missing optional SDKs also fail closed: `verify_quote`/`verify_gpu` raise
`AttestationError("… requires the optional 'dcap-qvl'/'nv-attestation-sdk'
package")` when the package is not importable.

---

## 8. Binding attested instances to selection

Strictness at the gate is necessary but not quite sufficient: the gate fetches
`/e2e/instances` itself, while the E2EE transport has its **own** discovery
cache. A race (or a malicious server returning different instance sets to the
two calls) could let the transport select an instance the gate never saw.

The fork adds an optional `instance_filter` hook to `DiscoveryManager`
(`chutes-e2ee-transport/src/chutes_e2ee/discovery.py`):

```python
DiscoveryManager(..., instance_filter=lambda chute_id, instances: allowed_instances)
```

- It is applied right after `_fetch_instances` / `_fetch_instances_async`, before
  the nonce pool is built.
- It may raise to refuse the chute.
- If it returns an empty set while instances were listed, the chute is refused
  (no silent fallback to an unapproved instance).
- When the gate is off, the proxy passes `None` — zero overhead.

The proxy wires it in `src/chutes_litellm/e2ee_litellm.py` and
`src/chutes_litellm/custom_provider.py`: the filter calls `verify_chute(chute_id)`
(cheaply cached after the pre-send `_maybe_verify`) and intersects the
transport's instances with the verified ids. The pre-send `_maybe_verify` is kept
as an explicit, cache-warm check so the failure message is emitted before any
discovery work; the filter is what actually constrains selection.

---

## 9. Caching and performance

Attestation is **off by default** and costs nothing unless
`CHUTES_VERIFY_ATTESTATION=true`. When on, steady-state requests do a dict
lookup, not a network round trip:

| Mechanism          | Env var (default)                          | Notes                                                                                                                                                                                         |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Positive cache     | `CHUTES_ATTESTATION_TTL` (300 s)           | Verified instance ids per `(chute_id, verify_quote, verify_gpu, check_signature)`. The mode is part of the key, so a no-quote verdict can never satisfy a `CHUTES_VERIFY_QUOTE=true` request. |
| Negative cache     | `CHUTES_ATTESTATION_FAILURE_TTL` (30 s)    | A failing chute fails fast instead of re-fetching on every request (self-inflicted DoS protection).                                                                                           |
| Model→chute map    | `CHUTES_ATTESTATION_MODEL_MAP_TTL` (300 s) | Avoids a `/v1/models` call per request.                                                                                                                                                       |
| Single-flight      | —                                          | Concurrent cold-cache callers are coalesced into one evidence fetch (a `threading.Event` per key).                                                                                            |
| Pooled HTTP client | —                                          | One shared `httpx.Client` keeps TLS sessions to `api.chutes.ai` alive.                                                                                                                        |

Measured on the reference host: a cold `verify_chute` (network-bound evidence
fetch + parse) ≈ **8 s**; warm ≈ **0 ms**; the GPU binding scan ≈ **0 ms** on
the captured evidence. `attested_body` (~139 KB) is base64-decoded and parsed
**once** per instance per verification and reused for every binding check.

---

## 10. Environment variables

| Variable                           | Default                                               | Meaning                                                           |
| ---------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| `CHUTES_VERIFY_ATTESTATION`        | `false`                                               | Master switch for the fail-closed gate                            |
| `CHUTES_VERIFY_QUOTE`              | `false`                                               | Require Intel DCAP TDX-quote verification (needs `dcap-qvl`)      |
| `CHUTES_VERIFY_GPU`                | `false`                                               | Require NVIDIA NRAS GPU verification (needs `nv-attestation-sdk`) |
| `CHUTES_ATTESTATION_TTL`           | `300`                                                 | Positive verdict TTL (seconds)                                    |
| `CHUTES_ATTESTATION_FAILURE_TTL`   | `30`                                                  | Negative verdict TTL (seconds)                                    |
| `CHUTES_ATTESTATION_MODEL_MAP_TTL` | `300`                                                 | Model→chute id cache TTL (seconds)                                |
| `CHUTES_DCAP_PCCS_URL`             | dcap-qvl default (Phala `https://pccs.phala.network`) | Collateral/PCCS base for DCAP                                     |
| `CHUTES_NVIDIA_NRAS_URL`           | `https://nras.attestation.nvidia.com/v3/attest/gpu`   | NVIDIA Remote Attestation Service endpoint                        |
| `CHUTES_E2EE_API_BASE`             | `https://api.chutes.ai`                               | E2EE/attestation API base                                         |
| `CHUTES_E2EE_MODELS_BASE`          | `https://llm.chutes.ai`                               | Model-listing base                                                |

The NVIDIA SDK additionally honours its own `NVIDIA_ATTESTATION_SERVICE_KEY`,
`NV_NRAS_GPU_URL`, `NV_OCSP_URL`, `NV_RIM_URL` and `NV_ALLOW_HOLD_CERT`; a
non-empty `CHUTES_NVIDIA_NRAS_URL` is passed explicitly and wins.

---

## 11. Manual verification (CLI)

Install the optional SDKs for hardware roots:

```sh
uv sync --extra attestation
```

Software binding for a model (default bases):

```sh
CHUTES_API_KEY=cpk_... chutes-verify-attestation --model "Qwen/Qwen3.5-397B-A17B-TEE"
```

Full hardware roots (needs outbound Intel PCCS + NVIDIA NRAS):

```sh
CHUTES_API_KEY=cpk_... chutes-verify-attestation \
  --chute <uuid> --verify-quote --verify-gpu --details
```

`--details` prints, per instance, the check map, the DCAP TCB status and the
NVIDIA arch/GPU count, e.g.:

```
chute <uuid> (model '…-TEE'): instances listed=5, evidence blobs=16, unmatched=11
  - <instance>: VERIFIED [signature=ok, tdx_binding=ok, gpu_binding=ok, body_digest=ok, pubkey_bound=ok | dcap_status=OK | gpu=1xHOPPER]
OK: 5 instance(s) passed TEE/GPU attestation
```

The default (non-`--details`) output is unchanged, so existing scripts keep
working. See [`scripts/verify_attestation.py`](../scripts/verify_attestation.py)
and [`src/chutes_litellm/verify.py`](../src/chutes_litellm/verify.py).

> **Software binding vs hardware roots.** Without `--verify-quote` /
> `--verify-gpu` (or without the SDKs), the CLI proves the signature over the
> evidence and the internal `sha256(nonce+pubkey)` binding. It does **not**
> validate the Intel/NVIDIA roots. When a hardware flag is set without the SDK
> installed, the command fails closed.

---

## 12. Sequence diagram

```mermaid
sequenceDiagram
    autonumber
    participant P as Proxy gate
    participant A as api.chutes.ai
    participant D as Intel DCAP (PCCS)
    participant N as NVIDIA NRAS

    P->>A: GET /v1/models (cached)
    A-->>P: model -> chute_id
    P->>A: GET /e2e/instances/{chute_id}
    A-->>P: instances [instance_id, e2e_pubkey, nonces, nonce_expires_in]
    P->>P: nonce = random 32 bytes; digest = sha256(nonce + e2e_pubkey)
    P->>A: GET /chutes/{chute_id}/evidence?nonce=...
    A-->>P: evidence[] (quote, gpu_evidence, certificate, signature, attested_body)

    loop each evidence blob
        P->>P: key_possession = RSA(signature, attested_body, cert)
        P->>P: quote_report_data = digest bytes in TDX report_data
        P->>P: gpu_nonce = digest in NVIDIA evidence
        opt CHUTES_VERIFY_QUOTE
            P->>D: get_collateral_and_verify(quote)
            D-->>P: report.status == OK/UpToDate
        end
        opt CHUTES_VERIFY_GPU
            P->>N: POST attest/GPU (nonce=digest, {arch,certificate,evidence})
            N-->>P: overall attestation result == true
        end
    end

    alt every listed instance verified
        P->>P: cache verified ids for TTL
        P->>P: instance_filter keeps only verified ids
        Note over P: encrypted request proceeds (see e2ee-encryption.md)
    else any failure / no binding / missing SDK
        P-->>P: raise AttestationError -> litellm.APIError 503
        Note over P: request is NOT sent (fail closed)
    end
```
