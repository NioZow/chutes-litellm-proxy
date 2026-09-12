# Implementation Plan — Chutes E2EE transport hardening + comprehensive security docs

This file is the **handoff context** for the next session. It is self-contained:
read it top to bottom before touching anything.

---

## 0. Context for the next session (read first)

### Repo / branch

- Working dir: `/home/user/chutes-litellm-proxy`
- Branch: `refactor`
- `src/`, `tests/`, `TEST.md` are currently **untracked** new work on this branch.
- **Do not commit or push anything.** The user handles all git operations themselves.
- The user maintains their own fork: `github.com:niozow/chutes-e2ee-transport`
  (currently pinned at rev `5630286de88d0797bf605279f19b974db25bded7`).

### Decisions already made (do not re-litigate)

1. **No CI.** Do not add any GitHub Actions workflow.
2. **Hardware SDKs are optional**, packaged the same way `pqcrypto` already is
   (they must not break the base build/run when absent).
3. **Fork workflow:** clone the fork **directly inside this repo folder** at
   `./chutes-e2ee-transport` (i.e. `/home/user/chutes-litellm-proxy/chutes-e2ee-transport`),
   use it locally as a `path` dependency, make transport changes **there directly**.
   It is **not** a submodule and **must not be committed** — add it to `.gitignore`.
   The user will push the fork changes later and re-pin the rev themselves.
   **STATUS: already cloned** at rev `5630286de88d0797bf605279f19b974db25bded7`
   (same as the pyproject pin) and `/chutes-e2ee-transport/` is already in
   `.gitignore`. It has its own `tests/` (only `tests/test_live.py` so far).
4. **Docs:** the user wants genuinely comprehensive, understandable docs on how the
   encryption and attestation work (they dislike not understanding AI-built things).
   This is a first-class deliverable (Phase 5), not an afterthought.

### Current state of the code (what was already changed this branch)

- `src/chutes_litellm/attestation.py`
  - `verify_evidence()` checks: RSA key-possession signature, TDX `report_data`
    binding, NVIDIA GPU nonce binding, diagnostic body digest. Fail-closed.
  - `verify_chute()` requires **every** `/e2e/instances` entry to verify (strict).
  - Cache keyed by `(chute_id, verify_quote, verify_gpu, check_signature)`.
  - Negative cache (`CHUTES_ATTESTATION_FAILURE_TTL`, default 30s) + single-flight
    coalescing. `attested_body` parsed once. `_run_async()` helper for DCAP.
  - `_verify_tdx_quote()` imports `dcap_qvl.get_collateral_and_verify`, checks
    `status == "UpToDate"`.
  - `_verify_nvidia_gpu()` imports `nv_attestation_sdk.attestation`, does
    `add_verifier(Devices.GPU, Environment.REMOTE, "", "")`, `set_nonce(digest)`,
    `client.attest(gpu_evidence)` — **this shape is suspect, see Phase 2**.
- `src/chutes_litellm/e2ee_litellm.py` wires `CHUTES_VERIFY_QUOTE` / `_GPU` into
  the sync + async scoped transports.
- `src/chutes_litellm/custom_provider.py` has `_maybe_verify_attestation()` called
  from `completion` / `acompletion` / `streaming` / `astreaming`.
- Offline suite: **40 passed, 5 skipped**. Live suite (real key): attestation
  **1 passed**, proxy **4 passed**. Lint clean.

### Test/run environment quirks (important)

- NixOS `libstdc++` workaround is required for importing litellm:
  ```sh
  LIB="$(dirname "$(gcc -print-file-name=libstdc++.so)")"
  LD_LIBRARY_PATH="$LIB" .venv/bin/python -m pytest tests/ -q
  ```
- `ruff` in `.venv` is a broken dynamic binary; use the profile one:
  `/etc/profiles/per-user/user/bin/ruff check src tests scripts`
- Chutes API key is supplied via env at run time (`CHUTES_API_KEY`). Never write it
  to a file or commit it.
- Offline tests set `CHUTES_E2EE_VERIFY_SSL=false` (mock is plain HTTP). Live runs
  use real TLS.

### Live evidence facts (captured from the real API, useful for Phase 2/4)

- `GET /v1/models` → 14 models under the test key, all `-TEE`.
- `GET /e2e/instances/{chute}` → instances with `instance_id`, `e2e_pubkey` (b64
  ML-KEM-768), `nonces`, plus `nonce_expires_in`.
- `GET /chutes/{chute_id}/evidence?nonce=...` → `{"evidence": [...]}`.
  Each blob keys: `instance_id`, `quote` (b64, ~5247 bytes decoded), `gpu_evidence`
  (list of `{arch, certificate, evidence}`), `certificate` (b64 DER), `signature`,
  `attested_body` (b64, ~139 KB decoded).
- Real chute for `Qwen/Qwen3.5-397B-A17B-TEE` had 5 listed instances and 16 evidence
  blobs (11 unmatched); all 5 matched instances passed software binding.
- Cold `verify_chute` ≈ 8 s (network-bound); warm ≈ 0 ms; GPU binding scan ≈ 0 ms.

---

## 1. Goals

- Make real **Intel DCAP** and **NVIDIA** hardware-root verification actually
  reachable and correct, gated by the existing `CHUTES_VERIFY_QUOTE` /
  `CHUTES_VERIFY_GPU` env vars.
- Bind the **attested instance set** to the transport's instance selection so the
  request can only go to an instance we actually verified (fork change).
- Add **comprehensive, understandable docs** on encryption + attestation.
- Keep the base (no-SDK) build/run working and the offline suite green.

Non-goals: no CI, no new cloud services, no changes to the OpenAI-compatible API
surface.

---

## 2. Phase 1 — Optional hardware SDKs (packaged like `pqcrypto`)

Packages confirmed on PyPI: `dcap-qvl` (0.6.3, `>=3.8`) and
`nv-attestation-sdk` (2.7.3, `>=3.9`).

### Tasks

1. `pyproject.toml`: add
   ```toml
   [project.optional-dependencies]
   attestation = ["dcap-qvl>=0.6.3", "nv-attestation-sdk>=2.7.3"]
   ```
   then `uv lock` (updates `uv.lock`).
2. `flake.nix`: add both to the Python env **optionally**. If not in nixpkgs,
   vendor wheels via `fetchurl` following the existing `pqcryptoWheel` pattern
   (see `flake.nix` lines ~34–112). Base `mkChutesPython` must still build with
   neither SDK present.
3. `Dockerfile`: install with the extra in both sync layers, e.g.
   `uv sync --locked --no-dev --extra attestation --no-install-project` and the
   final `uv sync --locked --no-dev --no-editable --extra attestation`.
4. `README.md`: document the optional extra.

### Acceptance

- `uv sync --extra attestation` succeeds; `python -c "import dcap_qvl, nv_attestation_sdk"` works.
- Without the extra, the proxy starts and offline tests pass exactly as today.
- Missing SDK at request time still fails **closed** with the existing clear error
  from `_verify_tdx_quote` / `_verify_nvidia_gpu`.

---

## 3. Phase 2 — Make the DCAP / NVIDIA verifier correct (highest risk)

The current `_verify_nvidia_gpu` almost certainly does not match the real SDK:

- Real `gpu_evidence` is a list of `{arch, certificate, evidence}` dicts; the
  NVIDIA SDK REMOTE verifier expects JWT-style evidence plus an NRAS verification
  URL, not that raw dict list.
- `add_verifier(Devices.GPU, Environment.REMOTE, "", "")` with empty URL must be
  validated (REMOTE mode needs the NRAS endpoint).

### Tasks

1. Install the SDKs into `.venv` (Phase 1) and inspect the real APIs:
   - `nv_attestation_sdk.attestation.Attestation` (`attest`, `add_verifier`,
     `set_nonce`, `Environment`, `Devices`) — read the installed source.
   - `dcap_qvl.get_collateral_and_verify` return object and `status` values.
2. Write a throwaway probe (in `/tmp/opencode`, not the repo) that feeds **real
   captured Chutes evidence** to the SDK to learn the exact acceptable shape.
3. Rewrite `_verify_nvidia_gpu(gpu_evidence, digest)`:
   - Transform each `{arch, certificate, evidence}` into whatever the SDK wants.
   - Configure the remote verifier URL (NRAS) correctly, ideally
     overridable by env (e.g. `CHUTES_NVIDIA_NRAS_URL`, defaults documented).
   - Require **all** GPUs to pass; fail closed with a per-GPU reason.
   - Keep `set_nonce(digest)` so the freshness check is meaningful.
4. Harden `_verify_tdx_quote`: keep `status == "UpToDate"`, but normalize case and
   surface PCCS-collateral failures distinctly. Make the collateral/PCCS endpoint
   overridable if the library supports it.
5. **Unit tests with fake SDKs** (monkeypatch `sys.modules["dcap_qvl"]` and
   `sys.modules["nv_attestation_sdk..."]`):
   - assert the raw `{arch, certificate, evidence}` dicts are transformed before
     reaching `attest()`;
   - assert a failing GPU fails the whole verification;
   - assert missing evidence / missing SDK raise `AttestationError` (fail closed).
     These run hermetically in the offline suite (no real SDK, no network).

### Acceptance

- Existing `test_hardware_verification_requires_optional_sdks` still passes.
- New fake-SDK unit tests pass in the offline suite.
- Manual `--verify-quote --verify-gpu` fails with actionable messages when
  PCCS/NRAS are unreachable.

---

## 4. Phase 3 — Fork `chutes-e2ee-transport` + bind attested instances to selection

### 4.1 Clone (directly in this folder)

Already done. If it needs re-cloning:

```sh
cd /home/user/chutes-litellm-proxy
git clone https://github.com/niozow/chutes-e2ee-transport.git chutes-e2ee-transport
```

- `chutes-e2ee-transport/` is already in `.gitignore` (not committed, not a submodule).
- For local use, point the proxy at the clone. Two options:
  - `pyproject.toml` + `[tool.uv.sources]` (e.g.
    `chutes-e2ee = { path = "chutes-e2ee-transport" }`) and `uv lock`; or
  - `pip install -e ./chutes-e2ee-transport` into `.venv` for the session.
    Prefer the `[tool.uv.sources]` path so `uv sync` is reproducible for the user.
- `flake.nix`: for local dev, switch `fetchFromGitHub` to a local `path:` (or leave
  the remote rev and only change it once the user pushes). Keep this clearly marked
  as a local-dev override the user will revert.
- Remember: the user pushes the fork later and re-pins the rev; leave a note/`TODO`.

### 4.2 Fork change design (the actual improvement)

Problem: the gate verifies a set of instances, but `DiscoveryManager.get_nonce`
(`chutes_e2ee/discovery.py`, ~lines 133–157 sync, 225–250 async) picks an instance
from its **own** `/e2e/instances` fetch. Strict all-instances enforcement narrows
the gap, but the two discovery caches are independent.

Cleanest fork change: add an optional **instance filter / verifier hook** to
`DiscoveryManager` (and thread it through `ChutesE2EETransport.__init__`):

```python
class DiscoveryManager:
    def __init__(self, ..., instance_filter: Callable[[str, list[InstanceInfo]], list[InstanceInfo]] | None = None):
        self._instance_filter = instance_filter

    def _apply_filter(self, chute_id, instances):
        if self._instance_filter is None:
            return instances
        allowed = self._instance_filter(chute_id, instances)   # may raise
        allowed_ids = {i.instance_id for i in allowed}
        return [i for i in instances if i.instance_id in allowed_ids]
```

- Apply it in `get_nonce` and `get_nonce_async` right after `_fetch_instances*`
  and before building `_CachedNonces`; raise a clear error if the filtered pool is
  empty.
- Thread through `ChutesE2EETransport` / `AsyncChutesE2EETransport` constructors.
- Add fork-side tests for the hook (the fork is a separate repo; add a
  `tests/` there if it doesn't have one).

### 4.3 Proxy-side wiring

- In `e2ee_litellm._ChutesScopedTransport` / `_ChutesScopedAsyncTransport`, pass an
  `instance_filter` that calls the attestation gate for that `chute_id`, computes
  the verified instance-id set (`verify_chute(...)`), and returns only those
  instances. When `CHUTES_VERIFY_ATTESTATION` is off, pass `None` (zero overhead).
- In `custom_provider._sync_transport` / `_async_transport`, same.
- This makes the gate intrinsic to selection: the instance used is provably one we
  verified in the same selection decision.
- Keep the existing pre-send `_maybe_verify()` as a cheap explicit check (or remove
  the redundancy once the filter owns it — decide and document).

### Acceptance

- Offline: a mock with **two** instances where only one verifies → the request is
  refused because the unverified instance cannot be selected.
- Offline: when attestation is off, behavior and per-request cost are unchanged
  (no filter call).
- Live: `CHUTES_VERIFY_ATTESTATION=true` request succeeds and the invoke's
  `X-Instance-Id` is in the verified set.

---

## 5. Phase 4 — Real end-to-end hardware verification + live test

### Tasks

1. Add an opt-in live test in `tests/test_live_attestation.py` guarded by
   `CHUTES_LIVE_TEE=1` **and** the SDKs being importable:
   - `verify_chute(model, verify_quote=True, verify_gpu=True)` must pass against a
     real `-TEE` chute (needs outbound Intel PCCS + NVIDIA NRAS).
   - `pytest.skip` (not fail) when SDKs are not installed.
2. Extend `scripts/verify_attestation.py` to surface per-GPU verdicts and the DCAP
   status, not just a boolean. Keep the CLI backward compatible.
3. Add an opt-in live test that drives a real request through the fork with the
   gate on and asserts the chosen `X-Instance-Id` was in the verified set.

### Acceptance

- On a host with the SDKs + network, the full hardware live test passes.
- Without the SDKs, it skips cleanly and the software-only live test still passes.

---

## 6. Phase 5 — Comprehensive security docs (first-class deliverable)

The user explicitly wants to **fully understand** the encryption and attestation.
Write docs that explain the _why_, the _cryptography_, the _wire format_, the
_trust chain_, and the _limits_ — with diagrams and concrete bytes/offsets.

### Deliverables

- `docs/README.md` — index / how to read.
- `docs/e2ee-encryption.md` — end-to-end encryption.
- `docs/attestation.md` — TEE/GPU attestation.
- `README.md` — add a short "Security model" section linking the two docs.

### `docs/e2ee-encryption.md` outline

1. **Threat model** — adversary (malicious/compromised Chutes host, network,
   relay), assets protected (prompt + completion confidentiality/integrity), and
   what is NOT protected (model id, timing, token counts, your local host, the API
   key, control-plane metadata).
2. **Primitives** — ML-KEM-768 (FIPS 203) KEM; HKDF-SHA256; ChaCha20-Poly1305 AEAD;
   why post-quantum KEM plus symmetric AEAD.
3. **Request key establishment** — ephemeral `response_pk/sk`; `mlkem_encrypt` to
   the instance `e2e_pubkey` → `(mlkem_ct, shared_secret)`; `HKDF(ss, salt=ct[:16],
info="e2e-req-v1")`; gzip the JSON; ChaCha20-Poly1305 with a random 12-byte
   nonce; `blob = mlkem_ct(1088) || nonce(12) || ct || tag(16)`. Reference
   `chutes_e2ee/crypto.py:build_e2ee_request`.
4. **Response** — request embeds `e2e_response_pk`; server encapsulates to it;
   client decapsulates with `response_sk`; `info="e2e-resp-v1"`.
5. **Streaming** — `e2e_init` frame carries an ML-KEM ct; derive
   `info="e2e-stream-v1"`; each SSE chunk is ChaCha20-Poly1305 with its own nonce;
   `usage`/errors handling. Reference `_iter_sse_from_e2ee`.
6. **Freshness / nonces / replay** — `X-E2E-Nonce` from the instance pool,
   `nonce_expires_in`, and the `sha256(nonce + e2e_pubkey)` binding digest.
7. **Sequence diagram (mermaid)**: client → `/v1/models`, `/e2e/instances`,
   `/e2e/invoke` (encrypted), decrypted reply.
8. **What each layer guarantees** and the “server never sees plaintext” property.
9. **Code map** (file → function) and **how to reproduce the capture proof**
   (intercept the pre-TLS bytes; assert the marker is absent).

### `docs/attestation.md` outline

1. **Why** — encryption alone does not prove the ML-KEM key belongs to genuine
   TDX + NVIDIA-CC hardware; a malicious relay could hand you its own key.
2. **Trust chain** — `e2e_pubkey` ↔ `sha256(nonce+pubkey)` in TDX `report_data`
   and NVIDIA evidence nonce ↔ attestation-proxy RSA signature ↔ Intel DCAP root
   ↔ NVIDIA NRAS/GPU root.
3. **Endpoints + evidence envelope** — real keys and sizes (from §0).
4. **Checks table** — `key_possession`, `quote_report_data`, `gpu_nonce`,
   `body_contains_digest`, `pubkey_bound`: exactly what each proves and what it
   does **not**.
5. **TDX `report_data` extraction** — offset math (`48 + 584 - 64`), why it is a
   heuristic, and how DCAP verifies the quote.
6. **NVIDIA evidence** — `{arch, certificate, evidence}` shape and how the SDK is
   invoked (cross-reference the Phase 2 rewrite).
7. **Fail-closed semantics** — strict all-instances, per-instance reasons, what a
   refused request looks like (`litellm.APIError` 503 body).
8. **Caching & performance** — TTLs, failure TTL, single-flight, per-mode cache
   keys; measured cold/warm numbers; why steady-state is a dict lookup.
9. **env var reference** — all `CHUTES_*` knobs.
10. **Manual verification** — `chutes-verify-attestation` CLI examples, and the
    distinction between **software binding** (default) vs **hardware roots**
    (requires SDKs).
11. **Sequence diagram (mermaid)** of the attestation gate.

### Acceptance

- A reader who understands basic crypto can follow request→reply and
  evidence→verdict without reading source.
- Every claim maps to a function/file reference and (where relevant) the observed
  live evidence shape.

---

## 7. Phase 6 — Final validation

1. Lint: `/etc/profiles/per-user/user/bin/ruff check src tests scripts`.
2. Offline suite (with the libstdc++ workaround): expect `40+n passed, 5 skipped`.
3. Live suite with the real key (attestation + proxy).
4. Live hardware path (`CHUTES_VERIFY_QUOTE=1 CHUTES_VERIFY_GPU=1`) — passes where
   PCCS/NRAS are reachable, skips otherwise.
5. Confirm docs links resolve and `README.md` Security section is accurate.

---

## 8. File map / where things are

| What                    | Path                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Attestation logic       | `src/chutes_litellm/attestation.py`                                                                                       |
| Native transport swap   | `src/chutes_litellm/e2ee_litellm.py`                                                                                      |
| Custom provider         | `src/chutes_litellm/custom_provider.py`                                                                                   |
| CLI verifier            | `src/chutes_litellm/verify.py` / `scripts/verify_attestation.py`                                                          |
| Offline tests           | `tests/test_attestation.py`, `tests/test_custom_provider.py`, `tests/test_e2ee_litellm.py`, `tests/mock_chutes_server.py` |
| Live tests              | `tests/test_live_attestation.py`, `tests/test_live_proxy.py`                                                              |
| Deps / build            | `pyproject.toml`, `uv.lock`, `Dockerfile`, `flake.nix`                                                                    |
| Docs                    | `README.md`, `TEST.md`, `docs/` (to create)                                                                               |
| Fork clone (local only) | `chutes-e2ee-transport/` (already cloned, gitignored)                                                                     |

### Fork internals to know

- `chutes_e2ee/discovery.py` — `DiscoveryManager`, `_CachedNonces`, `get_nonce`,
  `get_nonce_async`, `_fetch_instances(_async)`.
- `chutes_e2ee/transport.py` — `ChutesE2EETransport`,
  `AsyncChutesE2EETransport`; both create `self._discovery`.
- `chutes_e2ee/crypto.py` — `build_e2ee_request`, `decrypt_response`,
  `decrypt_stream_init`, `decrypt_stream_chunk`, `derive_key`.

---

## 9. Command cheat-sheet

```sh
# install (with hardware SDKs)
uv sync --extra attestation

# clone the fork locally (not committed)
cd /home/user/chutes-litellm-proxy && git clone https://github.com/niozow/chutes-e2ee-transport.git chutes-e2ee-transport

# lint
/etc/profiles/per-user/user/bin/ruff check src tests scripts

# offline tests (NixOS libstdc++ workaround)
LIB="$(dirname "$(gcc -print-file-name=libstdc++.so)")"
LD_LIBRARY_PATH="$LIB" .venv/bin/python -m pytest tests/ -q

# live tests (real key)
CHUTES_LIVE_TEE=1 CHUTES_API_KEY=<key> \
  LD_LIBRARY_PATH="$LIB" .venv/bin/python -m pytest tests/test_live_attestation.py tests/test_live_proxy.py -v -s

# manual attestation (software binding, then full hardware roots)
CHUTES_API_KEY=<key> uv run chutes-verify-attestation --model "Qwen/Qwen3.5-397B-A17B-TEE"
CHUTES_API_KEY=<key> uv run chutes-verify-attestation --chute <uuid> --verify-quote --verify-gpu
```

---

## 10. Execution order

**Phase 1 → 2 → 3 → 4 → 5 → 6**

Phases 2 and 3 are the risky ones; do them behind tests and keep the offline suite
green after each. Do **not** commit or push anything.
