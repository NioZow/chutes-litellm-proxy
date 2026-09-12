"""Chutes TEE / GPU attestation verification.

Chutes serves inference from Intel TDX confidential VMs that drive NVIDIA GPUs in
Confidential-Compute (CC) mode (see chutesai/sek8s, chutesai/chutes-api
docs/tee-verification.md).  ``chutes-e2ee`` (the ML-KEM-768 transport library
this proxy rides on) encrypts traffic to an *instance public key*, but it does
NOT verify that the instance key is bound to attested hardware.

This module closes that gap.  Before trusting a key it fetches the attestation
evidence Chutes exposes for an instance and checks, to the degree the platform
allows:

  * *Key possession* — the evidence payload is RSA-signed by the certificate of
    the attestation proxy that produced it.
  * *Key binding* — the instance ``e2e_pubkey`` is bound into the evidence via
    ``sha256(nonce + e2e_pubkey)`` (freshness + anti-substitution).  The same
    digest is expected inside the Intel TDX ``quote``'s ``report_data`` and as
    the NVIDIA attestation nonce of each ``gpu_evidence`` blob.
  * *Hardware roots (optional)* — verify the TDX quote with Intel DCAP
    (``dcap-qvl``) and the NVIDIA evidence with ``nv-attestation-sdk``.

The wire formats come from public Chutes docs / repos and are kept pluggable
(env-overridable paths) because they are not yet pinned to a public schema.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_API_BASE = "https://api.chutes.ai"
DEFAULT_MODELS_BASE = "https://llm.chutes.ai"
DEFAULT_EVIDENCE_PATH = "/instances/{instance_id}/evidence"

# Default NVIDIA remote verifier (NRAS) endpoint, matching the nv-attestation-sdk
# default.  Overridable with CHUTES_NVIDIA_NRAS_URL.
DEFAULT_NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"

# VerifiedReport.status values that mean "verified"; dcap-qvl <=0.6.3 reports the
# legacy string ("OK") and the policy API names the equivalent TCB state
# ("UpToDate"), so accept either, case-insensitively.
_DCAP_OK_STATUSES = frozenset({"ok", "uptodate"})

# The NVIDIA SDK's Attestation object is a process-wide singleton with mutable
# class state (verifier list + nonce), so serialise access to it.
_NVIDIA_LOCK = threading.Lock()


class AttestationError(RuntimeError):
    """Raised when a TEE/GPU attestation check fails (fail closed)."""


@dataclass
class InstanceInfo:
    instance_id: str
    e2e_pubkey: str  # base64 ML-KEM-768 public key
    nonces: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    instance_id: str
    verified: bool
    checks: dict[str, bool | None]  # True = passed, False = failed, None = not checked
    nonce: str
    digest: str
    detail: str = ""
    dcap_status: str | None = None  # TDX TCB status when verify_quote ran
    gpu: dict[str, Any] | None = None  # {"arch", "gpus", "nras_url"} when verify_gpu ran


def _digest_of(nonce: str, e2e_pubkey: str) -> str:
    """Freshness/binding digest: sha256(nonce + e2e_pubkey), hex-encoded."""
    return hashlib.sha256((nonce + e2e_pubkey).encode()).hexdigest().lower()


def _extract_report_data(tdx_quote: bytes) -> bytes | None:
    """Best-effort extraction of the 64-byte report_data from an Intel TDX quote.

    The DCAP quote wraps the TD-REPORT; ``report_data`` (64 bytes) sits at a
    fixed offset inside it.  This is a heuristic mirror of the Chutes
    ``tee-verification.md`` layout and may need adjusting if Intel changes the
    binary format.  Returns ``None`` when the blob does not look like a quote so
    callers can treat the hardware check as "not checked" instead of failing.
    """
    if len(tdx_quote) < 640:
        return None
    # TD-REPORT body begins at byte 48 of the quote; report_data is the last 64
    # bytes of the TD-REPORT (TD-REPORT is 584 bytes -> offset 48 + 584 - 64).
    report_data = tdx_quote[48 + 584 - 64 : 48 + 584]
    if not report_data or set(report_data) == {0}:
        return None
    return report_data


def _signature_ok(evidence: dict[str, Any]) -> bool | None:
    """Check the attestation proxy's RSA signature over ``attested_body``."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography import x509
    except ImportError:
        return None
    try:
        cert_b64 = evidence.get("certificate")
        sig_b64 = evidence.get("signature")
        body_b64 = evidence.get("attested_body")
        if not (cert_b64 and sig_b64 and body_b64):
            return None
        cert = x509.load_der_x509_certificate(base64.b64decode(cert_b64))
        cert.public_key().verify(
            base64.b64decode(sig_b64),
            base64.b64decode(body_b64),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def _body_text(evidence: dict[str, Any]) -> str:
    body_b64 = evidence.get("attested_body")
    if not body_b64:
        return ""
    try:
        return base64.b64decode(body_b64).decode(errors="replace")
    except Exception:
        return ""


def _body_json(evidence: dict[str, Any]) -> dict[str, Any] | None:
    try:
        parsed = json.loads(_body_text(evidence))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _find_quote(evidence: dict[str, Any], body: dict[str, Any] | None = None) -> bytes | None:
    """Locate the TDX quote (base64 DER) in the evidence blob, either at the
    top level or inside the signed ``attested_body`` payload."""
    if body is None:
        body = _body_json(evidence) or {}
    for candidate in (evidence.get("quote"), body.get("quote")):
        if not candidate:
            continue
        try:
            return base64.b64decode(candidate)
        except Exception:
            continue
    return None


def _collect_gpu(evidence: dict[str, Any], body: dict[str, Any] | None = None) -> list[Any]:
    """Gather every NVIDIA evidence candidate from the evidence envelope.

    Production nests the GPU evidence in the *signed* payload as
    ``attested_body.evidence.nvtrust_evidence`` (a JSON-encoded list) and also
    mirrors it at the top level as ``gpu_evidence`` (a list of
    ``{arch, certificate, evidence}`` dicts).  The mock instead uses a flat
    list of ``{nonce, gpu_uuid}`` dicts.  Strings may be raw JSON lists, so
    decode those one level before use.
    """
    blobs: list[Any] = []

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped[:1] in {"[", "{"}:
                try:
                    add(json.loads(stripped))
                    return
                except Exception:
                    pass
            blobs.append(candidate)
        elif isinstance(candidate, list):
            for item in candidate:
                add(item)
        elif isinstance(candidate, dict):
            blobs.append(candidate)

    add(evidence.get("gpu_evidence") or evidence.get("gpu_evidences"))
    if body is None:
        body = _body_json(evidence) or {}
    inner = body.get("evidence") if isinstance(body.get("evidence"), dict) else {}
    add(inner.get("gpu_evidence") or inner.get("gpu_evidences"))
    add(inner.get("nvtrust_evidence"))
    add(body.get("gpu_evidence") or body.get("gpu_evidences"))
    return blobs


def _find_gpu(evidence: dict[str, Any]) -> list[Any]:
    """Back-compat alias returning the discovered GPU evidence candidates."""
    return _collect_gpu(evidence)


def _quote_binds(evidence: dict[str, Any], digest: str, body: dict[str, Any] | None = None) -> bool | None:
    """True when the TDX quote's report_data embeds the raw binding digest.

    Per chutes-api/docs/tee-verification.md the instance places the *32 raw
    bytes* of ``sha256(nonce + e2e_pubkey)`` at the start of the TD-REPORT's
    64-byte ``report_data`` (TD-REPORT offset 520-584).
    """
    quote = _find_quote(evidence, body=body)
    if quote is None:
        return None
    rd = _extract_report_data(quote)
    if rd is None:
        return None
    try:
        digest_bytes = bytes.fromhex(digest)
    except ValueError:
        return False
    return digest_bytes in rd[:64]


def _gpu_binds(evidence: dict[str, Any], digest: str, blobs: list[Any] | None = None) -> bool | None:
    """True when any NVIDIA evidence blob is bound to the digest.

    Two encodings are handled:

    * the mock (and any JSON carrier) embeds the digest's *hex string*;
    * real Chutes evidence wraps an opaque, base64-encoded NVIDIA report whose
      decoded bytes contain the digest's *32 raw bytes* (freshness nonce).
    """
    if blobs is None:
        blobs = _collect_gpu(evidence)
    if not blobs:
        return None
    try:
        digest_bytes = bytes.fromhex(digest)
    except ValueError:
        return False

    for blob in blobs:
        if isinstance(blob, dict):
            if digest in json.dumps(blob):
                return True
            blob = blob.get("evidence") or blob.get("gpu_evidence") or blob.get("nonce") or ""
        if not isinstance(blob, str):
            continue
        if digest in blob or digest_bytes.hex() in blob:
            return True
        try:
            decoded = base64.b64decode(blob, validate=False)
        except Exception:
            continue
        if decoded and digest_bytes in decoded:
            return True
    return False


def verify_evidence(
    evidence: dict[str, Any],
    e2e_pubkey: str,
    *,
    nonce: str | None = None,
    verify_quote: bool = False,
    verify_gpu: bool = False,
    check_signature: bool = True,
) -> VerificationResult:
    """Verify a single instance's evidence blob.

    ``verify_quote`` / ``verify_gpu`` require ``dcap-qvl`` / ``nv-attestation-sdk``
    respectively and, for a real deployment, outbound access to the Intel PCCS /
    NVIDIA verifier servers.
    """
    instance_id = str(evidence.get("instance_id", ""))
    nonce = nonce or secrets.token_hex(32)
    digest = _digest_of(nonce, e2e_pubkey)

    checks: dict[str, bool | None] = {}

    if check_signature:
        checks["key_possession"] = _signature_ok(evidence)

    # Decode the signed payload once and reuse it for every binding check
    # (it can be >100 KB of base64 in production, so re-parsing it per check is
    # measurable work on the cold path).
    body_text = _body_text(evidence)
    body: dict[str, Any] | None = None
    if body_text:
        try:
            parsed = json.loads(body_text)
            body = parsed if isinstance(parsed, dict) else None
        except Exception:
            body = None
    bound = _quote_binds(evidence, digest, body=body)
    gpu = _gpu_binds(evidence, digest, blobs=_collect_gpu(evidence, body=body))
    body_has = digest in body_text.lower() if body_text else None

    checks["quote_report_data"] = bound
    checks["gpu_nonce"] = gpu
    checks["body_contains_digest"] = body_has  # diagnostic only, never a veto

    binding_sources = [b for b in (bound, gpu, body_has) if b is not None]
    any_pass = any(b is True for b in binding_sources)
    checks["pubkey_bound"] = True if any_pass else (False if binding_sources else None)

    # Hardware roots (optional).  These raise (fail closed) rather than returning
    # a boolean; the caller records the per-instance reason.
    dcap_status: str | None = None
    gpu_info: dict[str, Any] | None = None
    if verify_quote:
        quote_b64 = evidence.get("quote") or (body or {}).get("quote")
        dcap_status = _verify_tdx_quote(quote_b64)
        checks["dcap_quote"] = True
    if verify_gpu:
        gpu_info = _verify_nvidia_gpu(_collect_gpu(evidence, body=body), digest)
        checks["nv_gpu"] = True

    # Fail closed:
    #  - key possession present but invalid        -> fail
    #  - a present binding source that does NOT bind -> fail
    #  - no binding evidence at all (a dishonest    -> fail
    #    relay that forwards nothing to verify)
    failed: set[str] = set()
    for name, value in (
        ("key_possession", checks.get("key_possession")),
        ("quote_report_data", bound),
        ("gpu_nonce", gpu),
    ):
        if value is False:
            failed.add(name)
    if not binding_sources:
        failed.add("pubkey_bound")

    verified = not failed
    detail = ""
    if not verified:
        detail = "failed checks: " + ", ".join(sorted(failed))
    return VerificationResult(
        instance_id=instance_id,
        verified=verified,
        checks=checks,
        nonce=nonce,
        digest=digest,
        detail=detail,
        dcap_status=dcap_status,
        gpu=gpu_info,
    )


def _run_async(coro: Any) -> Any:
    """Run *coro* from sync code, even if an event loop is already running here.

    The old ``asyncio.get_event_loop().run_until_complete`` breaks (and is
    deprecated) when called from a thread that already has a running loop, or
    from a bare thread with no loop.  This handles both cases.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _pccs_url() -> str | None:
    """Intel/Phala collateral (PCCS) base URL; ``None`` uses the dcap-qvl default."""
    return os.environ.get("CHUTES_DCAP_PCCS_URL") or None


def _nras_url() -> str:
    """NVIDIA Remote Attestation Service (NRAS) endpoint for GPU evidence."""
    return os.environ.get("CHUTES_NVIDIA_NRAS_URL") or DEFAULT_NRAS_URL


def _verify_tdx_quote(quote_b64: str | None) -> str:
    """Verify an Intel TDX quote with DCAP, returning the accepted TCB status.

    Raises :class:`AttestationError` with a distinct message for a missing SDK,
    a malformed/failed quote (``dcap-qvl`` raises ``ValueError``) and an
    unreachable collateral service (``dcap-qvl`` raises ``RuntimeError``), so an
    operator can tell "the hardware failed" from "PCCS is down".
    """
    if not quote_b64:
        raise AttestationError("verify_quote requested but evidence has no 'quote'")
    try:
        import dcap_qvl  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise AttestationError("verify_quote requires the optional 'dcap-qvl' package") from e

    pccs_url = _pccs_url()

    async def _verify() -> Any:
        # dcap-qvl defaults to Phala's PCCS when pccs_url is falsy.
        return await dcap_qvl.get_collateral_and_verify(
            base64.b64decode(quote_b64), pccs_url=pccs_url
        )

    try:
        report = _run_async(_verify())
    except ValueError as e:
        raise AttestationError(f"TDX quote verification failed: {e}") from e
    except RuntimeError as e:
        where = pccs_url or "the default PCCS (pccs.phala.network)"
        raise AttestationError(f"TDX collateral fetch failed from {where}: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise AttestationError(f"TDX quote verification failed: {e}") from e

    status = getattr(report, "status", None)
    if status is None:  # pragma: no cover - older/alternate bindings
        try:
            status = json.loads(report.to_json()).get("status")
        except Exception:
            status = None
    if status is None:
        raise AttestationError("TDX quote verification returned no TCB status")
    if str(status).strip().lower() not in _DCAP_OK_STATUSES:
        advisories = list(getattr(report, "advisory_ids", None) or [])
        extra = f" (advisories: {', '.join(advisories)})" if advisories else ""
        raise AttestationError(
            f"TDX quote TCB status is {status!r}, expected one of "
            f"{sorted(_DCAP_OK_STATUSES)}{extra}"
        )
    return str(status)


def _normalize_gpu_evidence(blobs: list[Any]) -> list[dict[str, Any]]:
    """Normalize discovered GPU evidence into the shape ``nv-attestation-sdk`` wants.

    Production delivers each GPU entry as ``{"arch", "certificate", "evidence"}``
    (possibly nested under the signed body's ``evidence``/``nvtrust_evidence`` or
    JSON-encoded).  The REMOTE SDK verifier
    (``nv_attestation_sdk.gpu.attest_gpu_remote.build_payload``) requires exactly
    those three fields and a single consistent ``arch`` across the list.

    Anything else (e.g. the mock's ``{nonce, gpu_uuid}``) is dropped, extra keys
    are stripped, and duplicates are removed.  Returns ``[]`` when no usable
    entry exists so the caller can fail closed.
    """
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        arch = blob.get("arch")
        certificate = blob.get("certificate")
        evidence = blob.get("evidence")
        if isinstance(certificate, (dict, list)):
            certificate = json.dumps(certificate)
        if isinstance(evidence, (dict, list)):
            evidence = json.dumps(evidence)
        if not (
            isinstance(arch, str)
            and arch.strip()
            and isinstance(certificate, str)
            and certificate
            and isinstance(evidence, str)
            and evidence
        ):
            continue
        key = (arch, certificate, evidence)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"arch": arch, "evidence": evidence, "certificate": certificate})

    if not normalized:
        return []
    arches = sorted({entry["arch"] for entry in normalized})
    if len(arches) > 1:
        raise AttestationError(
            "GPU evidence mixes architectures, which the NVIDIA SDK cannot verify "
            f"in one call: {', '.join(arches)}"
        )
    return normalized


def _verify_nvidia_gpu(gpu_evidence: Any, digest: str) -> dict[str, Any]:
    """Verify NVIDIA GPU evidence against NRAS, returning a small verdict dict.

    ``gpu_evidence`` is any value :func:`_collect_gpu` produced (list of dicts,
    JSON strings, ...).  Every GPU must pass; the SDK's remote verifier returns a
    single overall verdict, so failure is reported with the arch/count and the
    NRAS URL used.
    """
    if not gpu_evidence:
        raise AttestationError("verify_gpu requested but evidence has no 'gpu_evidence'")
    try:
        from nv_attestation_sdk.attestation import (  # type: ignore
            Attestation,
            Devices,
            Environment,
        )
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise AttestationError("verify_gpu requires the optional 'nv-attestation-sdk' package") from e

    blobs = gpu_evidence if isinstance(gpu_evidence, list) else [gpu_evidence]
    normalized = _normalize_gpu_evidence(blobs)
    if not normalized:
        raise AttestationError(
            "verify_gpu requested but no usable NVIDIA GPU evidence was found "
            "(expected {'arch', 'certificate', 'evidence'} entries)"
        )

    url = _nras_url()
    with _NVIDIA_LOCK:
        # The SDK object is a singleton: clear the verifier list/nonce that a
        # previous (concurrent) verification may have left behind.
        client = Attestation()
        client.reset()
        client.add_verifier(Devices.GPU, Environment.REMOTE, url, "")
        client.set_nonce(digest)
        try:
            ok = client.attest(normalized)
        except Exception as e:  # noqa: BLE001
            raise AttestationError(f"NVIDIA GPU attestation raised: {e}") from e

    if not ok:
        raise AttestationError(
            f"NVIDIA GPU attestation did not pass for {len(normalized)} GPU(s) "
            f"(arch={normalized[0]['arch']}); NRAS={url}"
        )
    return {"arch": normalized[0]["arch"], "gpus": len(normalized), "nras_url": url}


# ---------------------------------------------------------------------------
# Network-facing helpers (discovery + evidence fetch)
# ---------------------------------------------------------------------------
#
# A single pooled ``httpx.Client`` is shared by every attestation call so the
# TCP/TLS sessions to api.chutes.ai (or llm.chutes.ai) are kept alive across
# requests instead of being re-established on each one.  Without it the
# per-request model->chute lookup alone would pay a fresh TLS handshake.


_http_lock = threading.Lock()
_shared_http: httpx.Client | None = None


def _client() -> httpx.Client:
    global _shared_http
    with _http_lock:
        if _shared_http is None:
            client = httpx.Client(
                timeout=30,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            atexit.register(client.close)
            _shared_http = client
        return _shared_http


def close_shared_http() -> None:
    """Close the shared attestation client (mainly for tests)."""
    global _shared_http
    with _http_lock:
        if _shared_http is not None:
            _shared_http.close()
            _shared_http = None


def fetch_model_map(api_key: str, models_base: str = DEFAULT_MODELS_BASE, http: httpx.Client | None = None) -> dict[str, str]:
    """id -> chute_id from the public model listing (same call the E2EE
    transport's discovery makes)."""
    client = http or _client()
    resp = client.get(
        f"{models_base.rstrip('/')}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return {
        e["id"]: e["chute_id"]
        for e in resp.json().get("data", [])
        if e.get("id") and e.get("chute_id")
    }


def fetch_instances(
    api_key: str,
    chute_id: str,
    api_base: str = DEFAULT_API_BASE,
    http: httpx.Client | None = None,
) -> list[InstanceInfo]:
    """List E2EE instances for a chute (mirrors DiscoveryManager._fetch_instances)."""
    client = http or _client()
    resp = client.get(
        f"{api_base.rstrip('/')}/e2e/instances/{chute_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        InstanceInfo(
            instance_id=inst["instance_id"],
            e2e_pubkey=inst["e2e_pubkey"],
            nonces=list(inst.get("nonces") or []),
        )
        for inst in data.get("instances", [])
    ]


def fetch_chute_evidence(
    api_key: str,
    chute_id: str,
    nonce: str,
    api_base: str = DEFAULT_API_BASE,
    http: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fetch TEE evidence for *all* live instances of a chute in one call.

    GET /chutes/{chute_id}/evidence?nonce={nonce}  (see tee-verification.md)
    """
    client = http or _client()
    resp = client.get(
        f"{api_base.rstrip('/')}/chutes/{chute_id}/evidence",
        params={"nonce": nonce},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json().get("evidence", [])


def fetch_evidence(
    api_key: str,
    instance_id: str,
    nonce: str,
    api_base: str = DEFAULT_API_BASE,
    evidence_path: str = DEFAULT_EVIDENCE_PATH,
    http: httpx.Client | None = None,
) -> dict[str, Any]:
    client = http or _client()
    path = evidence_path.format(instance_id=instance_id)
    resp = client.get(
        f"{api_base.rstrip('/')}{path}",
        params={"nonce": nonce},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# High-level gate used by the transport wrapper
# ---------------------------------------------------------------------------

_VERIFY_CACHE_TTL = int(os.environ.get("CHUTES_ATTESTATION_TTL", "300"))
# Failed attestations are cached too (shorter): without this, a chute that cannot
# be verified would make *every* request pay the full evidence fetch, turning a
# bad chute into a self-inflicted DoS on the proxy.
_VERIFY_FAILURE_TTL = int(os.environ.get("CHUTES_ATTESTATION_FAILURE_TTL", "30"))
_verify_lock = threading.Lock()
# key: (chute_id, verify_quote, verify_gpu, check_signature) -> (at, verified ids, error|None).
# The verification mode is part of the key: a chute verified without quote/GPU
# checks must not satisfy a later request that *does* require them.
_verify_cache: dict[tuple[str, bool, bool, bool], tuple[float, list[str], str | None]] = {}
# Single-flight set: cache_key -> event signalled when the leader finishes, so N
# concurrent cold-cache requests trigger one evidence fetch, not N.
_inflight: dict[tuple[str, bool, bool, bool], threading.Event] = {}

# model_name -> chute_id is stable, so cache it too; otherwise every request in
# attestation mode would re-hit /v1/models before it can even look up the
# verification cache (see verify_chute).
_MODEL_MAP_TTL = int(os.environ.get("CHUTES_ATTESTATION_MODEL_MAP_TTL", "300"))
_model_map_lock = threading.Lock()
_model_map_cache: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}


def _cached_model_map(api_key: str, models_base: str) -> dict[str, str]:
    key = (api_key, models_base)
    now = time.time()
    with _model_map_lock:
        cached = _model_map_cache.get(key)
        if cached and now - cached[0] < _MODEL_MAP_TTL:
            return cached[1]
    mapping = fetch_model_map(api_key, models_base=models_base)
    with _model_map_lock:
        _model_map_cache[key] = (time.time(), mapping)
    return mapping


_CHECK_LABELS = {
    "key_possession": "signature",
    "quote_report_data": "tdx_binding",
    "gpu_nonce": "gpu_binding",
    "body_contains_digest": "body_digest",
    "pubkey_bound": "pubkey_bound",
    "dcap_quote": "dcap_quote",
    "nv_gpu": "nv_gpu",
}


def _format_checks(checks: dict[str, bool | None]) -> str:
    """Render the per-check verdicts as a compact ``name=ok|FAIL|n/a`` list."""
    out: list[str] = []
    for name, value in checks.items():
        mark = "ok" if value is True else ("FAIL" if value is False else "n/a")
        out.append(f"{_CHECK_LABELS.get(name, name)}={mark}")
    return ", ".join(out)


def _describe_failure(result: VerificationResult) -> str:
    reason = result.detail or "no binding check passed"
    return f"{reason} [{_format_checks(result.checks)}]"


@dataclass
class ChuteVerificationReport:
    """Rich per-instance attestation result (used by the CLI and diagnostics)."""

    chute_id: str
    model: str
    api_base: str
    instances: int
    evidences: int
    unmatched: int
    verified: list[str] = field(default_factory=list)
    results: list[VerificationResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Fail closed: a chute is trusted only when every listed instance
        # verified and nothing failed.
        return bool(self.verified) and not self.failures


def _verify_instances(
    api_key: str,
    chute_id: str,
    api_base: str,
    model: str,
    *,
    verify_quote: bool,
    verify_gpu: bool,
    check_signature: bool,
) -> ChuteVerificationReport:
    """Fetch and verify evidence for *every* live instance of ``chute_id``.

    Fail-closed: every instance the transport could select from
    ``/e2e/instances`` must verify.  The transport picks an instance from its
    own discovery cache, so trusting a partially-verified chute would still let
    a request land on the unverified instance.
    """
    instances = fetch_instances(api_key, chute_id, api_base=api_base)
    pubkeys = {inst.instance_id: inst.e2e_pubkey for inst in instances}
    if not pubkeys:
        raise AttestationError(
            f"attestation failed for chute {chute_id} (model {model!r}): "
            f"no E2EE instances were listed for this chute at {api_base} — it may be "
            "down, not TEE-capable, or the chute id may be stale."
        )

    nonce = secrets.token_hex(32)
    evidences = fetch_chute_evidence(api_key, chute_id, nonce, api_base=api_base)

    report = ChuteVerificationReport(
        chute_id=chute_id,
        model=model,
        api_base=api_base,
        instances=len(pubkeys),
        evidences=len(evidences),
        unmatched=0,
    )
    seen: set[str] = set()
    for evidence in evidences:
        instance_id = evidence.get("instance_id")
        pubkey = pubkeys.get(instance_id)
        if pubkey is None:
            report.unmatched += 1  # evidence for an instance we cannot map to a key
            continue
        seen.add(instance_id)
        try:
            result = verify_evidence(
                evidence, pubkey, nonce=nonce,
                verify_quote=verify_quote, verify_gpu=verify_gpu,
                check_signature=check_signature,
            )
        except AttestationError as exc:
            # Hardware verification raises (missing SDK, failed DCAP/NVIDIA).
            # Record it per instance instead of aborting, so one bad instance
            # does not hide the verdict for the rest.
            report.failures.append(f"{instance_id}: {exc}")
            continue
        report.results.append(result)
        if result.verified:
            report.verified.append(instance_id)
        else:
            report.failures.append(
                f"{instance_id or '<unknown instance>'}: {_describe_failure(result)}"
            )

    for instance_id in sorted(iid for iid in pubkeys if iid not in seen):
        report.failures.append(
            f"{instance_id}: no attestation evidence returned for a listed E2EE instance"
        )
    return report


def _format_chute_failure(report: ChuteVerificationReport, models_base: str) -> str:
    lines = [
        f"attestation failed for chute {report.chute_id} (model {report.model!r}) — request refused.",
        f"  instances listed: {report.instances}, evidence blobs: {report.evidences}, "
        f"failed: {len(report.failures)}, unmatched: {report.unmatched}",
    ]
    if report.failures:
        lines.append("  per-instance reason(s):")
        lines += [f"    - {line}" for line in report.failures]
    if not report.evidences:
        lines.append("  the API returned no evidence blobs for this chute")
    if report.unmatched:
        lines.append(
            "  note: evidence returned for instances with no matching pubkey was ignored "
            "(stale /v1/models listing)"
        )
    lines.append(f"  api_base={report.api_base}  models_base={models_base}")
    return "\n".join(lines)


def _do_verify_chute(
    api_key: str,
    model: str,
    chute_id: str,
    api_base: str,
    models_base: str,
    *,
    verify_quote: bool,
    verify_gpu: bool,
    check_signature: bool,
) -> list[str]:
    report = _verify_instances(
        api_key, chute_id, api_base, model,
        verify_quote=verify_quote, verify_gpu=verify_gpu,
        check_signature=check_signature,
    )
    if not report.ok:
        raise AttestationError(_format_chute_failure(report, models_base))
    return report.verified


def verify_chute_report(
    api_key: str,
    model: str,
    *,
    api_base: str | None = None,
    models_base: str | None = None,
    verify_quote: bool = False,
    verify_gpu: bool = False,
    check_signature: bool = True,
) -> ChuteVerificationReport:
    """Like :func:`verify_chute` but returns the full per-instance report.

    Never consults the cache; intended for the CLI / diagnostics where an
    operator wants the per-instance verdicts (including the DCAP TCB status and
    NVIDIA arch/count) rather than a boolean.
    """
    api_base = api_base or os.environ.get("CHUTES_E2EE_API_BASE") or DEFAULT_API_BASE
    models_base = models_base or os.environ.get("CHUTES_E2EE_MODELS_BASE") or DEFAULT_MODELS_BASE

    if "-" in model and len(model) == 36 and model.replace("-", "").isalnum():
        chute_id = model
    else:
        chute_id = _cached_model_map(api_key, models_base).get(model)
        if chute_id is None:
            raise AttestationError(f"model {model!r} not found in {models_base}/v1/models")

    return _verify_instances(
        api_key, chute_id, api_base, model,
        verify_quote=verify_quote, verify_gpu=verify_gpu,
        check_signature=check_signature,
    )


def verify_chute(
    api_key: str,
    model: str,
    *,
    api_base: str | None = None,
    models_base: str | None = None,
    verify_quote: bool = False,
    verify_gpu: bool = False,
    check_signature: bool = True,
    force: bool = False,
) -> list[str]:
    """Verify every live instance of a chute before we encrypt to any of them.

    Returns the list of verified ``instance_id``s and raises
    :class:`AttestationError` unless *every* instance listed by
    ``/e2e/instances`` verifies.  Results (success and failure) are cached per
    chute + verification mode for ``CHUTES_ATTESTATION_TTL`` /
    ``CHUTES_ATTESTATION_FAILURE_TTL`` seconds; concurrent cold-cache calls are
    coalesced into a single evidence fetch.
    """
    api_base = api_base or os.environ.get("CHUTES_E2EE_API_BASE") or DEFAULT_API_BASE
    models_base = models_base or os.environ.get("CHUTES_E2EE_MODELS_BASE") or DEFAULT_MODELS_BASE

    # model may already be a chute UUID (transport accepts uuids directly).
    if "-" in model and len(model) == 36 and model.replace("-", "").isalnum():
        chute_id = model
    else:
        model_map = _cached_model_map(api_key, models_base)
        chute_id = model_map.get(model)
        if chute_id is None:
            raise AttestationError(f"model {model!r} not found in {models_base}/v1/models")

    cache_key = (chute_id, verify_quote, verify_gpu, check_signature)

    def _fresh() -> tuple[list[str], str | None] | None:
        cached = _verify_cache.get(cache_key)
        if cached is None or force:
            return None
        at, verified_ids, error = cached
        ttl = _VERIFY_FAILURE_TTL if error else _VERIFY_CACHE_TTL
        if time.time() - at >= ttl:
            return None
        return verified_ids, error

    with _verify_lock:
        hit = _fresh()
    if hit is not None:
        verified_ids, error = hit
        if error:
            raise AttestationError(error)
        return verified_ids

    # Single-flight: the first caller fetches; the rest wait for its outcome and
    # then re-check the cache (or, if the leader died without caching, try too).
    with _verify_lock:
        event = _inflight.get(cache_key)
        leader = event is None
        if leader:
            event = threading.Event()
            _inflight[cache_key] = event

    try:
        if not leader:
            event.wait(timeout=120)
            with _verify_lock:
                hit = _fresh()
            if hit is not None:
                verified_ids, error = hit
                if error:
                    raise AttestationError(error)
                return verified_ids

        try:
            verified = _do_verify_chute(
                api_key, model, chute_id, api_base, models_base,
                verify_quote=verify_quote, verify_gpu=verify_gpu,
                check_signature=check_signature,
            )
        except AttestationError as exc:
            with _verify_lock:
                _verify_cache[cache_key] = (time.time(), [], str(exc))
            raise
        with _verify_lock:
            _verify_cache[cache_key] = (time.time(), verified, None)
        return verified
    finally:
        if leader:
            with _verify_lock:
                _inflight.pop(cache_key, None)
            event.set()



def verify_model(api_key: str, model: str, **kwargs: Any) -> None:
    """Convenience wrapper used by the transport (raises on failure)."""
    verify_chute(api_key, model, **kwargs)
