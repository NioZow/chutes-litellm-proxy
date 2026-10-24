"""Chutes TEE / GPU attestation: network discovery, caching and the fail-closed gate.

The cryptographic verification lives in :mod:`chutes_litellm.tee` (a port of the
authoritative client verifier in ``chutesai/chutes-api``).  This module is the
network/caching/gate layer used by the E2EE transports:

  * fetch the instance list and per-chute evidence from Chutes;
  * delegate each envelope to :func:`chutes_litellm.tee.verify_evidence`, which
    (mandatorily) DCAP-verifies the TDX quote, binds it to
    ``sha256(nonce + e2e_pubkey)`` and to the certificate public key, matches
    a published Chutes measurement, and verifies NVIDIA GPU evidence against
    NRAS and the measurement's GPU count;
  * cache the verified ``instance_id -> e2e_pubkey`` mapping and expose it to the
    transport's instance filter so selection cannot diverge from verification.

Attestation is unconditional: there is no software-only "verified" path and no
per-quote toggle.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import endpoints
from .tee.errors import AttestationError, MeasurementMismatchError
from .tee.quote import verify_tdx_quote
from .tee.verifier import DEFAULT_NRAS_URL, VerifiedEvidence
from .tee.verifier import verify_evidence as _verify_evidence_async

__all__ = [
    "AttestationError",
    "ChuteEvidence",
    "ChuteVerificationReport",
    "InstanceInfo",
    "MeasurementMismatchError",
    "VerificationResult",
    "close_shared_http",
    "fetch_chute_evidence",
    "fetch_evidence",
    "fetch_instances",
    "fetch_model_map",
    "verified_ids",
    "verify_chute",
    "verify_chute_report",
    "verify_evidence",
    "verify_model",
]

DEFAULT_EVIDENCE_PATH = "/instances/{instance_id}/evidence"


@dataclass
class InstanceInfo:
    instance_id: str
    e2e_pubkey: str  # base64 ML-KEM-768 public key
    nonces: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Compatibility view of a successful :class:`VerifiedEvidence`."""

    instance_id: str
    verified: bool
    checks: dict[str, bool]
    nonce: str
    digest: str
    detail: str = ""
    dcap_status: str | None = None
    gpu: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChuteEvidence:
    evidence: list[dict[str, Any]]
    failed_instance_ids: list[str]


def _digest_of(nonce: str, e2e_pubkey: str) -> str:
    """Freshness/binding digest: sha256(nonce + e2e_pubkey), hex-encoded."""
    return hashlib.sha256((nonce + e2e_pubkey).encode()).hexdigest().lower()


def _run_async(coro: Any) -> Any:
    """Run *coro* from sync code, even if an event loop is already running here."""
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


def verify_evidence(
    evidence: dict[str, Any],
    e2e_pubkey: str,
    *,
    nonce: str,
    pccs_url: str | None = None,
    nras_url: str | None = None,
) -> VerificationResult:
    """Verify one instance's evidence (DCAP + measurement + GPU).

    Measurements are always fetched from the fixed Chutes API
    (``endpoints.API_BASE``).  Fails closed: every check is mandatory and any
    failure raises :class:`AttestationError`.
    """
    pccs_url = pccs_url if pccs_url is not None else _pccs_url()
    nras_url = nras_url if nras_url is not None else _nras_url()

    verified: VerifiedEvidence = _run_async(
        _verify_evidence_async(
            evidence,
            e2e_pubkey,
            nonce=nonce,
            pccs_url=pccs_url,
            nras_url=nras_url,
        )
    )
    return VerificationResult(
        instance_id=verified.instance_id,
        verified=True,
        checks=dict(verified.checks),
        nonce=verified.nonce,
        digest=verified.digest,
        dcap_status=verified.dcap_status,
        gpu=verified.gpu,
    )


def _verify_tdx_quote(quote_b64: str | None, *, pccs_url: str | None = None) -> str:
    """Compatibility wrapper: DCAP-verify a base64 quote, return its TCB status."""
    if not quote_b64:
        raise AttestationError("no TDX quote was provided")
    try:
        raw = base64.b64decode(quote_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise AttestationError(f"TDX quote is not valid base64: {exc}") from exc
    result = _run_async(verify_tdx_quote(raw, pccs_url=pccs_url if pccs_url is not None else _pccs_url()))
    return result.status


# ---------------------------------------------------------------------------
# Network-facing helpers (discovery + evidence fetch)
# ---------------------------------------------------------------------------

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


def fetch_model_map(
    api_key: str,
    http: httpx.Client | None = None,
) -> dict[str, str]:
    """id -> chute_id from the public model listing (``endpoints.MODELS_BASE``)."""
    client = http or _client()
    resp = client.get(
        f"{endpoints.MODELS_BASE.rstrip('/')}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return {e["id"]: e["chute_id"] for e in resp.json().get("data", []) if e.get("id") and e.get("chute_id")}


def fetch_instances(
    api_key: str,
    chute_id: str,
    http: httpx.Client | None = None,
) -> list[InstanceInfo]:
    """List E2EE instances for a chute (mirrors DiscoveryManager._fetch_instances).

    Always queries the fixed ``endpoints.API_BASE``.
    """
    client = http or _client()
    resp = client.get(
        f"{endpoints.API_BASE.rstrip('/')}/e2e/instances/{chute_id}",
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
    http: httpx.Client | None = None,
) -> ChuteEvidence:
    """Fetch TEE evidence for all live instances plus any failed instance ids.

    Always queries the fixed ``endpoints.API_BASE``.
    """
    client = http or _client()
    resp = client.get(
        f"{endpoints.API_BASE.rstrip('/')}/chutes/{chute_id}/evidence",
        params={"nonce": nonce},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()
    return ChuteEvidence(
        evidence=list(data.get("evidence") or []),
        failed_instance_ids=[str(i) for i in (data.get("failed_instance_ids") or [])],
    )


def fetch_evidence(
    api_key: str,
    instance_id: str,
    nonce: str,
    evidence_path: str = DEFAULT_EVIDENCE_PATH,
    http: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch one instance's raw evidence from the fixed ``endpoints.API_BASE``."""
    client = http or _client()
    path = evidence_path.format(instance_id=instance_id)
    resp = client.get(
        f"{endpoints.API_BASE.rstrip('/')}{path}",
        params={"nonce": nonce},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# High-level gate used by the transport wrapper
# ---------------------------------------------------------------------------

_VERIFY_CACHE_TTL = int(os.environ.get("CHUTES_ATTESTATION_TTL", "300"))
_VERIFY_FAILURE_TTL = int(os.environ.get("CHUTES_ATTESTATION_FAILURE_TTL", "30"))
_verify_lock = threading.Lock()
# key: chute_id -> (at, verified {id: pubkey}, error|None)
_verify_cache: dict[str, tuple[float, dict[str, str], str | None]] = {}
_inflight: dict[str, threading.Event] = {}

_MODEL_MAP_TTL = int(os.environ.get("CHUTES_ATTESTATION_MODEL_MAP_TTL", "300"))
_model_map_lock = threading.Lock()
_model_map_cache: dict[str, tuple[float, dict[str, str]]] = {}


def _cached_model_map(api_key: str) -> dict[str, str]:
    """Cached ``id -> chute_id`` map; the API key is the cache key."""
    models_base = endpoints.MODELS_BASE
    now = time.time()
    with _model_map_lock:
        cached = _model_map_cache.get(api_key)
        if cached and now - cached[0] < _MODEL_MAP_TTL:
            return cached[1]
    from chutes_litellm import get_logger

    get_logger("chutes_litellm.attestation").info("fetch_model_map: models_base=%s", models_base)
    mapping = fetch_model_map(api_key)
    with _model_map_lock:
        _model_map_cache[api_key] = (time.time(), mapping)
    return mapping


_CHECK_LABELS = {
    "signature": "signature",
    "quote_report_data": "tdx_binding",
    "certificate_binding": "cert_binding",
    "measurement": "measurement",
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
    verified: dict[str, str] = field(default_factory=dict)
    results: list[VerificationResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Fail closed: a chute is trusted only when every listed instance
        # verified and nothing failed.
        return bool(self.verified) and not self.failures


def _coerce_evidence(value: Any) -> ChuteEvidence:
    if isinstance(value, ChuteEvidence):
        return value
    return ChuteEvidence(evidence=list(value or []), failed_instance_ids=[])


def _verify_instances(
    api_key: str,
    chute_id: str,
    model: str,
) -> ChuteVerificationReport:
    """Fetch and verify evidence for every live instance of ``chute_id``.

    Network calls always use the fixed ``endpoints.API_BASE``.
    """
    from chutes_litellm import get_logger

    logger = get_logger("chutes_litellm.attestation")
    logger.info("_verify_instances: model=%s chute_id=%s", model, chute_id)
    instances = fetch_instances(api_key, chute_id)
    pubkeys = {inst.instance_id: inst.e2e_pubkey for inst in instances}
    if not pubkeys:
        logger.warning(
            "_verify_instances: model=%s chute_id=%s no instances listed at api=%s",
            model,
            chute_id,
            endpoints.API_BASE,
        )
        raise AttestationError(
            f"attestation failed for chute {chute_id} (model {model!r}): "
            f"no E2EE instances were listed for this chute at {endpoints.API_BASE} — it may be "
            "down, not TEE-capable, or the chute id may be stale."
        )

    nonce = secrets.token_hex(32)
    logger.info(
        "_verify_instances: model=%s chute_id=%s instances=%s nonce=%s",
        model,
        chute_id,
        len(pubkeys),
        nonce,
    )
    fetched = _coerce_evidence(fetch_chute_evidence(api_key, chute_id, nonce))

    report = ChuteVerificationReport(
        chute_id=chute_id,
        model=model,
        api_base=endpoints.API_BASE,
        instances=len(pubkeys),
        evidences=len(fetched.evidence),
        unmatched=0,
    )
    failed_ids = set(fetched.failed_instance_ids)
    seen: set[str] = set()
    for evidence in fetched.evidence:
        instance_id = evidence.get("instance_id")
        pubkey = pubkeys.get(instance_id)
        if pubkey is None:
            report.unmatched += 1
            continue
        seen.add(instance_id)
        if instance_id in failed_ids:
            report.failures.append(
                f"{instance_id}: evidence retrieval failed (API reported the instance in failed_instance_ids)"
            )
            continue
        try:
            result = verify_evidence(
                evidence,
                pubkey,
                nonce=nonce,
            )
        except AttestationError as exc:
            # Record per instance instead of aborting so one bad instance does
            # not hide the verdict for the rest.
            report.failures.append(f"{instance_id}: {exc}")
            continue
        report.results.append(result)
        if result.verified:
            report.verified[instance_id] = pubkey
        else:
            report.failures.append(f"{instance_id or '<unknown instance>'}: {_describe_failure(result)}")

    for instance_id in sorted(iid for iid in pubkeys if iid not in seen):
        if instance_id in failed_ids:
            reason = "evidence retrieval failed (API reported failed_instance_ids)"
        else:
            reason = "no attestation evidence returned for a listed E2EE instance"
        report.failures.append(f"{instance_id}: {reason}")

    status = "VERIFIED" if report.ok else "FAILED"
    logger.info(
        "_verify_instances: model=%s chute_id=%s status=%s verified=%s/%s failed=%s unmatched=%s",
        model,
        chute_id,
        status,
        len(report.verified),
        len(pubkeys),
        len(report.failures),
        report.unmatched,
    )
    return report


def _format_chute_failure(report: ChuteVerificationReport) -> str:
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
            "  note: evidence returned for instances with no matching pubkey was ignored (stale /v1/models listing)"
        )
    lines.append(f"  api_base={report.api_base}  models_base={endpoints.MODELS_BASE}")
    return "\n".join(lines)


def _do_verify_chute(
    api_key: str,
    model: str,
    chute_id: str,
) -> dict[str, str]:
    report = _verify_instances(api_key, chute_id, model)
    if not report.ok:
        raise AttestationError(_format_chute_failure(report))
    return report.verified


def _resolve_chute_id(api_key: str, model: str) -> str:
    """Resolve a model id (or pass through a chute UUID) to a chute id."""
    # model may already be a chute UUID (transport accepts uuids directly).
    if "-" in model and len(model) == 36 and model.replace("-", "").isalnum():
        return model
    chute_id = _cached_model_map(api_key).get(model)
    if chute_id is None:
        raise AttestationError(f"model {model!r} not found in {endpoints.MODELS_BASE}/v1/models")
    return chute_id


def verify_chute_report(
    api_key: str,
    model: str,
) -> ChuteVerificationReport:
    """Like :func:`verify_chute` but returns the full per-instance report."""
    chute_id = _resolve_chute_id(api_key, model)
    return _verify_instances(api_key, chute_id, model)


def verify_chute(
    api_key: str,
    model: str,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Verify every live instance of a chute; return ``{instance_id: e2e_pubkey}``.

    The model listing and evidence endpoints are fixed (``endpoints.MODELS_BASE``
    and ``endpoints.API_BASE``).  Raises :class:`AttestationError` unless *every*
    instance listed by ``/e2e/instances`` verifies.  Results (success and
    failure) are cached per chute.
    """
    from chutes_litellm import get_logger

    logger = get_logger("chutes_litellm.attestation")

    chute_id = _resolve_chute_id(api_key, model)

    cache_key = chute_id

    def _fresh() -> tuple[dict[str, str], str | None] | None:
        cached = _verify_cache.get(cache_key)
        if cached is None or force:
            return None
        at, verified, error = cached
        ttl = _VERIFY_FAILURE_TTL if error else _VERIFY_CACHE_TTL
        if time.time() - at >= ttl:
            return None
        return verified, error

    with _verify_lock:
        hit = _fresh()
    if hit is not None:
        verified, error = hit
        if error:
            logger.info("verify_chute: model=%s chute_id=%s cache=HIT (failure)", model, chute_id)
            raise AttestationError(error)
        logger.info(
            "verify_chute: model=%s chute_id=%s cache=HIT verified=%s",
            model,
            chute_id,
            len(verified),
        )
        return verified

    logger.info("verify_chute: model=%s chute_id=%s cache=MISS", model, chute_id)

    with _verify_lock:
        event = _inflight.get(cache_key)
        leader = event is None
        if leader:
            event = threading.Event()
            _inflight[cache_key] = event

    try:
        if not leader:
            logger.info("verify_chute: model=%s chute_id=%s waiting for leader", model, chute_id)
            event.wait(timeout=120)
            with _verify_lock:
                hit = _fresh()
            if hit is not None:
                verified, error = hit
                if error:
                    raise AttestationError(error)
                return verified

        try:
            verified = _do_verify_chute(api_key, model, chute_id)
        except AttestationError as exc:
            with _verify_lock:
                _verify_cache[cache_key] = (time.time(), {}, str(exc))
            logger.warning("verify_chute: model=%s chute_id=%s FAILED %s", model, chute_id, exc)
            raise
        with _verify_lock:
            _verify_cache[cache_key] = (time.time(), verified, None)
        logger.info(
            "verify_chute: model=%s chute_id=%s VERIFIED instances=%s",
            model,
            chute_id,
            len(verified),
        )
        return verified
    finally:
        if leader:
            with _verify_lock:
                _inflight.pop(cache_key, None)
            event.set()


def verified_ids(verified: dict[str, str]) -> list[str]:
    """Return the verified instance ids from a :func:`verify_chute` result."""
    return list(verified)


def verify_model(api_key: str, model: str, **kwargs: Any) -> None:
    """Convenience wrapper used by the transport (raises on failure)."""
    verify_chute(api_key, model, **kwargs)
