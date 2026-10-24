"""End-to-end attestation verification (mandatory, fail closed).

Order (any failure raises :class:`AttestationError`):

1. parse + RSA-verify the signed evidence body;
2. compute ``sha256(nonce + e2e_pubkey)``;
3. DCAP-verify the TDX quote (mandatory);
4. require ``report_data[:32] == digest`` and
   ``report_data[32:64] == sha256(cert_pubkey_DER)`` (exact, no substring);
5. require the quote to match a published Chutes measurement;
6. verify NVIDIA GPU evidence and its count against that measurement.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

from .errors import AttestationError, EvidenceError
from .evidence import (
    Evidence,
    cert_public_key_hash,
    extract_from_body,
    parse_evidence,
    verify_signed_body,
)
from .measurements import TeeMeasurement, get_measurements, match_measurement
from .quote import TdxQuote, verify_tdx_quote

DEFAULT_NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"

# The NVIDIA SDK's Attestation object is a process-wide singleton with mutable
# class state (verifier list + nonce), so serialise access to it.
_NVIDIA_LOCK = threading.Lock()


@dataclass
class VerifiedEvidence:
    instance_id: str
    nonce: str
    digest: str
    measurement: TeeMeasurement
    dcap_status: str
    gpu: dict[str, Any] | None
    checks: dict[str, bool]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_gpu_evidence(blobs: list[Any]) -> list[dict[str, str]]:
    """Keep only ``{arch, certificate, evidence}`` entries for the NVIDIA SDK."""
    normalized: list[dict[str, str]] = []
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
            f"GPU evidence mixes architectures, which the NVIDIA SDK cannot verify in one call: {', '.join(arches)}"
        )
    return normalized


def _nras_url(nras_url: str | None = None) -> str:
    return nras_url or os.environ.get("CHUTES_NVIDIA_NRAS_URL") or DEFAULT_NRAS_URL


def _verify_nvidia_gpu(
    gpu_evidence: Any,
    digest: str,
    *,
    expected_count: int,
    nras_url: str | None = None,
) -> dict[str, Any]:
    """Verify NVIDIA GPU evidence against NRAS and require the expected count."""
    if not gpu_evidence:
        raise AttestationError("GPU verification requested but evidence has no GPU data")
    try:
        from nv_attestation_sdk.attestation import (  # type: ignore
            Attestation,
            Devices,
            Environment,
        )
    except ImportError as exc:  # pragma: no cover - required dependency
        raise AttestationError("GPU verification requires the 'nv-attestation-sdk' package") from exc

    blobs = gpu_evidence if isinstance(gpu_evidence, list) else [gpu_evidence]
    normalized = _normalize_gpu_evidence(blobs)
    if not normalized:
        raise AttestationError(
            "GPU verification requested but no usable NVIDIA evidence was found "
            "(expected {'arch', 'certificate', 'evidence'} entries)"
        )
    if len(normalized) != expected_count:
        raise AttestationError(
            "GPU evidence count does not match the attested measurement: "
            f"got {len(normalized)}, expected {expected_count}"
        )

    url = _nras_url(nras_url)
    with _NVIDIA_LOCK:
        client = Attestation()
        client.reset()
        client.add_verifier(Devices.GPU, Environment.REMOTE, url, "")
        client.set_nonce(digest)
        try:
            ok = client.attest(normalized)
        except Exception as exc:  # noqa: BLE001
            raise AttestationError(f"NVIDIA GPU attestation raised: {exc}") from exc

    if not ok:
        raise AttestationError(
            f"NVIDIA GPU attestation did not pass for {len(normalized)} GPU(s) "
            f"(arch={normalized[0]['arch']}); NRAS={url}"
        )
    return {
        "arch": normalized[0]["arch"],
        "gpus": len(normalized),
        "expected": expected_count,
        "nras_url": url,
    }


def _decode_quote(b64_value: str) -> bytes:
    try:
        return base64.b64decode(b64_value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError("signed body TDX quote is not valid base64") from exc


async def verify_evidence(
    envelope: dict[str, Any],
    e2e_pubkey: str,
    *,
    nonce: str,
    pccs_url: str | None = None,
    nras_url: str | None = None,
) -> VerifiedEvidence:
    """Verify one instance's evidence.  Raises on any failure (fail closed).

    Published measurements are always fetched from the fixed Chutes API
    (``endpoints.API_BASE``) unless ``CHUTES_TEE_MEASUREMENTS_FILE`` is set.
    """
    ev: Evidence = parse_evidence(envelope)
    if not ev.instance_id:
        raise EvidenceError("evidence has no instance_id")

    body = verify_signed_body(ev)
    body_quote_b64, body_gpu = extract_from_body(body)

    # The signed body is authoritative; a top-level copy must agree exactly.
    if ev.quote_b64:
        if _decode_quote(ev.quote_b64) != _decode_quote(body_quote_b64):
            raise EvidenceError("top-level quote differs from the signed-body quote")
    if ev.gpu_evidence and _canonical(ev.gpu_evidence) != _canonical(body_gpu):
        raise EvidenceError("top-level GPU evidence differs from the signed-body GPU evidence")

    digest = hashlib.sha256((nonce + e2e_pubkey).encode()).hexdigest().lower()
    raw_quote = _decode_quote(body_quote_b64)
    quote = TdxQuote.from_bytes(raw_quote)

    dcap = await verify_tdx_quote(raw_quote, pccs_url=pccs_url)

    report_data = bytes.fromhex(quote.report_data)
    if len(report_data) != 64:
        raise AttestationError("TDX report_data must be 64 bytes")
    if report_data[:32] != bytes.fromhex(digest):
        raise AttestationError(
            "TDX report_data[0:32] does not bind sha256(nonce + e2e_pubkey); refusing legacy/raw-nonce evidence"
        )

    try:
        expected_cert_hash = bytes.fromhex(cert_public_key_hash(ev.certificate_b64))
    except EvidenceError:
        raise
    if report_data[32:64] != expected_cert_hash:
        raise AttestationError(
            "TDX report_data[32:64] does not bind the certificate public key; "
            "the evidence certificate is not the one attested by the quote"
        )

    measurements = await asyncio.to_thread(get_measurements)
    measurement = match_measurement(quote, measurements)

    checks: dict[str, bool] = {
        "signature": True,
        "quote_report_data": True,
        "certificate_binding": True,
        "measurement": True,
        "dcap_quote": True,
    }

    gpu = await asyncio.to_thread(
        _verify_nvidia_gpu,
        body_gpu,
        digest,
        expected_count=measurement.gpu_count,
        nras_url=nras_url,
    )
    checks["nv_gpu"] = True

    return VerifiedEvidence(
        instance_id=ev.instance_id,
        nonce=nonce,
        digest=digest,
        measurement=measurement,
        dcap_status=dcap.status,
        gpu=gpu,
        checks=checks,
    )
