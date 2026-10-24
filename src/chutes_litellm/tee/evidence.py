"""Attestation evidence envelope parsing, signature verification and cert binding.

The evidence comes from ``GET /chutes/{id}/evidence`` (see
``chutes-api/docs/tee-verification.md``).  The signed ``attested_body`` is the
source of truth: its quote/GPU evidence are extracted and compared with any
top-level copies, and the certificate that signed it must be bound into the TDX
quote's ``report_data[32:64]`` (checked by the verifier).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .errors import EvidenceError


@dataclass(frozen=True)
class Evidence:
    instance_id: str
    quote_b64: str | None
    gpu_evidence: list[dict[str, Any]]
    certificate_b64: str
    signature_b64: str | None
    attested_body_b64: str | None


def _b64decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"evidence field {field!r} is missing")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError(f"evidence field {field!r} is not valid base64") from exc


def _gpu_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def parse_evidence(envelope: dict[str, Any]) -> Evidence:
    """Validate the envelope and return its fields (signature still unverified)."""
    if not isinstance(envelope, dict):
        raise EvidenceError("attestation evidence must be an object")

    certificate = envelope.get("certificate")
    signature = envelope.get("signature")
    attested_body = envelope.get("attested_body")
    if not certificate:
        raise EvidenceError("evidence has no 'certificate'; refusing unsigned evidence")
    if not signature:
        raise EvidenceError("evidence has no 'signature'; refusing unsigned evidence")
    if not attested_body:
        raise EvidenceError("evidence has no 'attested_body'; refusing unsigned evidence")

    instance_id = str(envelope.get("instance_id") or "")
    quote = envelope.get("quote")
    gpu = envelope.get("gpu_evidence")
    if gpu is None:
        gpu = envelope.get("gpu_evidences")

    return Evidence(
        instance_id=instance_id,
        quote_b64=quote if isinstance(quote, str) and quote else None,
        gpu_evidence=_gpu_list(gpu),
        certificate_b64=str(certificate),
        signature_b64=str(signature),
        attested_body_b64=str(attested_body),
    )


def _load_certificate(certificate_b64: str) -> x509.Certificate:
    try:
        return x509.load_der_x509_certificate(_b64decode(certificate_b64, "certificate"))
    except EvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError(f"evidence 'certificate' is not a DER X.509 certificate: {exc}") from exc


def verify_signed_body(ev: Evidence) -> dict[str, Any]:
    """Verify RSA-PKCS1v15-SHA256 over ``attested_body`` and return the JSON body."""
    if not ev.signature_b64 or not ev.attested_body_b64:
        raise EvidenceError("evidence has no signed body")

    cert = _load_certificate(ev.certificate_b64)
    public_key = cert.public_key()
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise EvidenceError("evidence certificate public key is not RSA")

    body_bytes = _b64decode(ev.attested_body_b64, "attested_body")
    signature = _b64decode(ev.signature_b64, "signature")
    try:
        public_key.verify(signature, body_bytes, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise EvidenceError("attested_body signature does not verify") from exc
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError(f"attested_body signature verification failed: {exc}") from exc

    try:
        body = json.loads(body_bytes)
    except Exception as exc:  # noqa: BLE001
        raise EvidenceError(f"attested_body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise EvidenceError("attested_body JSON must be an object")
    return body


def _string_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise EvidenceError("signed GPU evidence is not valid JSON") from exc
    return _gpu_list(value)


def extract_from_body(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(tdx_quote_b64, nvtrust_evidence)`` from the signed body.

    Accepts the production shape ``{"evidence": {"tdx_quote": ...,
    "nvtrust_evidence": "<json>"}}`` and the flatter producer variants
    (``quote``/``gpu_evidence`` at top level or under ``evidence``).
    """
    inner = body.get("evidence") if isinstance(body.get("evidence"), dict) else {}

    quote = inner.get("tdx_quote") or inner.get("quote") or body.get("tdx_quote") or body.get("quote")
    if not isinstance(quote, str) or not quote:
        raise EvidenceError("signed body has no TDX quote")

    gpu = inner.get("nvtrust_evidence")
    if gpu is None:
        gpu = inner.get("gpu_evidence")
    if gpu is None:
        gpu = inner.get("gpu_evidences")
    if gpu is None:
        gpu = body.get("gpu_evidence")
    if gpu is None:
        gpu = body.get("gpu_evidences")

    return quote, _string_list(gpu)


def cert_public_key_hash(cert_b64: str) -> str:
    """SHA-256 of the certificate public key in DER SubjectPublicKeyInfo form."""
    cert = _load_certificate(cert_b64)
    der = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest().lower()
