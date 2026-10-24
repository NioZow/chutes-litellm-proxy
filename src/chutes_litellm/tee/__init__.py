"""TEE attestation primitives (quote, measurement policy, evidence, verifier)."""

from __future__ import annotations

from .errors import AttestationError, EvidenceError, MeasurementMismatchError
from .evidence import (
    Evidence,
    cert_public_key_hash,
    extract_from_body,
    parse_evidence,
    verify_signed_body,
)
from .measurements import (
    TeeMeasurement,
    get_measurements,
    match_measurement,
    parse_endpoint,
    parse_local_file,
    reset_cache,
)
from .quote import (
    TdxQuote,
    TdxVerificationResult,
    parse_and_crosscheck,
    resolve_tdx_tcb_status,
    verify_tdx_quote,
)
from .verifier import VerifiedEvidence, verify_evidence

__all__ = [
    "AttestationError",
    "Evidence",
    "EvidenceError",
    "MeasurementMismatchError",
    "TdxQuote",
    "TdxVerificationResult",
    "TeeMeasurement",
    "VerifiedEvidence",
    "cert_public_key_hash",
    "extract_from_body",
    "get_measurements",
    "match_measurement",
    "parse_and_crosscheck",
    "parse_endpoint",
    "parse_evidence",
    "parse_local_file",
    "reset_cache",
    "resolve_tdx_tcb_status",
    "verify_evidence",
    "verify_signed_body",
    "verify_tdx_quote",
]
