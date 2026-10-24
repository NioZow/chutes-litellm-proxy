"""Attestation error hierarchy (kept separate to avoid import cycles)."""

from __future__ import annotations


class AttestationError(RuntimeError):
    """Raised when a TEE/GPU attestation check fails (fail closed)."""


class MeasurementMismatchError(AttestationError):
    """The TDX quote does not match any published Chutes measurement."""


class EvidenceError(AttestationError):
    """The evidence envelope is missing, unsigned, or internally inconsistent."""
