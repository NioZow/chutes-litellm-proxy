"""Strict NEAR AI verification policy and post-verification assertions.

The upstream ``nearai-inference-sdk`` verifies cryptography correctly but its
defaults are permissive: it accepts stale TCB statuses, does not reject
advisories, treats GPU evidence as optional, and accepts a Gateway receipt for a
model that claims TEE execution.  This module fixes those defaults *outside* the
SDK and re-checks the verified objects after verification, so the guarantees
hold even if the SDK defaults change.

Everything here fails closed.  There is deliberately no toggle that weakens a
security-critical value:
``PLAN.md`` §3 lists the fixed defaults; ``PLAN.md`` §§4-5 describe the mode and
receipt policy.

TTLs and timeouts live here so later phases import them from a single module
without circular imports (``policy`` imports only the SDK and the standard
library).  Note that the SDK expresses cache lifetimes in milliseconds, hence
the ``*_MS`` companions.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast, get_args

import yaml
from nearai_inference_sdk import (
    AttestationPolicy,
    AttestationVerificationResult,
    AttestationVerifiers,
    DeploymentVerifier,
    GatewayVerificationOptions,
    MeasuredDeployment,
    ModelAttestationPolicy,
    ModelAttestationVerifiers,
    ModelVerificationOptions,
    TcbStatus,
    VerificationError,
    VerificationFailure,
    VerifiedCompletionResult,
    VerifiedGatewayAttestation,
    VerifiedModelAttestation,
)

# ---------------------------------------------------------------------------
# Fixed defaults (PLAN.md §3).  Not configurable and not overridable.
# ---------------------------------------------------------------------------

#: TEE models require a ``provider_tee`` receipt; routed models require
#: ``gateway``.  There is no fallback in either direction.
ATTESTATION_TTL_SECONDS = 900.0
FAILURE_TTL_SECONDS = 30.0
RESPONSE_TTL_SECONDS = 600.0
HARD_REQUEST_TIMEOUT_SECONDS = 900.0
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 600.0
PREWARM_CONCURRENCY = 4
#: Total wall-clock budget for one startup prewarm batch.  Individual
#: verifications still keep the 900 s hard timeout, but the batch as a whole
#: must never hold LiteLLM readiness for hours.
PREWARM_TOTAL_TIMEOUT_SECONDS = 60.0

#: Milliseconds for the SDK client options (``*_time_to_live_ms``).
ATTESTATION_TTL_MS = 900_000
RESPONSE_TTL_MS = 600_000

#: The only TCB status this provider accepts, for both the Gateway and models.
ACCEPTED_TCB_STATUSES: tuple[TcbStatus, ...] = ("UpToDate",)

#: Streaming defaults.  ``verified`` buffers and verifies before releasing any
#: content; ``tail`` streams live but withholds ``[DONE]`` on failure.
StreamingMode = Literal["verified", "tail"]
DEFAULT_STREAMING_MODE: StreamingMode = "verified"
ALLOWED_STREAMING_MODES: frozenset[str] = frozenset({"verified", "tail"})

#: Environment variable holding the forced-TEE allowlist (parsed in ``modes``).
FORCED_TEE_ENV = "NEARAI_TEE_MODELS"

#: The two modes a model can be classified as (PLAN.md §4).
ReceiptMode = Literal["tee", "routed"]
EXPECTED_RECEIPT_KINDS: Mapping[ReceiptMode, str] = MappingProxyType(
    {"tee": "provider_tee", "routed": "gateway"}
)

#: User-facing prefixes that are not part of the upstream model id.  Bare ids
#: are canonical for env lists, pins, and the wire body.
MODEL_PREFIXES: tuple[str, ...] = ("nearai_e2ee", "nearai")


def strip_model_prefix(model: str) -> str:
    """Strip one leading ``nearai_e2ee/`` or ``nearai/`` prefix from ``model``."""
    if "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in MODEL_PREFIXES and rest:
            return rest
    return model

# ---------------------------------------------------------------------------
# Stable machine-readable policy codes.  Phase 3 maps these to HTTP 502.
# ---------------------------------------------------------------------------

CODE_GATEWAY_TLS = "near.gateway.tls_not_attested"
CODE_GATEWAY_SPKI = "near.gateway.spki_missing"
CODE_GATEWAY_TCB = "near.gateway.tcb_not_uptodate"
CODE_GATEWAY_ADVISORIES = "near.gateway.advisories_present"
CODE_GATEWAY_SIGNER = "near.gateway.signer_algo"
CODE_MODEL_EMPTY = "near.model.none_verified"
CODE_MODEL_TCB = "near.model.tcb_not_uptodate"
CODE_MODEL_ADVISORIES = "near.model.advisories_present"
CODE_MODEL_GPU = "near.model.gpu_evidence_missing"
CODE_MODEL_KEY = "near.model.signing_key_missing"
CODE_RECEIPT_KIND = "near.receipt.kind_mismatch"
CODE_UNKNOWN_MODE = "near.policy.unknown_mode"
CODE_UNKNOWN_STREAMING_MODE = "near.policy.unknown_streaming_mode"

#: Measurement pin file problems (missing/unreadable file, bad schema, bad hex).
#: The error never contains file contents or pin values.
CODE_MEASUREMENTS_INVALID = "near.measurements.invalid"

#: Deployment measurement mismatches.  Distinct codes make a Gateway pin
#: failure distinguishable from a model pin failure.
CODE_MEASUREMENTS_GATEWAY_MISMATCH = "near.measurements.gateway_mismatch"
CODE_MEASUREMENTS_MODEL_MISMATCH = "near.measurements.model_mismatch"

#: Environment variable holding the optional measurement pin file path.
MEASUREMENTS_FILE_ENV = "NEARAI_MEASUREMENTS_FILE"

#: Pin file keys understood by :func:`load_measurement_pins`.
_MEASUREMENT_PIN_KEYS: frozenset[str] = frozenset({"os_image_hash", "compose_hash", "app_compose_sha256"})
_MEASUREMENT_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"gateway", "models"})
_MODEL_WILDCARD = "*"
_HEX_RE = re.compile(r"[0-9a-f]+\Z")

# Known-safe detail strings.  Anything else is redacted so neither prompts,
# API keys, nor raw evidence can reach a log line through ``describe()``.
_SAFE_DETAIL_STRINGS: frozenset[str] = frozenset(
    {
        *get_args(TcbStatus),
        "none",
        "attested",
        "verified",
        "not_provided",
        "if-present",
        "required",
        "tee",
        "routed",
        "provider_tee",
        "gateway",
        "tail",
        "model",
        "os_image_hash",
        "compose_hash",
        "app_compose_sha256",
        "ed25519",
    }
)
_REDACTED = "<redacted>"
_DETAIL_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class NearPolicyError(Exception):
    """A fail-closed NEAR AI policy violation.

    ``code`` is a stable machine-readable identifier (see the ``CODE_*``
    constants).  ``details`` contains only known-safe scalars derived from the
    verified objects -- never prompts, API keys, or raw evidence.  Unknown
    strings are redacted.  Phase 3 maps this error to an HTTP 502.

    ``str(error)`` is just the code, so an exception printed by an unrelated
    handler cannot leak request material.  Use :meth:`describe` for a compact
    log line that includes sanitized details.
    """

    def __init__(self, code: str, *, details: Mapping[str, object] | None = None) -> None:
        self.code = code
        self.details: Mapping[str, object] = MappingProxyType(dict(_sanitize_details(details)))
        super().__init__(code)

    def describe(self) -> str:
        """Return a short, log-safe representation of this violation."""
        if not self.details:
            return self.code
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.details.items()))
        return f"{self.code} ({rendered})"

    def __str__(self) -> str:
        return self.code


def _sanitize_details(details: Mapping[str, object] | None) -> dict[str, object]:
    """Keep only safe keys and scalar values; redact everything else."""
    if not details:
        return {}
    sanitized: dict[str, object] = {}
    for raw_key, raw_value in details.items():
        key = raw_key if isinstance(raw_key, str) and _DETAIL_KEY_RE.fullmatch(raw_key) else "value"
        sanitized[key] = _sanitize_detail_value(raw_value)
    return sanitized


def _sanitize_detail_value(value: object) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str) and value in _SAFE_DETAIL_STRINGS:
        return value
    return _REDACTED


# ---------------------------------------------------------------------------
# Strict SDK option objects
# ---------------------------------------------------------------------------

#: Gateway verification: capture the TLS SPKI and accept only ``UpToDate``.
GATEWAY_VERIFICATION_OPTIONS = GatewayVerificationOptions(
    include_spki_fingerprint=True,
    policy=AttestationPolicy(accepted_tcb_statuses=ACCEPTED_TCB_STATUSES),
)

#: TEE model verification: ``UpToDate`` and mandatory GPU evidence.
#:
#: Routed clients must pass **no** model verification options -- the SDK then
#: returns ``models == ()`` and only Gateway receipts can be verified.  There is
#: intentionally no permissive routed-model options object.
TEE_MODEL_VERIFICATION_OPTIONS = ModelVerificationOptions(
    policy=ModelAttestationPolicy(
        accepted_tcb_statuses=ACCEPTED_TCB_STATUSES,
        gpu_evidence="required",
    ),
)


# ---------------------------------------------------------------------------
# Optional deployment measurement pinning (PLAN.md section 7, phase 6)
# ---------------------------------------------------------------------------


def _normalize_hex(value: object, *, field_name: str) -> str:
    """Return a lowercase, prefix-free hex string or fail closed.

    Pins are configuration, so a malformed value is an operator error rather
    than a deployment mismatch.  The error never includes the offending value.
    """
    if not isinstance(value, str):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID, details={"field": field_name})
    normalized = value.strip()
    if normalized[:2].lower() == "0x":
        normalized = normalized[2:]
    normalized = normalized.lower()
    if not normalized or len(normalized) % 2 or not _HEX_RE.fullmatch(normalized):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID, details={"field": field_name})
    return normalized


def _normalize_runtime_hash(value: object) -> str | None:
    """Normalize an SDK-reported hash for comparison; ``None`` stays ``None``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized[:2].lower() == "0x":
        normalized = normalized[2:]
    normalized = normalized.lower()
    return normalized or None


def _normalize_hash_collection(value: object, *, field_name: str) -> frozenset[str]:
    """Normalize a collection of hex pins, rejecting strings and bad values."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID, details={"field": field_name})
    return frozenset(_normalize_hex(item, field_name=field_name) for item in value)


@dataclass(frozen=True)
class MeasurementPinSet:
    """Allowed deployment measurements for one scope (Gateway or one model).

    An empty set means "no constraint".  Every hash is normalized to lowercase
    hex without an optional ``0x`` prefix, so comparisons are exact.
    """

    os_image_hashes: frozenset[str] = frozenset()
    compose_hashes: frozenset[str] = frozenset()
    app_compose_sha256: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "os_image_hashes",
            _normalize_hash_collection(self.os_image_hashes, field_name="os_image_hash"),
        )
        object.__setattr__(
            self,
            "compose_hashes",
            _normalize_hash_collection(self.compose_hashes, field_name="compose_hash"),
        )
        object.__setattr__(
            self,
            "app_compose_sha256",
            _normalize_hash_collection(self.app_compose_sha256, field_name="app_compose_sha256"),
        )

    @property
    def is_empty(self) -> bool:
        """Whether this set constrains anything at all."""
        return not (self.os_image_hashes or self.compose_hashes or self.app_compose_sha256)


@dataclass(frozen=True)
class MeasurementPins:
    """Gateway and per-model measurement pins.

    ``models`` maps an exact model id to its pin set.  The optional ``"*"``
    entry is a fallback: an exact model id always wins, and a model with neither
    an exact entry nor ``"*"`` is unconstrained.  This is documented in
    ``docs/near-ai.md``.
    """

    gateway: MeasurementPinSet
    models: Mapping[str, MeasurementPinSet]

    def __post_init__(self) -> None:
        if not isinstance(self.gateway, MeasurementPinSet):
            raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
        if not isinstance(self.models, Mapping):
            raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
        normalized: dict[str, MeasurementPinSet] = {}
        for raw_model, pin_set in self.models.items():
            if not isinstance(raw_model, str) or not raw_model.strip():
                raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
            model = raw_model.strip()
            if model in normalized or not isinstance(pin_set, MeasurementPinSet):
                raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
            normalized[model] = pin_set
        object.__setattr__(self, "models", MappingProxyType(normalized))

    @property
    def wildcard(self) -> MeasurementPinSet | None:
        """The ``"*"`` fallback pin set, when configured."""
        return self.models.get(_MODEL_WILDCARD)

    @property
    def has_exact_models(self) -> bool:
        """Whether any exact (non-wildcard) model pins are configured."""
        return any(model != _MODEL_WILDCARD for model in self.models)

    @property
    def is_empty(self) -> bool:
        """Whether nothing is pinned (a valid but warning-worthy configuration)."""
        return self.gateway.is_empty and all(pin_set.is_empty for pin_set in self.models.values())

    def pins_for_model(self, model: str) -> MeasurementPinSet:
        """Return the pin set for ``model``: exact id, then ``"*"``, then empty."""
        if not isinstance(model, str):
            return MeasurementPinSet()
        normalized = model.strip()
        exact = self.models.get(normalized)
        if exact is not None:
            return exact
        return self.models.get(_MODEL_WILDCARD, MeasurementPinSet())


def _parse_pin_values(value: object, *, field_name: str) -> frozenset[str]:
    """Parse one pin value (a string or a list of strings)."""
    if isinstance(value, str):
        items: list[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID, details={"field": field_name})
    return frozenset(_normalize_hex(item, field_name=field_name) for item in items)


def _parse_pin_set(raw: object) -> MeasurementPinSet:
    """Parse one pin-set mapping, rejecting unknown keys and wrong types."""
    if not isinstance(raw, Mapping):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
    unknown = set(raw) - _MEASUREMENT_PIN_KEYS
    if unknown:
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
    return MeasurementPinSet(
        os_image_hashes=_parse_pin_values(raw.get("os_image_hash", []), field_name="os_image_hash"),
        compose_hashes=_parse_pin_values(raw.get("compose_hash", []), field_name="compose_hash"),
        app_compose_sha256=_parse_pin_values(
            raw.get("app_compose_sha256", []),
            field_name="app_compose_sha256",
        ),
    )


def _parse_measurement_pins(document: object) -> MeasurementPins:
    """Validate a decoded YAML/JSON pin document."""
    if not isinstance(document, Mapping):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
    unknown = set(document) - _MEASUREMENT_TOP_LEVEL_KEYS
    if unknown:
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)

    gateway = _parse_pin_set(document.get("gateway", {}))
    raw_models = document.get("models", {})
    if not isinstance(raw_models, Mapping):
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)

    models: dict[str, MeasurementPinSet] = {}
    for raw_model, raw_pin_set in raw_models.items():
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
        model = raw_model.strip()
        if model in models:
            raise NearPolicyError(CODE_MEASUREMENTS_INVALID)
        models[model] = _parse_pin_set(raw_pin_set)
    return MeasurementPins(gateway=gateway, models=models)


def load_measurement_pins(path: str | os.PathLike[str]) -> MeasurementPins:
    """Load optional deployment measurement pins from a YAML or JSON file.

    The schema is documented in ``docs/near-ai.md``::

        gateway:
          os_image_hash: ["<hex>"]
          compose_hash: ["<hex>"]
          app_compose_sha256: ["<hex>"]
        models:
          "z-ai/glm-5.3-flash":
            os_image_hash: ["<hex>"]
          "*":
            compose_hash: ["<hex>"]

    A missing/unreadable file, a non-mapping root, unknown keys, wrong types, or
    an empty/invalid hex value raises :class:`NearPolicyError` with
    :data:`CODE_MEASUREMENTS_INVALID`.  Errors never include file contents or
    pin values.  An empty mapping is valid and means "no constraints"; the
    runtime logs a warning for that case.
    """
    try:
        with open(os.fspath(path), encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, TypeError, ValueError):
        # ``from None`` matters: an OS or YAML error message could otherwise
        # carry a path or a fragment of the file into a log/traceback.
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID) from None

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID) from None
    return _parse_measurement_pins(document)


def _noop_deployment_verifier(_deployment: MeasuredDeployment) -> None:
    """Accept every deployment (used when the selected pin set is empty)."""


def measurement_verifier(
    pins: MeasurementPins,
    *,
    scope: Literal["gateway", "model"],
    model: str | None = None,
) -> DeploymentVerifier:
    """Build a deployment verifier for the Gateway or one model.

    The verifier is synchronous and raises
    ``VerificationError(VerificationFailure(code=..., details={"field", "scope"}))``
    on a mismatch.  It never puts the expected or actual hashes in the error.

    For ``scope="model"``, an exact model id wins over ``"*"``.  When no model
    id is supplied (the shared SDK option object has no model context), a
    wildcard-only configuration is enforced; if exact model entries exist, this
    returned verifier is a no-op and the runtime applies the per-model pins
    after verification via :func:`measurement_verifier` with the model id.
    """
    if scope == "gateway":
        pin_set = pins.gateway
        code = CODE_MEASUREMENTS_GATEWAY_MISMATCH
    elif scope == "model":
        if model is not None:
            pin_set = pins.pins_for_model(model)
        elif pins.has_exact_models:
            return _noop_deployment_verifier
        else:
            pin_set = pins.models.get(_MODEL_WILDCARD, MeasurementPinSet())
        code = CODE_MEASUREMENTS_MODEL_MISMATCH
    else:
        raise NearPolicyError(CODE_MEASUREMENTS_INVALID)

    if pin_set.is_empty:
        return _noop_deployment_verifier

    def verify(deployment: MeasuredDeployment) -> None:
        measurements = deployment.runtime_measurements
        checks = (
            (
                "os_image_hash",
                pin_set.os_image_hashes,
                _normalize_runtime_hash(measurements.os_image_hash),
            ),
            (
                "compose_hash",
                pin_set.compose_hashes,
                _normalize_runtime_hash(measurements.compose_hash),
            ),
            (
                "app_compose_sha256",
                pin_set.app_compose_sha256,
                _app_compose_sha256(deployment.app_compose),
            ),
        )
        for field_name, expected, actual in checks:
            if not expected:
                continue
            if actual is None or actual not in expected:
                raise VerificationError(
                    VerificationFailure(
                        code=code,
                        details={"field": field_name, "scope": scope},
                    )
                )

    return verify


def _app_compose_sha256(app_compose: object) -> str | None:
    """Return the SHA-256 of the measured app-compose string, if it is a string."""
    if not isinstance(app_compose, str):
        return None
    return hashlib.sha256(app_compose.encode("utf-8")).hexdigest()


def gateway_verification_options(
    pins: MeasurementPins | None = None,
) -> GatewayVerificationOptions:
    """Return strict Gateway options, optionally pinned to a measured deployment.

    With no pins this returns the module constant unchanged.  With pins it
    preserves the strict TCB policy and adds a deployment verifier, so the SDK
    reports ``deployment_provenance == "verified"`` only on a match.
    """
    if pins is None:
        return GATEWAY_VERIFICATION_OPTIONS
    return GatewayVerificationOptions(
        include_spki_fingerprint=True,
        policy=GATEWAY_VERIFICATION_OPTIONS.policy,
        verifiers=AttestationVerifiers(
            deployment=measurement_verifier(pins, scope="gateway"),
        ),
    )


def tee_model_verification_options(
    pins: MeasurementPins | None = None,
) -> ModelVerificationOptions:
    """Return strict TEE model options, optionally pinned to measured deployments.

    With no pins this returns the module constant unchanged.  With pins it
    preserves the strict TCB/advisory/GPU policy and adds a deployment verifier.
    """
    if pins is None:
        return TEE_MODEL_VERIFICATION_OPTIONS
    return ModelVerificationOptions(
        policy=TEE_MODEL_VERIFICATION_OPTIONS.policy,
        verifiers=ModelAttestationVerifiers(
            deployment=measurement_verifier(pins, scope="model"),
        ),
    )


# ---------------------------------------------------------------------------
# Post-verification assertions
# ---------------------------------------------------------------------------


def assert_gateway_verified(gateway: VerifiedGatewayAttestation) -> None:
    """Require an attested TLS binding, ``ed25519`` signer, ``UpToDate`` TCB,
    and no advisories.

    Raises:
        NearPolicyError: If any Gateway invariant does not hold.
    """
    if gateway.signer.signing_algo != "ed25519":
        raise NearPolicyError(
            CODE_GATEWAY_SIGNER,
            details={"signing_algo": cast(str, gateway.signer.signing_algo)},
        )
    if gateway.tls_binding.kind != "attested":
        raise NearPolicyError(
            CODE_GATEWAY_TLS,
            details={"kind": gateway.tls_binding.kind},
        )
    if not gateway.tls_binding.spki_fingerprint:
        raise NearPolicyError(CODE_GATEWAY_SPKI)
    if gateway.tcb_status != "UpToDate":
        raise NearPolicyError(
            CODE_GATEWAY_TCB,
            details={"tcb_status": cast(str, gateway.tcb_status)},
        )
    if gateway.advisory_ids:
        raise NearPolicyError(
            CODE_GATEWAY_ADVISORIES,
            details={"count": len(gateway.advisory_ids)},
        )


def assert_tee_models_verified(
    result: AttestationVerificationResult | Sequence[VerifiedModelAttestation],
) -> tuple[VerifiedModelAttestation, ...]:
    """Require at least one complete TEE model report and validate every one.

    Accepts either an :class:`AttestationVerificationResult` (the SDK's
    ``verify()`` return value) or a tuple of verified model reports.  Returns
    the validated tuple for convenience.

    Raises:
        NearPolicyError: On the first failing report, with a stable code.
    """
    models = result.models if isinstance(result, AttestationVerificationResult) else tuple(result)
    if not models:
        raise NearPolicyError(CODE_MODEL_EMPTY)
    for index, model in enumerate(models):
        if model.tcb_status != "UpToDate":
            raise NearPolicyError(
                CODE_MODEL_TCB,
                details={"index": index, "tcb_status": cast(str, model.tcb_status)},
            )
        if model.advisory_ids:
            raise NearPolicyError(
                CODE_MODEL_ADVISORIES,
                details={"index": index, "count": len(model.advisory_ids)},
            )
        if model.gpu_evidence != "verified":
            raise NearPolicyError(
                CODE_MODEL_GPU,
                details={"index": index, "gpu_evidence": model.gpu_evidence},
            )
        if not model.signing_public_key:
            raise NearPolicyError(CODE_MODEL_KEY, details={"index": index})
    return tuple(models)


def assert_receipt_kind(result: VerifiedCompletionResult, expected_mode: ReceiptMode) -> None:
    """Require the response receipt kind implied by the model's mode.

    TEE models must return ``provider_tee``; routed models must return
    ``gateway``.  A receipt of the other kind is a hard failure -- there is no
    fallback and no downgrade.

    Raises:
        NearPolicyError: For an unknown mode or a mismatched receipt kind.
    """
    expected = EXPECTED_RECEIPT_KINDS.get(expected_mode)
    if expected is None:
        raise NearPolicyError(
            CODE_UNKNOWN_MODE,
            details={"mode": cast(str, expected_mode)},
        )
    if result.signature_kind != expected:
        raise NearPolicyError(
            CODE_RECEIPT_KIND,
            details={
                "mode": expected_mode,
                "expected": expected,
                "actual": cast(str, result.signature_kind),
            },
        )


# ---------------------------------------------------------------------------
# Streaming mode resolution
# ---------------------------------------------------------------------------


def resolve_streaming_mode(value: str | None = None) -> StreamingMode:
    """Resolve a configured streaming mode, defaulting to ``verified``.

    Unknown values fail closed rather than silently falling back to a weaker
    mode.

    Raises:
        NearPolicyError: If ``value`` is not an allowed streaming mode.
    """
    if value is None or not value.strip():
        return DEFAULT_STREAMING_MODE
    normalized = value.strip().lower()
    if normalized not in ALLOWED_STREAMING_MODES:
        raise NearPolicyError(
            CODE_UNKNOWN_STREAMING_MODE,
            details={"mode": cast(str, normalized)},
        )
    return cast(StreamingMode, normalized)


# ---------------------------------------------------------------------------
# Negative cache
# ---------------------------------------------------------------------------


class FailureCache:
    """A tiny TTL cache for failed attestations (the plan's 30 s negative cache).

    Later phases use it to suppress repeated verification attempts against a
    failing model.  Keys are caller-provided (typically the model id), so
    distinct models never collide.  Expiry uses a monotonic clock by default;
    ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        ttl: float = FAILURE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl)
        self._clock = clock
        self._entries: dict[str, tuple[float, BaseException]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> BaseException | None:
        """Return the stored error for ``key``, or ``None`` if absent/expired."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, error = entry
            if now >= expires_at:
                del self._entries[key]
                return None
            return error

    def put(self, key: str, error: BaseException) -> None:
        """Store ``error`` for ``key`` for one TTL from now."""
        with self._lock:
            self._entries[key] = (self._clock() + self._ttl, error)

    def clear(self) -> None:
        """Drop every cached failure."""
        with self._lock:
            self._entries.clear()


__all__ = [
    "ACCEPTED_TCB_STATUSES",
    "ALLOWED_STREAMING_MODES",
    "ATTESTATION_TTL_MS",
    "ATTESTATION_TTL_SECONDS",
    "CODE_GATEWAY_ADVISORIES",
    "CODE_GATEWAY_SIGNER",
    "CODE_GATEWAY_SPKI",
    "CODE_GATEWAY_TCB",
    "CODE_GATEWAY_TLS",
    "CODE_MEASUREMENTS_GATEWAY_MISMATCH",
    "CODE_MEASUREMENTS_INVALID",
    "CODE_MEASUREMENTS_MODEL_MISMATCH",
    "CODE_MODEL_ADVISORIES",
    "CODE_MODEL_EMPTY",
    "CODE_MODEL_GPU",
    "CODE_MODEL_KEY",
    "CODE_MODEL_TCB",
    "CODE_RECEIPT_KIND",
    "CODE_UNKNOWN_MODE",
    "CODE_UNKNOWN_STREAMING_MODE",
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_STREAMING_MODE",
    "EXPECTED_RECEIPT_KINDS",
    "FAILURE_TTL_SECONDS",
    "FORCED_TEE_ENV",
    "GATEWAY_VERIFICATION_OPTIONS",
    "HARD_REQUEST_TIMEOUT_SECONDS",
    "MEASUREMENTS_FILE_ENV",
    "MODEL_PREFIXES",
    "PREWARM_CONCURRENCY",
    "PREWARM_TOTAL_TIMEOUT_SECONDS",
    "READ_TIMEOUT_SECONDS",
    "RESPONSE_TTL_MS",
    "RESPONSE_TTL_SECONDS",
    "TEE_MODEL_VERIFICATION_OPTIONS",
    "FailureCache",
    "MeasurementPinSet",
    "MeasurementPins",
    "NearPolicyError",
    "ReceiptMode",
    "StreamingMode",
    "assert_gateway_verified",
    "assert_receipt_kind",
    "assert_tee_models_verified",
    "gateway_verification_options",
    "load_measurement_pins",
    "measurement_verifier",
    "resolve_streaming_mode",
    "strip_model_prefix",
    "tee_model_verification_options",
]
