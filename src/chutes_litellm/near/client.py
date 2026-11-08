"""Process-wide NEAR AI runtime clients, one instance per API key.

This is the Phase 2 runtime described in ``PLAN.md`` §5 and §9.  It composes the
audited upstream SDK instead of forking it:

* two ``InferenceClient`` instances -- ``e2ee=True`` for true TEE models and
  ``e2ee=False`` for Gateway-routed models -- shared by every request that uses
  the same API key;
* one pinned ``AttestationClient`` for mode detection, bootstrapped from a
  verified Gateway TLS binding and re-bootstrapped once if the pin goes stale;
* a frozen :class:`~chutes_litellm.near.modes.ModelModeRegistry`, a positive
  verification cache aligned with the SDK's attestation TTL, and a 30 s
  negative cache;
* per-model task coalescing plus hard ``asyncio.timeout`` around every
  network-facing operation;
* prewarming bounded by concurrency and a 60 s total batch budget.

Nothing here creates a client per request, logs prompts/keys/evidence, or
offers a toggle that weakens verification.  Failures are split into two
families: :class:`~chutes_litellm.near.policy.NearPolicyError` for cryptographic
or policy violations (HTTP 502 in Phase 3) and :class:`NearUnavailableError` for
timeouts, undeterminable modes, bootstrap failures, and transport problems
(HTTP 503 in Phase 3).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import httpx
from nearai_inference_sdk import (
    ApiError,
    AttestationClient,
    GatewayVerificationOptions,
    InferenceClient,
    ModelVerificationOptions,
    VerificationError,
    VerifiedModelAttestation,
    create_pinned_tls_client,
    verify_gateway_attestation,
)

from chutes_litellm.near import endpoints, policy
from chutes_litellm.near.modes import (
    CODE_MODE_CHANGED,
    ModelMode,
    ModelModeRegistry,
)
from chutes_litellm.near.policy import (
    ATTESTATION_TTL_MS,
    ATTESTATION_TTL_SECONDS,
    CODE_MEASUREMENTS_GATEWAY_MISMATCH,
    CODE_MEASUREMENTS_MODEL_MISMATCH,
    CODE_UNKNOWN_MODE,
    CONNECT_TIMEOUT_SECONDS,
    FAILURE_TTL_SECONDS,
    HARD_REQUEST_TIMEOUT_SECONDS,
    MEASUREMENTS_FILE_ENV,
    PREWARM_CONCURRENCY,
    PREWARM_TOTAL_TIMEOUT_SECONDS,
    READ_TIMEOUT_SECONDS,
    RESPONSE_TTL_MS,
    FailureCache,
    MeasurementPins,
    NearPolicyError,
    assert_gateway_verified,
    assert_tee_models_verified,
    gateway_verification_options,
    load_measurement_pins,
    measurement_verifier,
    strip_model_prefix,
    tee_model_verification_options,
)

#: Local aliases for the two stable measurement-mismatch codes.  They are
#: re-exported so Phase 3 and tests can map them without importing ``policy``.
_CODE_MEASUREMENT_MISMATCHES: frozenset[str] = frozenset(
    {CODE_MEASUREMENTS_GATEWAY_MISMATCH, CODE_MEASUREMENTS_MODEL_MISMATCH}
)


def _verification_policy_error(error: VerificationError) -> NearPolicyError | None:
    """Return a 502-family error for a measurement pin mismatch, else ``None``.

    Other metadata ``VerificationError`` codes may mean a rotated Gateway
    SPKI/TLS pin rather than a policy violation; the caller re-bootstraps once
    before treating the second failure as an availability problem.
    """
    code = getattr(error.failure, "code", None)
    if code in _CODE_MEASUREMENT_MISMATCHES:
        return NearPolicyError(code, details=error.failure.details)
    return None

# The runtime-level positive cache must not outlive the SDK's attestation
# session cache (or vice versa); keep the two representations in lockstep.
if ATTESTATION_TTL_MS != int(ATTESTATION_TTL_SECONDS * 1000):  # pragma: no cover - import guard
    raise RuntimeError(
        "NEAR AI attestation TTLs diverged: "
        f"policy ms={ATTESTATION_TTL_MS!r} seconds={ATTESTATION_TTL_SECONDS!r}"
    )

#: Environment variable holding the API key used when none is passed explicitly.
API_KEY_ENV = "NEARAI_API_KEY"

#: Environment variable holding a comma-separated prewarm list.
PREWARM_MODELS_ENV = "NEARAI_PREWARM_MODELS"

#: Stable machine-readable codes for :class:`NearUnavailableError`.
#: Phase 3 maps the family to HTTP 503.
CODE_UNAVAILABLE_TIMEOUT = "near.unavailable.timeout"
CODE_UNAVAILABLE_MODE_UNKNOWN = "near.unavailable.mode_unknown"
CODE_UNAVAILABLE_ATTESTATION = "near.unavailable.attestation"
CODE_UNAVAILABLE_TRANSPORT = "near.unavailable.transport"
CODE_UNAVAILABLE_API_KEY = "near.unavailable.api_key_missing"
CODE_UNAVAILABLE_CLOSED = "near.unavailable.closed"

#: A local cryptographic failure surfaced by the SDK's ``VerificationError``.
#: Kept in the policy family because it is a verification failure, not an
#: availability problem; Phase 3 maps it to HTTP 502.
CODE_ATTESTATION_FAILED = "near.attestation.verification_failed"

_LOGGER_NAME = "chutes_litellm.near.client"

_T = TypeVar("_T")


class NearUnavailableError(Exception):
    """A fail-closed service-unavailable NEAR AI failure (HTTP 503).

    Use this for conditions where the provider cannot safely serve the request
    but no policy was violated: the model mode cannot be determined, an
    operation exceeded its hard timeout, Gateway/metadata bootstrap failed, or
    the upstream transport failed with no more specific SDK error.  Never use it
    for cryptographic or policy violations -- those stay
    :class:`~chutes_litellm.near.policy.NearPolicyError` (HTTP 502).

    ``code`` is stable and machine-readable.  ``model`` is the model id, when
    known; model ids are safe to log.  ``str(error)`` is just the code, so an
    exception printed by an unrelated handler cannot leak request material.
    """

    def __init__(self, code: str, *, model: str | None = None) -> None:
        self.code = code
        self.model = model
        super().__init__(code)

    def describe(self) -> str:
        """Return a short, log-safe representation of this failure."""
        if self.model is None:
            return self.code
        return f"{self.code} (model={self.model!r})"

    def __str__(self) -> str:
        return self.code


def _safe_error_code(error: BaseException) -> str:
    """Return a stable, log-safe code for ``error`` (never its message)."""
    code = getattr(error, "code", None)
    return code if isinstance(code, str) else type(error).__name__


def _dedupe_models(models: Sequence[str]) -> tuple[str, ...]:
    """Return non-empty, stripped, prefix-free model ids in input order.

    A leading ``nearai/`` or ``nearai_e2ee/`` is removed so copying a
    user-facing model name into ``NEARAI_PREWARM_MODELS`` works.
    """
    unique: dict[str, None] = {}
    for raw in models:
        model = strip_model_prefix(raw.strip())
        if model and model not in unique:
            unique[model] = None
    return tuple(unique)


def _log_construction_warning(message: str) -> None:
    """Log one log-safe construction warning (never hashes or keys)."""
    from chutes_litellm import get_logger

    get_logger(_LOGGER_NAME).warning(message)


def prewarm_models_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Read the comma-separated prewarm list from ``NEARAI_PREWARM_MODELS``.

    ``environ`` defaults to ``os.environ``; whitespace is ignored, empty entries
    are dropped, and duplicates keep their first position.  This phase never
    fetches the model catalog, so an unset or empty variable yields ``()``.
    """
    source = os.environ if environ is None else environ
    raw = source.get(PREWARM_MODELS_ENV) or ""
    return _dedupe_models(tuple(raw.split(",")))


class NearRuntime:
    """The persistent per-API-key NEAR AI clients and verification state.

    Construct **one** instance per API key and reuse it for every request (see
    :func:`get_runtime`).  Construction builds both SDK clients eagerly: it is
    synchronous and does no network I/O, and :meth:`aclose` releases them.

    Args:
        api_key: The NEAR AI API key.  Required and never logged.
        registry: Optional pre-built mode registry (tests).  When omitted, one
            is built from ``environ`` so ``NEARAI_TEE_MODELS`` is honoured.
        environ: Environment mapping used for the forced-TEE allowlist, the
            prewarm list, and ``NEARAI_MEASUREMENTS_FILE``.  Defaults to
            ``os.environ`` at construction time.  A set but invalid measurement
            pin file raises :class:`NearPolicyError` here, before any client is
            built; there is no toggle to ignore it.
        clock: Monotonic clock used for the positive and negative caches;
            injectable for deterministic tests.
    """

    def __init__(
        self,
        api_key: str,
        *,
        registry: ModelModeRegistry | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        key = api_key.strip() if isinstance(api_key, str) else ""
        if not key:
            raise NearUnavailableError(CODE_UNAVAILABLE_API_KEY)
        self._api_key = key
        self._environ = environ
        self._clock = clock
        self._closed = False

        # Measurement pinning is optional.  A set, non-empty path is authoritative:
        # an invalid file fails closed at construction (NearPolicyError), and
        # there is deliberately no toggle that ignores a present file.
        source = os.environ if self._environ is None else self._environ
        raw_measurements_file = (source.get(MEASUREMENTS_FILE_ENV) or "").strip()
        if raw_measurements_file:
            pins = load_measurement_pins(raw_measurements_file)
            self._pins: MeasurementPins | None = pins
            self._gateway_options = gateway_verification_options(pins)
            self._model_options = tee_model_verification_options(pins)
            if pins.is_empty:
                _log_construction_warning(
                    "NEAR AI measurement pin file defines no pins; deployment provenance is not pinned"
                )
        else:
            self._pins = None
            self._gateway_options = gateway_verification_options()
            self._model_options = tee_model_verification_options()
            _log_construction_warning(
                "NEAR AI deployment provenance is not pinned because "
                f"{MEASUREMENTS_FILE_ENV} is unset"
            )

        base_url = endpoints.API_BASE  # read at construction time, never earlier
        self._tee_client = InferenceClient(
            key,
            base_url=base_url,
            e2ee=True,
            signing_algo="ed25519",
            attestation_cache_time_to_live_ms=ATTESTATION_TTL_MS,
            response_cache_time_to_live_ms=RESPONSE_TTL_MS,
            gateway_verification=self._gateway_options,
            model_verification=self._model_options,
        )
        # Routed models cannot use model E2EE.  Passing a model policy here
        # would make routed models fail verification, so it is deliberately
        # ``None``: the SDK then returns ``models == ()`` and accepts a
        # Gateway receipt.
        self._routed_client = InferenceClient(
            key,
            base_url=base_url,
            e2ee=False,
            signing_algo="ed25519",
            attestation_cache_time_to_live_ms=ATTESTATION_TTL_MS,
            response_cache_time_to_live_ms=RESPONSE_TTL_MS,
            gateway_verification=self._gateway_options,
            model_verification=None,
        )

        self._registry = registry if registry is not None else ModelModeRegistry(environ=environ)
        self._failure_cache = FailureCache(FAILURE_TTL_SECONDS, clock=clock)
        self._verified: dict[str, tuple[ModelMode, float]] = {}
        self._cache_lock = threading.Lock()

        # Bootstrap/metadata state (created lazily after Gateway verification).
        self._metadata_client: AttestationClient | None = None
        self._pinned_http_client: httpx.AsyncClient | None = None

        # In-flight coalescing map: one task per (operation, model) key.
        self._inflight: dict[tuple[str, str], asyncio.Task[Any]] = {}

    # ------------------------------------------------------------------
    # Read-only surface for Phase 3
    # ------------------------------------------------------------------

    @property
    def tee_client(self) -> InferenceClient:
        """The E2EE client used for true TEE models."""
        return self._tee_client

    @property
    def routed_client(self) -> InferenceClient:
        """The non-E2EE client used for Gateway-routed models."""
        return self._routed_client

    @property
    def registry(self) -> ModelModeRegistry:
        """The frozen model-mode registry."""
        return self._registry

    @property
    def failure_cache(self) -> FailureCache:
        """The 30 s negative cache."""
        return self._failure_cache

    @property
    def measurement_pins(self) -> MeasurementPins | None:
        """The loaded measurement pins, or ``None`` when the file is unset."""
        return self._pins

    @property
    def gateway_options(self) -> GatewayVerificationOptions:
        """The Gateway verification options used by both SDK clients and bootstrap."""
        return self._gateway_options

    @property
    def model_options(self) -> ModelVerificationOptions:
        """The TEE model verification options (pinned or the strict constant)."""
        return self._model_options

    def client_for(self, mode: ModelMode) -> InferenceClient:
        """Return the SDK client for ``mode``.

        Raises:
            NearPolicyError: If ``mode`` is not ``"tee"`` or ``"routed"``.
        """
        if mode == "tee":
            return self._tee_client
        if mode == "routed":
            return self._routed_client
        raise NearPolicyError(CODE_UNKNOWN_MODE, details={"mode": str(mode)})

    def request_timeout(self) -> httpx.Timeout:
        """Return the per-request timeout to pass to Phase 3 sends.

        The SDK itself constructs its clients with ``timeout=None``, so callers
        must supply this on every request in addition to :meth:`hard_timeout`.
        """
        return httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)

    @asynccontextmanager
    async def hard_timeout(self, seconds: float | None = None) -> AsyncIterator[None]:
        """Bound an operation and translate ``TimeoutError`` to a clean 503.

        Phase 3 wraps the whole send-plus-verify operation in this context
        manager.  ``seconds`` defaults to the module-level
        :data:`~chutes_litellm.near.policy.HARD_REQUEST_TIMEOUT_SECONDS` at
        call time.
        """
        limit = HARD_REQUEST_TIMEOUT_SECONDS if seconds is None else seconds
        try:
            async with asyncio.timeout(limit):
                yield
        except TimeoutError as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_TIMEOUT) from exc

    # ------------------------------------------------------------------
    # Mode detection and attestation
    # ------------------------------------------------------------------

    async def verify_model(self, model: str) -> ModelMode:
        """Ensure ``model`` has a valid attestation and return its frozen mode.

        The warm path (a successful verification inside the attestation TTL)
        performs no network calls at all.  The cold path coalesces concurrent
        callers per model, fetches metadata once through the pinned client,
        freezes the mode, and calls the matching SDK client's ``verify``.
        Failures are put in the 30 s negative cache and re-raised unchanged.

        Returns:
            ``"tee"`` for true TEE models, ``"routed"`` for Gateway-routed ones.

        Raises:
            NearUnavailableError: On timeout, undeterminable mode, bootstrap
                failure, or transport failure (HTTP 503 in Phase 3).
            NearPolicyError: On a cryptographic or policy violation, including
                mode changes and missing/stale TEE evidence (HTTP 502).
        """
        self._require_open(model)
        normalized = model.strip()
        if not normalized:
            raise NearUnavailableError(CODE_UNAVAILABLE_MODE_UNKNOWN)

        cached_failure = self._failure_cache.get(normalized)
        if cached_failure is not None:
            raise cached_failure

        cached_mode = self._verified_mode(normalized)
        if cached_mode is not None:
            return cached_mode

        return await self._coalesced(("verify", normalized), lambda: self._verify_cold(normalized))

    async def observe_mode(self, model: str) -> ModelMode:
        """Fetch metadata through the pinned client and freeze/validate the mode.

        Re-observing a frozen model is how a runtime metadata change is detected:
        the registry rejects any disagreement, including a TEE model that starts
        reporting routed.  Phase 4 discovery may also use this directly.
        """
        self._require_open(model)
        normalized = model.strip()
        if not normalized:
            raise NearUnavailableError(CODE_UNAVAILABLE_MODE_UNKNOWN)
        return await self._coalesced(("mode", normalized), lambda: self._fetch_and_observe(normalized))

    async def _verify_cold(self, model: str) -> ModelMode:
        """Run one cold verification attempt and populate the caches."""
        try:
            mode = await self._resolve_mode(model)
            await self._verify_attestation(model, mode)
        except (NearPolicyError, NearUnavailableError) as exc:
            self._failure_cache.put(model, exc)
            raise
        except Exception as exc:  # last-resort fail-closed mapping
            error = NearUnavailableError(CODE_UNAVAILABLE_TRANSPORT, model=model)
            self._failure_cache.put(model, error)
            raise error from exc

        with self._cache_lock:
            self._verified[model] = (mode, self._clock())
        return mode

    async def _resolve_mode(self, model: str) -> ModelMode:
        """Return the frozen mode after revalidating metadata on a cold attempt.

        The frozen mode is authoritative for choosing the client, but a cold
        attempt re-reads metadata and runs it through
        :meth:`ModelModeRegistry.observe`, so a Gateway that starts reporting a
        different mode fails closed instead of silently using a stale answer.
        Warm requests never reach this method.
        """
        return await self.observe_mode(model)

    async def _fetch_and_observe(self, model: str) -> ModelMode:
        """Fetch metadata once (coalesced by the caller) and observe it.

        A metadata ``VerificationError`` usually means the pinned Gateway
        SPKI/TLS pin is stale after a rotation.  In that case the pinned client
        is closed, the full strict Gateway bootstrap is repeated exactly once,
        and the metadata fetch is retried once.  A second failure is mapped to
        503, never retried again.  Measurement-pin mismatches are policy
        failures and are never retried.
        """
        metadata_client = await self._ensure_metadata_client()
        try:
            metadata = await self._fetch_model_metadata(metadata_client, model)
        except VerificationError as exc:
            policy_error = _verification_policy_error(exc)
            if policy_error is not None:
                raise policy_error from exc

            await self._reset_metadata_client()
            metadata_client = await self._ensure_metadata_client()
            try:
                metadata = await self._fetch_model_metadata(metadata_client, model)
            except VerificationError as retry_exc:
                policy_error = _verification_policy_error(retry_exc)
                if policy_error is not None:
                    raise policy_error from retry_exc
                raise NearUnavailableError(CODE_UNAVAILABLE_MODE_UNKNOWN, model=model) from retry_exc

        # Enforces forced-TEE and no-downgrade; raises NearPolicyError.
        return self._registry.observe(model, metadata)

    async def _fetch_model_metadata(self, metadata_client: AttestationClient, model: str) -> Any:
        """Fetch one model's metadata under the hard timeout.

        Transport/timeout failures map to :class:`NearUnavailableError`;
        ``VerificationError`` is propagated for the caller's single
        re-bootstrap retry.
        """
        try:
            async with asyncio.timeout(HARD_REQUEST_TIMEOUT_SECONDS):
                return await metadata_client.fetch_model_metadata(model)
        except TimeoutError as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_TIMEOUT, model=model) from exc
        except VerificationError:
            raise
        except (ApiError, httpx.HTTPError, OSError) as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_MODE_UNKNOWN, model=model) from exc

    async def _verify_attestation(self, model: str, mode: ModelMode) -> None:
        """Call the matching client's ``verify`` under a hard timeout and assert policy."""
        sdk_client = self.client_for(mode)
        try:
            async with asyncio.timeout(HARD_REQUEST_TIMEOUT_SECONDS):
                result = await sdk_client.verify(model)
        except TimeoutError as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_TIMEOUT, model=model) from exc
        except VerificationError as exc:
            # A local cryptographic/binding failure is a 502-family violation.
            # Preserve the specific deployment-mismatch codes; everything else
            # collapses to the generic attestation-failure code.
            if exc.failure.code in _CODE_MEASUREMENT_MISMATCHES:
                raise NearPolicyError(exc.failure.code, details=exc.failure.details) from exc
            raise NearPolicyError(CODE_ATTESTATION_FAILED) from exc
        except (ApiError, httpx.HTTPError, OSError) as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_ATTESTATION, model=model) from exc

        self._assert_frozen_mode(model, mode)
        assert_gateway_verified(result.gateway)
        if mode == "tee":
            reports = assert_tee_models_verified(result.models)
            self._assert_model_measurements(model, reports)
        # Routed models are not expected to carry model reports; the Gateway
        # receipt check happens in Phase 3 via ``assert_receipt_kind``.

    def _assert_model_measurements(
        self,
        model: str,
        reports: Sequence[VerifiedModelAttestation],
    ) -> None:
        """Apply exact/wildcard model pins after the SDK's own verification.

        The shared ``ModelVerificationOptions`` object has no per-request model
        identity, so the SDK can only enforce a wildcard-only pin set there.  The
        runtime applies the selected model's pins here, which makes an exact
        model id win over ``"*"`` as documented.
        """
        if self._pins is None:
            return
        verifier = measurement_verifier(self._pins, scope="model", model=model)
        for report in reports:
            try:
                verifier(report.deployment)
            except VerificationError as exc:
                raise NearPolicyError(exc.failure.code, details=exc.failure.details) from exc

    def _assert_frozen_mode(self, model: str, mode: ModelMode) -> None:
        """Require that the mode used is the mode frozen for ``model``."""
        frozen = self._registry.get(model)
        if frozen != mode:
            raise NearPolicyError(
                CODE_MODE_CHANGED,
                details={"previous_mode": frozen, "observed_mode": mode},
            )

    def _verified_mode(self, model: str) -> ModelMode | None:
        """Return the cached mode if verification is still inside the TTL."""
        now = self._clock()
        with self._cache_lock:
            entry = self._verified.get(model)
            if entry is None:
                return None
            mode, verified_at = entry
            if now - verified_at >= ATTESTATION_TTL_SECONDS:
                del self._verified[model]
                return None
            return mode

    # ------------------------------------------------------------------
    # Pinned metadata client
    # ------------------------------------------------------------------

    async def _ensure_metadata_client(self) -> AttestationClient:
        """Return the pinned metadata client, bootstrapping it exactly once."""
        existing = self._metadata_client
        if existing is not None:
            return existing
        return await self._coalesced(("metadata-client", ""), self._bootstrap_metadata_client)

    async def _bootstrap_metadata_client(self) -> AttestationClient:
        """Verify the Gateway, pin its SPKI, and build the authenticated client.

        The bootstrap evidence fetch is intentionally unauthenticated: the API
        key must never be sent to a Gateway whose attestation has not been
        verified yet.  Only after ``assert_gateway_verified`` and a non-empty
        SPKI fingerprint do we create a pinned TLS client and pass the key to an
        ``AttestationClient`` that uses it.

        Policy and cryptographic failures are :class:`NearPolicyError` (HTTP
        502).  Transport, timeout, and API availability failures remain
        :class:`NearUnavailableError` (HTTP 503).

        Raises:
            NearPolicyError: If the Gateway policy or cryptography fails.
            NearUnavailableError: If a transport or verification-service step
                fails.
        """
        base_url = endpoints.API_BASE
        bootstrap = AttestationClient(base_url=base_url)
        try:
            async with asyncio.timeout(HARD_REQUEST_TIMEOUT_SECONDS):
                fetched = await bootstrap.fetch_gateway_attestation(
                    signing_algo="ed25519",
                    include_spki_fingerprint=True,
                )
                gateway = await verify_gateway_attestation(
                    fetched.attestation,
                    fetched.client_binding,
                    policy=self._gateway_options.policy,
                    verifiers=self._gateway_options.verifiers,
                )
        except TimeoutError as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_TIMEOUT) from exc
        except VerificationError as exc:
            policy_error = _verification_policy_error(exc)
            if policy_error is not None:
                raise policy_error from exc
            raise NearPolicyError(CODE_ATTESTATION_FAILED) from exc
        except (ApiError, httpx.HTTPError, OSError) as exc:
            raise NearUnavailableError(CODE_UNAVAILABLE_ATTESTATION) from exc

        # Propagate the specific Gateway policy code (TLS, SPKI, TCB,
        # advisories, signer algorithm) so it maps to 502.
        assert_gateway_verified(gateway)

        fingerprint = gateway.tls_binding.spki_fingerprint
        if not fingerprint:
            raise NearPolicyError(policy.CODE_GATEWAY_SPKI)

        try:
            pinned = create_pinned_tls_client(fingerprint)
        except VerificationError as exc:
            policy_error = _verification_policy_error(exc)
            if policy_error is not None:
                raise policy_error from exc
            raise NearPolicyError(CODE_ATTESTATION_FAILED) from exc
        except (TypeError, ValueError) as exc:
            raise NearPolicyError(policy.CODE_GATEWAY_SPKI) from exc

        # The SDK creates the pinned client with ``timeout=None``; metadata and
        # signature requests still need the fixed connect/read budget.  The
        # outer ``asyncio.timeout`` remains the hard wall-clock bound.
        pinned.timeout = self.request_timeout()

        self._pinned_http_client = pinned
        self._metadata_client = AttestationClient(self._api_key, base_url=base_url, http_client=pinned)
        return self._metadata_client

    async def _reset_metadata_client(self) -> None:
        """Close and forget the pinned metadata client.  Idempotent and safe.

        Used after a metadata ``VerificationError`` so the next bootstrap can
        pick up a rotated Gateway SPKI/TLS fingerprint, and by :meth:`aclose`.
        Close errors are logged with safe codes only and never propagate.
        """
        pinned = self._pinned_http_client
        self._metadata_client = None
        self._pinned_http_client = None
        if pinned is None:
            return
        try:
            await pinned.aclose()
        except Exception as exc:  # noqa: BLE001 - close must not mask the caller
            from chutes_litellm import get_logger

            get_logger(_LOGGER_NAME).warning(
                "NEAR AI metadata client close failed: %s",
                _safe_error_code(exc),
            )

    # ------------------------------------------------------------------
    # Prewarm
    # ------------------------------------------------------------------

    async def prewarm(
        self,
        models: Sequence[str],
        *,
        concurrency: int = PREWARM_CONCURRENCY,
        timeout: float = PREWARM_TOTAL_TIMEOUT_SECONDS,
    ) -> dict[str, BaseException | None]:
        """Verify ``models`` with bounded concurrency, never raising per model.

        The whole batch is bounded by ``timeout`` (60 s by default) so the
        awaited LiteLLM startup hook can never hold readiness for the 900 s
        per-model hard timeout times the catalog size.  On expiry the remaining
        tasks are cancelled, any missing result is filled with a safe
        ``near.unavailable.timeout`` error, and the method returns normally.

        Args:
            models: Model ids to verify.  Empty or whitespace-only ids are
                ignored; duplicates are collapsed and a leading ``nearai/`` or
                ``nearai_e2ee/`` prefix is stripped.
            concurrency: Maximum number of verifications in flight at once.
                Values below one are treated as one.
            timeout: Total wall-clock budget for the whole batch, in seconds.

        Returns:
            A mapping of bare model id to ``None`` on success or the safe
            exception on failure.  Individual failures never abort the batch.
        """
        unique = _dedupe_models(tuple(models))
        if not unique:
            return {}

        semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        results: dict[str, BaseException | None] = {}

        async def run(model: str) -> None:
            async with semaphore:
                try:
                    async with self.hard_timeout():
                        await self.verify_model(model)
                except Exception as exc:  # noqa: BLE001 - per-model isolation
                    results[model] = exc
                else:
                    results[model] = None

        tasks = [asyncio.create_task(run(model), name=f"near:prewarm:{model}") for model in unique]
        timed_out = False
        try:
            async with asyncio.timeout(timeout):
                await asyncio.gather(*tasks)
        except TimeoutError:
            timed_out = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if timed_out:
            for model in unique:
                results.setdefault(model, NearUnavailableError(CODE_UNAVAILABLE_TIMEOUT, model=model))
            self._log_prewarm_incomplete(results, timeout)
        self._log_prewarm(results)
        return results

    async def prewarm_from_env(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        concurrency: int = PREWARM_CONCURRENCY,
    ) -> dict[str, BaseException | None]:
        """Prewarm the models listed in ``NEARAI_PREWARM_MODELS``."""
        source = self._environ if environ is None else environ
        return await self.prewarm(prewarm_models_from_env(source), concurrency=concurrency)

    @staticmethod
    def _log_prewarm_incomplete(results: Mapping[str, BaseException | None], timeout: float) -> None:
        """Log one concise, log-safe summary for an expired prewarm budget."""
        from chutes_litellm import get_logger

        failures = sum(1 for error in results.values() if error is not None)
        get_logger(_LOGGER_NAME).warning(
            "NEAR AI prewarm incomplete after %s s: %d/%d models unverified",
            timeout,
            failures,
            len(results),
        )

    @staticmethod
    def _log_prewarm(results: Mapping[str, BaseException | None]) -> None:
        """Log a concise summary: model ids and safe codes only."""
        from chutes_litellm import get_logger

        logger = get_logger(_LOGGER_NAME)
        failures = {model: error for model, error in results.items() if error is not None}
        logger.info(
            "NEAR AI prewarm: %d/%d models verified",
            len(results) - len(failures),
            len(results),
        )
        for model, error in failures.items():
            logger.warning("NEAR AI prewarm failed for model %s: %s", model, _safe_error_code(error))

    # ------------------------------------------------------------------
    # Coalescing helper and lifecycle
    # ------------------------------------------------------------------

    async def _coalesced(
        self,
        key: tuple[str, str],
        factory: Callable[[], Coroutine[Any, Any, _T]],
    ) -> _T:
        """Run ``factory`` once per in-flight ``key`` and share its result.

        The first caller creates a task; concurrent callers await the same task.
        The task is removed when it completes.  ``asyncio.shield`` keeps the
        shared work alive if a particular waiter is cancelled.
        """
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(factory(), name=f"near:{key[0]}:{key[1]}")
            self._inflight[key] = task

            def discard(completed: asyncio.Task[Any], *, coalesce_key: tuple[str, str] = key) -> None:
                if self._inflight.get(coalesce_key) is completed:
                    del self._inflight[coalesce_key]

            task.add_done_callback(discard)
        return await asyncio.shield(task)

    def reset_caches(self) -> None:
        """Clear the positive and negative caches (tests and forced re-checks)."""
        with self._cache_lock:
            self._verified.clear()
        self._failure_cache.clear()

    async def aclose(self) -> None:
        """Close both SDK clients and the pinned metadata client.  Idempotent.

        Close errors are logged and swallowed so shutdown releases every
        resource even when one client fails; calling this twice is a no-op.
        """
        if self._closed:
            return
        self._closed = True

        for task in tuple(self._inflight.values()):
            task.cancel()
        self._inflight.clear()

        failures: list[tuple[str, BaseException]] = []
        for label, sdk_client in (("tee", self._tee_client), ("routed", self._routed_client)):
            try:
                await sdk_client.aclose()
            except Exception as exc:  # noqa: BLE001 - close must not mask other closes
                failures.append((label, exc))

        await self._reset_metadata_client()

        self.reset_caches()

        if failures:
            from chutes_litellm import get_logger

            logger = get_logger(_LOGGER_NAME)
            for label, error in failures:
                logger.warning("NEAR AI runtime close failed for %s client: %s", label, _safe_error_code(error))

    def _require_open(self, model: str) -> None:
        if self._closed:
            raise NearUnavailableError(CODE_UNAVAILABLE_CLOSED, model=model or None)


# ---------------------------------------------------------------------------
# Process-wide runtime registry: one runtime per API key
# ---------------------------------------------------------------------------

_RUNTIMES: dict[str, NearRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime(
    api_key: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> NearRuntime:
    """Return the process-wide runtime for ``api_key``, creating it once.

    The key comes from ``api_key`` or ``NEARAI_API_KEY``.  A missing or empty
    key raises :class:`NearUnavailableError` with a stable code; Phase 3 maps
    that to an authentication error.  Nothing is instantiated at import time.
    """
    source = os.environ if environ is None else environ
    candidate = api_key if api_key is not None else source.get(API_KEY_ENV)
    key = candidate.strip() if isinstance(candidate, str) else ""
    if not key:
        raise NearUnavailableError(CODE_UNAVAILABLE_API_KEY)

    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = NearRuntime(key, environ=environ)
            _RUNTIMES[key] = runtime
        return runtime


async def close_runtimes() -> None:
    """Close and forget every process-wide runtime.  Safe to call twice."""
    with _RUNTIMES_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()

    for runtime in runtimes:
        try:
            await runtime.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown keeps closing the rest
            from chutes_litellm import get_logger

            get_logger(_LOGGER_NAME).warning(
                "NEAR AI runtime shutdown failed: %s",
                _safe_error_code(exc),
            )


def reset_runtimes() -> None:
    """Forget every runtime, closing them on a best-effort basis (tests).

    Outside a running event loop each runtime is closed with ``asyncio.run``.
    Inside one, closure is scheduled as a task and is **not** awaited: clearing
    the map is immediate, but resource release happens when that task runs.  Use
    :func:`close_runtimes` from async shutdown paths when awaiting matters.
    """
    with _RUNTIMES_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()

    for runtime in runtimes:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(runtime.aclose())
            except Exception as exc:  # noqa: BLE001 - best effort in tests
                from chutes_litellm import get_logger

                get_logger(_LOGGER_NAME).debug(
                    "NEAR AI best-effort runtime close failed: %s",
                    _safe_error_code(exc),
                )
        else:
            try:
                asyncio.get_running_loop().create_task(runtime.aclose())
            except Exception as exc:  # noqa: BLE001 - loop may be shutting down
                from chutes_litellm import get_logger

                get_logger(_LOGGER_NAME).debug(
                    "NEAR AI runtime close could not be scheduled: %s",
                    _safe_error_code(exc),
                )


__all__ = [
    "API_KEY_ENV",
    "CODE_ATTESTATION_FAILED",
    "CODE_UNAVAILABLE_API_KEY",
    "CODE_UNAVAILABLE_ATTESTATION",
    "CODE_UNAVAILABLE_CLOSED",
    "CODE_UNAVAILABLE_MODE_UNKNOWN",
    "CODE_UNAVAILABLE_TIMEOUT",
    "CODE_UNAVAILABLE_TRANSPORT",
    "MEASUREMENTS_FILE_ENV",
    "PREWARM_MODELS_ENV",
    "NearRuntime",
    "NearUnavailableError",
    "close_runtimes",
    "get_runtime",
    "prewarm_models_from_env",
    "reset_runtimes",
]
