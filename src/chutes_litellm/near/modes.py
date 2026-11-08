"""Model classification (TEE vs. routed) and the frozen per-process mode map.

NEAR AI Cloud serves two kinds of models (``PLAN.md`` §4):

* **TEE** -- ``provider_type == "vllm"`` and ``attestation_supported``.  These
  require E2EE, strict model attestation, and a ``provider_tee`` receipt.
* **Routed** -- everything else.  These rely on Gateway attestation, TLS
  pinning, and a ``gateway`` receipt.

The first observed mode for a model is frozen for the lifetime of the process.
A later observation that disagrees fails closed, including routed -> TEE: a
runtime mode change is never trusted.  ``NEARAI_TEE_MODELS`` can only *increase*
strictness; if a forced-TEE model's metadata says routed, the observation is
rejected.  There is deliberately no "force routed" option.
"""

from __future__ import annotations

import fnmatch
import os
import threading
from collections.abc import Iterable, Mapping
from typing import Literal

from nearai_inference_sdk import ModelMetadata

from chutes_litellm.near.policy import FORCED_TEE_ENV, NearPolicyError, strip_model_prefix

ModelMode = Literal["tee", "routed"]

CODE_FORCED_TEE_MISMATCH = "near.mode.forced_tee_mismatch"
CODE_MODE_CHANGED = "near.mode.changed"

_LOGGER_NAME = "chutes_litellm.near.modes"


def mode_from_metadata(metadata: ModelMetadata) -> ModelMode:
    """Classify a model from its Cloud API metadata.

    TEE iff the provider is NEAR's ``vllm`` TEE deployment *and* it advertises
    attestation support; any other combination is routed.
    """
    if metadata.provider_type == "vllm" and metadata.attestation_supported:
        return "tee"
    return "routed"


class ForcedTeeMatcher:
    """Matcher for the ``NEARAI_TEE_MODELS`` allowlist.

    The spec is a comma-separated list of exact model ids or globs matched with
    :func:`fnmatch.fnmatchcase` (case-sensitive).  Whitespace is ignored, empty
    entries are dropped, and an empty or unset value matches nothing.
    """

    def __init__(self, patterns: Iterable[str] | str = ()) -> None:
        if isinstance(patterns, str):
            patterns = patterns.split(",")
        self._patterns: tuple[str, ...] = tuple(
            strip_model_prefix(pattern.strip()) for pattern in patterns if pattern and pattern.strip()
        )

    @classmethod
    def from_spec(cls, spec: str | None) -> ForcedTeeMatcher:
        """Build a matcher from a raw comma-separated spec (``None`` -> empty)."""
        return cls(spec or "")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ForcedTeeMatcher:
        """Read ``NEARAI_TEE_MODELS`` from ``environ`` (default: ``os.environ``)."""
        source = os.environ if environ is None else environ
        return cls.from_spec(source.get(FORCED_TEE_ENV))

    @property
    def patterns(self) -> tuple[str, ...]:
        """The normalized patterns in configuration order."""
        return self._patterns

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def matches(self, model: str) -> bool:
        """Return ``True`` if ``model`` matches any configured pattern.

        A leading ``nearai/`` or ``nearai_e2ee/`` is stripped so a user-facing
        model name can be copied into ``NEARAI_TEE_MODELS``.  Bare upstream ids
        remain canonical.
        """
        candidate = strip_model_prefix(model.strip())
        return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in self._patterns)


class ModelModeRegistry:
    """Process-local, frozen model-mode map.

    Construct one per runtime (Phase 2 creates one per API-key runtime).  The
    forced-TEE environment is read at construction time, not import time.
    ``observe`` is safe to call concurrently; the critical section is short and
    protected by a plain lock (callers are async but the work is synchronous).

    Security invariants:

    * The first observed mode is frozen.
    * Any later mode change fails closed -- TEE -> routed is a downgrade attempt,
      routed -> TEE is an untrusted mode change.
    * A forced-TEE model whose metadata reports routed is rejected immediately.
    """

    def __init__(
        self,
        forced_tee: ForcedTeeMatcher | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._forced_tee = forced_tee if forced_tee is not None else ForcedTeeMatcher.from_env(environ)
        self._modes: dict[str, ModelMode] = {}
        self._lock = threading.Lock()

    @property
    def forced_tee(self) -> ForcedTeeMatcher:
        """The forced-TEE matcher in use."""
        return self._forced_tee

    def observe(self, model: str, metadata: ModelMetadata) -> ModelMode:
        """Freeze and return the mode for ``model``.

        If the mode was already frozen, the observed mode must match it.

        Raises:
            NearPolicyError: If a forced-TEE model reports routed metadata, or
                if the observed mode differs from the frozen mode.
        """
        observed = mode_from_metadata(metadata)
        with self._lock:
            previous = self._modes.get(model)
            if previous is not None and previous != observed:
                self._log_mode_change(model, previous, observed)
                raise NearPolicyError(
                    CODE_MODE_CHANGED,
                    details={"previous_mode": previous, "observed_mode": observed},
                )
            if observed == "routed" and self._forced_tee.matches(model):
                self._log_forced_mismatch(model)
                raise NearPolicyError(
                    CODE_FORCED_TEE_MISMATCH,
                    details={"observed_mode": observed},
                )
            if previous is None:
                self._modes[model] = observed
            return observed

    def get(self, model: str) -> ModelMode | None:
        """Return the frozen mode for ``model``, or ``None`` if never observed."""
        with self._lock:
            return self._modes.get(model)

    def is_frozen(self, model: str) -> bool:
        """Return ``True`` if a mode has been observed (and thus frozen)."""
        with self._lock:
            return model in self._modes

    def reset(self) -> None:
        """Forget every observed mode (tests only; the forced list is kept)."""
        with self._lock:
            self._modes.clear()

    @staticmethod
    def _log_mode_change(model: str, previous: ModelMode, observed: ModelMode) -> None:
        from chutes_litellm import get_logger

        logger = get_logger(_LOGGER_NAME)
        if previous == "tee" and observed == "routed":
            logger.error(
                "NEAR AI downgrade attempt for model %s: frozen tee, metadata reports routed; refusing",
                model,
            )
        else:
            logger.error(
                "NEAR AI model mode change for model %s: %s -> %s; refusing",
                model,
                previous,
                observed,
            )

    @staticmethod
    def _log_forced_mismatch(model: str) -> None:
        from chutes_litellm import get_logger

        get_logger(_LOGGER_NAME).error(
            "NEAR AI forced-TEE mismatch for model %s: %s matched but metadata reports routed; refusing",
            model,
            FORCED_TEE_ENV,
        )


__all__ = [
    "CODE_FORCED_TEE_MISMATCH",
    "CODE_MODE_CHANGED",
    "ForcedTeeMatcher",
    "ModelMode",
    "ModelModeRegistry",
    "mode_from_metadata",
]
