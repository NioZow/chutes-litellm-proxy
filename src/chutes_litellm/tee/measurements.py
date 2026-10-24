"""Published Chutes TEE measurement policy: fetch, cache, parse and match.

Ported from ``chutesai/chutes-api`` @ ``3b5609f42f84e29dea374382ef1fa95b0fda329c``
(``api/config/__init__.py: TeeMeasurementConfig`` and
``api/server/router.py: GET /servers/tee/measurements``).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
import yaml

from .. import endpoints
from .errors import AttestationError, MeasurementMismatchError
from .quote import TdxQuote

ZERO_RTMR = "0" * 96
_HEX_CHARS = frozenset("0123456789ABCDEF")

_cache_lock = threading.Lock()
# source -> (fetched_at, measurements)
_cache: dict[str, tuple[float, list["TeeMeasurement"]]] = {}
# source -> last successful measurements (served on transient endpoint failure)
_last_good: dict[str, list["TeeMeasurement"]] = {}


@dataclass(frozen=True)
class TeeMeasurement:
    """One approved VM version + hardware variant."""

    version: str
    name: str
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    runtime_rtmr3: str
    expected_gpus: list[str]
    gpu_count: int
    fingerprint: str | None = None

    @property
    def boot_rtmrs(self) -> dict[str, str]:
        return {
            "RTMR0": self.rtmr0,
            "RTMR1": self.rtmr1,
            "RTMR2": self.rtmr2,
            "RTMR3": ZERO_RTMR,
        }

    @property
    def runtime_rtmrs(self) -> dict[str, str]:
        return {
            "RTMR0": self.rtmr0,
            "RTMR1": self.rtmr1,
            "RTMR2": self.rtmr2,
            "RTMR3": self.runtime_rtmr3,
        }

    def matches(self, quote: TdxQuote, *, runtime: bool = True) -> bool:
        """Exact MRTD + all four RTMRs (runtime set for chute evidence)."""
        if quote.mrtd.upper() != self.mrtd.upper():
            return False
        expected = self.runtime_rtmrs if runtime else self.boot_rtmrs
        for name, value in expected.items():
            actual = quote.rtmrs.get(name.lower())
            if not actual or actual.upper() != value.upper():
                return False
        return True


def _hex96(value: Any, field_name: str, owner: str) -> str:
    cleaned = str(value).strip().upper()
    if len(cleaned) != 96 or any(c not in _HEX_CHARS for c in cleaned):
        raise AttestationError(
            f"invalid {field_name} for measurement '{owner}': expected 96 hex characters, got {cleaned!r}"
        )
    return cleaned


def _fingerprint(value: Any, owner: str) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned:
        return None
    if len(cleaned) != 64 or any(c not in "0123456789abcdef" for c in cleaned):
        raise AttestationError(f"invalid fingerprint for measurement '{owner}': expected 64 hex chars")
    return cleaned


def _gpu_count(value: Any, owner: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise AttestationError(f"missing or invalid gpu_count for measurement '{owner}'") from exc
    if count < 0:
        raise AttestationError(f"invalid gpu_count for measurement '{owner}': {count}")
    return count


def _gpu_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(gpu).strip().lower() for gpu in value if str(gpu).strip()]


def parse_local_file(path: str) -> list[TeeMeasurement]:
    """Parse a locally pinned YAML file using the Chutes measurement schema."""
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except Exception as exc:  # noqa: BLE001
        raise AttestationError(f"failed to read TEE measurements file {path!r}: {exc}") from exc

    if not isinstance(config, dict):
        raise AttestationError(f"TEE measurements file {path!r} must contain a mapping")

    measurements: list[TeeMeasurement] = []
    for version_config in config.get("measurements") or []:
        version = str(version_config.get("version") or "").strip()
        if not version:
            raise AttestationError(f"measurement in {path!r} has no version")
        mrtd = _hex96(version_config.get("mrtd"), "MRTD", version)
        rtmr1 = _hex96(version_config.get("rtmr1"), "RTMR1", version)
        rtmr2 = _hex96(version_config.get("rtmr2"), "RTMR2", version)
        runtime_rtmr3 = _hex96(version_config.get("runtime_rtmr3"), "runtime RTMR3", version)

        hardware = version_config.get("hardware") or []
        if not hardware:
            raise AttestationError(f"measurement version '{version}' must define at least one hardware variant")
        for hw in hardware:
            name = str(hw.get("name") or "unnamed")
            owner = f"{version}/{name}"
            measurements.append(
                TeeMeasurement(
                    version=version,
                    name=name,
                    mrtd=mrtd,
                    rtmr0=_hex96(hw.get("rtmr0"), "RTMR0", owner),
                    rtmr1=rtmr1,
                    rtmr2=rtmr2,
                    runtime_rtmr3=runtime_rtmr3,
                    expected_gpus=_gpu_list(hw.get("expected_gpus")),
                    gpu_count=_gpu_count(hw.get("gpu_count"), owner),
                    fingerprint=_fingerprint(hw.get("fingerprint"), owner),
                )
            )

    if not measurements:
        raise AttestationError(f"TEE measurements file {path!r} defines no measurements")
    return measurements


def _rtmr(rtmrs: Any, name: str, owner: str) -> str:
    if not isinstance(rtmrs, dict):
        raise AttestationError(f"measurement '{owner}' is missing {name} RTMRs")
    wanted = name.upper()
    for key, value in rtmrs.items():
        if str(key).upper() == wanted:
            return _hex96(value, name, owner)
    raise AttestationError(f"measurement '{owner}' is missing {name}")


def parse_endpoint(payload: Any) -> list[TeeMeasurement]:
    """Parse ``GET /servers/tee/measurements`` (a list of response objects)."""
    if isinstance(payload, dict):
        payload = payload.get("measurements")
    if not isinstance(payload, list):
        raise AttestationError("TEE measurements endpoint returned an unexpected payload")

    measurements: list[TeeMeasurement] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        version = str(entry.get("version") or "").strip()
        name = str(entry.get("name") or "unnamed")
        owner = f"{version or '?'}/{name}"
        runtime = entry.get("runtime_rtmrs")
        boot = entry.get("boot_rtmrs")
        rtmrs = runtime if runtime is not None else boot
        measurements.append(
            TeeMeasurement(
                version=version,
                name=name,
                mrtd=_hex96(entry.get("mrtd"), "MRTD", owner),
                rtmr0=_rtmr(rtmrs, "RTMR0", owner),
                rtmr1=_rtmr(rtmrs, "RTMR1", owner),
                rtmr2=_rtmr(rtmrs, "RTMR2", owner),
                runtime_rtmr3=_rtmr(rtmrs, "RTMR3", owner),
                expected_gpus=_gpu_list(entry.get("expected_gpus")),
                gpu_count=_gpu_count(entry.get("gpu_count"), owner),
                fingerprint=_fingerprint(entry.get("fingerprint"), owner),
            )
        )

    if not measurements:
        raise AttestationError("TEE measurements endpoint returned no entries")
    return measurements


def _fetch_json(url: str) -> Any:
    """HTTP GET isolated so tests can replace it without a network."""
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _source() -> tuple[str, str]:
    """Return ``(cache_key, path_or_base)`` for the measurement source.

    A pinned ``CHUTES_TEE_MEASUREMENTS_FILE`` wins; otherwise the fixed
    ``endpoints.API_BASE`` is used.  The API source key is the constant string
    ``"api"`` because the base itself cannot vary.
    """
    local = os.environ.get("CHUTES_TEE_MEASUREMENTS_FILE")
    if local:
        return f"file:{local}", local
    return "api", endpoints.API_BASE


def _ttl() -> int:
    try:
        return int(os.environ.get("CHUTES_TEE_MEASUREMENTS_TTL", "3600"))
    except ValueError:
        return 3600


def _load(source: str, path_or_base: str) -> list[TeeMeasurement]:
    if source.startswith("file:"):
        return parse_local_file(path_or_base)
    return parse_endpoint(_fetch_json(f"{path_or_base}/servers/tee/measurements"))


def get_measurements(*, force: bool = False) -> list[TeeMeasurement]:
    """Return the approved measurement list (cached, fail closed if unavailable).

    A configured ``CHUTES_TEE_MEASUREMENTS_FILE`` is the sole source; otherwise
    the fixed public Chutes API (``endpoints.API_BASE``) is used.  A transient
    endpoint failure serves the last-known-good list; with no cached list it
    fails closed.
    """
    source, path_or_base = _source()
    now = time.time()
    ttl = _ttl()

    if not force:
        with _cache_lock:
            cached = _cache.get(source)
        if cached is not None and now - cached[0] < ttl:
            return cached[1]

    try:
        measurements = _load(source, path_or_base)
    except Exception as exc:  # noqa: BLE001
        if not source.startswith("file:"):
            with _cache_lock:
                previous = _last_good.get(source)
            if previous is not None:
                from chutes_litellm import get_logger

                get_logger("chutes_litellm.tee.measurements").warning(
                    "TEE measurements endpoint unavailable; using last-known-good cache: %s",
                    exc,
                )
                return previous
        raise AttestationError(f"TEE measurements are unavailable (source {source!r}): {exc}") from exc

    with _cache_lock:
        _cache[source] = (now, measurements)
        _last_good[source] = measurements
    return measurements


def match_measurement(quote: TdxQuote, measurements: list[TeeMeasurement]) -> TeeMeasurement:
    """Return the unique measurement matching *quote* (runtime RTMR set)."""
    matches = [m for m in measurements if m.matches(quote, runtime=True)]
    if not matches:
        raise MeasurementMismatchError(
            "TDX quote does not match any published Chutes measurement "
            f"(MRTD={quote.mrtd[:16]}…, RTMR0={quote.rtmr0[:16]}…)"
        )

    first = matches[0]
    for other in matches[1:]:
        if sorted(other.expected_gpus) != sorted(first.expected_gpus) or other.gpu_count != first.gpu_count:
            raise MeasurementMismatchError(
                "TDX quote matches multiple measurements with conflicting GPU "
                f"expectations ({first.version}/{first.name} vs "
                f"{other.version}/{other.name}); failing closed as ambiguous"
            )
    return first


def reset_cache() -> None:
    """Clear caches (mainly for tests)."""
    with _cache_lock:
        _cache.clear()
        _last_good.clear()
