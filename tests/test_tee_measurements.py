"""Unit tests for the Chutes measurement policy (parse, match, cache)."""

from __future__ import annotations

import uuid

import pytest
import yaml

from chutes_litellm import endpoints
from chutes_litellm.tee import measurements
from chutes_litellm.tee.errors import AttestationError, MeasurementMismatchError
from chutes_litellm.tee.measurements import (
    ZERO_RTMR,
    TeeMeasurement,
    get_measurements,
    match_measurement,
    parse_endpoint,
    parse_local_file,
)
from chutes_litellm.tee.quote import TdxQuote

MRTD = "AA" * 48
RTMR0 = "BB" * 48
RTMR1 = "CC" * 48
RTMR2 = "DD" * 48
RTMR3 = "EE" * 48
FINGERPRINT = "ab" * 32


def make_quote(
    *,
    mrtd: str = MRTD,
    rtmr0: str = RTMR0,
    rtmr1: str = RTMR1,
    rtmr2: str = RTMR2,
    rtmr3: str = RTMR3,
) -> TdxQuote:
    """Build a parsed quote with the test measurement values."""
    return TdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd=mrtd,
        rtmr0=rtmr0,
        rtmr1=rtmr1,
        rtmr2=rtmr2,
        rtmr3=rtmr3,
        report_data="00" * 64,
        td_attributes="00" * 8,
        raw_bytes=b"",
    )


def make_measurement(
    *,
    version: str = "v1",
    name: str = "h100",
    mrtd: str = MRTD,
    rtmr0: str = RTMR0,
    rtmr1: str = RTMR1,
    rtmr2: str = RTMR2,
    runtime_rtmr3: str = RTMR3,
    expected_gpus: list[str] | None = None,
    gpu_count: int = 8,
    fingerprint: str | None = None,
) -> TeeMeasurement:
    """Build a measurement with defaults matching :func:`make_quote`."""
    return TeeMeasurement(
        version=version,
        name=name,
        mrtd=mrtd,
        rtmr0=rtmr0,
        rtmr1=rtmr1,
        rtmr2=rtmr2,
        runtime_rtmr3=runtime_rtmr3,
        expected_gpus=list(expected_gpus or ["h100"]),
        gpu_count=gpu_count,
        fingerprint=fingerprint,
    )


def endpoint_entry(**overrides: object) -> dict[str, object]:
    """Build one ``/servers/tee/measurements`` response entry."""
    entry: dict[str, object] = {
        "version": "v1",
        "name": "h100",
        "mrtd": MRTD,
        "runtime_rtmrs": {
            "rtmr0": RTMR0,
            "rtmr1": RTMR1,
            "rtmr2": RTMR2,
            "rtmr3": RTMR3,
        },
        "expected_gpus": ["h100"],
        "gpu_count": 8,
    }
    entry.update(overrides)
    return entry


def local_entry(**overrides: object) -> dict[str, object]:
    """Build one measurement version with a single hardware variant."""
    entry: dict[str, object] = {
        "version": "v1",
        "mrtd": MRTD,
        "rtmr1": RTMR1,
        "rtmr2": RTMR2,
        "runtime_rtmr3": RTMR3,
        "hardware": [
            {
                "name": "h100",
                "rtmr0": RTMR0,
                "expected_gpus": ["h100"],
                "gpu_count": 8,
            }
        ],
    }
    entry.update(overrides)
    return entry


def write_local_file(tmp_path, entries: list[dict[str, object]]) -> str:
    """Write a Chutes-schema measurements YAML file and return its path."""
    path = tmp_path / "measurements.yaml"
    path.write_text(yaml.safe_dump({"measurements": entries}))
    return str(path)


def api_base() -> str:
    """Return a unique fake API base so cache keys never collide across tests."""
    return f"https://{uuid.uuid4().hex}.test"


@pytest.fixture()
def clean_measurement_cache(monkeypatch):
    """Reset module caches and remove the local-file override around a test."""
    monkeypatch.delenv("CHUTES_TEE_MEASUREMENTS_FILE", raising=False)
    measurements.reset_cache()
    try:
        yield
    finally:
        measurements.reset_cache()


# ---------------------------------------------------------------------------
# parse_endpoint
# ---------------------------------------------------------------------------


def test_parse_endpoint_normalizes_valid_entry() -> None:
    payload = [
        endpoint_entry(
            mrtd=MRTD.lower(),
            runtime_rtmrs={
                "rtmr0": RTMR0.lower(),
                "RTMR1": RTMR1.lower(),
                "rtmr2": RTMR2,
                "rtmr3": RTMR3,
            },
            expected_gpus=["H100", "B200"],
            gpu_count=8,
            fingerprint=FINGERPRINT,
        )
    ]

    (measurement,) = parse_endpoint(payload)

    assert measurement.mrtd == MRTD
    assert measurement.rtmr0 == RTMR0
    assert measurement.rtmr1 == RTMR1
    assert measurement.rtmr2 == RTMR2
    assert measurement.runtime_rtmr3 == RTMR3
    assert measurement.runtime_rtmrs == {
        "RTMR0": RTMR0,
        "RTMR1": RTMR1,
        "RTMR2": RTMR2,
        "RTMR3": RTMR3,
    }
    assert measurement.expected_gpus == ["h100", "b200"]
    assert measurement.gpu_count == 8
    assert measurement.fingerprint == FINGERPRINT


def test_parse_endpoint_falls_back_to_boot_rtmrs() -> None:
    entry = endpoint_entry(
        runtime_rtmrs=None,
        boot_rtmrs={
            "rtmr0": RTMR0,
            "rtmr1": RTMR1,
            "rtmr2": RTMR2,
            "rtmr3": RTMR3,
        },
    )

    (measurement,) = parse_endpoint([entry])

    assert measurement.rtmr0 == RTMR0
    assert measurement.runtime_rtmr3 == RTMR3


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-a-list", "unexpected payload"),
        ({}, "unexpected payload"),
        ([], "no entries"),
        ([endpoint_entry(mrtd="zz" * 48)], "invalid MRTD"),
        ([endpoint_entry(gpu_count="many")], "gpu_count"),
    ],
)
def test_parse_endpoint_rejects_invalid_payloads(payload: object, message: str) -> None:
    with pytest.raises(AttestationError, match=message):
        parse_endpoint(payload)


# ---------------------------------------------------------------------------
# parse_local_file
# ---------------------------------------------------------------------------


def test_parse_local_file_expands_hardware_variants(tmp_path) -> None:
    second_rtmr0 = "12" * 48
    entries = [
        local_entry(
            hardware=[
                {
                    "name": "h100",
                    "rtmr0": RTMR0,
                    "expected_gpus": ["H100"],
                    "gpu_count": 8,
                    "fingerprint": FINGERPRINT,
                },
                {
                    "name": "b200",
                    "rtmr0": second_rtmr0,
                    "expected_gpus": ["B200"],
                    "gpu_count": 8,
                },
            ]
        )
    ]
    path = write_local_file(tmp_path, entries)

    parsed = parse_local_file(path)

    assert [measurement.name for measurement in parsed] == ["h100", "b200"]
    assert all(measurement.version == "v1" for measurement in parsed)
    assert all(measurement.mrtd == MRTD for measurement in parsed)
    assert all(measurement.rtmr1 == RTMR1 for measurement in parsed)
    assert all(measurement.rtmr2 == RTMR2 for measurement in parsed)
    assert all(measurement.runtime_rtmr3 == RTMR3 for measurement in parsed)
    assert parsed[0].rtmr0 == RTMR0
    assert parsed[1].rtmr0 == second_rtmr0
    assert parsed[0].expected_gpus == ["h100"]
    assert parsed[1].expected_gpus == ["b200"]
    assert parsed[0].fingerprint == FINGERPRINT
    assert parsed[1].fingerprint is None


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([local_entry(hardware=[])], "at least one hardware"),
        ([local_entry(version="")], "no version"),
        (
            [
                local_entry(
                    hardware=[
                        {
                            "name": "h100",
                            "rtmr0": "zz" * 48,
                            "gpu_count": 8,
                        }
                    ]
                )
            ],
            "invalid RTMR0",
        ),
    ],
)
def test_parse_local_file_rejects_invalid_entries(tmp_path, entries: list[dict[str, object]], message: str) -> None:
    path = write_local_file(tmp_path, entries)
    with pytest.raises(AttestationError, match=message):
        parse_local_file(path)


# ---------------------------------------------------------------------------
# TeeMeasurement.matches
# ---------------------------------------------------------------------------


def test_matches_returns_true_on_exact_runtime_match() -> None:
    assert make_measurement().matches(make_quote()) is True


@pytest.mark.parametrize("field", ["mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"])
def test_matches_rejects_each_single_field_mismatch(field: str) -> None:
    measurement = make_measurement()
    quote = make_quote(**{field: "FF" * 48})
    assert measurement.matches(quote) is False


def test_matches_boot_mode_uses_zero_rtmr3() -> None:
    measurement = make_measurement(runtime_rtmr3=RTMR3)

    assert measurement.matches(make_quote(rtmr3=ZERO_RTMR), runtime=False) is True
    assert measurement.matches(make_quote(rtmr3=RTMR3), runtime=False) is False
    assert measurement.matches(make_quote(rtmr3=ZERO_RTMR), runtime=True) is False


# ---------------------------------------------------------------------------
# match_measurement
# ---------------------------------------------------------------------------


def test_match_measurement_returns_unique_match() -> None:
    measurement = make_measurement(name="matching")
    other = make_measurement(name="other", mrtd="FF" * 48)

    assert match_measurement(make_quote(), [other, measurement]) is measurement


def test_match_measurement_raises_when_nothing_matches() -> None:
    other = make_measurement(mrtd="FF" * 48)

    with pytest.raises(MeasurementMismatchError, match="does not match any"):
        match_measurement(make_quote(), [other])


def test_match_measurement_rejects_ambiguous_gpu_expectations() -> None:
    first = make_measurement(name="h100", expected_gpus=["h100"], gpu_count=8)
    second = make_measurement(name="b200", expected_gpus=["b200"], gpu_count=8)

    with pytest.raises(MeasurementMismatchError, match="ambiguous"):
        match_measurement(make_quote(), [first, second])


def test_match_measurement_allows_identical_gpu_expectations() -> None:
    first = make_measurement(name="first", expected_gpus=["h100"], gpu_count=8)
    second = make_measurement(name="second", expected_gpus=["h100"], gpu_count=8)

    assert match_measurement(make_quote(), [first, second]) is first


# ---------------------------------------------------------------------------
# get_measurements caching / fail-closed
# ---------------------------------------------------------------------------


def test_get_measurements_caches_within_ttl_and_force_refetches(clean_measurement_cache, monkeypatch) -> None:
    monkeypatch.setenv("CHUTES_TEE_MEASUREMENTS_TTL", "3600")
    calls: list[str] = []

    def fake_fetch(url: str) -> list[dict[str, object]]:
        calls.append(url)
        return [endpoint_entry()]

    monkeypatch.setattr(measurements, "_fetch_json", fake_fetch)
    base = api_base()
    monkeypatch.setattr(endpoints, "API_BASE", base)

    first = get_measurements()
    second = get_measurements()

    assert len(calls) == 1
    assert second == first
    assert calls[0] == f"{base}/servers/tee/measurements"

    forced = get_measurements(force=True)

    assert len(calls) == 2
    assert forced == first


def test_get_measurements_ttl_zero_forces_refetch(clean_measurement_cache, monkeypatch) -> None:
    monkeypatch.setenv("CHUTES_TEE_MEASUREMENTS_TTL", "0")
    calls: list[str] = []

    def fake_fetch(url: str) -> list[dict[str, object]]:
        calls.append(url)
        return [endpoint_entry()]

    monkeypatch.setattr(measurements, "_fetch_json", fake_fetch)
    monkeypatch.setattr(endpoints, "API_BASE", api_base())

    get_measurements()
    get_measurements()

    assert len(calls) == 2


def test_get_measurements_fails_closed_without_cache(clean_measurement_cache, monkeypatch) -> None:
    def fake_fetch(url: str) -> list[dict[str, object]]:
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(measurements, "_fetch_json", fake_fetch)
    monkeypatch.setattr(endpoints, "API_BASE", api_base())

    with pytest.raises(AttestationError, match="unavailable"):
        get_measurements()


def test_get_measurements_serves_last_known_good(clean_measurement_cache, monkeypatch) -> None:
    monkeypatch.setenv("CHUTES_TEE_MEASUREMENTS_TTL", "0")
    state = {"fail": False}
    calls: list[str] = []

    def fake_fetch(url: str) -> list[dict[str, object]]:
        calls.append(url)
        if state["fail"]:
            raise RuntimeError("endpoint down")
        return [endpoint_entry()]

    monkeypatch.setattr(measurements, "_fetch_json", fake_fetch)
    monkeypatch.setattr(endpoints, "API_BASE", api_base())

    first = get_measurements()
    state["fail"] = True
    second = get_measurements()

    assert len(calls) == 2
    assert second == first


def test_get_measurements_local_file_is_sole_source(clean_measurement_cache, monkeypatch, tmp_path) -> None:
    path = write_local_file(tmp_path, [local_entry()])
    monkeypatch.setenv("CHUTES_TEE_MEASUREMENTS_FILE", path)
    calls: list[str] = []

    def forbidden_fetch(url: str) -> list[dict[str, object]]:
        calls.append(url)
        return []

    monkeypatch.setattr(measurements, "_fetch_json", forbidden_fetch)
    monkeypatch.setattr(endpoints, "API_BASE", api_base())

    parsed = get_measurements()

    assert calls == []
    assert len(parsed) == 1
    assert parsed[0].mrtd == MRTD
    assert parsed[0].rtmr0 == RTMR0
    assert parsed[0].name == "h100"
