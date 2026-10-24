"""Unit tests for TDX quote parsing, DCAP cross-checks and TCB policy."""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from chutes_litellm.tee import quote as tee_quote
from chutes_litellm.tee.errors import AttestationError
from chutes_litellm.tee.quote import (
    TdxQuote,
    TdxVerificationResult,
    parse_and_crosscheck,
    resolve_tdx_tcb_status,
    verify_tdx_quote,
)

HEADER_LEN = 48
REPORT_LEN = 584
QUOTE_LEN = HEADER_LEN + REPORT_LEN
RTMR_FIELDS = ("rtmr0", "rtmr1", "rtmr2", "rtmr3")
TD10_KEYS = {
    "mrtd": "mr_td",
    "rtmr0": "rt_mr0",
    "rtmr1": "rt_mr1",
    "rtmr2": "rt_mr2",
    "rtmr3": "rt_mr3",
}

DEFAULT_MRTD = "ab" * 48
DEFAULT_RTMRS = ("cd" * 48, "ef" * 48, "01" * 48, "23" * 48)
DEFAULT_REPORT_DATA = "45" * 64
DEFAULT_ATTRIBUTES = "00" * 8

ZERO_MRSIGNER = "00" * 48
ZERO_ATTRIBUTES = "00" * 8
FULL_MASK = "ff" * 8


def build_quote_bytes(
    *,
    version: int = 4,
    att_key_type: int = 2,
    tee_type: int = 0x81,
    mrtd: str = DEFAULT_MRTD,
    rtmrs: tuple[str, str, str, str] = DEFAULT_RTMRS,
    report_data: str = DEFAULT_REPORT_DATA,
    td_attributes: str = DEFAULT_ATTRIBUTES,
    length: int = QUOTE_LEN,
) -> bytes:
    """Build a quote with the 48-byte header + 584-byte TD-REPORT layout."""
    raw = bytearray(length)
    struct.pack_into("<HHI16s20s", raw, 0, version, att_key_type, tee_type, b"\x00" * 16, b"\x00" * 20)
    if length >= QUOTE_LEN:
        report = memoryview(raw)[HEADER_LEN:QUOTE_LEN]
        report[120:128] = bytes.fromhex(td_attributes)
        report[136:184] = bytes.fromhex(mrtd)
        for index, value in enumerate(rtmrs):
            start = 328 + index * 48
            report[start : start + 48] = bytes.fromhex(value)
        report[520:584] = bytes.fromhex(report_data)
    return bytes(raw)


def make_result(**overrides: object) -> TdxVerificationResult:
    """Build a verification result with valid defaults."""
    values: dict[str, object] = {
        "mrtd": DEFAULT_MRTD.upper(),
        "rtmr0": DEFAULT_RTMRS[0].upper(),
        "rtmr1": DEFAULT_RTMRS[1].upper(),
        "rtmr2": DEFAULT_RTMRS[2].upper(),
        "rtmr3": DEFAULT_RTMRS[3].upper(),
        "report_data": DEFAULT_REPORT_DATA.upper(),
        "td_attributes": DEFAULT_ATTRIBUTES.upper(),
        "status": "UpToDate",
        "advisory_ids": [],
    }
    values.update(overrides)
    return TdxVerificationResult(**values)  # type: ignore[arg-type]


def td10_for(quote: TdxQuote, *, td_attributes: str | None = None) -> dict[str, str]:
    """Build the TD10 report dict DCAP would return for *quote*."""
    td10 = {
        "mr_td": quote.mrtd.lower(),
        "rt_mr0": quote.rtmr0.lower(),
        "rt_mr1": quote.rtmr1.lower(),
        "rt_mr2": quote.rtmr2.lower(),
        "rt_mr3": quote.rtmr3.lower(),
        "report_data": quote.report_data.lower(),
        "td_attributes": (td_attributes or quote.td_attributes).lower(),
    }
    return td10


def report_payload(
    quote: TdxQuote,
    *,
    status: str = "UpToDate",
    advisory_ids: list[str] | None = None,
    td_attributes: str | None = None,
) -> dict[str, object]:
    """Build the decoded ``VerifiedReport.to_json()`` payload."""
    return {
        "report": {"TD10": td10_for(quote, td_attributes=td_attributes)},
        "status": status,
        "advisory_ids": list(advisory_ids or []),
    }


class FakeReport:
    """Minimal stand-in for the object returned by dcap-qvl."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_json(self) -> str:
        return json.dumps(self._payload)


def pad16(values: list[int]) -> list[int]:
    """Pad or truncate a component list to the 16 TDX TCB components."""
    return (list(values) + [0] * 16)[:16]


def build_tcb_info(
    *,
    platform_status: str = "UpToDate",
    module_status: str = "UpToDate",
    module_advisories: list[str] | None = None,
    module_isvsvn: int = 3,
    identity_id: str = "TDX_03",
) -> dict[str, object]:
    """Build a minimal but structurally valid Intel TDX TCB Info document."""
    return {
        "id": "TDX",
        "version": 3,
        "tcbLevels": [
            {
                "tcb": {
                    "sgxtcbcomponents": [{"svn": 2} for _ in range(16)],
                    "pcesvn": 13,
                    "tdxtcbcomponents": [{"svn": 3}] + [{"svn": 0} for _ in range(15)],
                },
                "tcbStatus": platform_status,
                "advisoryIDs": [],
            }
        ],
        "tdxModule": {
            "mrsigner": ZERO_MRSIGNER,
            "attributes": ZERO_ATTRIBUTES,
            "attributesMask": FULL_MASK,
        },
        "tdxModuleIdentities": [
            {
                "id": identity_id,
                "mrsigner": ZERO_MRSIGNER,
                "attributes": ZERO_ATTRIBUTES,
                "attributesMask": FULL_MASK,
                "tcbLevels": [
                    {
                        "tcb": {"isvsvn": module_isvsvn},
                        "tcbStatus": module_status,
                        "advisoryIDs": list(module_advisories or []),
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_from_bytes_extracts_tdx_v4_fields() -> None:
    mrtd = "ab" * 48
    rtmrs = ("cd" * 48, "ef" * 48, "01" * 48, "23" * 48)
    report_data = "45" * 64
    attributes = "08" + "00" * 7
    raw = build_quote_bytes(mrtd=mrtd, rtmrs=rtmrs, report_data=report_data, td_attributes=attributes)

    quote = TdxQuote.from_bytes(raw)

    assert quote.version == 4
    assert quote.att_key_type == 2
    assert quote.tee_type == 0x81
    assert quote.mrtd == mrtd.upper()
    assert quote.rtmr0 == rtmrs[0].upper()
    assert quote.rtmr1 == rtmrs[1].upper()
    assert quote.rtmr2 == rtmrs[2].upper()
    assert quote.rtmr3 == rtmrs[3].upper()
    assert quote.report_data == report_data.upper()
    assert quote.td_attributes == attributes.upper()
    assert quote.raw_bytes == raw

    assert len(quote.mrtd) == 96
    assert all(len(getattr(quote, name)) == 96 for name in RTMR_FIELDS)
    assert len(quote.report_data) == 128
    assert len(quote.td_attributes) == 16

    for name in ("mrtd", *RTMR_FIELDS, "report_data", "td_attributes"):
        value = getattr(quote, name)
        assert value == value.upper()
        assert all(char in "0123456789ABCDEF" for char in value)

    assert quote.rtmrs == {
        "rtmr0": rtmrs[0].upper(),
        "rtmr1": rtmrs[1].upper(),
        "rtmr2": rtmrs[2].upper(),
        "rtmr3": rtmrs[3].upper(),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length": 631}, "too short"),
        ({"version": 3}, "version"),
        ({"version": 6}, "version"),
        ({"tee_type": 0x82}, "TEE type"),
        ({"att_key_type": 1}, "attestation key type"),
        ({"att_key_type": 4}, "attestation key type"),
    ],
)
def test_from_bytes_rejects_malformed_quotes(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(AttestationError, match=message):
        TdxQuote.from_bytes(build_quote_bytes(**kwargs))  # type: ignore[arg-type]


def test_from_bytes_accepts_version_five() -> None:
    quote = TdxQuote.from_bytes(build_quote_bytes(version=5))
    assert quote.version == 5


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def test_debug_enabled_reads_little_endian_bit_zero() -> None:
    assert make_result(td_attributes="01" + "00" * 7).debug_enabled is True
    assert make_result(td_attributes="00" * 8).debug_enabled is False
    # fail safe: missing or unparseable attributes are treated as debug on
    assert make_result(td_attributes="").debug_enabled is True
    assert make_result(td_attributes="not-hex").debug_enabled is True


def test_is_valid_requires_clean_uptodate_verdict() -> None:
    assert make_result().is_valid is True
    assert make_result(status="OutOfDate").is_valid is False
    assert make_result(advisory_ids=["INTEL-SA-01036"]).is_valid is False
    assert make_result(td_attributes="01" + "00" * 7).is_valid is False


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------


def test_parse_and_crosscheck_accepts_wrapped_payload() -> None:
    quote = TdxQuote.from_bytes(build_quote_bytes())
    payload = report_payload(quote)

    result = parse_and_crosscheck(payload, quote)

    assert result.mrtd == quote.mrtd
    assert result.rtmr0 == quote.rtmr0
    assert result.rtmr1 == quote.rtmr1
    assert result.rtmr2 == quote.rtmr2
    assert result.rtmr3 == quote.rtmr3
    assert result.status == "UpToDate"
    assert result.advisory_ids == []
    assert result.debug_enabled is False


def test_parse_and_crosscheck_accepts_direct_td10_dict() -> None:
    quote = TdxQuote.from_bytes(build_quote_bytes())

    result = parse_and_crosscheck(td10_for(quote), quote)

    assert result.status == "UpToDate"
    assert result.mrtd == quote.mrtd
    assert result.rtmr3 == quote.rtmr3


@pytest.mark.parametrize(
    ("field", "report_key"),
    [
        ("mrtd", "mr_td"),
        ("rtmr0", "rt_mr0"),
        ("rtmr1", "rt_mr1"),
        ("rtmr2", "rt_mr2"),
        ("rtmr3", "rt_mr3"),
    ],
)
def test_parse_and_crosscheck_rejects_measurement_mismatch(field: str, report_key: str) -> None:
    quote = TdxQuote.from_bytes(build_quote_bytes())
    td10 = td10_for(quote)
    td10[report_key] = "ff" * 48
    payload = {"report": {"TD10": td10}, "status": "UpToDate", "advisory_ids": []}

    with pytest.raises(AttestationError, match=field.upper()):
        parse_and_crosscheck(payload, quote)


def test_parse_and_crosscheck_rejects_missing_td10() -> None:
    quote = TdxQuote.from_bytes(build_quote_bytes())
    with pytest.raises(AttestationError, match="no TD10"):
        parse_and_crosscheck({"report": {}, "status": "UpToDate"}, quote)


# ---------------------------------------------------------------------------
# verify_tdx_quote
# ---------------------------------------------------------------------------


def test_verify_tdx_quote_returns_result_on_valid_verdict(monkeypatch) -> None:
    raw = build_quote_bytes()
    quote = TdxQuote.from_bytes(raw)

    async def fake_dcap(raw_quote, *, pccs_url=None):
        assert raw_quote == raw
        return FakeReport(report_payload(quote))

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)

    result = asyncio.run(verify_tdx_quote(raw))

    assert isinstance(result, TdxVerificationResult)
    assert result.is_valid is True
    assert result.mrtd == quote.mrtd


@pytest.mark.parametrize(
    ("status", "advisories", "attributes"),
    [
        ("OutOfDate", [], None),
        ("UpToDate", ["INTEL-SA-01036"], None),
        ("UpToDate", [], "01" + "00" * 7),
    ],
)
def test_verify_tdx_quote_rejects_invalid_verdict(
    monkeypatch, status: str, advisories: list[str], attributes: str | None
) -> None:
    raw = build_quote_bytes()
    quote = TdxQuote.from_bytes(raw)

    async def fake_dcap(raw_quote, *, pccs_url=None):
        return FakeReport(
            report_payload(
                quote,
                status=status,
                advisory_ids=advisories,
                td_attributes=attributes,
            )
        )

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)

    with pytest.raises(AttestationError, match="not valid"):
        asyncio.run(verify_tdx_quote(raw))


def test_verify_tdx_quote_reports_collateral_fetch_failure(monkeypatch) -> None:
    raw = build_quote_bytes()

    async def fake_dcap(raw_quote, *, pccs_url=None):
        raise RuntimeError("pccs unreachable")

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)

    with pytest.raises(AttestationError, match="collateral fetch failed"):
        asyncio.run(verify_tdx_quote(raw))


def test_verify_tdx_quote_reports_generic_failure(monkeypatch) -> None:
    raw = build_quote_bytes()

    async def fake_dcap(raw_quote, *, pccs_url=None):
        raise Exception("unexpected dcap failure")

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)

    with pytest.raises(AttestationError, match="verification failed"):
        asyncio.run(verify_tdx_quote(raw))


def test_verify_tdx_quote_uses_module_identity_fallback(monkeypatch) -> None:
    raw = build_quote_bytes()
    quote = TdxQuote.from_bytes(raw)
    calls: dict[str, object] = {}

    async def fake_dcap(raw_quote, *, pccs_url=None):
        raise ValueError("No matching TCB level found for this quote")

    async def fake_resolve(raw_quote, *, pccs_url, original_error):
        calls["raw"] = raw_quote
        calls["pccs_url"] = pccs_url
        calls["original_error"] = original_error
        return make_result(
            mrtd=quote.mrtd,
            rtmr0=quote.rtmr0,
            rtmr1=quote.rtmr1,
            rtmr2=quote.rtmr2,
            rtmr3=quote.rtmr3,
            report_data=quote.report_data,
            td_attributes=quote.td_attributes,
        )

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)
    monkeypatch.setattr(tee_quote, "_resolve_tdx_tcb_via_module_identity", fake_resolve)

    result = asyncio.run(verify_tdx_quote(raw, pccs_url="https://pccs.test"))

    assert result.status == "UpToDate"
    assert result.is_valid is True
    assert calls["raw"] == raw
    assert calls["pccs_url"] == "https://pccs.test"
    assert isinstance(calls["original_error"], ValueError)


def test_verify_tdx_quote_does_not_use_fallback_for_other_value_errors(
    monkeypatch,
) -> None:
    raw = build_quote_bytes()
    calls: list[bytes] = []

    async def fake_dcap(raw_quote, *, pccs_url=None):
        raise ValueError("malformed collateral")

    async def fake_resolve(raw_quote, *, pccs_url, original_error):
        calls.append(raw_quote)
        return make_result()

    monkeypatch.setattr(tee_quote, "_dcap_verify", fake_dcap)
    monkeypatch.setattr(tee_quote, "_resolve_tdx_tcb_via_module_identity", fake_resolve)

    with pytest.raises(AttestationError, match="verification failed"):
        asyncio.run(verify_tdx_quote(raw))

    assert calls == []


# ---------------------------------------------------------------------------
# resolve_tdx_tcb_status
# ---------------------------------------------------------------------------


def test_resolve_tdx_tcb_status_returns_uptodate() -> None:
    status, advisories = resolve_tdx_tcb_status(
        tcb_info=build_tcb_info(),
        tee_tcb_svn=pad16([3, 3]),
        sgx_tcb_components=pad16([2] * 16),
        pce_svn=13,
        mr_signer_seam=ZERO_MRSIGNER,
        seam_attributes=ZERO_ATTRIBUTES,
    )

    assert status == "UpToDate"
    assert advisories == []


def test_resolve_tdx_tcb_status_merges_module_severity() -> None:
    status, advisories = resolve_tdx_tcb_status(
        tcb_info=build_tcb_info(module_status="OutOfDate", module_advisories=["INTEL-SA-01036"]),
        tee_tcb_svn=pad16([3, 3]),
        sgx_tcb_components=pad16([2] * 16),
        pce_svn=13,
        mr_signer_seam=ZERO_MRSIGNER,
        seam_attributes=ZERO_ATTRIBUTES,
    )

    assert status == "OutOfDate"
    assert advisories == ["INTEL-SA-01036"]


def test_resolve_tdx_tcb_status_fails_on_platform_mismatch() -> None:
    with pytest.raises(AttestationError, match="platform TCB level"):
        resolve_tdx_tcb_status(
            tcb_info=build_tcb_info(),
            tee_tcb_svn=pad16([3, 3]),
            sgx_tcb_components=pad16([2] * 16),
            pce_svn=0,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_ATTRIBUTES,
        )


def test_resolve_tdx_tcb_status_fails_on_missing_module_version() -> None:
    with pytest.raises(AttestationError, match="unsupported TDX module version"):
        resolve_tdx_tcb_status(
            tcb_info=build_tcb_info(),
            tee_tcb_svn=pad16([3, 99]),
            sgx_tcb_components=pad16([2] * 16),
            pce_svn=13,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_ATTRIBUTES,
        )
