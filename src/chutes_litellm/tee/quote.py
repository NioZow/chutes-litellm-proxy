"""Intel TDX quote parsing, DCAP verification and TCB policy.

Ported from ``chutesai/chutes-api`` @ ``3b5609f42f84e29dea374382ef1fa95b0fda329c``:

* ``api/server/quote.py`` (``TdxQuote`` / ``TdxVerificationResult`` /
  ``resolve_tdx_tcb_status``)
* ``api/server/util.py`` (``verify_quote_signature`` and the TDX-module TCB
  fallback)

The upstream verifier is the authority for the evidence this proxy consumes;
this module is a pure-Python port so users only need to fork this repository.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

from .errors import AttestationError

RTMR_KEYS = ("rtmr0", "rtmr1", "rtmr2", "rtmr3")

# Intel TDX: debug attribute is bit 0 of td_attributes (Linux TDX_ATTR_DEBUG_BIT).
TDX_ATTR_DEBUG_BIT = 0

# TCB status severity, least-current wins when merging platform and TDX-module
# statuses.  Mirrors dcap-qvl's TcbStatus::severity ordering.
_TCB_STATUS_SEVERITY = {
    "UpToDate": 0,
    "SWHardeningNeeded": 1,
    "ConfigurationNeeded": 2,
    "ConfigurationAndSWHardeningNeeded": 3,
    "OutOfDate": 4,
    "OutOfDateConfigurationNeeded": 5,
    "Revoked": 6,
}


@dataclass(frozen=True)
class TdxQuote:
    """Parsed TDX quote with the measurements the verifier cares about."""

    version: int
    att_key_type: int
    tee_type: int
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    report_data: str
    td_attributes: str
    raw_bytes: bytes

    @property
    def rtmrs(self) -> dict[str, str]:
        return {
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "rtmr3": self.rtmr3,
        }

    @classmethod
    def from_bytes(cls, data: bytes) -> "TdxQuote":
        """Parse the fixed TDX quote layout (header + 584-byte TD-REPORT)."""
        if len(data) < 632:
            raise AttestationError(f"TDX quote too short: {len(data)} bytes")
        try:
            version, att_key_type, tee_type = struct.unpack_from("<HHI", data, 0)
        except struct.error as exc:  # pragma: no cover - guarded by the length check
            raise AttestationError(f"invalid TDX quote header: {exc}") from exc

        if version not in (4, 5):
            raise AttestationError(f"invalid TDX quote version: {version} (expected 4 or 5)")
        if tee_type != 0x81:
            raise AttestationError(f"invalid TEE type: {tee_type:08x} (expected 0x81 for TDX)")
        if att_key_type not in (2, 3):
            raise AttestationError(f"invalid attestation key type: {att_key_type}")

        td_report = data[48:632]
        return cls(
            version=version,
            att_key_type=att_key_type,
            tee_type=tee_type,
            mrtd=td_report[136:184].hex().upper(),
            rtmr0=td_report[328:376].hex().upper(),
            rtmr1=td_report[376:424].hex().upper(),
            rtmr2=td_report[424:472].hex().upper(),
            rtmr3=td_report[472:520].hex().upper(),
            report_data=td_report[520:584].hex().upper(),
            td_attributes=td_report[120:128].hex().upper(),
            raw_bytes=data,
        )


@dataclass(frozen=True)
class TdxVerificationResult:
    """The DCAP-verified measurements + policy verdict for a quote."""

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    report_data: str
    td_attributes: str
    status: str
    advisory_ids: list[str]

    @property
    def debug_enabled(self) -> bool:
        """True if the TD has debug mode enabled (bit 0 of td_attributes)."""
        if not self.td_attributes:
            return True  # treat missing as unsafe
        try:
            value = int.from_bytes(bytes.fromhex(self.td_attributes), byteorder="little")
        except (ValueError, TypeError):
            return True  # treat unparseable as unsafe
        return bool(value & (1 << TDX_ATTR_DEBUG_BIT))

    @property
    def is_valid(self) -> bool:
        return self.status == "UpToDate" and not self.advisory_ids and not self.debug_enabled


def _norm_hex(value: Any) -> str:
    return str(value or "").upper()


def _result_from_td10(td10: dict[str, Any], *, status: str, advisory_ids: list[str]) -> TdxVerificationResult:
    return TdxVerificationResult(
        mrtd=_norm_hex(td10.get("mr_td")),
        rtmr0=_norm_hex(td10.get("rt_mr0")),
        rtmr1=_norm_hex(td10.get("rt_mr1")),
        rtmr2=_norm_hex(td10.get("rt_mr2")),
        rtmr3=_norm_hex(td10.get("rt_mr3")),
        report_data=_norm_hex(td10.get("report_data")),
        td_attributes=_norm_hex(td10.get("td_attributes")),
        status=status,
        advisory_ids=list(advisory_ids or []),
    )


def parse_and_crosscheck(report_json: dict[str, Any], quote: TdxQuote) -> TdxVerificationResult:
    """Build the verification result and prove DCAP saw the same quote bytes.

    ``report_json`` is the decoded ``VerifiedReport.to_json()`` payload
    (``{"report": {"TD10": {...}}, "status": ..., "advisory_ids": ...}``).  A
    direct ``TD10`` dict is also accepted for convenience.
    """
    if "report" in report_json or "status" in report_json:
        report = report_json.get("report") or {}
        td10 = report.get("TD10") or {}
        status = str(report_json.get("status") or "Unknown")
        advisory_ids = list(report_json.get("advisory_ids") or [])
    else:
        td10 = report_json
        status = "UpToDate"
        advisory_ids = []

    if not td10:
        raise AttestationError("DCAP verification result has no TD10 report")

    result = _result_from_td10(td10, status=status, advisory_ids=advisory_ids)

    mismatches: list[str] = []
    if _norm_hex(quote.mrtd) != result.mrtd:
        mismatches.append("MRTD")
    for name in RTMR_KEYS:
        if _norm_hex(getattr(quote, name)) != getattr(result, name):
            mismatches.append(name.upper())
    if mismatches:
        raise AttestationError("TDX quote does not match the DCAP verification result: " + ", ".join(mismatches))
    return result


async def _dcap_verify(raw_quote: bytes, *, pccs_url: str | None = None) -> Any:
    """Run DCAP.  Isolated so the offline suite can monkeypatch it."""
    try:
        import dcap_qvl  # type: ignore
    except ImportError as exc:  # pragma: no cover - required dependency
        raise AttestationError("TDX verification requires the 'dcap-qvl' package") from exc
    return await dcap_qvl.get_collateral_and_verify(raw_quote, pccs_url=pccs_url)


async def _resolve_tdx_tcb_via_module_identity(
    raw_quote: bytes, *, pccs_url: str | None, original_error: Exception
) -> TdxVerificationResult:
    """Re-resolve a TCB verdict via Intel's TDX module-identity algorithm.

    ``dcap-qvl`` compares all 16 TDX TCB components at platform level instead of
    skipping the two module-governed bytes (Phala dcap-qvl issue #124), so newer
    hosts fail to match a level.  This is invoked only after every signature has
    already been checked by ``verify``; it only recomputes the TCB status.
    """
    try:
        import dcap_qvl  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise AttestationError("TDX verification requires the 'dcap-qvl' package") from exc

    try:
        collateral = await dcap_qvl.get_collateral(pccs_url or dcap_qvl.PHALA_PCCS_URL, raw_quote)
        parsed = dcap_qvl.Quote.parse(raw_quote)
        if not parsed.is_tdx():
            raise original_error
        report = parsed.report
        pck = parsed.pck_extension()
        if pck is None:
            raise AttestationError("DCAP quote has no parseable PCK extension")
        tcb_info = json.loads(collateral.tcb_info)
        status, advisory_ids = resolve_tdx_tcb_status(
            tcb_info=tcb_info,
            tee_tcb_svn=list(report.tee_tcb_svn),
            sgx_tcb_components=list(pck.cpu_svn),
            pce_svn=pck.pce_svn,
            mr_signer_seam=report.mr_signer_seam,
            seam_attributes=report.seam_attributes,
        )
    except AttestationError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
        raise AttestationError(f"TDX TCB fallback failed: {exc}") from exc

    return TdxVerificationResult(
        mrtd=report.mr_td.hex().upper(),
        rtmr0=report.rt_mr0.hex().upper(),
        rtmr1=report.rt_mr1.hex().upper(),
        rtmr2=report.rt_mr2.hex().upper(),
        rtmr3=report.rt_mr3.hex().upper(),
        report_data=report.report_data.hex().upper(),
        td_attributes=report.td_attributes.hex().upper(),
        status=status,
        advisory_ids=advisory_ids,
    )


async def verify_tdx_quote(raw_quote: bytes, *, pccs_url: str | None = None) -> TdxVerificationResult:
    """DCAP-verify *raw_quote*, require a valid TCB verdict, and cross-check it."""
    quote = TdxQuote.from_bytes(raw_quote)

    try:
        report = await _dcap_verify(raw_quote, pccs_url=pccs_url)
    except ValueError as exc:
        if "No matching TCB level found" in str(exc):
            result = await _resolve_tdx_tcb_via_module_identity(raw_quote, pccs_url=pccs_url, original_error=exc)
        else:
            raise AttestationError(f"TDX quote verification failed: {exc}") from exc
    except RuntimeError as exc:
        where = pccs_url or "the default PCCS (pccs.phala.network)"
        raise AttestationError(f"TDX collateral fetch failed from {where}: {exc}") from exc
    except AttestationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AttestationError(f"TDX quote verification failed: {exc}") from exc
    else:
        try:
            payload = json.loads(report.to_json())
        except Exception as exc:  # noqa: BLE001
            raise AttestationError(f"DCAP returned an unreadable report: {exc}") from exc
        result = parse_and_crosscheck(payload, quote)

    if not result.is_valid:
        problems: list[str] = [f"status={result.status!r}"]
        if result.advisory_ids:
            problems.append("advisories=" + ",".join(result.advisory_ids))
        if result.debug_enabled:
            problems.append("debug=on")
        raise AttestationError("TDX quote is not valid: " + "; ".join(problems))
    return result


# ---------------------------------------------------------------------------
# Intel TDX Module Identity TCB fallback (ported from chutes-api)
# ---------------------------------------------------------------------------


def _hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, str):
        return value.lower()
    raise AttestationError(f"cannot interpret {type(value).__name__} as hex bytes")


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return bytes.fromhex(value)
        except ValueError as exc:
            raise AttestationError("invalid hex value in TDX module field") from exc
    raise AttestationError(f"cannot interpret {type(value).__name__} as bytes")


def _merge_tcb_status(a: tuple[str, list[str]], b: tuple[str, list[str]] | None) -> tuple[str, list[str]]:
    if b is None:
        return a
    a_status, a_adv = a
    b_status, b_adv = b
    unknown = max(_TCB_STATUS_SEVERITY.values()) + 1
    a_sev = _TCB_STATUS_SEVERITY.get(a_status, unknown)
    b_sev = _TCB_STATUS_SEVERITY.get(b_status, unknown)
    status = b_status if b_sev > a_sev else a_status
    advisory_ids = list(a_adv)
    for adv in b_adv:
        if adv not in advisory_ids:
            advisory_ids.append(adv)
    return status, advisory_ids


def _match_platform_tcb_level(
    tcb_info: dict[str, Any],
    tee_tcb_svn: list[int],
    sgx_tcb_components: list[int],
    pce_svn: int,
) -> tuple[str, list[str]]:
    tdx_start = 2 if tee_tcb_svn[1] > 0 else 0
    for level in tcb_info.get("tcbLevels", []):
        tcb = level["tcb"]
        if pce_svn < tcb["pcesvn"]:
            continue
        sgx_components = [c["svn"] for c in tcb.get("sgxtcbcomponents", [])]
        if len(sgx_components) != len(sgx_tcb_components):
            raise AttestationError("SGX TCB component count mismatch in TCB Info")
        if any(a < b for a, b in zip(sgx_tcb_components, sgx_components)):
            continue
        tdx_components = [c["svn"] for c in tcb.get("tdxtcbcomponents", [])]
        if len(tdx_components) != len(tee_tcb_svn):
            raise AttestationError("TDX TCB component count mismatch in TCB Info")
        if any(a < b for a, b in zip(tee_tcb_svn[tdx_start:], tdx_components[tdx_start:])):
            continue
        return level["tcbStatus"], level.get("advisoryIDs", []) or []
    raise AttestationError("no matching platform TCB level found")


def _verify_tdx_module(
    tcb_info: dict[str, Any],
    tee_tcb_svn: list[int],
    mr_signer_seam: Any,
    seam_attributes: Any,
) -> tuple[str, list[str]] | None:
    if tcb_info.get("id") != "TDX" or tcb_info.get("version", 0) < 3:
        return None

    base = tcb_info.get("tdxModule")
    if base is None:
        raise AttestationError("TDX TCB Info is missing tdxModule field")

    module_version = tee_tcb_svn[1]
    module_isvsvn = tee_tcb_svn[0]

    expected_mrsigner = base["mrsigner"]
    expected_attributes = base["attributes"]
    attributes_mask = base["attributesMask"]
    identity_levels: list[dict[str, Any]] | None = None

    identities = tcb_info.get("tdxModuleIdentities") or []
    if module_version > 0 and identities:
        wanted_id = f"TDX_{module_version:02X}"
        identity = next(
            (i for i in identities if i.get("id", "").upper() == wanted_id.upper()),
            None,
        )
        if identity is None:
            raise AttestationError(f"unsupported TDX module version: no identity '{wanted_id}' in TCB Info")
        expected_mrsigner = identity["mrsigner"]
        expected_attributes = identity["attributes"]
        attributes_mask = identity["attributesMask"]
        identity_levels = identity.get("tcbLevels", [])

    if _as_bytes(expected_mrsigner) != _as_bytes(mr_signer_seam):
        raise AttestationError("TDX module MRSIGNER mismatch")

    exp = _as_bytes(expected_attributes)
    mask = _as_bytes(attributes_mask)
    act = _as_bytes(seam_attributes)
    if not (len(exp) == len(mask) == len(act)):
        raise AttestationError("TDX module SEAMATTRIBUTES length mismatch")
    for e, m, a in zip(exp, mask, act):
        if (e & m) != (a & m) or (a & (~m & 0xFF)) != 0:
            raise AttestationError("TDX module SEAMATTRIBUTES mismatch")

    if identity_levels is not None:
        for level in identity_levels:
            if module_isvsvn >= level["tcb"]["isvsvn"]:
                return level["tcbStatus"], level.get("advisoryIDs", []) or []
        raise AttestationError(f"TDX module ISVSVN {module_isvsvn} below minimum in TDX module TCB levels")
    return None


def resolve_tdx_tcb_status(
    tcb_info: dict[str, Any],
    tee_tcb_svn: list[int],
    sgx_tcb_components: list[int],
    pce_svn: int,
    mr_signer_seam: Any,
    seam_attributes: Any,
) -> tuple[str, list[str]]:
    """Resolve the overall TCB status using Intel's TDX Module Identity algorithm."""
    if len(tee_tcb_svn) < 2:
        raise AttestationError("TEE_TCB_SVN too short to resolve TDX TCB status")
    platform = _match_platform_tcb_level(tcb_info, tee_tcb_svn, sgx_tcb_components, pce_svn)
    module = _verify_tdx_module(tcb_info, tee_tcb_svn, mr_signer_seam, seam_attributes)
    return _merge_tcb_status(platform, module)
