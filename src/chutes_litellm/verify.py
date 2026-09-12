"""Verify Chutes TEE/GPU attestation for a model or chute before encrypting to it.

Uses the same evidence endpoints chutes-api documents (docs/tee-verification.md)
and chutes_litellm.attestation.verify_evidence():

  * GET /e2e/instances/{chute_id}          -> instance ML-KEM pubkeys
  * GET /chutes/{chute_id}/evidence?nonce= -> TDX quote + per-GPU NVIDIA evidence
  * key-possession signature + SHA256(nonce + e2e_pubkey) binding checks

Full Intel DCAP quote / NVIDIA attestation verification requires the optional
``dcap-qvl`` and ``nv-attestation-sdk`` packages (add --verify-quote/--verify-gpu).

Examples:
  CHUTES_API_KEY=cpk_... chutes-verify-attestation --model "Qwen/Qwen3.5-397B-A17B-TEE"
  CHUTES_API_KEY=cpk_... chutes-verify-attestation --chute 51a4284a-a5a0-5e44-a9cc-6af5a2abfbcf --verify-quote --verify-gpu
"""

from __future__ import annotations

import argparse
import os
import sys

from chutes_litellm import attestation as att


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chutes-verify-attestation", description=__doc__
    )
    parser.add_argument("--model", help="Chutes model id (as listed at llm.chutes.ai/v1/models)")
    parser.add_argument("--chute", help="Chute id (uuid); takes precedence over --model")
    parser.add_argument("--api-key", default=os.environ.get("CHUTES_API_KEY"), help="Chutes API key")
    parser.add_argument("--api-base", default=att.DEFAULT_API_BASE)
    parser.add_argument("--models-base", default=att.DEFAULT_MODELS_BASE)
    parser.add_argument("--verify-quote", action="store_true", help="verify TDX quote with Intel DCAP (dcap-qvl)")
    parser.add_argument("--verify-gpu", action="store_true", help="verify NVIDIA evidence (nv-attestation-sdk)")
    parser.add_argument("--force", action="store_true", help="ignore the verification cache")
    parser.add_argument(
        "--details",
        action="store_true",
        help="print per-instance checks, DCAP TCB status and NVIDIA GPU verdicts",
    )
    return parser


def _print_details(report: att.ChuteVerificationReport) -> None:
    print(
        f"chute {report.chute_id} (model {report.model!r}): "
        f"instances listed={report.instances}, evidence blobs={report.evidences}, "
        f"unmatched={report.unmatched}"
    )
    for result in report.results:
        parts = [att._format_checks(result.checks)]
        if result.dcap_status:
            parts.append(f"dcap_status={result.dcap_status}")
        if result.gpu:
            parts.append(f"gpu={result.gpu['gpus']}x{result.gpu['arch']}")
        verdict = "VERIFIED" if result.verified else "FAILED"
        print(f"  - {result.instance_id}: {verdict} [{' | '.join(parts)}]")
    if report.failures:
        print("  per-instance failure(s):", file=sys.stderr)
        for line in report.failures:
            print(f"    - {line}", file=sys.stderr)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        print("error: set CHUTES_API_KEY or pass --api-key", file=sys.stderr)
        return 2
    if not args.chute and not args.model:
        print("error: pass --chute <uuid> or --model <id>", file=sys.stderr)
        return 2

    lookup = args.chute if args.chute else args.model

    if args.details:
        try:
            report = att.verify_chute_report(
                args.api_key,
                lookup,
                api_base=args.api_base,
                models_base=args.models_base,
                verify_quote=args.verify_quote,
                verify_gpu=args.verify_gpu,
            )
        except att.AttestationError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        _print_details(report)
        if not report.ok:
            print("FAILED: not every instance verified (fail closed)", file=sys.stderr)
            return 1
        print(f"OK: {len(report.verified)} instance(s) passed TEE/GPU attestation")
        return 0

    try:
        verified = att.verify_chute(
            args.api_key,
            lookup,
            api_base=args.api_base,
            models_base=args.models_base,
            verify_quote=args.verify_quote,
            verify_gpu=args.verify_gpu,
            force=args.force,
        )
    except att.AttestationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {len(verified)} instance(s) passed TEE/GPU attestation:")
    for instance_id in verified:
        print(f"  - {instance_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
