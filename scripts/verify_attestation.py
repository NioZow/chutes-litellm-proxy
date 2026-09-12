#!/usr/bin/env python3
"""Verify Chutes TEE/GPU attestation for a model or chute.

Thin wrapper around :mod:`chutes_litellm.verify` (also exposed as the
``chutes-verify-attestation`` console script after a uv/nix install).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chutes_litellm.verify import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
