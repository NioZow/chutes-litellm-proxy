"""Pins for the mandatory chutes-e2ee transport hook.

The proxy has two dependency sources: uv (dev/test) and Nix (deployed).  They
drifted once: uv used a fork revision with ``instance_filter`` while the Nix
package pinned an older revision, so the Nix-built proxy failed every Chutes
request with an unexpected-keyword TypeError.  Keep the two pins identical.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _flake_rev() -> str:
    text = (ROOT / "flake.nix").read_text()
    match = re.search(
        r'chutes-e2ee\s*=\s*python\.pkgs\.buildPythonPackage.*?rev\s*=\s*"([0-9a-f]{40})"',
        text,
        re.DOTALL,
    )
    assert match, "chutes-e2ee rev not found in flake.nix"
    return match.group(1)


def _uv_lock_rev() -> str:
    text = (ROOT / "uv.lock").read_text()
    match = re.search(
        r'name = "chutes-e2ee".*?'
        r'source = \{ git = "https://github\.com/niozow/chutes-e2ee-transport\.git#([0-9a-f]{40})" \}',
        text,
        re.DOTALL,
    )
    assert match, "chutes-e2ee revision not found in uv.lock"
    return match.group(1)


def test_nix_and_uv_pin_the_same_chutes_e2ee_revision():
    assert _flake_rev() == _uv_lock_rev()
