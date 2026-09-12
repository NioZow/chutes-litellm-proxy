#!/usr/bin/env python3
"""Generate config.generated.yml from the bundled config.template.yml.

Thin wrapper around :mod:`chutes_litellm.generate_config` so the operation stays
available as a standalone command in a source checkout.  The ``litellm-proxy``
console script runs the same generation at startup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chutes_litellm.generate_config import main  # noqa: E402

if __name__ == "__main__":
    main()
