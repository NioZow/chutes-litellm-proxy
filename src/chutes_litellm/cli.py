"""Console entry point for running the proxy via uv (and the Docker/Nix paths).

Orchestrates the three steps the deployment scripts used to do in shell:

1. generate the concrete model config (``chutes_litellm.generate_config``),
2. install the host-scoped Chutes E2EE transport (``e2ee_litellm.install()``),
3. exec the ``litellm`` CLI against the generated config.

Environment overrides:

* ``LITELLM_TEMPLATE``  - template path (default: packaged ``config.template.yml``)
* ``LITELLM_OUTPUT``    - generated config path (default: ``./config.generated.yml``)
* ``LITELLM_HOST``      - bind host (default ``0.0.0.0``)
* ``LITELLM_PORT``      - bind port (default ``4000``)
"""

from __future__ import annotations

import os
import sys

from chutes_litellm import e2ee_litellm
from chutes_litellm.generate_config import generate


def litellm_proxy() -> int:
    output = generate()
    # No-op (keeps serving the other providers) when CHUTES_API_KEY is unset or
    # the chutes_e2ee package is missing.
    e2ee_litellm.install()

    # Opt-in: serve Chutes through the self-contained custom provider
    # (chutes_e2ee/<model>) instead of the builtin chutes provider + factory
    # swap.  Requires models configured under the chutes_e2ee/ prefix.
    if os.environ.get("CHUTES_CUSTOM_PROVIDER", "").strip().lower() in {"1", "true", "yes", "on"}:
        from chutes_litellm.custom_provider import register as register_custom_provider

        register_custom_provider()

    host = os.environ.get("LITELLM_HOST", "0.0.0.0")
    port = os.environ.get("LITELLM_PORT", "4000")
    argv = ["litellm", "--config", output, "--host", host, "--port", port]
    print(f"[litellm-proxy] exec: {' '.join(argv)}", file=sys.stderr)
    try:
        os.execvp("litellm", argv)
    except FileNotFoundError:  # pragma: no cover - depends on the environment
        print("error: `litellm` not found on PATH (is litellm installed?)", file=sys.stderr)
        return 1


def main() -> None:
    sys.exit(litellm_proxy())


if __name__ == "__main__":
    main()
