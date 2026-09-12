#!/usr/bin/env python3
"""Run the Chutes-aware LiteLLM proxy in-process, for live/E2E testing.

Unlike the deployed console entry point (``chutes-litellm-proxy``, which
``os.execvp``-es the ``litellm`` CLI and therefore *discards* in-memory
registrations), this launcher keeps the custom ``chutes_e2ee`` provider and the
optional host-scoped E2EE transport installed in the same process that serves
HTTP.  It is what the live integration tests spawn.

Environment:

* ``CHUTES_API_KEY``          - real API key (required for completions)
* ``CONFIG_FILE_PATH`` or ``--config`` - proxy YAML (model list)
* ``CHUTES_E2EE_INSTALL``     - "1" to also apply ``e2ee_litellm.install()``
* ``LITELLM_MASTER_KEY``      - optional; the config's ``master_key`` wins

Usage:

    CONFIG_FILE_PATH=config.generated.yml CHUTES_API_KEY=... \\
        python scripts/live_server.py --port 4001
"""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("CONFIG_FILE_PATH"), help="proxy config YAML")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    args = parser.parse_args()

    if not args.config or not os.path.isfile(args.config):
        print(f"error: config file not found: {args.config!r}", file=sys.stderr)
        return 2
    os.environ["CONFIG_FILE_PATH"] = os.path.abspath(args.config)

    # Keep the custom provider + (optionally) the host-scoped transport alive in
    # this process so proxied chutes_e2ee requests are served here.
    from chutes_litellm.custom_provider import register as register_custom_provider

    register_custom_provider()
    if _truthy(os.environ.get("CHUTES_E2EE_INSTALL")):
        from chutes_litellm import e2ee_litellm

        if not e2ee_litellm.install():
            print("warning: e2ee_litellm.install() was a no-op", file=sys.stderr)

    import uvicorn
    from litellm.proxy.proxy_server import app

    port = args.port or _free_port()
    print(f"live-server: http://{args.host}:{port} config={args.config}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
