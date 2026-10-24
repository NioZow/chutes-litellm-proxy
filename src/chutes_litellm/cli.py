"""Console entry point for running the proxy via uv (and the Docker/Nix paths).

Orchestrates the three steps the deployment scripts used to do in shell:

1. generate the concrete model config (``chutes_litellm.generate_config``),
2. install the host-scoped Chutes E2EE transport (``e2ee_litellm.install()``),
3. run the LiteLLM proxy server **in this process** against the generated config.

Serving must happen in-process: ``e2ee_litellm.install()`` and the optional
custom-provider ``register()`` mutate process-global LiteLLM/OpenAI-SDK hooks.
Replacing the process image (or forking fresh workers) would discard those
hooks, so the default ``chutes/`` provider would send plaintext to Chutes and
defeat the mandatory E2EE/attestation design.  For the same reason the server
is pinned to a single uvicorn worker.

Command-line options take precedence over the environment:

* ``--config PATH``     - template to expand (overrides ``LITELLM_TEMPLATE``;
                          default: packaged ``config.template.yml``)
* ``--host HOST``       - bind host (overrides ``LITELLM_HOST``, default ``0.0.0.0``)
* ``--port PORT``       - bind port (overrides ``LITELLM_PORT``, default ``4000``)

Environment overrides (used when the matching flag is omitted):

* ``LITELLM_TEMPLATE``  - template path (default: packaged ``config.template.yml``)
* ``LITELLM_OUTPUT``    - generated config path (default: ``./config.generated.yml``)
* ``LITELLM_HOST``      - bind host (default ``0.0.0.0``)
* ``LITELLM_PORT``      - bind port (default ``4000``)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from chutes_litellm import e2ee_litellm, get_logger
from chutes_litellm.generate_config import generate

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4000

_EPILOG = (
    "TEE/GPU attestation, Chutes end-to-end encryption and TLS verification are "
    "always enforced at runtime; there are no flags to disable them."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``chutes-litellm-proxy``.

    The parser is deliberately limited to startup plumbing: template selection
    and the LiteLLM bind address. Security behaviour is not configurable.

    Returns:
        An ``argparse.ArgumentParser`` whose defaults leave every option unset,
        so the caller can distinguish a CLI value from an environment fallback.
    """
    parser = argparse.ArgumentParser(
        prog="chutes-litellm-proxy",
        description="Generate the LiteLLM config and serve the LiteLLM proxy in-process.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "LiteLLM YAML config/template to expand at startup; overrides "
            "LITELLM_TEMPLATE (default: the packaged config.template.yml)"
        ),
    )
    parser.add_argument(
        "--host",
        metavar="HOST",
        help="LiteLLM bind host; overrides LITELLM_HOST (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        help="LiteLLM bind port; overrides LITELLM_PORT (default: 4000)",
    )
    return parser


def _resolve_port(parser: argparse.ArgumentParser, cli_port: int | None) -> int:
    """Resolve the bind port from the CLI, the environment, then the default.

    Args:
        parser: Parser used to report a clean usage error (exit code 2).
        cli_port: Value supplied via ``--port``, or ``None`` when omitted.

    Returns:
        The port to pass to the LiteLLM CLI.
    """
    if cli_port is not None:
        return cli_port

    raw_port = os.environ.get("LITELLM_PORT")
    if raw_port is None or not raw_port.strip():
        return DEFAULT_PORT

    try:
        return int(raw_port)
    except ValueError:
        parser.error(f"invalid LITELLM_PORT value: {raw_port!r} (expected an integer)")


def _serve(config_path: str, host: str, port: int) -> int:
    """Run LiteLLM's proxy server inside the current process.

    This deliberately calls ``litellm.run_server`` in-process instead of
    launching the ``litellm`` executable: the E2EE client swap and the custom
    provider registration are live Python state, so a process replacement or
    fresh subprocess workers would lose them and send requests unencrypted.
    ``--num_workers 1`` keeps uvicorn in the current process (default workers
    would fork children that do not inherit those hooks).

    Args:
        config_path: Generated LiteLLM config to serve.
        host: Bind host for the HTTP server.
        port: Bind port for the HTTP server.

    Returns:
        ``0`` after a clean server shutdown, ``1`` when LiteLLM or click is not
        importable, ``1`` or ``2`` for click runtime/usage errors, and ``130``
        for ``KeyboardInterrupt``.

    Raises:
        SystemExit: If LiteLLM itself exits during startup (``sys.exit``).
    """
    try:
        import click
        from litellm import run_server
    except ImportError:
        print(
            "error: `litellm` is not importable in this process (install it, e.g. `uv sync`)",
            file=sys.stderr,
        )
        return 1

    try:
        run_server.main(
            args=[
                "--config",
                config_path,
                "--host",
                host,
                "--port",
                str(port),
                # A single uvicorn worker: forked workers would not inherit the
                # in-memory E2EE transport/custom-provider hooks installed below.
                "--num_workers",
                "1",
            ],
            prog_name="litellm",
            standalone_mode=False,
        )
    except click.UsageError as exc:
        # argparse-style clean failure: message on stderr, exit code 2, no traceback.
        exc.show()
        return 2
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        print("Aborted!", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def litellm_proxy(argv: Sequence[str] | None = None) -> int:
    """Generate the config and serve the LiteLLM proxy in this process.

    The server is started in-process (not via a process replacement) so the
    E2EE transport installed here remains active for every proxied request.

    Args:
        argv: Optional argument sequence for tests. ``None`` lets argparse read
            ``sys.argv`` as usual.

    Returns:
        ``0`` after a clean server shutdown, otherwise the server's non-zero
        exit code.

    Raises:
        SystemExit: For ``--help`` (code 0) and invalid usage (code 2).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config is not None and not os.path.isfile(args.config):
        parser.error(f"--config file not found: {args.config}")

    # CLI > LITELLM_HOST/LITELLM_PORT > the historical defaults.  Resolve and
    # validate the bind address before generating anything, so usage errors do
    # not leave a half-written config behind.
    host = args.host if args.host is not None else (os.environ.get("LITELLM_HOST") or DEFAULT_HOST)
    port = _resolve_port(parser, args.port)

    # CLI > LITELLM_TEMPLATE > generate()'s packaged default.  Passing None
    # keeps the output path behaviour (LITELLM_OUTPUT / ./config.generated.yml).
    template = args.config if args.config is not None else os.environ.get("LITELLM_TEMPLATE")
    output = generate(template=template, output=None)

    # Initialise the shared logger early so startup, E2EE install and attestation
    # events are emitted to both stderr and CHUTES_LOG_FILE.
    logger = get_logger("chutes_litellm")
    logger.info("litellm_proxy: starting (config=%s)", output)
    # No-op (keeps serving the other providers) when CHUTES_API_KEY is unset or
    # the chutes_e2ee package is missing.
    e2ee_litellm.install()
    logger.info("litellm_proxy: E2EE transport installed")

    # Opt-in: serve Chutes through the self-contained custom provider
    # (chutes_e2ee/<model>) instead of the builtin chutes provider + factory
    # swap.  Requires models configured under the chutes_e2ee/ prefix.
    if os.environ.get("CHUTES_CUSTOM_PROVIDER", "").strip().lower() in {"1", "true", "yes", "on"}:
        from chutes_litellm.custom_provider import register as register_custom_provider

        register_custom_provider()
        logger.info("litellm_proxy: custom provider registered")

    serve_args = ["--config", output, "--host", host, "--port", str(port)]
    logger.info("litellm_proxy: serving in-process (%s)", " ".join(serve_args))
    return _serve(output, host, port)


def main() -> None:
    sys.exit(litellm_proxy())


if __name__ == "__main__":
    main()
