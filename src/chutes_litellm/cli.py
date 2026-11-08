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

import yaml

from chutes_litellm import e2ee_litellm, get_logger
from chutes_litellm.generate_config import generate

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4000

#: NEAR AI startup wiring.  The env names mirror
#: :mod:`chutes_litellm.near.client`; they are duplicated rather than imported
#: so a deployment without NEAR AI never imports the SDK at CLI startup.
NEAR_API_KEY_ENV = "NEAR_API_KEY"
NEAR_PREWARM_MODELS_ENV = "NEARAI_PREWARM_MODELS"
WORKER_STARTUP_HOOKS_ENV = "LITELLM_WORKER_STARTUP_HOOKS"
NEAR_STARTUP_HOOK = "chutes_litellm.near.startup:prewarm"

_NEAR_MODEL_PREFIX = "nearai/"
_NEAR_WIRE_PREFIX = "nearai_e2ee/"

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


def register_near() -> None:
    """Register the ``nearai_e2ee`` CustomLLM provider.

    The import is lazy so deployments without NEAR AI do not pay for importing
    the NEAR AI SDK.  Registration does not require an API key and only mutates
    LiteLLM's in-process provider map, so it must happen before serving in this
    process (the single-worker invariant in :func:`_serve`).
    """
    from chutes_litellm.near.provider import register

    register()


def _read_generated_config(config_path: str) -> dict:
    """Read a generated YAML config, returning ``{}`` when it is unusable."""
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return config if isinstance(config, dict) else {}


def _strip_model_prefix(value: object, prefix: str) -> str | None:
    """Return the model id after ``prefix``, or ``None`` when it does not match."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate.startswith(prefix):
        return None
    model_id = candidate[len(prefix):].strip()
    if not model_id or model_id == "*":
        return None
    return model_id


def _chutes_tee_flags(config: dict) -> tuple[list[str], list[str]]:
    """Return ``(tee_models, non_tee_models)`` from a generated config.

    Chutes now serves both confidential-compute and plain models under the
    same ``chutes/`` prefix.  The E2EE transport must only wrap the TEE ones,
    so the startup wiring forwards the catalog's ``confidential_compute`` flag.
    """
    tee: list[str] = []
    non_tee: list[str] = []
    for entry in config.get("model_list") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params")
        wire_model = params.get("model") if isinstance(params, dict) else None
        if not isinstance(wire_model, str) or not wire_model.startswith("chutes/"):
            continue
        model_id = wire_model[len("chutes/"):].strip()
        if not model_id:
            continue
        info = entry.get("model_info")
        flag = info.get("confidential_compute") if isinstance(info, dict) else None
        if flag is True:
            tee.append(model_id)
        elif flag is False:
            non_tee.append(model_id)
    return tee, non_tee


def _near_model_ids(config: dict) -> list[str]:
    """Collect bare NEAR model ids from a generated config's ``model_list``."""
    ids: list[str] = []
    for entry in config.get("model_list") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params")
        wire_model = params.get("model") if isinstance(params, dict) else None
        for value, prefix in ((entry.get("model_name"), _NEAR_MODEL_PREFIX), (wire_model, _NEAR_WIRE_PREFIX)):
            model_id = _strip_model_prefix(value, prefix)
            if model_id is not None:
                ids.append(model_id)
                break
    return ids


def _merge_model_ids(*groups: Sequence[str]) -> list[str]:
    """Return non-empty, stripped, de-duplicated model ids in input order."""
    unique: dict[str, None] = {}
    for group in groups:
        for raw in group:
            if not isinstance(raw, str):
                continue
            model = raw.strip()
            if model and model not in unique:
                unique[model] = None
    return list(unique)


def _configure_near_startup(config_path: str) -> list[str]:
    """Export the NEAR prewarm list and the LiteLLM worker startup hook.

    The models come from the generated config (``nearai/<id>`` entries and/or
    ``nearai_e2ee/<id>`` wire models), unioned with any pre-existing
    ``NEARAI_PREWARM_MODELS`` value.  The hook is appended to any existing
    ``LITELLM_WORKER_STARTUP_HOOKS`` exactly once, so prewarming runs in the
    serving event loop rather than at CLI import time.

    Returns:
        The merged model ids; empty when the config declares no NEAR models.
    """
    config = _read_generated_config(config_path)
    existing = (os.environ.get(NEAR_PREWARM_MODELS_ENV) or "").split(",")
    models = _merge_model_ids(_near_model_ids(config), existing)
    if not models:
        return []

    os.environ[NEAR_PREWARM_MODELS_ENV] = ",".join(models)
    hooks = [hook.strip() for hook in (os.environ.get(WORKER_STARTUP_HOOKS_ENV) or "").split(",") if hook.strip()]
    if NEAR_STARTUP_HOOK not in hooks:
        hooks.append(NEAR_STARTUP_HOOK)
    os.environ[WORKER_STARTUP_HOOKS_ENV] = ",".join(hooks)
    return models


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
    # Non-TEE Chutes deployments share the API host but cannot be encrypted;
    # tell the scoped transport which model ids are TEE before it is installed.
    tee_models, non_tee_models = _chutes_tee_flags(_read_generated_config(output))
    e2ee_litellm.configure_tee_models(tee_models, non_tee_models)
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

    # NEAR AI is opt-in by key presence.  Registration and the prewarm hook
    # must happen before serving so they live in this process/event loop.
    if (os.environ.get(NEAR_API_KEY_ENV) or "").strip():
        register_near()
        logger.info("litellm_proxy: NEAR AI provider registered")
        near_models = _configure_near_startup(output)
        if near_models:
            logger.info(
                "litellm_proxy: NEAR AI startup prewarm scheduled for %d model(s)",
                len(near_models),
            )

    serve_args = ["--config", output, "--host", host, "--port", str(port)]
    logger.info("litellm_proxy: serving in-process (%s)", " ".join(serve_args))
    return _serve(output, host, port)


def main() -> None:
    sys.exit(litellm_proxy())


if __name__ == "__main__":
    main()
