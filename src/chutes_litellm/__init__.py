"""Chutes TEE/E2EE integration for LiteLLM.

``e2ee_litellm``   host-scoped swap of LiteLLM's OpenAI-SDK http clients so
                   native ``chutes`` provider traffic is end-to-end encrypted.
``attestation``    optional fail-closed TDX/GPU attestation gate + verifier.
"""

from __future__ import annotations

import logging
import os
import pathlib as _pathlib

from chutes_litellm import attestation, e2ee_litellm

__all__ = ["attestation", "e2ee_litellm", "get_logger"]


# ---------------------------------------------------------------------------
# Structured logger wired to both stdout (container logs / journalctl) and an
# optional on-disk file.  All sub-modules import this singleton.
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = _pathlib.Path.home() / ".local" / "state" / "chutes_litellm"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "proxy.log"

_LOG_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_logger: logging.Logger | None = None


def get_logger(name: str = "chutes_litellm") -> logging.Logger:
    """Return the shared, configured ``chutes_litellm`` logger.

    * stdout/stderr handler (so logs show up in ``docker logs``, ``journalctl``,
      launchd logfiles, etc.)
    * optional file handler at ``CHUTES_LOG_FILE`` (defaults to
      ``~/.local/state/chutes_litellm/proxy.log``)
    * level set by ``CHUTES_LOG_LEVEL`` (defaults to ``INFO``)

    Idempotent: calling this more than once returns the same logger instance.
    """
    global _logger
    if _logger is not None:
        return _logger

    log = logging.getLogger(name)
    log.setLevel(_LOG_LEVEL_MAP.get((os.environ.get("CHUTES_LOG_LEVEL") or "info").strip().lower(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always emit to stderr so container / systemd / launchd capture it.
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(fmt)
        log.addHandler(stderr_handler)

    # Optional file handler (env-overridable path).
    raw_path = os.environ.get("CHUTES_LOG_FILE")
    if raw_path is not None and raw_path.strip().lower() in {"", "none", "null", "false", "0"}:
        raw_path = None
    log_path = _pathlib.Path(raw_path) if raw_path else _DEFAULT_LOG_FILE
    if log_path and not any(
        isinstance(h, logging.FileHandler) and _pathlib.Path(h.baseFilename) == log_path
        for h in log.handlers
    ):
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path))
            file_handler.setFormatter(fmt)
            log.addHandler(file_handler)
        except OSError:
            # If the log directory is not writable (e.g. in a hardened container),
            # degrade gracefully to stderr-only logging.
            pass

    log.propagate = False
    _logger = log
    return log
