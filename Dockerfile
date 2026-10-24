# syntax=docker/dockerfile:1
# Build with the same toolchain the repo uses locally: a uv-managed project
# (pyproject.toml + committed uv.lock), installed with `uv sync --locked`.
# The base is uv's Python 3.13 image so no interpreter download is needed and
# the runtime matches the version pqcrypto ships wheels for.
FROM ghcr.io/astral-sh/uv:0.11.21-python3.13-trixie-slim@sha256:29b42d1b8d1b38d2240abd2e636bb90b42c72d729e6f9c10983e503184027f78

# Reproducible, cache-friendly uv behaviour.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# `git` is required for the chutes-e2ee git dependency in uv.lock; everything
# else installs from prebuilt wheels (no compiler needed).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies only, so this layer is cached until the lock changes.
#    dcap-qvl (Intel DCAP) and nv-attestation-sdk (NVIDIA NRAS) are ALWAYS base
#    dependencies of this image: hardware-root attestation, DCAP quote
#    verification, GPU/NRAS verification, measurement matching and TLS
#    verification are always enforced. There is no extra or flag that disables
#    them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2. Install the proxy itself (non-editable: the wheel + config template are
#    baked into .venv, so the runtime only needs this image, not the sources).
#    README.md is required by the build backend (project metadata).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

# Runs generate_config -> E2EE transport install -> litellm (see chutes_litellm.cli).
ENTRYPOINT ["python", "-m", "chutes_litellm.cli"]
