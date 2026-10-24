# Setup and running

## Who this is for

Operators and developers who want to install the proxy, run it with `uv`,
Docker/Podman, or Nix, connect an OpenAI-compatible client, and diagnose a
startup or attestation failure.

## What you'll learn

- the prerequisites and the three supported install paths,
- how CLI flags, environment variables, and the config template interact,
- how to supply API keys without baking them into an image,
- how to verify attestation manually and what a fail-closed error means.

## Prerequisites

- Python 3.13 or 3.14. The project requires `>=3.13,<3.15`; `pqcrypto` ships
  prebuilt wheels only for those versions.
- [uv](https://docs.astral.sh/uv) for the source install.
- Docker or Podman for the container path. `scripts/run.sh` picks Podman when
  available, otherwise Docker.
- Optional: Nix with flakes for the native package and service modules.
- A `CHUTES_API_KEY` if you want `chutes/` models. Hardware attestation also
  needs outbound access to `https://api.chutes.ai`, `https://llm.chutes.ai`,
  the Intel/Phala PCCS used by DCAP, and NVIDIA NRAS.

The Intel DCAP (`dcap-qvl`) and NVIDIA (`nv-attestation-sdk`) SDKs are base
dependencies. `uv sync --extra attestation` still works, but it is only a
compatibility alias; it does not enable or disable anything.

## Run with uv

```sh
uv sync
export CHUTES_API_KEY=cpk_...
uv run chutes-litellm-proxy --host 127.0.0.1 --port 4000
```

The entry point performs the same startup sequence on every path:

1. expand the template into a generated LiteLLM config,
2. install the host-scoped Chutes E2EE transport (a no-op without
   `CHUTES_API_KEY`),
3. optionally register the `chutes_e2ee/` custom provider,
4. serve the LiteLLM proxy in the same process.

### CLI flags

```text
chutes-litellm-proxy [--config PATH] [--host HOST] [--port PORT]
```

| Flag       | Overrides        | Default                          |
| ---------- | ---------------- | -------------------------------- |
| `--config` | `LITELLM_TEMPLATE` | packaged `config.template.yml` |
| `--host`   | `LITELLM_HOST`   | `0.0.0.0`                        |
| `--port`   | `LITELLM_PORT`   | `4000`                           |

Precedence is **CLI flag > environment variable > default**:

```sh
LITELLM_HOST=0.0.0.0 LITELLM_PORT=4000 \
  uv run chutes-litellm-proxy --host 127.0.0.1 --port 8080
# binds 127.0.0.1:8080
```

`--config` names the input template that is expanded at startup; it is not the
already-generated `config.generated.yml`. It must point to an existing file. A
missing `--config` file, a non-integer `--port`, or a non-integer
`LITELLM_PORT` exits with status `2`. If `litellm` (or `click`) cannot be
imported in the process, the entry point exits with status `1` and prints a
message to install it, for example with `uv sync`.

### Common examples

```sh
# Keep the generated config next to the checkout
LITELLM_OUTPUT=./config.generated.yml uv run chutes-litellm-proxy

# Use a custom template
uv run chutes-litellm-proxy --config ./config.custom.yml --host 127.0.0.1

# Regenerate the config without starting the server
uv run scripts/generate_config.py

# Show the exact CLI contract
uv run chutes-litellm-proxy --help
```

## Docker and Podman

Build the image first:

```sh
docker build -t litellm-proxy .
```

Then run it through `scripts/run.sh`:

```sh
RUNTIME=docker CHUTES_API_KEY=cpk_... ./scripts/run.sh
```

If both engines are installed, set `RUNTIME` to the one you built with. With
only one engine installed, `auto` (the default) is fine.

`run.sh` reads these variables:

| Variable          | Meaning                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `LITELLM_BIND`    | Host publish address, default `127.0.0.1:4000`. Comma-separated values are allowed. |
| `RUNTIME`         | `podman`, `docker`, or `auto` (default).                                |
| `*_API_KEY`       | Forwarded into the container unchanged.                                 |
| `*_API_KEY_PATH`  | Read on the host; its contents are forwarded as the matching `*_API_KEY`. |

Each `LITELLM_BIND` entry is published to container port `4000`. The container
CLI therefore listens on `0.0.0.0:4000` while the host side is loopback-only by
default.

`run.sh` forwards only `*_API_KEY` and `*_API_KEY_PATH`. It does **not** forward
`LITELLM_HOST`, `LITELLM_PORT`, `LITELLM_TEMPLATE`, `LITELLM_OUTPUT`, or the
`CHUTES_*` tuning variables. Use Compose or a direct `docker run -e ...` when
you need those.

### Docker Compose

`docker-compose.yml` builds the same image and starts LiteLLM plus Open WebUI:

```sh
CHUTES_API_KEY=cpk_... docker compose up --build
```

- LiteLLM is published on host port `4000` (`4000:4000`).
- The generated config is written to `./litellm-data/config.generated.yml`
  (`LITELLM_OUTPUT=/app/data/config.generated.yml`).
- The Compose file forwards `CHUTES_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `META_API_KEY`,
  `MISTRAL_API_KEY`, `XAI_API_KEY`, and `PERPLEXITY_API_KEY` from your shell.
- Open WebUI is published on port `3000`.

Add `http://litellm:4000/v1` as an OpenAI connection inside Open WebUI, using a
dummy key when the proxy has no `master_key`.

Compose does not resolve `*_API_KEY_PATH` for you; it forwards the variables it
lists. Pass a raw key in the environment, extend the Compose `environment`
block, or use `run.sh` for file-backed keys.

## Nix

The flake supports `x86_64-linux`, `aarch64-linux`, and `aarch64-darwin`.

```sh
nix build .#                 # result/bin/litellm-proxy
nix run .#                   # run the packaged proxy
nix develop                  # dev shell: Python env, uv, ruff, git
```

The `attestation` package alias is the same derivation as the default package.
`nix build .#attestation` still works for existing scripts.

The Nix launcher forwards command-line arguments to the Python CLI. Use either
form:

```sh
CHUTES_API_KEY=cpk_... nix run .# -- --host 127.0.0.1 --port 8000

# Equivalent, environment-only form
CHUTES_API_KEY=cpk_... LITELLM_HOST=127.0.0.1 LITELLM_PORT=8000 nix run .#
```

`nix develop` creates `.venv` on first entry, sets `LITELLM_HOST=127.0.0.1`,
`LITELLM_PORT=4000`, `PYTHONPATH`, and a `libstdc++` search path.

### Service modules

All four modules share [`options.nix`](../options.nix):

| Flake output          | Runs as                         | Declared in             |
| --------------------- | ------------------------------- | ----------------------- |
| `nixosModules.default` | system service (`DynamicUser`) | NixOS configuration     |
| `nixosModules.user`    | `systemd --user`                | NixOS configuration     |
| `homeModules.default`  | `systemd --user`                | home-manager (Linux)    |
| `homeModules.darwin`   | launchd agent                   | home-manager (macOS)    |

Root service state lives in `/var/lib/litellm`; user-scope state lives in
`~/.local/state/litellm`. The NixOS user module is installed system-wide
(`/etc/systemd/user`) and starts in every logged-in user's systemd instance.

```nix
{ inputs, ... }: {
  imports = [ inputs.chutes-litellm.nixosModules.default ];

  services.litellm = {
    enable = true;
    apiKeys.CHUTES_API_KEY_PATH = config.age.secrets."litellm/chutes".path;
  };
}
```

For home-manager, import `homeModules.default` on Linux or
`homeModules.darwin` on macOS instead.

If you also import nixpkgs' own `services.litellm` module, disable it first:

```nix
disabledModules = [ "services/misc/litellm.nix" ];
```

### Manual systemd and launchd templates

The [`services/`](../services) directory contains `litellm.service` and
`local.litellm-proxy.plist`. Both launch `scripts/run.sh`; edit the placeholder
path, add `*_API_KEY_PATH` environment entries, then install the unit the usual
way for your platform.

### Important options

| Option                   | Default     | Meaning                                                              |
| ------------------------ | ----------- | -------------------------------------------------------------------- |
| `enable`                 | `false`     | Enable the service.                                                  |
| `package`                | flake build | Package to run.                                                      |
| `host`                   | `127.0.0.1` | Bind address (note: differs from the CLI default `0.0.0.0`).          |
| `port`                   | `4000`      | Bind port.                                                           |
| `apiKeys`                | `{}`        | Env var name to raw value or secret-file path.                       |
| `configTemplate`         | `null`      | Custom template; `null` uses the packaged one.                       |
| `openFirewall`           | `false`     | Open `port` in the NixOS firewall (system scope only).               |
| `httpProxy`, `httpsProxy`, `allProxy`, `noProxy` | `null` | Proxy variables exported to the service.                    |
| `teeMeasurementsFile`    | `null`      | Local measurement YAML; when set it is the sole source.              |
| `teeMeasurementsTtl`     | `3600`      | Measurement cache TTL in seconds.                                    |
| `attestationTtl`         | `300`       | Positive verdict cache TTL in seconds.                               |
| `attestationFailureTtl`  | `30`        | Failed verdict cache TTL in seconds.                                 |
| `attestationModelMapTtl` | `300`       | Model-to-chute lookup cache TTL in seconds.                          |
| `dcapPccsUrl`            | `null`      | DCAP collateral/PCCS base; `null` uses the dcap-qvl default.         |
| `nvidiaNrasUrl`          | `null`      | NVIDIA NRAS endpoint; `null` uses the SDK default.                   |
| `customProvider`         | `false`     | Serve models under the `chutes_e2ee/` prefix.                        |
| `logLevel`               | `"info"`    | `debug`, `info`, `warning`, `error`, or `critical`.                  |
| `logFile`                | `null`      | Log path; `null` uses the default, `"none"` disables file logging.   |

The Chutes API and model-listing endpoints are fixed constants, not Nix
options. There is no option that disables attestation, DCAP, measurement
matching, NRAS, E2EE, or TLS verification.

## API keys and secrets

Every provider is optional except that `chutes/` models need
`CHUTES_API_KEY`. Prefixes and variable names are listed in the
[provider table](../README.md#supported-providers).

`*_API_KEY_PATH` is resolved by the proxy at startup: any environment variable
ending in `_API_KEY_PATH` is read from disk and its contents are placed in the
matching `*_API_KEY` variable. An explicitly set `*_API_KEY` wins.

```sh
export ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic
export CHUTES_API_KEY_PATH=/run/secrets/chutes
```

`run.sh` reads the file on the host and writes the resolved value into a
temporary env file for the container; the temporary file is removed when the
script exits. The Nix `apiKeys` option accepts either a raw string or a path, so
the same `_PATH` variables work there.

Do not bake keys into an image, a Compose file committed to version control, or
a Nix store path that other users can read. Prefer a runtime environment,
Docker/Podman secrets, systemd `LoadCredential`, or agenix/sops-nix.

## Config template, wildcards, and auth

The bundled template is
[`src/chutes_litellm/config/config.template.yml`](../src/chutes_litellm/config/config.template.yml).
Each provider is a wildcard entry such as `openai/*`. At startup:

- a wildcard provider with a key set is expanded by querying that provider's
  model API,
- a provider without a key is skipped with a log line,
- Chutes keeps only ids ending in `-TEE`,
- fixed (non-wildcard) entries are copied through unchanged.

The generated file is written to `LITELLM_OUTPUT` (default
`./config.generated.yml`). Its parent directory is created if needed.

Use `--config PATH` or `LITELLM_TEMPLATE` to supply a custom template. Add
fixed models, or a provider wildcard, in the usual LiteLLM format:

```yaml
model_list:
  - model_name: "chutes/Qwen/Qwen3-32B-TEE"
    litellm_params:
      model: "chutes/Qwen/Qwen3-32B-TEE"
      api_key: "os.environ/CHUTES_API_KEY"

  - model_name: "openai/*"
    litellm_params:
      model: "openai/*"
      api_key: "os.environ/OPENAI_API_KEY"

general_settings:
  # null disables proxy authentication (the bundled default)
  master_key: "sk-local-change-me"
```

With a non-null `master_key`, clients must send
`Authorization: Bearer sk-local-change-me`.

### Optional custom provider

`CHUTES_CUSTOM_PROVIDER=true` (also `1`, `yes`, or `on`) registers a
self-contained `chutes_e2ee/` provider. Its models must be present in the
template; the bundled template does not include them:

```yaml
model_list:
  - model_name: "chutes_e2ee/Qwen/Qwen3-32B-TEE"
    litellm_params:
      model: "chutes_e2ee/Qwen/Qwen3-32B-TEE"
      api_key: "os.environ/CHUTES_API_KEY"
```

## Attestation and logging environment

These variables tune caching and endpoints; none of them weakens or disables a
verification step.

| Variable                           | Default                                      | Meaning                                                        |
| ---------------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| `CHUTES_TEE_MEASUREMENTS_FILE`     | unset                                        | Local measurement YAML; when set it is the sole source.         |
| `CHUTES_TEE_MEASUREMENTS_TTL`      | `3600`                                       | Published measurement cache TTL (seconds).                      |
| `CHUTES_ATTESTATION_TTL`           | `300`                                        | Positive verdict cache TTL (seconds).                           |
| `CHUTES_ATTESTATION_FAILURE_TTL`   | `30`                                         | Failed verdict cache TTL (seconds).                             |
| `CHUTES_ATTESTATION_MODEL_MAP_TTL` | `300`                                        | Model-to-chute lookup cache TTL (seconds).                      |
| `CHUTES_DCAP_PCCS_URL`             | dcap-qvl default                             | DCAP collateral/PCCS base.                                      |
| `CHUTES_NVIDIA_NRAS_URL`           | `https://nras.attestation.nvidia.com/v3/attest/gpu` | NVIDIA NRAS endpoint.                                   |
| `CHUTES_CUSTOM_PROVIDER`           | `false`                                      | Register the optional `chutes_e2ee/` provider.                  |
| `CHUTES_LOG_LEVEL`                 | `info`                                       | `debug`, `info`, `warning`, `error`, or `critical`.             |
| `CHUTES_LOG_FILE`                  | `~/.local/state/chutes_litellm/proxy.log`    | Log path; set `none` to disable file logging.                   |

The Chutes API and model-listing endpoints are fixed and are not in this table.

## Connecting clients

List models and send a chat completion:

```sh
curl -s http://127.0.0.1:4000/v1/models

curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chutes/Qwen/Qwen3-32B-TEE","messages":[{"role":"user","content":"hi"}]}'
```

Add `-H 'authorization: Bearer <master_key>'` when the template enables auth.

opencode (`~/.config/opencode/opencode.jsonc`):

```jsonc
"provider": {
  "litellm": {
    "options": {
      "baseURL": "http://127.0.0.1:4000/v1",
      "type": "openai",
      "apiKey": "dummy" // use the master key when auth is enabled
    }
  }
}
```

For Open WebUI, start the Compose stack and add `http://litellm:4000/v1` as an
OpenAI connection. `scripts/gen_opencode_models.py` can write the per-model
opencode block from the running proxy:

```sh
uv run scripts/gen_opencode_models.py --chutes-api-key "$CHUTES_API_KEY"          # preview
uv run scripts/gen_opencode_models.py --chutes-api-key "$CHUTES_API_KEY" --apply  # write
```

### Reasoning and thinking

Chutes exposes thinking differently by model family. The proxy normalises it so
one client setting works:

| Family | Enable thinking with | Notes |
| ------ | -------------------- | ----- |
| DeepSeek V3.x / V4-Flash | `reasoning_effort: low\|medium\|high\|xhigh\|max` | `none` disables. |
| Gemma 4 | `reasoning_effort`, or `chat_template_kwargs.enable_thinking=true` | A live `reasoning_effort` is translated to the template flag. |
| Kimi K2.6 and other reasoners | reasons by default | No parameter required. |

Thinking is returned in `reasoning_content`, both in the non-streaming message
and as streamed deltas. Generated Chutes entries declare
`litellm_params.allowed_openai_params` so LiteLLM forwards these parameters;
direct SDK calls can use `extra_body={...}` or the equivalent.

## Verifying attestation manually

`chutes-verify-attestation` uses the same gate as the proxy:

```sh
export CHUTES_API_KEY=cpk_...
uv run chutes-verify-attestation --model "Qwen/Qwen3-32B-TEE"

# per-instance checks, DCAP status, and GPU verdicts
uv run chutes-verify-attestation --chute <uuid> --details
```

| Flag        | Meaning                                                        |
| ----------- | -------------------------------------------------------------- |
| `--model`   | Chutes model id as listed at `https://llm.chutes.ai/v1/models`. |
| `--chute`   | Chute UUID; takes precedence over `--model`.                   |
| `--api-key` | Key to use instead of `CHUTES_API_KEY`.                        |
| `--force`   | Ignore the verification cache (default, non-`--details` path). |
| `--details` | Print per-instance checks, DCAP TCB status, and GPU counts.    |

Exit status is `0` on success, `1` when any instance fails, and `2` for missing
arguments or a missing key. The CLI always runs the full chain: signed evidence,
DCAP, certificate and `report_data` binding, measurement matching, and NRAS.

The Nix package installs the `litellm-proxy` launcher only; run
`chutes-verify-attestation` from a `uv` checkout, or use
`python -m chutes_litellm.verify` in a `nix develop` shell.

## Troubleshooting

**`litellm` is not importable (exit 1).** This is an import failure, not a PATH
lookup. Run `uv sync` in the checkout, or use the Docker image or Nix package,
which include `litellm` in their environment.

**A provider is skipped.** At startup `generate_config` logs
`<ENV> not set — skipping <provider>`. Set the corresponding `*_API_KEY` or
`*_API_KEY_PATH`.

**`--config` file not found (exit 2).** `--config` must name an existing file,
not a directory. `LITELLM_TEMPLATE` is not pre-validated by the CLI; a missing
template path fails when `generate_config` tries to read it.

**No TEE models, or a 503 with attestation reasons.** This is the expected
fail-closed behaviour, not a configuration switch. Check that:
`CHUTES_API_KEY` is valid; the requested model id ends in `-TEE`; the host can
reach `api.chutes.ai` and `llm.chutes.ai`; and, if you pinned
`CHUTES_TEE_MEASUREMENTS_FILE`, the local YAML is readable and matches the
published measurement. Run `chutes-verify-attestation --details` to see which
per-instance check failed. The failure is intentionally not bypassable.

**`run.sh` cannot find the image.** Build it first with
`docker build -t litellm-proxy .`.

**`run.sh` ignores `LITELLM_HOST`/`LITELLM_PORT`.** It forwards only API keys
and publishes `LITELLM_BIND` to container port `4000`. Use Compose or
`docker run -e ...` for other runtime variables.

**Nix CLI flags.** The launcher forwards arguments:
`nix run .# -- --host 127.0.0.1 --port 8000`. The environment-only form
`LITELLM_HOST=127.0.0.1 LITELLM_PORT=8000 nix run .#` is equivalent.

**Importing `litellm` fails with `libstdc++.so.6` on NixOS.** Run the venv Python
with the compiler runtime on the library path:

```sh
LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so)")" \
  .venv/bin/python -m pytest tests/ -q
```

The flake's `nix develop` shell sets this path for you.
