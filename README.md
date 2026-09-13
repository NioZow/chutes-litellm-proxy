# Chutes LiteLLM Proxy

A LiteLLM proxy that exposes a unified OpenAI-compatible API for all major providers.

## Quickstart

```sh
uv sync                                             # Python 3.13 / 3.14

# Start the proxy (it generates config, installs the Chutes E2EE transport, then serves)
CHUTES_API_KEY=cpk_... uv run chutes-litellm-proxy  # → http://127.0.0.1:4000

# Talk to it with any OpenAI-compatible client
curl -s localhost:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chutes/Qwen/Qwen3-32B-TEE","messages":[{"role":"user","content":"hi"}]}'
```

Or run the container:

```sh
docker build -t litellm-proxy . && CHUTES_API_KEY=cpk_... ./scripts/run.sh
```

- Tests: see [TEST.md](./TEST.md) (`uv run pytest tests/ -q`).
- Deployment (systemd, launchd, Nix, Open WebUI): see [Deployment](#deployment).
- Usage and config knobs: see [Usage](#usage); model generation: see
  [Adding models](#adding-models).

## Security model

Chutes `*-TEE` traffic is end-to-end encrypted with post-quantum ML-KEM-768 +
ChaCha20-Poly1305, and (optionally) only sent to instances whose TDX/GPU
attestation verifies. The full explanation lives in [`docs/`](./docs/README.md):

- **[docs/e2ee-encryption.md](./docs/e2ee-encryption.md)** — threat model, the
  primitives, the exact wire/blob format with byte offsets, request/reply key
  establishment, streaming, nonces and replay, and what is *not* protected.
- **[docs/attestation.md](./docs/attestation.md)** — the evidence envelope, the
  binding checks, Intel DCAP + NVIDIA NRAS hardware roots, fail-closed
  semantics, caching, the `CHUTES_*` knobs and the `chutes-verify-attestation`
  CLI.

Hardware-root verification is optional and packaged like any other extra:

```sh
uv sync --extra attestation     # adds dcap-qvl + nv-attestation-sdk
nix build .#attestation         # same, as a native Nix package
```

The base install, the Docker image and the offline test suite work without it;
asking for `CHUTES_VERIFY_QUOTE=true` / `CHUTES_VERIFY_GPU=true` without the
SDKs fails closed (the request is refused).

## Supported providers

| Provider    | Env var              | Model prefix   |
| ----------- | -------------------- | -------------- |
| Anthropic   | `ANTHROPIC_API_KEY`  | `anthropic/`   |
| OpenAI      | `OPENAI_API_KEY`     | `openai/`      |
| Google      | `GEMINI_API_KEY`     | `gemini/`      |
| Perplexity  | `PERPLEXITY_API_KEY` | `perplexity/`  |
| xAI         | `XAI_API_KEY`        | `xai/`         |
| DeepSeek    | `DEEPSEEK_API_KEY`   | `deepseek/`    |
| Meta        | `META_API_KEY`       | `meta/`        |
| Chutes TEE (E2EE) | `CHUTES_API_KEY`    | `chutes/`      |

> **Native `_PATH` support:** Instead of setting `ANTHROPIC_API_KEY=sk-…` directly,
> you may set `ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic` (or any path to a file
> containing the raw key). The proxy will read the file and treat its content as the
> key. This works for every provider above and is the cleanest way to inject secrets
> from systemd service files.

## Deployment

> [!WARNING]
> The docker image must first be built before the services can start.
> Both `podman` and `docker` are supported, the image can be built using `docker build -t litellm-proxy .`.
>
> The bundled [config.template.yml](./src/chutes_litellm/config/config.template.yml)
> already lists every supported provider. At startup the proxy queries each
> provider that has an API key set and expands it into concrete models; providers
> without a key are skipped, so the same image serves OpenAI/Anthropic/Gemini/
> DeepSeek/Meta/… as soon as you pass the matching `*_API_KEY` to
> [run.sh](./scripts/run.sh) (or `docker-compose.yml`).

### Manual

You can find a manual launchd file and systemd service file in the [./services](./services) folder.

Otherwise you can launch the [run.sh](./scripts/run.sh) script directly.

```
$ head scripts/run.sh
#!/usr/bin/env bash
# Runs the litellm-proxy container. Configuration is supplied entirely through
# environment variables (no CLI arguments are accepted).
#
# Env vars:
#   LITELLM_BIND      Host:port to publish (default: 127.0.0.1:4000; comma-separated)
#   RUNTIME           Container engine: podman | docker | auto (default: auto)
#   *_API_KEY          Raw API keys forwarded as-is.
#   *_API_KEY_PATH     File path containing API key; forwarded as *_API_KEY.
# Example:
#   LITELLM_BIND=127.0.0.1:4000 ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic ./scripts/run.sh
```

#### macOS

To install manually :

```sh
cp services/local.litellm-proxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local.litellm-proxy.plist
```

To start/stop :

```sh
launchctl load ~/Library/LaunchAgents/local.litellm-proxy.plist
launchctl bootout gui/$UID/local.litellm-proxy
```

#### linux

Install :

```sh
cp services/litellm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now litellm
```

### uv

The repository is a normal Python project (`pyproject.toml` + committed
`uv.lock`), so it can also be installed and run with [uv](https://docs.astral.sh/uv)
— no container, no Nix. Requires Python 3.13 or 3.14 (`pqcrypto` ships
prebuilt wheels for those only).

```sh
uv sync                                   # create .venv from uv.lock
uv run chutes-litellm-proxy               # generate config, install E2EE, run the proxy
# or tune it through env vars, e.g.:
LITELLM_HOST=127.0.0.1 LITELLM_PORT=4000 CHUTES_API_KEY=cpk_... uv run chutes-litellm-proxy
```

`uv run chutes-litellm-proxy` is the same entry point the Docker and Nix paths
use: it expands the bundled template into `config.generated.yml`, installs the
host-scoped Chutes E2EE transport, then execs `litellm`.

> The console script is named `chutes-litellm-proxy` because the `litellm`
> package itself already ships a `litellm-proxy` command (its interactive
> client). The Docker/Nix launchers are unaffected — they use their own
> `litellm-proxy` wrapper, not the Python entry point.

Useful one-liners:

```sh
uv run chutes-verify-attestation --model "Qwen/Qwen3.5-397B-A17B-TEE"   # attestation CLI
uv run scripts/generate_config.py                                        # regenerate config only
uv run scripts/proxy_test.py                                             # smoke test the running proxy
uv run pytest tests/ -q                                                  # protocol/E2EE test suite
uv run ruff check src scripts tests                                      # lint
```

### Nix

#### Declarative services (`nix-services`)

Service units are generated by the [`nix-services`](https://github.com/niozow/nix-services)
library (`mkService`), so the same proxy can run as a **root system service**, a
**NixOS-declared user service**, or a **home-manager user service**:

| Deployment | Flake output | Runs as | Declared in |
| --- | --- | --- | --- |
| NixOS system service | `nixosModules.default` | root (`DynamicUser`) | NixOS configuration |
| NixOS user service | `nixosModules.user` | unprivileged user (`systemd --user`) | NixOS configuration |
| home-manager (Linux) | `homeModules.default` | your user (`systemd --user`) | home-manager configuration |
| home-manager (macOS) | `homeModules.darwin` | your user (`launchd.agents`) | home-manager configuration |

All four share the same options (see [`./options.nix`](./options.nix)) and the
same package; they differ in *who runs the unit* and *who can declare it*:

- **root — `nixosModules.default`** starts at boot without a login, applies
  systemd hardening (`DynamicUser`, `StateDirectory`, `ProtectSystem=strict`, …)
  and keeps state in `/var/lib/litellm`. Needs root to build/deploy.
- **NixOS user — `nixosModules.user`** is a `systemd --user` unit declared in the
  **system** config. It runs unprivileged, but declaring/activating it still
  needs a `nixos-rebuild` (root). Good when the host is managed declaratively but
  you don't use home-manager.
- **home-manager — `homeModules.default` / `homeModules.darwin`** is a per-user
  unit declared in **your own** home-manager config. This is the one you can
  enable as an ordinary user (`home-manager switch`), and it is scoped to *your*
  account only.

User-scope variants keep config and logs under `~/.local/state/litellm`.

##### NixOS, run as root

```nix
{ inputs, ... }: {
  imports = [ inputs.chutes-litellm.nixosModules.default ];

  services.litellm = {
    enable = true;
    apiKeys.CHUTES_API_KEY_PATH = config.age.secrets."litellm/chutes".path;
    # Or pass a raw key directly
    # OPENAI_API_KEY = "sk-...";
  };
}
```

##### NixOS, run as an unprivileged user (no home-manager)

```nix
{ inputs, ... }: {
  imports = [ inputs.chutes-litellm.nixosModules.user ];

  services.litellm = {
    enable = true;
    apiKeys.CHUTES_API_KEY_PATH = config.age.secrets."litellm/chutes".path;
  };
}
```

> The NixOS user unit is installed system-wide (`/etc/systemd/user`) with
> `WantedBy=default.target`, so it starts in **every** logged-in user's systemd
> instance. Use the home-manager variant below if you want it scoped to one user.

##### home-manager, run as your user

```nix
{ inputs, config, ... }: {
  # Use homeModules.darwin on macOS and homeModules.default on Linux.
  imports = [ inputs.chutes-litellm.homeModules.darwin ];

  services.litellm = {
    enable = true;
    apiKeys.CHUTES_API_KEY_PATH = config.age.secrets."litellm/chutes".path;
  };
}
```

Under the hood, [`options.nix`](./options.nix) is a function of a small
deployment record; [`flake.nix`](./flake.nix) binds it to
`nix-services.lib.mkService` with a constant `scope`/`homeManager` (these decide
the emitted unit schema and cannot be read from `config`). The `nix-services`
README has the full schema matrix.

> [!NOTE]
> nixpkgs unstable now ships its own `services.litellm` module. If you import it
> too, disable it first: `disabledModules = [ "services/misc/litellm.nix" ];`

#### Nix flake (native build, no container)

The repository is also a Nix flake that builds the proxy as a native Python
environment (no podman/docker). It builds `litellm` from nixpkgs, the
`chutes-e2ee` transport from source, and fetches the `pqcrypto==0.4.0` wheel
that `chutes-e2ee` requires.

```sh
# Development shell (litellm + chutes provider + tools on $PATH)
nix develop

# Build the proxy package
nix build .#

# Run the proxy (generate_config.py then litellm on 127.0.0.1:4000)
nix run .# -- LITELLM_PORT=4000
# or set env vars when launching:
#   LITELLM_HOST=127.0.0.1 LITELLM_PORT=4000 CHUTES_API_KEY=<key> nix run .#
```

To install it declaratively, import one of the service modules from the flake
(`nixosModules.default`, `nixosModules.user`, `homeModules.default` or
`homeModules.darwin` — see [Declarative services](#declarative-services-nix-services)).

> The `apiKeys` option maps env var names directly to values. The proxy natively
> resolves `_PATH` suffixes at runtime (e.g. `CHUTES_API_KEY_PATH` is read as a
> file), so you can mix raw keys and secret file paths in the same config.

On first build, fill in the `chutes-e2ee` source hash in `flake.nix` (the
placeholder is intentional):

```sh
nix build .#default 2>&1 | grep -A2 'got:'
```

## Usage

The proxy is OpenAI-compatible. Point any tool at `http://127.0.0.1:<port>` with any API key (auth is disabled by default but can be enabled through the bundled [config.template.yml](./src/chutes_litellm/config/config.template.yml) file).

**List available models:**

```sh
curl http://127.0.0.1:4000/v1/models | jq '.data[].id'
```

**Chat (curl):**

```sh
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**opencode config (`~/.config/opencode/opencode.jsonc`):**

```jsonc
"provider": {
  "litellm": {
    "options": {
      "baseURL": "http://127.0.0.1:4000/v1",
      "type": "openai",
      "apiKey": "dummy" // only if didn't configure any in the litellm config and disabled auth
    }
  }
}
```

## Adding models

The models will be automatically queried from each provider API and made available through the LiteLLM proxy. This is done through the [generate_config.py](./scripts/generate_config.py) script.

### Generating opencode models

opencode needs per-model metadata (capabilities, limits, pricing), so it cannot
discover them on its own. [gen_opencode_models.py](./scripts/gen_opencode_models.py)
scrapes LiteLLM's `/model/info` (plus the Chutes listing for TEE models) and
rewrites the `provider.litellm.models` block of your opencode config in place:

```sh
# preview the generated block
uv run scripts/gen_opencode_models.py --chutes-api-key "$CHUTES_API_KEY"

# write it into ~/.config/opencode/opencode.jsonc (other config preserved)
uv run scripts/gen_opencode_models.py --apply
```

Each entry includes the extended fields opencode understands (validated against
the published [config schema](https://opencode.ai/config.json)):

- `tool_call`, `temperature`, `attachment`, `reasoning` capabilities,
- `interleaved` (`reasoning_content`) for reasoning models,
- `limit.context` / `limit.output` (LiteLLM → Chutes → models.dev → existing),
- `cost` (input/output/cache-read USD per 1M tokens, from Chutes pricing),
- `modalities` (`input`/`output`: text, image, video, …),
- `release_date` and `status`,
- `variants` for reasoning models — a thinking-level map (`off`, `low`,
  `medium`, `high`, …) that sends the matching `reasoning_effort` /
  `chat_template_kwargs` to the proxy.

## Reasoning / thinking

Chutes exposes model thinking differently per family. The custom `chutes_e2ee`
provider normalises it so one client setting works, and `generate_config`
declares the parameters LiteLLM needs to forward them (instead of dropping them):

| Family | Enable thinking with | Notes |
| ------ | -------------------- | ----- |
| DeepSeek V3.x / V4-Flash | `reasoning_effort: low\|medium\|high\|xhigh\|max` | `none`/`off` disables |
| Gemma 4 | `reasoning_effort`, or `chat_template_kwargs.enable_thinking=true` | a live `reasoning_effort` is auto-translated to the template flag |
| Kimi K2.6 and other reasoners | reasons by default | no parameter required |

Thinking is returned in the standard `reasoning_content` field — on the
non-streaming message and as streamed deltas. Because LiteLLM strips unknown
parameters for custom providers, the generated config sets
`litellm_params.allowed_openai_params`; in direct SDK calls pass
`allowed_openai_params=[...]` (or use `extra_body={...}`). The live tests assert
this for Kimi K2.6, Gemma 4 and DeepSeek V4-Flash (see [TEST.md](./TEST.md)).

## Testing

Full documentation lives in [TEST.md](./TEST.md).

### Unit / protocol tests (no proxy, no network)

`tests/` contains a protocol-faithful mock of the Chutes E2EE control plane
(`tests/mock_chutes_server.py`).  Because the mock holds the instance ML-KEM-768
*private* key, the tests are cryptographic proof that the native `chutes`
provider + this integration really do end-to-end encryption, and that the
attestation gate blocks unverified instances:

```sh
# inside `nix develop`, or after `uv sync` (the dev dependency group adds pytest)
uv run pytest tests/ -q
```

To exercise the *real* attestation/evidence endpoints against api.chutes.ai
(needs a `CHUTES_API_KEY`), add live checks guarded by `CHUTES_LIVE_TEE=1`.

### Live end-to-end tests (real key + network)

Run the live module on its own — the offline suite points at the mock and would
otherwise clobber your real credentials:

```sh
CHUTES_LIVE_TEE=1 CHUTES_API_KEY=<key> \
  .venv/bin/python -m pytest tests/test_live_proxy.py -v -s
```

`tests/test_live_proxy.py` covers, in order:

1. direct custom-provider invoke over the real E2EE transport (non-streaming),
2. the same in streaming mode,
3. thinking visibility (`reasoning_content`, non-stream + stream) for Kimi K2.6,
   Gemma 4 and DeepSeek V4-Flash,
4. an actual in-process LiteLLM proxy (`scripts/live_server.py`) reached over
   HTTP: health check, a streamed and non-streamed `/v1/chat/completions` call,
   and a reasoning request through the `chutes_e2ee/<model>` custom provider.

The target model defaults to a DeepSeek `-TEE` model auto-selected from the
live `/v1/models` listing; override with `CHUTES_LIVE_MODEL=<id>`. The existing
`tests/test_live_attestation.py` additionally verifies live evidence binding.

`scripts/live_server.py` exists because the deployed `chutes-litellm-proxy`
entry point `os.execvp`s the `litellm` CLI, which discards the in-memory
provider/transport registrations.  The launcher keeps them in the serving
process instead, and is what the live proxy test spawns.

### Smoke tests against the running proxy

Tests both non-streaming and streaming:

```sh
# Test Chutes TEE (default)
./scripts/proxy_test.py

# Test any other model
./scripts/proxy_test.py anthropic/claude-sonnet-4-6
./scripts/proxy_test.py openai/gpt-4o
```

## Chutes TEE / end-to-end encryption

Chutes serves `*-TEE` models from Intel TDX confidential VMs driving NVIDIA GPUs
in Confidential-Compute mode, and encrypts inference with post-quantum ML-KEM-768.
This proxy relies on LiteLLM's **native `chutes` provider** (declarative
OpenAI-compatible provider) and layers the encryption on underneath it:

* `src/chutes_litellm/e2ee_litellm.py` — swaps LiteLLM's OpenAI-SDK HTTP client
  factories for shared clients whose transport is the `chutes-e2ee-transport`
  library (ML-KEM-768 + ChaCha20-Poly1305), scoped so only `llm.chutes.ai` /
  `api.chutes.ai` traffic is encrypted and every other provider is untouched.
  There is **no custom LiteLLM provider** to maintain: params, transforms,
  streaming, usage and retries are all upstream LiteLLM code.
* `src/chutes_litellm/custom_provider.py` — optional **self-contained custom
  provider** (`chutes_e2ee/<model>`).  A `CustomLLM` handler registered through
  LiteLLM's documented `custom_provider_map` extension point; it owns its own
  `chutes-e2ee` transport and speaks OpenAI chat protocol over it, with **no**
  LiteLLM-internal patching (non-stream + stream, tool calls, reasoning_content,
  usage, error mapping).  It also honours the same
  `CHUTES_VERIFY_ATTESTATION` / `CHUTES_VERIFY_QUOTE` / `CHUTES_VERIFY_GPU`
  gate as the native transport.  Enable with `CHUTES_CUSTOM_PROVIDER=true` and configure
  models under the `chutes_e2ee/` prefix (e.g. `chutes_e2ee/Qwen/Qwen3.5-397B-A17B-TEE`).
* `src/chutes_litellm/attestation.py` — optional **GPU/TEE attestation
  verification** (`CHUTES_VERIFY_ATTESTATION=true`): before encrypting, it
  fetches the instance's TDX + NVIDIA evidence, verifies the key-possession
  signature and that the instance public key is bound to attested hardware, and
  refuses to send (fail closed) otherwise.  **Every** instance listed by
  `/e2e/instances` must verify — the transport chooses an instance itself, so a
  partially-verified chute would still let a request land on the unverified
  instance.  The gate also runs as the transport's **instance-selection filter**
  (see `chutes-e2ee-transport/src/chutes_e2ee/discovery.py`), so the encrypted
  request can only go to an instance that was verified in the same decision.
  For full Intel DCAP / NVIDIA NRAS verification install the `attestation` extra
  (`uv sync --extra attestation`, which adds `dcap-qvl` + `nv-attestation-sdk`)
  and set `CHUTES_VERIFY_QUOTE=true` / `CHUTES_VERIFY_GPU=true` (honoured by both
  the native transport and the custom provider).  See
  [docs/attestation.md](./docs/attestation.md) for the full trust chain.

  > **Note — `python-ecdsa` (CVE-2024-23342).** The Nix build pins
  > `python-ecdsa`, a transitive dependency of `nv-attestation-sdk` /
  > `nv-local-gpu-verifier`, which nixpkgs flags as insecure for
  > CVE-2024-23342 (the "Minerva" timing side-channel). That flaw lives in the
  > library's **private-key** operations (ECDSA signing / nonce derivation) and
  > requires the secret key to be present. The NVIDIA code only ever *verifies*
  > signatures: it imports just `VerifyingKey` + `BadSignatureError` and calls
  > `VerifyingKey.from_pem(cert_public_key).verify(...)`, which uses public key
  > material only and is not exposed to the side channel. The flake therefore
  > drops the advisory for this verification-only dependency (`ecdsa` override
  > in `flake.nix`) rather than globally permitting insecure packages. No code
  > path signs with this library.

* `scripts/verify_attestation.py` — standalone CLI to verify a model/chute
  against the live API (see below); `--details` prints per-instance checks, the
  DCAP TCB status and NVIDIA GPU verdicts.

Relevant environment variables:

| Variable                    | Default                     | Meaning                                              |
| --------------------------- | --------------------------- | ---------------------------------------------------- |
| `CHUTES_E2EE_HOSTS`         | `llm.chutes.ai,api.chutes.ai` | Hosts whose traffic is end-to-end encrypted        |
| `CHUTES_E2EE_API_BASE`      | `https://api.chutes.ai`     | E2EE/attestation API base (override for tests)       |
| `CHUTES_E2EE_MODELS_BASE`   | `https://llm.chutes.ai`     | Model-listing base used for model→chute resolution   |
| `CHUTES_VERIFY_ATTESTATION` | `true`                      | Enable the fail-closed attestation gate              |
| `CHUTES_VERIFY_QUOTE`       | `true`                      | Require Intel DCAP TDX-quote verification            |
| `CHUTES_VERIFY_GPU`         | `true`                      | Require NVIDIA attestation-SDK GPU verification      |
| `CHUTES_ATTESTATION_TTL`    | `300`                       | How long a verified chute stays trusted (seconds)    |
| `CHUTES_ATTESTATION_FAILURE_TTL` | `30`                   | How long a failed verdict is cached (seconds)        |
| `CHUTES_ATTESTATION_MODEL_MAP_TTL` | `300`               | How long a model→chute id lookup is cached (seconds) |
| `CHUTES_DCAP_PCCS_URL`      | dcap-qvl default (Phala `pccs.phala.network`) | Intel/Phala collateral (PCCS) base for DCAP |
| `CHUTES_NVIDIA_NRAS_URL`    | `https://nras.attestation.nvidia.com/v3/attest/gpu` | NVIDIA remote attestation (NRAS) endpoint |
| `CHUTES_LOG_LEVEL`          | `info`                      | Logger level: debug, info, warning, error, critical  |
| `CHUTES_LOG_FILE`           | `~/.local/state/chutes_litellm/proxy.log` | Optional log file path; set `none` to disable |

**Logging.** All attestation and E2EE events are emitted through a shared logger that writes to both stderr (visible in `docker logs`, `journalctl`, launchd logfiles, etc.) and an optional on-disk file.  The file path defaults to `~/.local/state/chutes_litellm/proxy.log` and is overridable via `CHUTES_LOG_FILE` (set to `none` to disable file logging).  The level is controlled by `CHUTES_LOG_LEVEL` (default `info`).

**Overhead.** Attestation is **on by default**, so every request is guarded. This
trusts that the proxy verifies TEE/GPU evidence before encrypting traffic. You can opt out by setting `CHUTES_VERIFY_ATTESTATION=false`, but that removes all guarantees that the instance you are talking to is running on trusted hardware.


The E2EE transport itself is also connection-oriented: HTTP clients/transports
are cached per API key (see `e2ee_litellm._clients` / `custom_provider`), so
requests ride a keep-alive connection to `llm.chutes.ai` / `api.chutes.ai`
instead of a fresh TCP + TLS handshake per call. The ML-KEM-768 + ChaCha20
encryption is performed locally per request — CPU work, no extra network round
trip.

Manual attestation check:

```sh
CHUTES_API_KEY=cpk_... ./scripts/verify_attestation.py --model "Qwen/Qwen3.5-397B-A17B-TEE"
# full hardware roots (needs dcap-qvl + nv-attestation-sdk):
CHUTES_API_KEY=cpk_... ./scripts/verify_attestation.py --chute <uuid> --verify-quote --verify-gpu
```

The `chutes-e2ee-transport` package is installed in the image at build time.

> Note: the `chutes-e2ee` transport itself only *encrypts* — it does **not**
> verify GPU attestation. That gap is what `chutes_litellm.attestation` closes.

## Open WebUI

You can also start the litellm proxy along with an open-webui container using docker compose. In the open-webui settings, it should then possible to add `http://litellm:4000/v1` to your OpenAI API Connections.

## Structure

```
src/chutes_litellm/     # The Python package (installable via uv or nix)
  config/config.template.yml  # source of truth for the model list (ships as package data)
  generate_config.py    #   expands wildcard providers into a generated config
  e2ee_litellm.py       #   host-scoped swap of litellm's OpenAI-SDK http clients (install())
  custom_provider.py    #   optional self-contained chutes_e2ee CustomLLM provider (no patching)
  attestation.py        #   optional TDX/GPU attestation verification (fail-closed)
  verify.py             #   attestation CLI (chutes-verify-attestation)
  cli.py                #   chutes-litellm-proxy entry: generate -> E2EE install -> litellm
scripts/                # Runnable tools & thin wrappers (source checkouts)
  generate_config.py    #   wrapper: regenerate config.generated.yml only
  proxy_test.py         #   smoke test against the running proxy (streaming + non-streaming)
  gen_opencode_models.py#   generates the opencode.jsonc models block from /model/info
  live_server.py        #   in-process proxy launcher for live/E2E tests
  verify_attestation.py #   wrapper: verify a chute/model's TEE+GPU attestation
  run.sh                #   runs the container via rootless Podman / Docker
tests/                  # Protocol-faithful E2EE + attestation tests (mock server, no network)
docs/                   # Security docs: e2ee-encryption.md, attestation.md (start at docs/README.md)
services/               # launchd + systemd templates to run the proxy on boot
options.nix             # shared service options/config (NixOS + home-manager)
pyproject.toml + uv.lock# uv project definition (chutes-litellm-proxy / chutes-verify-attestation)
Dockerfile              # Image definition: uv base + `uv sync --locked --extra attestation` (runs `python -m chutes_litellm.cli`)
docker-compose.yml      # Optional litellm + open-webui compose stack
flake.nix               # Nix flake: native package, dev shell, service modules
```

### Why two `generate_config.py` files?

They are not duplicates:

- `src/chutes_litellm/generate_config.py` is the **library** — the real
  implementation. It is imported by the `chutes-litellm-proxy` entry point
  (`chutes_litellm.cli`), which regenerates the config on every startup.
- `scripts/generate_config.py` is a **17-line wrapper** for source checkouts: it
  adds `src/` to `sys.path`, imports the library, and calls `main()`. It exists
  so you can run `uv run scripts/generate_config.py` without going through the
  console script or importing the package manually.

The same pattern applies to `verify_attestation.py` (wrapper) over
`chutes_litellm.verify` (library).

## Attestation performance

Attestation incurs a small, predictable latency that is **not** on the hot path of
every request. The design front-loads verification into a one-time scan and then
caches the verdict so that steady-state traffic is a fast local lookup.

### What is scanned and when

Every time the proxy sees a model it has not yet attested—or a cached verdict
expires—it performs one scan of the entire chute:

1. **Model→chute lookup** (`GET /v1/models`) — resolves the model id to the
   stable chute UUID (cached for `CHUTES_ATTESTATION_MODEL_MAP_TTL`).
2. **Instance enumeration** (`GET /e2e/instances/<chute>`) — lists every live
   instance with its ML-KEM public key.
3. **Evidence fetch** (`GET /chutes/<chute>/evidence`) — fetches the TDX quote and
   NVIDIA GPU evidence blobs for *all* instances in a single request.
4. **Per-instance verification** — for each instance the gate checks:
   - key-possession signature (software binding),
   - TDX `report_data` binds the instance pubkey,
   - GPU evidence binds the instance pubkey,
   - optional Intel DCAP quote root verification,
   - optional NVIDIA NRAS GPU root verification.

Only when **every** instance passes is the chute accepted. If any instance fails,
the entire chute is rejected (fail-closed). The scan is
[coalesced](https://en.wikipedia.org/wiki/Cache_stampede): if N requests arrive
simultaneously for the same cold chute, only one performs the evidence fetch; the
rest block until the leader stores the result in the shared cache.

### Steady-state overhead

After a chute is verified, requests are gated by a **dict lookup** on a cache key
`(chute_id, verify_quote, verify_gpu, check_signature)`. The lookup is
`O(1)`, adds ~0.05–0.1 ms of local CPU, and involves **no network traffic**.

| TTL knob | Default | Effect |
|----------|---------|--------|
| `CHUTES_ATTESTATION_TTL` | `300` | How long a positive verdict (all instances verified) is reused |
| `CHUTES_ATTESTATION_FAILURE_TTL` | `30` | How long a rejected chute is black-listed (prevents retry storms) |
| `CHUTES_ATTESTATION_MODEL_MAP_TTL` | `300` | How long the model→chute lookup is reused |

### Why latency is dominated by network, not crypto

The cryptographic work per instance (signature verification, SHA-256 digest
checks, DCAP quote parsing) is measured in sub-milliseconds on modern hardware.
The dominant cost is the **three sequential HTTPS round trips** (model listing,
instances, evidence) to Chutes' control plane, which together take **150–400 ms**
under typical network conditions. The optional DCAP / NVIDIA verifier calls add
one or two extra HTTPS round trips each (Intel PCCS and NVIDIA NRAS).

Re-verification only happens on cache expiry, so a busy deployment quickly
settles into the fast path. You can increase `CHUTES_ATTESTATION_TTL` to trade
freshness for lower scanning frequency; many production deployments run with
600–1800 s.

### Disabling attestation

If you trust the network and wish to eliminate all scanning latency, set:

```sh
CHUTES_VERIFY_ATTESTATION=false
```

This also disables the custom provider's pre-send gate and the transport's
instance filter. The E2EE encryption layer itself remains active—traffic is
still encrypted with ML-KEM-768 + ChaCha20-Poly1305—only the hardware-trust
check is bypassed.

### Why this matters for *-TEE models

Attestation is the mechanism that lets you trust that the `*-TEE` instance is
truly running inside an Intel TDX confidential VM with a GPU in
confidential-compute mode. Without it, the proxy encrypts to an arbitrary
instance's public key with no guarantee that the remote machine is the hardware
it claims to be. The default-on behaviour keeps that guarantee without requiring
operators to remember to enable it.

| Path       | Command                                              | Generates config + installs E2EE |
| ---------- | ---------------------------------------------------- | ------------------------------- |
| uv         | `uv sync && uv run chutes-litellm-proxy`          | yes (`chutes_litellm.cli`)       |
| Nix        | `nix run .#`                                         | yes (same CLI entry point)       |
| Docker     | `docker build -t litellm-proxy . && ./scripts/run.sh`| yes (same CLI entry point)       |
