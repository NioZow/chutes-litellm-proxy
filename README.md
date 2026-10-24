# Chutes LiteLLM Proxy

An OpenAI-compatible LiteLLM proxy for Chutes and the major model providers. It
exposes one `/v1` API, expands each provider's model catalog at startup, and
forwards non-Chutes traffic normally. Chutes `*-TEE` requests are different:
they are end-to-end encrypted with ML-KEM-768 + ChaCha20-Poly1305 and are sent
only to instances whose Intel TDX and NVIDIA GPU attestation verifies. TEE
attestation, Intel DCAP quote verification, NVIDIA NRAS GPU verification,
measurement matching, E2EE and TLS verification are mandatory and cannot be
disabled.

## Quickstart

### uv

```sh
uv sync
export CHUTES_API_KEY=cpk_...
uv run chutes-litellm-proxy --host 127.0.0.1 --port 4000
```

```sh
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chutes/Qwen/Qwen3-32B-TEE","messages":[{"role":"user","content":"hi"}]}'
```

### Docker / Podman

```sh
docker build -t litellm-proxy .
RUNTIME=docker CHUTES_API_KEY=cpk_... ./scripts/run.sh
```

For Podman, use `podman build -t litellm-proxy .` and `RUNTIME=podman`.
`run.sh` publishes `127.0.0.1:4000` by default and forwards `*_API_KEY` /
`*_API_KEY_PATH` variables. See [docs/setup.md](docs/setup.md) for Compose and
deployment details.

### Nix

`CHUTES_API_KEY=cpk_... LITELLM_HOST=127.0.0.1 nix run .#` — see
[docs/setup.md](docs/setup.md#nix).

## Supported providers

Only providers whose API key is set are expanded; the others are skipped.

| Provider          | Env var              | Model prefix  |
| ----------------- | -------------------- | ------------- |
| Anthropic         | `ANTHROPIC_API_KEY`  | `anthropic/`  |
| OpenAI            | `OPENAI_API_KEY`     | `openai/`     |
| Google Gemini     | `GEMINI_API_KEY`     | `gemini/`     |
| Perplexity        | `PERPLEXITY_API_KEY` | `perplexity/` |
| xAI               | `XAI_API_KEY`        | `xai/`        |
| DeepSeek          | `DEEPSEEK_API_KEY`   | `deepseek/`   |
| Meta              | `META_API_KEY`       | `meta/`       |
| Mistral           | `MISTRAL_API_KEY`    | `mistral/`    |
| Chutes TEE (E2EE) | `CHUTES_API_KEY`     | `chutes/`     |

Chutes model discovery keeps only models whose id ends in `-TEE`.

## CLI and environment

```text
chutes-litellm-proxy [--config PATH] [--host HOST] [--port PORT]
```

Command-line values override the environment. Defaults are the packaged
template, `0.0.0.0`, and port `4000`. A missing `--config` file or an invalid
port exits with status `2`.

| Variable            | Default                     | Meaning                                                    |
| ------------------- | --------------------------- | ---------------------------------------------------------- |
| `CHUTES_API_KEY`    | unset                       | Chutes API key; required for `chutes/` models              |
| `*_API_KEY`         | per provider                | Provider key, as listed above                              |
| `*_API_KEY_PATH`    | unset                       | File containing the key; read at startup                   |
| `LITELLM_HOST`      | `0.0.0.0`                   | Bind host (overridden by `--host`)                         |
| `LITELLM_PORT`      | `4000`                      | Bind port (overridden by `--port`)                         |
| `LITELLM_TEMPLATE`  | packaged `config.template.yml` | YAML template to expand (overridden by `--config`)       |
| `LITELLM_OUTPUT`    | `./config.generated.yml`    | Where the generated config is written                      |

An explicit `*_API_KEY` wins over its `*_API_KEY_PATH`. The generated config is
written on every start; use a custom `--config` template for extra models or to
set `general_settings.master_key`.

## Security

Chutes `*-TEE` traffic is protected by two always-on layers:

1. **End-to-end encryption.** The proxy encapsulates a fresh ML-KEM-768 secret
   to the instance key, derives an HKDF-SHA256 key, and encrypts the request
   with ChaCha20-Poly1305. The reply and streamed chunks come back encrypted to
   an ephemeral client key.
2. **Hardware attestation.** Before encrypting, the proxy verifies the RSA
   signature over the evidence body, runs Intel DCAP on the TDX quote, binds
   `report_data` to the instance key and certificate, matches MRTD + RTMR0-3
   against a published Chutes measurement, verifies NVIDIA NRAS GPU evidence,
   and pins the verified instance id + key. Any failure refuses the request
   (fail closed). Every listed instance must verify.

TLS certificate verification is always on. The Chutes API and model-listing
endpoints are fixed at `https://api.chutes.ai` and `https://llm.chutes.ai`;
there is no endpoint or host override. There is no CLI flag, environment
variable, or Nix option that disables any part of the chain.

Attestation covers the VM firmware, boot chain, rootfs and GPU hardware. It does
**not** bind a model id, revision, or weight digest; see
[Model weights are not attested](docs/attestation.md#14-model-weights-are-not-attested).

Deep dives:

- [End-to-end encryption](docs/e2ee-encryption.md) — primitives, wire format,
  key establishment, streaming, nonces, and what encryption does not hide.
- [TEE and GPU attestation](docs/attestation.md) — evidence envelope, binding
  checks, DCAP, NRAS, measurement policy, caching, and fail-closed rules.

## Documentation

| Document                                         | Start here when you want to…                              |
| ------------------------------------------------ | --------------------------------------------------------- |
| [docs/setup.md](docs/setup.md)                   | install, run, configure, deploy, or troubleshoot          |
| [docs/how-it-works.md](docs/how-it-works.md)     | understand the startup sequence and request path          |
| [docs/e2ee-encryption.md](docs/e2ee-encryption.md) | understand the cryptography on the wire                 |
| [docs/attestation.md](docs/attestation.md)       | understand the TDX/GPU trust chain                        |
| [docs/README.md](docs/README.md)                  | choose the right document                                 |
| [TEST.md](TEST.md)                               | run the offline and live test suites                      |

## Testing

```sh
uv run pytest tests/ -q
uv run ruff check src tests scripts
```

The offline suite uses a protocol-faithful mock and needs no network or
credentials. Live tests are opt-in; see [TEST.md](TEST.md).

## License

MIT.
