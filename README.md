# Chutes LiteLLM Proxy

An OpenAI-compatible LiteLLM proxy for Chutes, NEAR AI Cloud, and the major
model providers. It exposes one `/v1` API, expands each provider's model catalog
at startup, and forwards all other traffic normally. Two providers are
different:

- **Chutes `*-TEE`** requests are end-to-end encrypted with ML-KEM-768 +
  ChaCha20-Poly1305 and are sent only to instances whose Intel TDX and NVIDIA GPU
  attestation verifies.
- **NEAR AI `<model>`** requests go through Gateway and (for true TEE models)
  model attestation, TLS SPKI pinning and response-signature verification, with
  E2EE v3 for TEE models.

TEE attestation, Intel DCAP quote verification, NVIDIA NRAS GPU verification,
measurement matching, E2EE, TLS verification and response receipts are mandatory
and cannot be disabled.

## Quickstart

### uv

```sh
uv sync
export CHUTES_API_KEY=cpk_...
export NEARAI_API_KEY=near-...          # optional
uv run chutes-litellm-proxy --host 127.0.0.1 --port 4000
```

```sh
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chutes/Qwen/Qwen3-32B-TEE","messages":[{"role":"user","content":"hi"}]}'

# NEAR AI uses the same API shape; `nearai/` selects the attested provider.
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"nearai/Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"hi"}]}'
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
| NEAR AI Cloud     | `NEARAI_API_KEY`     | `nearai/`     |

Chutes model discovery keeps only models whose id ends in `-TEE`. NEAR AI
discovery keeps chat models and serves them through the `nearai_e2ee` provider;
TEE and Gateway-routed models are both supported.

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
| `NEARAI_STREAM_MODE` | `verified`                 | `verified` (buffer + verify) or `tail` (terminal verify)   |
| `NEARAI_PREWARM_MODELS` | unset                   | Comma-separated bare NEAR model ids to attest at startup   |
| `NEARAI_TEE_MODELS` | unset                        | Comma-separated ids/globs that must verify as true TEE     |
| `NEARAI_MEASUREMENTS_FILE` | unset                 | Optional YAML/JSON pin file for Gateway + model builds     |

An explicit `*_API_KEY` wins over its `*_API_KEY_PATH`. For NEAR AI,
`NEARAI_API_KEY` is canonical and wins over a request-level key so callers
cannot create additional runtimes. A leading `nearai/` or `nearai_e2ee/` is
stripped from prewarm/forced-TEE model lists and from measurement-pin model
keys; bare upstream ids are canonical. The generated config is written on every
start; use a custom `--config` template for extra models or to set
`general_settings.master_key`.

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

### NEAR AI

NEAR AI traffic is verified in two layers:

1. **Gateway attestation and TLS pinning.** The proxy verifies the Gateway's
   Intel DCAP TDX quote, binds the nonce/report data/signer and deployment
   `mr_config_id`/`app_compose`, replays dstack RTMR3, accepts only `UpToDate`
   with no advisories and an `ed25519` Gateway signer, and pins the attested TLS
   SPKI before sending the key for inference and mode-detection metadata.
   `/v1/models` discovery is TLS-only and is not a trust input for inference;
   the mode is re-verified at runtime.
2. **Model and response policy.** True TEE models (`providerType == "vllm"` with
   attestation support) additionally require model TDX attestation, NVIDIA NRAS
   GPU evidence, a signing key, E2EE v3 field-level encryption, and a
   `provider_tee` receipt. Gateway-routed models do not use E2EE and must return
   a `gateway` receipt. A mode change, a wrong receipt kind, or missing
   evidence fails closed.

The default streaming mode is `verified`: the proxy buffers and verifies the
whole stream before releasing content. `tail` keeps model TTFT but withholds the
terminal chunk until verification succeeds. Verification TTLs and timeouts are
fixed in code (900 s attestation, 30 s failure, 600 s response, 900 s hard
timeout) and are not configurable.

Deployment provenance is unpinned by default and logs a warning when the NEAR
runtime is constructed (startup prewarm or the first NEAR request). Set
`NEARAI_MEASUREMENTS_FILE` to a YAML/JSON file pinning the Gateway and TEE model
`os_image_hash`, `compose_hash`, and `app_compose_sha256`; an exact model id
wins over `"*"`. A mismatch returns `502`, and an invalid file fails closed when
the runtime is constructed (on the first NEAR request, or during startup prewarm
where it is logged), so NEAR requests fail rather than running unpinned. See
[docs/near-ai.md](docs/near-ai.md#measurement-pinning) for the
schema and the command that prints the current measurements. Image provenance
(Sigstore), OHTTP, and upstream SDK hardening PRs are intentionally not
implemented.

TLS certificate verification is always on. The Chutes API and model-listing
endpoints are fixed at `https://api.chutes.ai` and `https://llm.chutes.ai`; the
NEAR AI base URL is fixed at `https://cloud-api.near.ai/v1`. There is no
endpoint or host override. There is no CLI flag, environment variable, or Nix
option that disables any part of either chain.

Attestation covers the VM firmware, boot chain, rootfs and GPU hardware. It does
**not** bind a model id, revision, or weight digest; see
[Model weights are not attested](docs/attestation.md#14-model-weights-are-not-attested).

Deep dives:

- [End-to-end encryption](docs/e2ee-encryption.md) — primitives, wire format,
  key establishment, streaming, nonces, and what encryption does not hide.
- [TEE and GPU attestation](docs/attestation.md) — evidence envelope, binding
  checks, DCAP, NRAS, measurement policy, caching, and fail-closed rules.
- [NEAR AI Cloud](docs/near-ai.md) — Gateway/model verification, TEE vs routed
  mode policy, streaming tradeoffs, configuration, and accepted limitations.

## Documentation

| Document                                         | Start here when you want to…                              |
| ------------------------------------------------ | --------------------------------------------------------- |
| [docs/setup.md](docs/setup.md)                   | install, run, configure, deploy, or troubleshoot          |
| [docs/how-it-works.md](docs/how-it-works.md)     | understand the startup sequence and request path          |
| [docs/e2ee-encryption.md](docs/e2ee-encryption.md) | understand the cryptography on the wire                 |
| [docs/attestation.md](docs/attestation.md)       | understand the TDX/GPU trust chain                        |
| [docs/near-ai.md](docs/near-ai.md)               | understand or operate the NEAR AI provider                |
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
