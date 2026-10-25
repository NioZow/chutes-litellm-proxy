# Documentation

## Who this is for

Anyone who needs to install the proxy, understand how a Chutes `*-TEE` request
is protected, or maintain the attestation and E2EE code. Start with the question
you have, not with the file order.

## What you'll learn

- which document answers setup, architecture, encryption, or attestation
  questions,
- the properties that protect Chutes traffic and where each is explained,
- where the main code lives.

## Choose a document

| Question | Document |
| -------- | -------- |
| How do I install, run, configure, deploy, or troubleshoot it? | [setup.md](setup.md) |
| How does startup work, how are models routed, and what happens on a `*-TEE` request? | [how-it-works.md](how-it-works.md) |
| How is the prompt encrypted, what is the wire format, and what does encryption not hide? | [e2ee-encryption.md](e2ee-encryption.md) |
| How does the proxy decide that an instance key belongs to genuine TDX/GPU hardware? | [attestation.md](attestation.md) |
| How do I run the offline or live test suites? | [TEST.md](../TEST.md) |
| What is this project in one page? | [README.md](../README.md) |

If you are new, read [setup.md](setup.md) first. If you are changing the
request path or debugging a verification failure, read
[how-it-works.md](how-it-works.md), then the relevant deep dive.

## The 60-second version

1. **Encryption.** The client generates a throwaway ML-KEM-768 key pair. It
   encapsulates a shared secret to the Chutes **instance** public key, derives a
   symmetric key with HKDF-SHA256, gzips the OpenAI JSON, and encrypts it with
   ChaCha20-Poly1305. The instance private key lives inside an Intel TDX
   confidential VM. The reply comes back encrypted to the client's throwaway
   key. The Chutes control plane, the network, and any relay see ciphertext.
2. **Attestation.** Encryption alone does not prove whose key you used. Before
   trusting an instance key, the proxy verifies a signed evidence envelope, runs
   Intel DCAP on the TDX quote, binds `report_data` to the instance key and
   certificate, matches MRTD + RTMR0-3 against a published Chutes measurement,
   and verifies NVIDIA NRAS GPU evidence. All checks are mandatory; any failure
   refuses the request (fail closed).
3. **Scope.** Attestation covers the VM image and GPU hardware, not model
   weights. See
   [Model weights are not attested](attestation.md#14-model-weights-are-not-attested).

The Chutes API and model-listing endpoints are fixed at
`https://api.chutes.ai` and `https://llm.chutes.ai`, and TLS verification is
always on. There is no endpoint, host, or verification override.

## Code map

| Concern | Where |
| ------- | ----- |
| ML-KEM/HKDF/ChaCha helpers, blob format | `chutes-e2ee-transport/src/chutes_e2ee/crypto.py` |
| httpx transport, headers, streaming, reply decryption | `chutes-e2ee-transport/src/chutes_e2ee/transport.py` |
| Instance discovery, nonce pool, authenticated-instance filter | `chutes-e2ee-transport/src/chutes_e2ee/discovery.py` |
| Evidence parsing, binding checks, gate/cache | `src/chutes_litellm/attestation.py` |
| DCAP, measurements, NRAS verification | `src/chutes_litellm/tee/` |
| Startup and CLI | `src/chutes_litellm/cli.py` |
| Config generation | `src/chutes_litellm/generate_config.py` |
| Native LiteLLM client swap | `src/chutes_litellm/e2ee_litellm.py` |
| Self-contained `chutes_e2ee/` provider | `src/chutes_litellm/custom_provider.py` |
| Attestation CLI | `src/chutes_litellm/verify.py`, `scripts/verify_attestation.py` |
| Protocol-faithful offline mock + tests | `tests/mock_chutes_server.py`, `tests/test_e2ee_litellm.py`, `tests/test_attestation.py` |

> The `chutes-e2ee` transport is vendored from the fork
> [`niozow/chutes-e2ee-transport`](https://github.com/niozow/chutes-e2ee-transport).
> The proxy pins a specific commit in both `uv.lock` and `flake.nix`; the two
> must stay identical, and `tests/test_packaging.py` enforces that. The fork
> adds the authenticated-instance hook described in
> [attestation.md §9](attestation.md#9-binding-attested-instances-to-selection).
