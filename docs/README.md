# Security documentation

This directory explains **how the Chutes TEE traffic this proxy sends is
encrypted** and **how the remote hardware is attested**, in enough detail that
you can follow a request from plaintext JSON to ciphertext on the wire to a
verified TDX + NVIDIA verdict without reading the source first.

| Document | Read it when you want to know… |
| -------- | ------------------------------ |
| [e2ee-encryption.md](./e2ee-encryption.md) | How a prompt becomes an ML-KEM/ChaCha20 blob, how the reply is decrypted, how streaming works, and what the encryption does *not* hide. |
| [attestation.md](./attestation.md) | How the proxy decides it is safe to encrypt to an instance public key in the first place: the evidence envelope, the binding checks, Intel DCAP and NVIDIA NRAS verification, fail-closed rules, caching and the CLI. |

## The 60-second version

1. **Encryption.** The client generates a throwaway ML-KEM-768 key pair. It
   encapsulates a shared secret to the Chutes **instance** public key, derives a
   symmetric key with HKDF-SHA256, gzips the OpenAI JSON, and encrypts it with
   ChaCha20-Poly1305. The server can only read it with the instance private key,
   which lives inside an **Intel TDX** confidential VM. The reply comes back
   encrypted to the client's throwaway key. The Chutes control plane, network,
   and any relay only ever see ciphertext.
2. **Attestation.** Encryption alone proves nothing about *whose* key you
   encrypted to. Before trusting an instance key, the proxy fetches hardware
   evidence and checks that the key is bound into a **TDX quote** and into
   **NVIDIA GPU** evidence, optionally verifying those roots with Intel DCAP and
   NVIDIA NRAS. If anything does not line up, the request is refused (fail
   closed).

## Code map

| Concern | Where |
| ------- | ----- |
| ML-KEM/HKDF/ChaCha helpers, blob format | `chutes-e2ee-transport/src/chutes_e2ee/crypto.py` |
| httpx transport, headers, streaming, reply decryption | `chutes-e2ee-transport/src/chutes_e2ee/transport.py` |
| Instance discovery, nonce pool, authenticated-instance filter | `chutes-e2ee-transport/src/chutes_e2ee/discovery.py` |
| Evidence parsing + binding checks + gate/cache | `src/chutes_litellm/attestation.py` |
| Native LiteLLM client swap | `src/chutes_litellm/e2ee_litellm.py` |
| Self-contained `chutes_e2ee/` provider | `src/chutes_litellm/custom_provider.py` |
| Attestation CLI | `src/chutes_litellm/verify.py`, `scripts/verify_attestation.py` |
| Protocol-faithful offline mock + tests | `tests/mock_chutes_server.py`, `tests/test_e2ee_litellm.py`, `tests/test_attestation.py` |

> The two `chutes-e2ee` files above are vendored from the fork
> [`niozow/chutes-e2ee-transport`](https://github.com/niozow/chutes-e2ee-transport),
> cloned locally at `./chutes-e2ee-transport` (gitignored). The proxy pins a
> specific commit; the fork adds the authenticated-instance hook described in
> [attestation.md §8](./attestation.md#8-binding-attested-instances-to-selection).
