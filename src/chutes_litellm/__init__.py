"""Chutes TEE/E2EE integration for LiteLLM.

``e2ee_litellm``   host-scoped swap of LiteLLM's OpenAI-SDK http clients so
                   native ``chutes`` provider traffic is end-to-end encrypted.
``attestation``    optional fail-closed TDX/GPU attestation gate + verifier.
"""

from chutes_litellm import attestation, e2ee_litellm

__all__ = ["attestation", "e2ee_litellm"]
