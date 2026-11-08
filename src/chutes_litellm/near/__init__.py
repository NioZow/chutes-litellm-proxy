"""NEAR AI Cloud provider support for this LiteLLM proxy.

The package is split so later phases can import the policy foundation without
pulling in HTTP clients or the LiteLLM provider:

``endpoints``  fixed Cloud API base URL and path helpers (no override surface).
``policy``     strict verification options, post-verification assertions,
               timeouts/TTLs, and the negative failure cache.
``modes``      metadata -> ``tee``/``routed`` classification and the frozen,
               process-local mode registry driven by ``NEARAI_TEE_MODELS``.

The upstream ``nearai-inference-sdk`` is used as-is; this package only fixes its
permissive defaults and enforces the fail-closed policy described in
``PLAN.md``.
"""

from __future__ import annotations

__all__ = ["endpoints", "modes", "policy"]
