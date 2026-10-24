"""Fixed Chutes service endpoints.

These are module-level constants on purpose: the Chutes API and model-listing
hosts are not configurable.  Production modules import this module and read
``endpoints.API_BASE`` / ``endpoints.MODELS_BASE`` at call time (never binding
them into local constants or default arguments at import time) so a test can
monkeypatch the module attributes.
"""

from __future__ import annotations

API_BASE = "https://api.chutes.ai"
MODELS_BASE = "https://llm.chutes.ai"

__all__ = ["API_BASE", "MODELS_BASE"]
