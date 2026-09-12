import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
os.environ.setdefault("CHUTES_E2EE_VERIFY_SSL", "false")

from chutes_litellm import attestation, e2ee_litellm  # noqa: E402
from mock_chutes_server import MockChutesServer  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _point_env_at_mock():
    # Live runs (CHUTES_LIVE_TEE=1) must keep the user's real credentials and
    # production defaults; the mock pointers are only injected for offline runs.
    if os.environ.get("CHUTES_LIVE_TEE"):
        yield
        return
    os.environ["CHUTES_API_KEY"] = "cpk_test"
    os.environ["CHUTES_E2EE_HOSTS"] = "127.0.0.1"
    yield


@pytest.fixture()
def mock_server():
    server = MockChutesServer()
    os.environ["CHUTES_E2EE_API_BASE"] = server.base_url
    os.environ["CHUTES_E2EE_MODELS_BASE"] = server.base_url
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _reset_litellm_clients():
    # Isolate tests: drop per-key transport clients + the attestation cache so a
    # fresh (env-faithful) transport is built for the next test.
    e2ee_litellm._clients.clear()
    e2ee_litellm._aclients.clear()
    from chutes_litellm import custom_provider

    custom_provider.reset_clients()
    attestation._verify_cache.clear()
    attestation._model_map_cache.clear()
    attestation.close_shared_http()
    yield


def ensure_installed():
    assert e2ee_litellm.install(), "install() returned False (CHUTES_API_KEY / chutes_e2ee missing)"
