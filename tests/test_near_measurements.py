"""Optional NEAR AI deployment measurement pinning (offline, Phase 6).

The tests use real SDK dataclasses and a local YAML/JSON file written to
``tmp_path``.  Runtime tests inject fakes at ``chutes_litellm.near.client`` so no
network, TLS, DCAP, NRAS, or API key is involved.

Pins never expose hash values in errors: several tests assert that the actual
and expected measurements cannot be recovered from ``str(error)``, ``repr`` or
``to_dict()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from nearai_inference_sdk import (
    AttestationVerificationResult,
    GatewayTlsBinding,
    MeasuredDeployment,
    ModelMetadata,
    RuntimeMeasurements,
    SigningIdentity,
    VerificationError,
    VerificationFailure,
    VerifiedGatewayAttestation,
    VerifiedModelAttestation,
)

from chutes_litellm.near import client, policy

# Distinct, even-length hex values (32 bytes) so a leak is easy to spot.
HEX_A = "aa" * 32
HEX_B = "bb" * 32
HEX_C = "cc" * 32
HEX_D = "dd" * 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, content: str, *, suffix: str = ".yaml") -> Path:
    path = tmp_path / f"pins{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


def _deployment(
    *,
    app_compose: str = "app-compose",
    os_image_hash: str | None = None,
    compose_hash: str | None = None,
) -> MeasuredDeployment:
    return MeasuredDeployment(
        app_compose=app_compose,
        runtime_measurements=RuntimeMeasurements(
            os_image_hash=os_image_hash,
            compose_hash=compose_hash,
        ),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pins(
    *,
    gateway: policy.MeasurementPinSet | None = None,
    models: dict[str, policy.MeasurementPinSet] | None = None,
) -> policy.MeasurementPins:
    return policy.MeasurementPins(
        gateway=gateway or policy.MeasurementPinSet(),
        models=models or {},
    )


#: Pin file whose Gateway and exact-model pins match :func:`_deployment`.
VALID_PINS = f"""\
gateway:
  os_image_hash: ["0x{HEX_A.upper()}"]
models:
  "exact/model":
    compose_hash: "{HEX_B}"
  "*":
    os_image_hash: ["{HEX_C}"]
"""


# ---------------------------------------------------------------------------
# Pin loading and normalization
# ---------------------------------------------------------------------------


def test_load_yaml_accepts_string_and_list_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        f"""\
gateway:
  os_image_hash: "0x{HEX_A.upper()}"
  compose_hash: ["{HEX_B.upper()}", "{HEX_C}"]
models:
  "exact/model":
    app_compose_sha256: "{HEX_D.upper()}"
  "*":
    os_image_hash: ["{HEX_A}"]
""",
    )

    pins = policy.load_measurement_pins(path)

    assert pins.gateway.os_image_hashes == frozenset({HEX_A})
    assert pins.gateway.compose_hashes == frozenset({HEX_B, HEX_C})
    assert pins.models["exact/model"].app_compose_sha256 == frozenset({HEX_D})
    assert pins.models["*"].os_image_hashes == frozenset({HEX_A})


def test_load_json_is_supported(tmp_path: Path) -> None:
    path = _write(tmp_path, json.dumps({"gateway": {"compose_hash": [HEX_A]}}), suffix=".json")
    pins = policy.load_measurement_pins(path)
    assert pins.gateway.compose_hashes == frozenset({HEX_A})


def test_empty_mapping_is_valid_and_unconstrained(tmp_path: Path) -> None:
    pins = policy.load_measurement_pins(_write(tmp_path, "{}"))
    assert pins.is_empty
    assert pins.models == {}
    assert pins.gateway.is_empty


def test_missing_sections_mean_no_constraints(tmp_path: Path) -> None:
    pins = policy.load_measurement_pins(_write(tmp_path, "models: {}\n"))
    assert pins.is_empty
    assert pins.wildcard is None


def test_exact_model_id_wins_over_wildcard(tmp_path: Path) -> None:
    pins = policy.load_measurement_pins(_write(tmp_path, VALID_PINS))

    exact = pins.pins_for_model("exact/model")
    fallback = pins.pins_for_model("other/model")

    assert exact.compose_hashes == frozenset({HEX_B})
    assert fallback.os_image_hashes == frozenset({HEX_C})
    assert pins.pins_for_model("third/model") == fallback

    no_wildcard = policy.load_measurement_pins(
        _write(tmp_path, f'models:\n  "exact/model":\n    os_image_hash: ["{HEX_A}"]\n', suffix="-2.yaml")
    )
    assert no_wildcard.pins_for_model("other/model").is_empty


def test_model_ids_are_stripped_and_mapping_is_read_only(tmp_path: Path) -> None:
    pins = policy.load_measurement_pins(_write(tmp_path, f'models:\n  "  m  ":\n    os_image_hash: ["{HEX_A}"]\n'))
    assert pins.pins_for_model("m").os_image_hashes == frozenset({HEX_A})
    with pytest.raises(TypeError):
        pins.models["m"] = policy.MeasurementPinSet()  # type: ignore[index]


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "'string'",
        "gateway: []",
        "gateway: null",
        "models: []",
        "models:\n  m: []",
        "gateway:\n  unknown: ['aa']",
        "unknown_top: {}",
        "gateway:\n  os_image_hash: 12",
        "gateway:\n  os_image_hash: ['aa', 12]",
        "gateway:\n  os_image_hash: ['']",
        "gateway:\n  os_image_hash: ['0x']",
        "gateway:\n  os_image_hash: ['abc']",
        "gateway:\n  os_image_hash: ['nothex']",
        "gateway:\n  os_image_hash: [null]",
        "models:\n  '':\n    os_image_hash: ['aa']",
        "models:\n  12:\n    os_image_hash: ['aa']",
        "models:\n  m:\n    os_image_hash: ['aa']\n  ' m ':\n    os_image_hash: ['bb']",
    ],
)
def test_invalid_pin_documents_fail_closed(tmp_path: Path, document: str) -> None:
    with pytest.raises(policy.NearPolicyError) as excinfo:
        policy.load_measurement_pins(_write(tmp_path, document))
    assert excinfo.value.code == policy.CODE_MEASUREMENTS_INVALID


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(policy.NearPolicyError) as excinfo:
        policy.load_measurement_pins(tmp_path / "missing.yaml")
    assert excinfo.value.code == policy.CODE_MEASUREMENTS_INVALID


def test_unreadable_path_fails_closed(tmp_path: Path) -> None:
    # A directory is reliably unreadable as a file on every platform.
    with pytest.raises(policy.NearPolicyError) as excinfo:
        policy.load_measurement_pins(tmp_path)
    assert excinfo.value.code == policy.CODE_MEASUREMENTS_INVALID


def test_invalid_pin_error_does_not_leak_values(tmp_path: Path) -> None:
    secret_value = "deadbeefzz"
    with pytest.raises(policy.NearPolicyError) as excinfo:
        policy.load_measurement_pins(_write(tmp_path, f"gateway:\n  os_image_hash: ['{secret_value}']"))

    error = excinfo.value
    rendered = f"{error} {error.describe()} {error!r}"
    assert error.code == policy.CODE_MEASUREMENTS_INVALID
    assert secret_value not in rendered


# ---------------------------------------------------------------------------
# Deployment verifier
# ---------------------------------------------------------------------------


def test_gateway_verifier_accepts_all_matching_measurements() -> None:
    pins = _pins(
        gateway=policy.MeasurementPinSet(
            os_image_hashes=frozenset({HEX_A}),
            compose_hashes=frozenset({HEX_B}),
            app_compose_sha256=frozenset({_sha256("app-compose")}),
        )
    )

    policy.measurement_verifier(pins, scope="gateway")(
        _deployment(os_image_hash=HEX_A, compose_hash=HEX_B)
    )


@pytest.mark.parametrize(
    ("field", "pin_kwargs", "deployment_kwargs"),
    [
        ("os_image_hash", {"os_image_hashes": frozenset({HEX_A})}, {"os_image_hash": HEX_B}),
        ("compose_hash", {"compose_hashes": frozenset({HEX_A})}, {"compose_hash": HEX_B}),
        (
            "app_compose_sha256",
            {"app_compose_sha256": frozenset({HEX_A})},
            {"app_compose": "different-compose"},
        ),
    ],
)
def test_gateway_verifier_rejects_each_mismatch(
    field: str,
    pin_kwargs: dict[str, frozenset[str]],
    deployment_kwargs: dict[str, str],
) -> None:
    pins = _pins(gateway=policy.MeasurementPinSet(**pin_kwargs))

    with pytest.raises(VerificationError) as excinfo:
        policy.measurement_verifier(pins, scope="gateway")(_deployment(**deployment_kwargs))

    failure = excinfo.value.failure
    assert failure.code == policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH
    assert failure.details == {"field": field, "scope": "gateway"}


def test_gateway_verifier_rejects_missing_deployment_value() -> None:
    pins = _pins(gateway=policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A})))

    with pytest.raises(VerificationError) as excinfo:
        policy.measurement_verifier(pins, scope="gateway")(_deployment())

    assert excinfo.value.failure.code == policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH
    assert excinfo.value.failure.details == {"field": "os_image_hash", "scope": "gateway"}


def test_empty_gateway_pins_are_a_noop() -> None:
    policy.measurement_verifier(_pins(), scope="gateway")(_deployment())


def test_model_verifier_uses_a_distinct_code() -> None:
    pins = _pins(models={"m": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A}))})

    with pytest.raises(VerificationError) as excinfo:
        policy.measurement_verifier(pins, scope="model", model="m")(_deployment())

    assert excinfo.value.failure.code == policy.CODE_MEASUREMENTS_MODEL_MISMATCH
    assert excinfo.value.failure.details == {"field": "os_image_hash", "scope": "model"}


def test_model_verifier_exact_and_wildcard_selection() -> None:
    pins = _pins(
        models={
            "exact/model": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A})),
            "*": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_B})),
        }
    )

    policy.measurement_verifier(pins, scope="model", model="exact/model")(
        _deployment(os_image_hash=HEX_A)
    )
    policy.measurement_verifier(pins, scope="model", model="other/model")(
        _deployment(os_image_hash=HEX_B)
    )

    with pytest.raises(VerificationError):
        policy.measurement_verifier(pins, scope="model", model="exact/model")(
            _deployment(os_image_hash=HEX_B)
        )
    with pytest.raises(VerificationError):
        policy.measurement_verifier(pins, scope="model", model="other/model")(
            _deployment(os_image_hash=HEX_A)
        )


def test_model_verifier_without_pins_is_a_noop() -> None:
    policy.measurement_verifier(_pins(), scope="model", model="unpinned")(_deployment())


def test_measurement_mismatch_does_not_leak_hashes() -> None:
    pins = _pins(gateway=policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A})))

    with pytest.raises(VerificationError) as excinfo:
        policy.measurement_verifier(pins, scope="gateway")(_deployment(os_image_hash=HEX_B))

    error = excinfo.value
    rendered = f"{error} {error!r} {error.to_dict()}"
    assert HEX_A not in rendered
    assert HEX_B not in rendered


def test_runtime_hash_comparison_normalizes_case_and_prefix() -> None:
    pins = _pins(gateway=policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A})))

    policy.measurement_verifier(pins, scope="gateway")(
        _deployment(os_image_hash=f"0x{HEX_A.upper()}")
    )


# ---------------------------------------------------------------------------
# Option factories
# ---------------------------------------------------------------------------


def test_no_pin_factories_return_the_strict_constants() -> None:
    assert policy.gateway_verification_options() is policy.GATEWAY_VERIFICATION_OPTIONS
    assert policy.tee_model_verification_options() is policy.TEE_MODEL_VERIFICATION_OPTIONS


def test_pinned_gateway_options_preserve_strict_policy_and_attach_verifier() -> None:
    pins = _pins(gateway=policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A})))
    options = policy.gateway_verification_options(pins)

    assert options.include_spki_fingerprint is True
    assert options.policy is policy.GATEWAY_VERIFICATION_OPTIONS.policy
    assert options.policy is not None
    assert options.policy.accepted_tcb_statuses == ("UpToDate",)
    assert options.verifiers is not None
    assert options.verifiers.deployment is not None


def test_pinned_model_options_preserve_strict_policy_and_attach_verifier() -> None:
    pins = _pins(models={"*": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A}))})
    options = policy.tee_model_verification_options(pins)

    assert options.policy is policy.TEE_MODEL_VERIFICATION_OPTIONS.policy
    assert options.policy is not None
    assert options.policy.accepted_tcb_statuses == ("UpToDate",)
    assert options.policy.gpu_evidence == "required"
    assert options.verifiers is not None
    assert options.verifiers.deployment is not None


def test_wildcard_only_factory_verifier_rejects_a_mismatch() -> None:
    pins = _pins(models={"*": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A}))})
    verifier = policy.tee_model_verification_options(pins).verifiers
    assert verifier is not None and verifier.deployment is not None

    with pytest.raises(VerificationError):
        verifier.deployment(_deployment(os_image_hash=HEX_B))


def test_exact_only_factory_verifier_defers_to_per_model_checks() -> None:
    pins = _pins(models={"m": policy.MeasurementPinSet(os_image_hashes=frozenset({HEX_A}))})
    verifier = policy.tee_model_verification_options(pins).verifiers
    assert verifier is not None and verifier.deployment is not None

    # The shared option object has no model identity, so this is a no-op; the
    # runtime applies the exact pin after verification.
    verifier.deployment(_deployment(os_image_hash=HEX_B))


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------


class _FakeInferenceClient:
    def __init__(self, api_key: str | None, kwargs: dict[str, Any]) -> None:
        self.api_key = api_key
        self.kwargs = kwargs
        self.verify_calls: list[str] = []
        self.handler: Any = None
        self.close_calls = 0

    async def verify(self, model: str) -> AttestationVerificationResult:
        self.verify_calls.append(model)
        if self.handler is None:
            raise AssertionError(f"unexpected verify({model!r})")
        return await self.handler(model)

    async def aclose(self) -> None:
        self.close_calls += 1


class _FetchedGateway:
    def __init__(self) -> None:
        self.attestation = object()
        self.client_binding = object()


class _FakeAttestationClient:
    def __init__(self, harness: Harness, api_key: str | None, kwargs: dict[str, Any]) -> None:
        self.harness = harness
        self.api_key = api_key
        self.kwargs = kwargs

    async def fetch_gateway_attestation(self, **kwargs: Any) -> _FetchedGateway:
        return _FetchedGateway()

    async def fetch_model_metadata(self, model: str) -> ModelMetadata:
        return self.harness.metadata


class _FakePinnedClient:
    async def aclose(self) -> None:
        return None


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.created_inference: list[_FakeInferenceClient] = []
        self.gateway_calls: list[tuple[object, object]] = []
        self.warnings: list[str] = []
        self.metadata = ModelMetadata(provider_type="vllm", attestation_supported=True)
        self.runtimes: list[client.NearRuntime] = []

        def inference_factory(api_key: str | None = None, **kwargs: Any) -> _FakeInferenceClient:
            fake = _FakeInferenceClient(api_key, kwargs)
            self.created_inference.append(fake)
            return fake

        def attestation_factory(api_key: str | None = None, **kwargs: Any) -> _FakeAttestationClient:
            return _FakeAttestationClient(self, api_key, kwargs)

        def pinned_factory(_fingerprint: str) -> _FakePinnedClient:
            return _FakePinnedClient()

        async def verify_gateway(
            _attestation: object,
            _client_binding: object,
            *,
            policy: object = None,
            verifiers: object = None,
        ) -> VerifiedGatewayAttestation:
            self.gateway_calls.append((policy, verifiers))
            return _safe_gateway()

        monkeypatch.setattr(client, "InferenceClient", inference_factory)
        monkeypatch.setattr(client, "AttestationClient", attestation_factory)
        monkeypatch.setattr(client, "create_pinned_tls_client", pinned_factory)
        monkeypatch.setattr(client, "verify_gateway_attestation", verify_gateway)
        monkeypatch.setattr(client, "_log_construction_warning", self.warnings.append)

    def make_runtime(self, api_key: str = "test-key", **kwargs: Any) -> client.NearRuntime:
        kwargs.setdefault("environ", {})
        runtime = client.NearRuntime(api_key, **kwargs)
        self.runtimes.append(runtime)
        return runtime


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(monkeypatch)


@pytest.fixture(autouse=True)
def _clean_process_runtimes() -> Any:
    client.reset_runtimes()
    yield
    client.reset_runtimes()


def _signer() -> SigningIdentity:
    return SigningIdentity(signing_algo="ed25519", signing_address="0xsigner")


def _safe_gateway() -> VerifiedGatewayAttestation:
    return VerifiedGatewayAttestation(
        signer=_signer(),
        tcb_status="UpToDate",
        advisory_ids=(),
        deployment=_deployment(),
        deployment_provenance="not_checked",
        tls_binding=GatewayTlsBinding(kind="attested", spki_fingerprint="ab" * 32),
    )


def _reports(*, deployment: MeasuredDeployment | None = None) -> AttestationVerificationResult:
    report = VerifiedModelAttestation(
        signer=_signer(),
        tcb_status="UpToDate",
        advisory_ids=(),
        deployment=deployment or _deployment(),
        deployment_provenance="not_checked",
        gpu_evidence="verified",
        signing_public_key="cd" * 32,
    )
    return AttestationVerificationResult(
        gateway=_safe_gateway(),
        models=(report,),
        verified_at=1_700_000_000_000,
    )


def _tee_handler(result: AttestationVerificationResult) -> Any:
    async def handler(_model: str) -> AttestationVerificationResult:
        return result

    return handler


def test_unset_measurements_file_uses_strict_constants_and_warns_once(harness: Harness) -> None:
    runtime = harness.make_runtime()

    assert runtime.measurement_pins is None
    assert runtime.gateway_options is policy.GATEWAY_VERIFICATION_OPTIONS
    assert runtime.model_options is policy.TEE_MODEL_VERIFICATION_OPTIONS
    assert runtime.tee_client.kwargs["gateway_verification"] is policy.GATEWAY_VERIFICATION_OPTIONS
    assert runtime.tee_client.kwargs["model_verification"] is policy.TEE_MODEL_VERIFICATION_OPTIONS
    assert runtime.routed_client.kwargs["gateway_verification"] is policy.GATEWAY_VERIFICATION_OPTIONS
    assert runtime.routed_client.kwargs["model_verification"] is None

    assert len(harness.warnings) == 1
    assert client.MEASUREMENTS_FILE_ENV in harness.warnings[0]
    assert "not pinned" in harness.warnings[0]


def test_valid_pin_file_is_loaded_for_both_clients(tmp_path: Path, harness: Harness) -> None:
    path = _write(
        tmp_path,
        f"""\
gateway:
  os_image_hash: ["{HEX_A}"]
models:
  "*":
    os_image_hash: ["{HEX_C}"]
""",
    )
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})

    assert runtime.measurement_pins is not None
    assert runtime.gateway_options is not policy.GATEWAY_VERIFICATION_OPTIONS
    assert runtime.model_options is not policy.TEE_MODEL_VERIFICATION_OPTIONS
    assert runtime.gateway_options.verifiers is not None
    assert runtime.gateway_options.verifiers.deployment is not None
    assert runtime.model_options.verifiers is not None
    assert runtime.model_options.verifiers.deployment is not None

    assert runtime.tee_client.kwargs["gateway_verification"] is runtime.gateway_options
    assert runtime.tee_client.kwargs["model_verification"] is runtime.model_options
    assert runtime.routed_client.kwargs["gateway_verification"] is runtime.gateway_options
    assert runtime.routed_client.kwargs["model_verification"] is None
    assert harness.warnings == []


def test_empty_pin_file_warns_but_is_valid(tmp_path: Path, harness: Harness) -> None:
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(_write(tmp_path, "{}"))})

    assert runtime.measurement_pins is not None
    assert runtime.measurement_pins.is_empty
    assert len(harness.warnings) == 1
    assert "defines no pins" in harness.warnings[0]


def test_invalid_pin_file_fails_closed_at_construction(tmp_path: Path, harness: Harness) -> None:
    path = _write(tmp_path, "gateway:\n  unknown: ['aa']\n")

    with pytest.raises(policy.NearPolicyError) as excinfo:
        harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})

    assert excinfo.value.code == policy.CODE_MEASUREMENTS_INVALID
    assert harness.created_inference == []


def test_bootstrap_uses_the_pinned_gateway_options(tmp_path: Path, harness: Harness) -> None:
    path = _write(tmp_path, f"gateway:\n  os_image_hash: ['{HEX_A}']\n")
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})

    async def scenario() -> None:
        await runtime.observe_mode("m")
        assert len(harness.gateway_calls) == 1
        policy_arg, verifiers_arg = harness.gateway_calls[0]
        assert policy_arg is runtime.gateway_options.policy
        assert verifiers_arg is runtime.gateway_options.verifiers

    asyncio.run(scenario())


def test_runtime_applies_exact_model_pins_after_verification(tmp_path: Path, harness: Harness) -> None:
    path = _write(tmp_path, f'models:\n  "m":\n    os_image_hash: ["{HEX_A}"]\n')
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})
    runtime.tee_client.handler = _tee_handler(
        _reports(deployment=_deployment(os_image_hash=HEX_A))
    )

    async def scenario() -> None:
        assert await runtime.verify_model("m") == "tee"

    asyncio.run(scenario())


def test_runtime_rejects_an_exact_model_pin_mismatch(tmp_path: Path, harness: Harness) -> None:
    path = _write(tmp_path, f'models:\n  "m":\n    os_image_hash: ["{HEX_A}"]\n')
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})
    runtime.tee_client.handler = _tee_handler(
        _reports(deployment=_deployment(os_image_hash=HEX_B))
    )

    async def scenario() -> None:
        with pytest.raises(policy.NearPolicyError) as excinfo:
            await runtime.verify_model("m")
        assert excinfo.value.code == policy.CODE_MEASUREMENTS_MODEL_MISMATCH

    asyncio.run(scenario())


def test_unpinned_model_is_not_constrained_by_other_models(tmp_path: Path, harness: Harness) -> None:
    path = _write(tmp_path, f'models:\n  "pinned":\n    os_image_hash: ["{HEX_A}"]\n')
    runtime = harness.make_runtime(environ={client.MEASUREMENTS_FILE_ENV: str(path)})
    runtime.tee_client.handler = _tee_handler(_reports(deployment=_deployment(os_image_hash=HEX_B)))

    async def scenario() -> None:
        assert await runtime.verify_model("unpinned") == "tee"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code",
    [policy.CODE_MEASUREMENTS_GATEWAY_MISMATCH, policy.CODE_MEASUREMENTS_MODEL_MISMATCH],
)
def test_sdk_measurement_mismatch_preserves_the_specific_policy_code(
    harness: Harness, code: str
) -> None:
    runtime = harness.make_runtime()

    async def failing(_model: str) -> AttestationVerificationResult:
        raise VerificationError(
            VerificationFailure(code=code, details={"field": "os_image_hash", "scope": "gateway"})
        )

    runtime.tee_client.handler = failing

    async def scenario() -> None:
        with pytest.raises(policy.NearPolicyError) as excinfo:
            await runtime.verify_model("m")
        assert excinfo.value.code == code

    asyncio.run(scenario())
