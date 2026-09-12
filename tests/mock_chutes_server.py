#!/usr/bin/env python3
"""A protocol-faithful mock of the Chutes E2EE control plane for tests.

Implements the wire protocol that ``chutes-e2ee``'s transports speak
(see https://github.com/niozow/chutes-e2ee-transport):

  * ``GET /v1/models``            -> model listing (id + chute_id)
  * ``GET /e2e/instances/{chute}``-> discovery: instances + nonce pools
  * ``POST /e2e/invoke``          -> an ML-KEM-768 encrypted blob; decrypted here
                                     and answered with an encrypted reply
  * ``GET /instances/{id}/evidence?nonce=`` -> attestation evidence (mocked)

The ML-KEM-768/HKDF/ChaCha20 crypto is real (``pqcrypto`` + ``cryptography``),
so a passing test is proof that a request really was end-to-end encrypted: the
server can only read it because it holds the instance private key, and the
client can only read the reply because it holds the ephemeral response key.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID
from pqcrypto.kem.ml_kem_768 import decrypt as mlkem_decrypt
from pqcrypto.kem.ml_kem_768 import encrypt as mlkem_encrypt
from pqcrypto.kem.ml_kem_768 import generate_keypair as mlkem_generate_keypair

from chutes_e2ee.crypto import (
    INFO_REQ,
    INFO_RESP,
    INFO_STREAM,
    TAG_SIZE,
    chacha_decrypt,
    chacha_encrypt,
    derive_key,
)

MODEL_ID = "Test/Model-TEE"
CHUTE_ID = "11111111-2222-4333-8444-555555555555"
INSTANCE_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

QUOTE_REPORT_DATA_OFFSET = 48 + 584 - 64  # see chutes_litellm.attestation._extract_report_data
QUOTE_LEN = 640


@dataclass
class CapturedInvoke:
    path: str
    headers: dict[str, str]
    blob: bytes
    plaintext: dict[str, Any]
    stream: bool


@dataclass
class MockState:
    instance_pk: bytes
    instance_sk: bytes
    scenario: str = "chat"  # chat | tools | reasoning | rate_limit
    nonce_pool: list[str] = field(default_factory=lambda: [uuid.uuid4().hex for _ in range(64)])
    invokes: list[CapturedInvoke] = field(default_factory=list)
    seen_paths: list[str] = field(default_factory=list)
    evidence_ok: bool = True  # set False to serve unbound/tampered evidence
    model_id: str = MODEL_ID
    chute_id: str = CHUTE_ID
    instance_id: str = INSTANCE_ID

    # attestation signing identity
    _sign_key: rsa.RSAPrivateKey = None  # type: ignore[assignment]
    _sign_cert: x509.Certificate = None  # type: ignore[assignment]

    def ensure_signing_identity(self) -> None:
        if self._sign_key is not None:
            return
        import datetime as _dt

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "chutes-test-attestor")])
        now = _dt.datetime.now(_dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now.replace(year=2099))
            .sign(key, hashes.SHA256())
        )
        self._sign_key = key
        self._sign_cert = cert

    # -- helpers used by tests --------------------------------------------
    def cert_der(self) -> bytes:
        self.ensure_signing_identity()
        return self._sign_cert.public_bytes(serialization.Encoding.DER)

    def sign(self, data: bytes) -> bytes:
        self.ensure_signing_identity()
        return self._sign_key.sign(data, padding.PKCS1v15(), hashes.SHA256())

    @property
    def pubkey_b64(self) -> str:
        return base64.b64encode(self.instance_pk).decode()

    def last_invoke(self) -> CapturedInvoke:
        return self.invokes[-1]


def build_evidence(state: MockState, digest: str, nonce: str) -> dict[str, Any]:
    """Build a plausible evidence blob.  When ``state.evidence_ok`` is False the
    binding digest is replaced so every binding check must fail."""
    state.ensure_signing_identity()
    bind = digest if state.evidence_ok else ("0" * 64)

    quote = bytearray(QUOTE_LEN)
    # report_data holds the raw 32-byte digest (see chutes-api tee-verification.md)
    quote[QUOTE_REPORT_DATA_OFFSET : QUOTE_REPORT_DATA_OFFSET + 64] = bytes.fromhex(bind).ljust(64, b"\x00")

    gpu_evidence = [{"nonce": bind, "gpu_uuid": "00000000-0000-0000-0000-000000000001"}]

    attested_body = json.dumps(
        {
            "instance_id": state.instance_id,
            "digest": bind,
            "quote": base64.b64encode(bytes(quote)).decode(),
            "gpu_evidence": gpu_evidence,
        }
    ).encode()
    cert_der = state.cert_der()
    evidence = {
        "instance_id": state.instance_id,
        "quote": base64.b64encode(bytes(quote)).decode(),
        "gpu_evidence": gpu_evidence,
        "certificate": base64.b64encode(cert_der).decode(),
        "signature": base64.b64encode(state.sign(attested_body)).decode(),
        "attested_body": base64.b64encode(attested_body).decode(),
    }
    return evidence


def _chat_completion_body(model_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": model_payload.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong from mock"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def _tool_completion_body(model_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock-tools",
        "object": "chat.completion",
        "created": 0,
        "model": model_payload.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }


def _reasoning_completion_body(model_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock-reasoning",
        "object": "chat.completion",
        "created": 0,
        "model": model_payload.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": "let me think carefully",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


def _response_body_for(model_payload: dict[str, Any], scenario: str) -> bytes:
    if scenario == "tools":
        body = _tool_completion_body(model_payload)
    elif scenario == "reasoning":
        body = _reasoning_completion_body(model_payload)
    else:
        body = _chat_completion_body(model_payload)
    return json.dumps(body).encode()


def _stream_chunks_for(model_payload: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    """OpenAI chat.completion.chunk events for the given scenario (without [DONE])."""
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": MODEL_ID}
    if scenario == "tools":
        return [
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {"index": 0, "id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Par'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'is"}'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
    if scenario == "reasoning":
        return [
            {
                **base,
                "choices": [{"index": 0, "delta": {"reasoning_content": "step one "}, "finish_reason": None}],
            },
            {
                **base,
                "choices": [{"index": 0, "delta": {"reasoning_content": "step two"}, "finish_reason": None}],
            },
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": "final answer"}, "finish_reason": None}],
            },
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": " from mock"}, "finish_reason": "stop"}]},
    ]


def _stream_usage_for(scenario: str) -> dict[str, Any]:
    if scenario == "reasoning":
        return {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    return {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}


def _encrypt_response(payload: dict[str, Any], body: bytes) -> bytes:
    """Encrypt ``body`` to the client's embedded response public key."""
    response_pk = base64.b64decode(payload["e2e_response_pk"])
    ct, shared = mlkem_encrypt(response_pk)
    key = derive_key(shared, ct, INFO_RESP)
    nonce = os.urandom(12)
    ciphertext, tag = chacha_encrypt(key, nonce, gzip.compress(body))
    return ct + nonce + ciphertext + tag


def _decrypt_request(blob: bytes, state: MockState) -> dict[str, Any]:
    ct = blob[:1088]
    nonce = blob[1088 : 1088 + 12]
    ciphertext = blob[1088 + 12 : -TAG_SIZE]
    tag = blob[-TAG_SIZE:]
    shared = mlkem_decrypt(state.instance_sk, ct)
    key = derive_key(shared, ct, INFO_REQ)
    plaintext = gzip.decompress(chacha_decrypt(key, nonce, ciphertext, tag))
    return json.loads(plaintext)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:  # silence
        return

    @property
    def state(self) -> MockState:
        return self.server.state  # type: ignore[attr-defined,no-any-return]

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        s = self.state
        path = self.path.split("?")[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        s.seen_paths.append(f"GET {path}")
        if path == "/v1/models":
            self._send(
                200,
                json.dumps(
                    {"object": "list", "data": [{"id": s.model_id, "chute_id": s.chute_id}]}
                ).encode(),
            )
        elif path.startswith("/e2e/instances/"):
            body = json.dumps(
                {
                    "instances": [
                        {
                            "instance_id": s.instance_id,
                            "e2e_pubkey": s.pubkey_b64,
                            "nonces": list(s.nonce_pool),
                        }
                    ],
                    "nonce_expires_in": 60,
                }
            ).encode()
            self._send(200, body)
        elif path.startswith("/chutes/") and path.endswith("/evidence"):
            params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
            nonce = params.get("nonce", "")
            from chutes_litellm.attestation import _digest_of

            digest = _digest_of(nonce, s.pubkey_b64)
            body = json.dumps(
                {
                    "evidence": [build_evidence(s, digest, nonce)],
                    "failed_instance_ids": [],
                }
            ).encode()
            self._send(200, body)
        elif path.startswith("/instances/") and path.endswith("/evidence"):
            params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
            nonce = params.get("nonce", "")
            from chutes_litellm.attestation import _digest_of

            digest = _digest_of(nonce, s.pubkey_b64)
            self._send(200, json.dumps(build_evidence(s, digest, nonce)).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        s = self.state
        s.seen_paths.append(f"POST {self.path}")
        if self.path != "/e2e/invoke":
            self._send(404, b'{"error":"not found"}')
            return
        blob = self._read_body()
        payload = _decrypt_request(blob, s)
        is_stream = bool(payload.get("stream"))
        s.invokes.append(
            CapturedInvoke(
                path=self.path,
                headers={k: v for k, v in self.headers.items()},
                blob=blob,
                plaintext=payload,
                stream=is_stream,
            )
        )
        if s.scenario == "rate_limit":
            self._send(
                429,
                json.dumps({"error": {"message": "mock rate limit", "type": "rate_limit_error"}}).encode(),
            )
            return
        if not is_stream:
            self._send(200, _encrypt_response(payload, _response_body_for(payload, s.scenario)), "application/octet-stream")
            return

        # Streaming reply: e2e_init frame + encrypted OpenAI SSE events.
        response_pk = base64.b64decode(payload["e2e_response_pk"])
        init_ct, shared = mlkem_encrypt(response_pk)
        stream_key = derive_key(shared, init_ct, INFO_STREAM)

        def sse_frame(event: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(event)}\n\n".encode()

        chunks = _stream_chunks_for(payload, s.scenario)
        usage = _stream_usage_for(s.scenario)
        out = bytearray(sse_frame({"e2e_init": base64.b64encode(init_ct).decode()}))
        for chunk in chunks:
            nonce = os.urandom(12)
            ciphertext, tag = chacha_encrypt(stream_key, nonce, sse_frame(chunk))
            enc = nonce + ciphertext + tag
            out += sse_frame({"e2e": base64.b64encode(enc).decode()})
        out += sse_frame({"usage": usage})
        out += b"data: [DONE]\n\n"
        self._send(200, bytes(out), "text/event-stream")


class MockChutesServer:
    def __init__(self) -> None:
        pk, sk = mlkem_generate_keypair()
        self.state = MockState(instance_pk=pk, instance_sk=sk)
        self.state.ensure_signing_identity()

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.state = self.state  # type: ignore[attr-defined]
        self._server = server
        self.port = server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def last_invoke(self) -> CapturedInvoke:
        return self.state.invokes[-1]
