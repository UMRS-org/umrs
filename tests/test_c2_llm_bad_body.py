"""Cycle-2 umrs-api lane (2026-10-21): a body that is not a JSON object is
a 400 on the OpenAI-compatible LLM routes, not a 500.

Found live: POST /umrs/llm/v1/chat/completions (also /v1/completions,
/v1/embeddings) with a verified identity and an empty or non-JSON body
answered 500: ``await request.json()`` raised JSONDecodeError (and a JSON
array reached ``body.get`` as AttributeError).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import llm

from r4_contracts.handshake_hmac import sign_claims

SECRET = "c2-llm-body-secret"
HEADERS = {
    "X-User-ID": "alice",
    "X-Verified-Project-ID": "pa",
    "X-Verified-Is-Admin": "false",
    "X-Verified-HMAC": sign_claims(SECRET, "alice", "pa", False),
    "Content-Type": "application/json",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    app = FastAPI()
    app.include_router(llm.router)
    app.dependency_overrides[llm.get_registry] = lambda: object()
    return TestClient(app)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"])
@pytest.mark.parametrize("payload", [b"", b"not json", b"[1, 2]"])
def test_non_object_body_is_400(client: TestClient, path: str, payload: bytes) -> None:
    r = client.post(f"/umrs/llm{path}", content=payload, headers=HEADERS)
    assert r.status_code == 400, r.text
    assert r.headers.get("x-platform-error-code") == "llm.bad_request"
