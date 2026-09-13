"""Tests for persisted UMRS egress audit records."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import egress_audit
from umrs.api.egress_audit import EgressAuditRecord


# pylint: disable=protected-access,too-few-public-methods


def _audit_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": "11111111-1111-4111-8111-111111111111",
        "user_id": "alice",
        "project_id": "project-alpha",
        "agent_session_id": "not-a-uuid-session-id",
        "tool_name": "web_fetch",
        "destination_url": "https://api.example.com/data",
        "destination_domain": "api.example.com",
        "method": "GET",
        "payload_hash": "sha256:empty",
        "payload_size_bytes": 0,
        "sensitivity_flags": {},
        "user_tags": ["smoke"],
        "action": "allowed",
        "response_status": 200,
        "response_size_bytes": 42,
    }
    payload.update(overrides)
    return payload


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(egress_audit.router)
    return app


def test_egress_audit_post_then_list_memory_store(monkeypatch) -> None:
    monkeypatch.delenv("UMRS_DB_URL", raising=False)
    egress_audit._store = None
    client = TestClient(_make_app())

    created = client.post("/umrs/egress/audit", json=_audit_payload())
    assert created.status_code == 201
    assert created.json() == {
        "request_id": "11111111-1111-4111-8111-111111111111",
        "persisted": False,
    }

    listed = client.get(
        "/umrs/egress/audit",
        params={"user_id": "alice", "project_id": "project-alpha", "limit": 10},
    )
    assert listed.status_code == 200
    body = listed.json()
    assert body["count"] == 1
    assert body["items"][0]["agent_session_id"] == "not-a-uuid-session-id"
    assert body["items"][0]["destination_domain"] == "api.example.com"


def test_egress_audit_store_uses_configured_database(monkeypatch) -> None:
    rows: list[tuple[Any, ...]] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def execute(self, sql: str, params: list[Any] | tuple[Any, ...]) -> None:
            if sql.lstrip().upper().startswith("INSERT"):
                rows.append(tuple(params))

        def fetchall(self) -> list[tuple[Any, ...]]:
            return rows

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    monkeypatch.setenv("UMRS_DB_URL", "postgresql://audit-db.example/umrs")
    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(connect=lambda _url, **_kw: FakeConnection()),
    )

    store = egress_audit._EgressAuditStore()
    record = EgressAuditRecord(
        **_audit_payload(
            request_id="22222222-2222-4222-8222-222222222222",
            created_at=datetime.now(timezone.utc),
        )
    )

    assert store.db_enabled
    assert store.insert(record) is True

    items = store.query(user_id="alice", project_id="project-alpha", limit=10)
    assert len(items) == 1
    assert items[0].request_id == "22222222-2222-4222-8222-222222222222"
    assert items[0].agent_session_id == "not-a-uuid-session-id"
