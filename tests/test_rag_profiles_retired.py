"""R3 (2026-08-22): rag-profiles CRUD is retired → 410 Gone everywhere."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.pipelines import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_list_retired(client):
    resp = client.get("/umrs/rag-profiles", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 410


def test_get_retired(client):
    resp = client.get("/umrs/rag-profiles/abc", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 410


def test_create_retired(client):
    resp = client.post(
        "/umrs/rag-profiles",
        json={"name": "x"},
        headers={"x-user-id": "u", "x-project-id": "p"},
    )
    assert resp.status_code == 410


def test_update_retired(client):
    resp = client.patch(
        "/umrs/rag-profiles/abc",
        json={"name": "x"},
        headers={"x-user-id": "u", "x-project-id": "p"},
    )
    assert resp.status_code == 410


def test_delete_retired(client):
    resp = client.delete("/umrs/rag-profiles/abc", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 410


def test_retirement_message_is_actionable(client):
    resp = client.get("/umrs/rag-profiles", headers={"x-user-id": "u", "x-project-id": "p"})
    detail = resp.json()["detail"]
    assert "ephemeral" in detail
    assert "retired" in detail
