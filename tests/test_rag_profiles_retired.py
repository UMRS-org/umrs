"""R3 (2026-08-22): rag-profiles CRUD retired; 2026-10-18 (H-10.1
rag-profiles wave): the stubs are deleted with their last client, so every
verb answers 404 — the route resolves no identity and carries no message.
On the retired-stub sources each of these answered 410."""

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


def test_list_deleted(client):
    resp = client.get("/umrs/rag-profiles", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 404


def test_get_deleted(client):
    resp = client.get("/umrs/rag-profiles/abc", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 404


def test_create_deleted(client):
    resp = client.post(
        "/umrs/rag-profiles",
        json={"name": "x"},
        headers={"x-user-id": "u", "x-project-id": "p"},
    )
    assert resp.status_code == 404


def test_update_deleted(client):
    resp = client.patch(
        "/umrs/rag-profiles/abc",
        json={"name": "x"},
        headers={"x-user-id": "u", "x-project-id": "p"},
    )
    assert resp.status_code == 404


def test_delete_deleted(client):
    resp = client.delete("/umrs/rag-profiles/abc", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.status_code == 404


def test_deletion_has_no_bespoke_message(client):
    """The old 410 detail ('ephemeral', 'retired') pointed at site config;
    a deleted route has no handler, so FastAPI's stock 404 answers."""
    resp = client.get("/umrs/rag-profiles", headers={"x-user-id": "u", "x-project-id": "p"})
    assert resp.json() == {"detail": "Not Found"}
