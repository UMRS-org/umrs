"""Satellite router durability tests — evidence + experiment registry HTTP.

These legacy suites exercise the router WITHOUT identity headers, i.e. the
anonymous mvp-mode contract. Wave-6 tenant-binding behavior under signed
identity is covered by test_wave6_caller_gate.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.evidence import router
from umrs.api.evidence import set_store as set_evidence_store
from umrs.api.experiment_registry import router as exp_router
from umrs.api.experiment_registry import set_registry
from umrs.evidence import build_default_store
from umrs.experiment_registry import build_default_store as build_registry


@pytest.fixture(autouse=True)
def _mvp_mode(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    yield


def _evidence_client(monkeypatch):
    store = build_default_store()
    set_evidence_store(store)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), store


def _experiment_client(monkeypatch):
    registry = build_registry()
    set_registry(registry)
    app = FastAPI()
    app.include_router(exp_router)
    return TestClient(app), registry


def _post_supporting_evidence(client) -> str:
    payload = {
        "project_id": "proj-1",
        "claim_id": "hyp-1",
        "source_type": "experiment_result",
        "source_id": "exp-1",
        "content": "supporting-dataset-A",
        "content_uri": "s3://bucket/evidence-a.json",
        "relevance_score": 0.91,
        "citation": "Experiment 2026-08-01",
    }
    resp = client.post("/umrs/evidence", json=payload)
    assert resp.status_code == 200
    evidence_id = resp.json()["evidence_id"]
    assert evidence_id
    return evidence_id


def test_evidence_create_get_lt_search():
    client, _ = _evidence_client(None)
    evidence_id = _post_supporting_evidence(client)

    got = client.get(f"/umrs/evidence/{evidence_id}")
    assert got.status_code == 200
    assert got.json()["claim_id"] == "hyp-1"
    assert isinstance(got.json()["source_type"], str)

    ar = client.get("/umrs/evidence", params={"project_id": "proj-1"})
    assert ar.status_code == 200
    assert any(e["evidence_id"] == evidence_id for e in ar.json()["evidence"])

    search = client.get("/umrs/evidence/search", params={"query": "supporting"})
    assert search.status_code == 200
    assert search.json()["evidence"][0]["content"].startswith("supporting")


def test_evidence_404_missing():
    client, _ = _evidence_client(None)
    assert client.get("/umrs/evidence/missing").status_code == 404


def test_evidence_update_status_and_count():
    client, _ = _evidence_client(None)
    payload = {
        "project_id": "p1",
        "claim_id": "c",
        "source_type": "manual",
        "source_id": "s",
        "content": "x",
        "content_uri": "",
        "relevance_score": 0.5,
        "citation": "",
    }
    resp = client.post("/umrs/evidence", json=payload)
    eid = resp.json()["evidence_id"]
    upd = client.put(f"/umrs/evidence/{eid}/status", json={"status": "retracted"})
    assert upd.status_code == 200
    assert upd.json()["status"] == "retracted"
    count = client.get("/umrs/evidence/count")
    assert count.status_code == 200
    assert count.json()["count"] >= 1


def test_experiment_create_list_status_transition():
    client, _ = _experiment_client(None)
    payload = {
        "project_id": "proj-1",
        "hypothesis_id": "hyp-1",
        "protocol": {"slurm": {}},
        "parameters": {"lr": 0.01},
        "validation_method": "statistical_test",
        "created_by": "alice",
        "notes": "guard against regressions",
    }
    resp = client.post("/umrs/experiment-registry", json=payload)
    assert resp.status_code == 200
    exp_id = resp.json()["experiment_id"]
    assert exp_id

    lst = client.get("/umrs/experiment-registry", params={"project_id": "proj-1"})
    assert lst.status_code == 200
    assert lst.json()["experiments"][0]["experiment_id"] == exp_id

    designed = client.put(f"/umrs/experiment-registry/{exp_id}/status", json={"status": "designed"})
    assert designed.status_code in (200, 204)
    detail = client.get(f"/umrs/experiment-registry/{exp_id}")
    assert detail.json()["status"] == "designed"


def test_experiment_invalid_transition_raises_409():
    client, _ = _experiment_client(None)
    payload = {
        "project_id": "proj-1",
        "hypothesis_id": "hyp-1",
        "protocol": {},
        "parameters": {},
        "validation_method": "manual",
        "created_by": "alice",
        "notes": "",
    }
    resp = client.post("/umrs/experiment-registry", json=payload)
    exp_id = resp.json()["experiment_id"]
    invalid = client.put(f"/umrs/experiment-registry/{exp_id}/status", json={"status": "RUNNING"})
    assert invalid.status_code in (400, 409, 422)


def test_experiment_result_roundtrip():
    client, _ = _experiment_client(None)
    payload = {
        "project_id": "proj-1",
        "hypothesis_id": "hyp-1",
        "protocol": {},
        "parameters": {},
        "validation_method": "manual",
        "created_by": "alice",
        "notes": "",
    }
    resp = client.post("/umrs/experiment-registry", json=payload)
    exp_id = resp.json()["experiment_id"]
    upd = client.post(
        f"/umrs/experiment-registry/{exp_id}/results",
        json={"metrics": {"acc": 0.9}, "artifacts": ["s3://x"], "logs": "ok"},
    )
    assert upd.status_code in (200, 201)
    detail = client.get(f"/umrs/experiment-registry/{exp_id}")
    assert any(r.get("logs") == "ok" for r in detail.json().get("results", []))
