"""Cycle-3 memfix N7 (2026-10-22): a caller-supplied ``spec.project_id``
never moves a job into another project.

Found by the mem review (``.stab-cycle3/memreview/memreview-v0-1``): a caller
verified for project A posted ``{job_type: knowledge_understand, spec:
{project_id: B}}`` to POST /umrs/jobs. The job row was A's, but every
knowledge backend resolved ``job.spec.get("project_id", job.project_id)``:
understand wrote graph nodes into B's graph, verify wrote B's reports and an
export bundled B's graph with A as its submitter. Now the submit refuses the
spec (HTTP 403, ZMQ error; the same answer whether B exists) and every
backend runs in the job's own project whatever the spec says.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs import data_ingestion
from umrs.jobs import knowledge_adapters as ka
from umrs.server.job_tracking_handler import DurableJobCommandBackend

_PROJECT_SCOPED = (
    "knowledge_ingest",
    "knowledge_understand",
    "knowledge_embed",
    "knowledge_verify",
    "knowledge_export",
    "knowledge_reindex",
    "domain_benchmark",
)


class _JobStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kw: Any) -> str:
        self.created.append(kw)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id: str) -> None:
        return None

    def get_job(self, _job_id: str) -> Any:
        return SimpleNamespace(to_dict=lambda: {"job_id": "job-1"})

    def get_attempts_for_job(self, _job_id: str) -> list[Any]:
        return []

    def _get_conn(self) -> Any:
        raise RuntimeError("no catalog store in this test")


def _post(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> tuple[Any, _JobStore]:
    from umrs.api import jobs as jobs_api

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
    store = _JobStore()
    app = FastAPI()
    app.include_router(jobs_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    body = {"project_id": "proj-A", "user_id": "alice", **body}
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/umrs/jobs",
        json=body,
        headers={"X-Verified-Project-Id": "proj-A", "X-User-Id": "alice"},
    )
    return resp, store


@pytest.mark.parametrize("job_type", _PROJECT_SCOPED)
def test_submit_refuses_a_spec_naming_another_project(monkeypatch, job_type):
    resp, store = _post(monkeypatch, {"job_type": job_type, "spec": {"project_id": "proj-B"}})
    assert resp.status_code == 403, resp.text
    assert "spec.project_id 'proj-B'" in resp.text
    assert store.created == []


def test_the_refusal_is_the_same_whether_the_named_project_exists(monkeypatch):
    texts = {
        named: _post(monkeypatch, {"job_type": "knowledge_export", "spec": {"project_id": named}})[
            0
        ].text.replace(named, "<named>")
        for named in ("proj-B", "no-such-project-9f3")
    }
    assert len(set(texts.values())) == 1


@pytest.mark.parametrize("spec", [{"project_id": "proj-A"}, {}])
def test_submit_takes_a_spec_naming_its_own_project_or_none(monkeypatch, spec):
    resp, store = _post(monkeypatch, {"job_type": "knowledge_verify", "spec": spec})
    assert resp.status_code == 201, resp.text
    assert store.created[0]["project_id"] == "proj-A"


def test_submit_refuses_a_fan_out_child_naming_another_project(monkeypatch):
    child = {"job_type": "knowledge_understand", "spec": {"project_id": "proj-B"}}
    resp, store = _post(
        monkeypatch,
        {"job_type": "knowledge_verify", "spec": {}, "fan_out_jobs": [child]},
    )
    assert resp.status_code == 403, resp.text
    assert store.created == []


def test_the_zmq_submit_refuses_a_spec_naming_another_project():
    """ZMQ takes no knowledge job types; data_ingestion is project-scoped too
    (an admin, so the bucket check passes and the project check answers)."""
    store = _JobStore()
    spec = {"bucket": "b", "s3_key": "k", "project_id": "proj-B"}
    payload = {"project_id": "proj-A", "job_type": "data_ingestion", "spec": spec}
    answer = json.loads(
        DurableJobCommandBackend(store).submit("alice", json.dumps(payload), is_admin=True)
    )
    assert answer["status"] == "error" and "spec.project_id 'proj-B'" in answer["message"]
    assert store.created == []


# ── run time: the job's own project, whatever the spec says ────────────────


def _job(**over: Any) -> Any:
    fields = {
        "job_id": "job-x",
        "project_id": "proj-A",
        "user_id": "alice",
        "priority": 0,
        "spec": {"project_id": "proj-B", "sources": [{"artifact_id": "a1"}]},
    }
    return SimpleNamespace(**{**fields, **over})


def _attempt() -> Any:
    return SimpleNamespace(attempt_id="att", metrics={})


async def _no_pool() -> None:
    return None


def _recording(seen: list[str], position: int) -> Any:
    async def record(*args: Any, **_kw: Any) -> str:
        seen.append(args[position])
        return "ok"

    return record


@pytest.mark.parametrize(
    ("backend", "method", "position"),
    [
        (ka.KnowledgeIngestBackend, "_submit_ingest", 4),
        (ka.KnowledgeUnderstandBackend, "_submit_understand", 3),
        (ka.KnowledgeEmbedBackend, "_submit_embed", 3),
    ],
)
def test_ingest_understand_and_embed_run_in_the_jobs_own_project(
    monkeypatch, backend, method, position
):
    seen: list[str] = []
    monkeypatch.setattr(backend, method, _recording(seen, position))
    monkeypatch.setattr(backend, "_init_s3", lambda *_a: None, raising=False)
    asyncio.run(backend().submit_job(_job(), _attempt()))
    assert seen == ["proj-A"]


def test_verify_runs_in_the_jobs_own_project(monkeypatch):
    monkeypatch.setattr(ka, "_created_pool", _no_pool)
    ref = asyncio.run(ka.KnowledgeVerifyBackend().submit_job(_job(), _attempt()))
    assert "project=proj-A|" in ref


def test_export_runs_in_the_jobs_own_project(monkeypatch):
    seen: list[str] = []

    async def export(_self: Any, _pool: Any, project_id: str, *_a: Any, **_k: Any) -> Any:
        seen.append(project_id)
        raise RuntimeError("stop here")

    monkeypatch.setattr(ka, "_created_pool", _no_pool)
    monkeypatch.setattr(ka.KnowledgeExportBackend, "_export_with_integrity", export)
    with contextlib.suppress(RuntimeError):
        asyncio.run(ka.KnowledgeExportBackend().submit_job(_job(), _attempt()))
    assert seen == ["proj-A"]


def test_benchmark_runs_in_the_jobs_own_project(monkeypatch):
    monkeypatch.setattr(ka, "_created_pool", _no_pool)
    spec = {
        "project_id": "proj-B",
        "benchmark": [{"question": "q", "expected_node_ids": ["n1"]}],
    }
    ref = asyncio.run(ka.DomainBenchmarkBackend().submit_job(_job(spec=spec), _attempt()))
    assert "project=proj-A|" in ref


def test_the_understand_row_probes_ask_about_the_jobs_own_project(monkeypatch):
    asked: list[Any] = []

    async def fetchval(*args: Any) -> int:
        asked.append(args[1:])
        return 1

    def scope(project_id: str, _graph_id: str) -> str:
        asked.append(project_id)
        return ""

    monkeypatch.setattr(ka, "_node_id_scope", scope)
    store = SimpleNamespace(_pool=SimpleNamespace(fetchval=fetchval))
    assert asyncio.run(ka._artifact_row_absent(store, _job(), "a1")) == ""
    assert asyncio.run(ka._graph_written(store, _job(), "a1"))
    assert asked[:2] == [("a1", "proj-A"), "proj-A"]


def test_data_ingestion_runs_in_the_jobs_own_project(monkeypatch):
    import umrs.data_catalog.ingest as ingest
    import umrs.shared.catalog_store_access as access

    seen: list[str] = []

    def ingest_object(*_a: Any, project_id: str, **_k: Any) -> dict[str, str]:
        seen.append(project_id)
        return {"action": "created"}

    monkeypatch.setattr(ingest, "ingest_object", ingest_object)
    monkeypatch.setattr(access, "get_catalog_meta_store", lambda: object())
    monkeypatch.setattr(data_ingestion, "_build_s3_client", lambda: object())
    spec = {"bucket": "b", "s3_key": "k", "project_id": "proj-B"}
    asyncio.run(data_ingestion.DataIngestionBackend().submit_job(_job(spec=spec), _attempt()))
    assert seen == ["proj-A"]


def test_a_knowledge_job_without_a_project_fails_by_name():
    with pytest.raises(ValueError, match="has none"):
        ka.job_project(_job(project_id=""))
