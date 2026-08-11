"""Phase 2H regression tests: node_type contract + descriptor-based UNDERSTAND.

Two live failures covered:

1. ``knowledge_nodes_node_type_check``: produced-by/failure nodes used
   ``node_type="job"`` / ``"job_failure"`` (rejected by the DB CHECK → insert failed silently).
2. provenance → UNDERSTAND race: fire-and-forget provenance persisted ~1.2s
   after `_enqueue_understand_job` queried `knowledge_artifacts` (empty) → UNDERSTAND skipped.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge.store import KnowledgeNode, KnowledgeStore

_DEDUPED = [
    {
        "artifact_id": "wfart-95c825952512cddb",
        "content_hash": "95c825952512cddb",
        "kind": "s3_uri",
        "source_ref": "s3://r4-data/jobs/job-x/attempt_1/demo08_yield_eval.json",
    }
]


class _FakeKnowledgeStore:
    """KnowledgeStore stand-in: captures every upsert_node/upsert_edge."""

    def __init__(self, _pool: object) -> None:
        self.pool = _pool

    async def upsert_node(self, node: KnowledgeNode) -> None:
        CAPTURED.append(node)

    async def upsert_edge(self, edge: object) -> None:
        CAPTURED.append(edge)


CAPTURED: list[object] = []


@asynccontextmanager
async def _pool_ctx(pool):
    yield pool


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-x",
        user_id="alice",
        project_id="default",
        priority=0,
        job_type=SimpleNamespace(value="workflow_run"),
        spec={},
    )


def _attempt() -> SimpleNamespace:
    return SimpleNamespace(attempt_id="att-x", attempt_no=1)


# ── Fix A: graph nodes comply with the CHECK constraint ──────────────────────


def test_produced_by_node_is_entity_within_allowed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global CAPTURED
    CAPTURED = []
    monkeypatch.setenv("DB_HOST", "127.0.0.1")  # satisfy the DSN gate
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _FakeKnowledgeStore)
    monkeypatch.setattr(ka, "_open_pool", lambda: _pool_ctx(object()))

    ka._create_produced_by_edges(
        list(_DEDUPED), "default", "alice", _job(), _attempt(), "demo08_yield_eval", "*"
    )

    produced_nodes = [n for n in CAPTURED if isinstance(n, KnowledgeNode)]
    assert produced_nodes, "expected provenance node to be written"
    for node in produced_nodes:
        assert node.node_type == "entity"
        assert node.node_type in KnowledgeStore.ALLOWED_NODE_TYPES
        # Semantic role preserved in label, not DB ontology.
        assert node.label.startswith(("job:", "artifact:"))

    # FK invariant: every emitted edge endpoint must reference an upserted node.
    from umrs.knowledge.store import KnowledgeEdge

    node_ids = {n.node_id for n in produced_nodes}
    assert any(aid in node_ids for aid in ("wfart-95c825952512cddb",)), (
        "artifact node must be upserted before its edge"
    )
    edges = [e for e in CAPTURED if isinstance(e, KnowledgeEdge)]
    assert edges
    for edge in edges:
        assert edge.source_id in node_ids, edge.source_id
        assert edge.target_id in node_ids, edge.target_id


def test_failure_node_is_entity_within_allowed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global CAPTURED
    CAPTURED = []
    monkeypatch.setenv("DB_HOST", "127.0.0.1")  # satisfy the DSN gate
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _FakeKnowledgeStore)
    monkeypatch.setattr(ka, "_open_pool", lambda: _pool_ctx(object()))

    ka.enqueue_ingest_failure(
        job_id="job-x",
        attempt_no=2,
        exit_reason="boom",
        user_id="alice",
        project_id="default",
    )

    failure_nodes = [n for n in CAPTURED if isinstance(n, KnowledgeNode) and "FAILED" in n.label]
    assert failure_nodes, "expected failure node"
    for node in failure_nodes:
        assert node.node_type == "entity"
        assert node.node_type in KnowledgeStore.ALLOWED_NODE_TYPES

    # FK invariant: caused_by edge endpoints must also reference upserted nodes.
    from umrs.knowledge.store import KnowledgeEdge

    all_node_ids = {n.node_id for n in CAPTURED if isinstance(n, KnowledgeNode)}
    assert "prod-job-job-x" in all_node_ids
    edges = [e for e in CAPTURED if isinstance(e, KnowledgeEdge)]
    assert edges
    for edge in edges:
        assert edge.source_id in all_node_ids, edge.source_id
        assert edge.target_id in all_node_ids, edge.target_id


def test_allowed_node_types_is_canonical_db_ontology() -> None:
    assert {
        "class",
        "function",
        "file",
        "module",
        "concept",
        "entity",
        "relationship",
    } == KnowledgeStore.ALLOWED_NODE_TYPES


# ── Fix B: UNDERSTAND receives descriptors even when the table is empty ──────


def test_understand_submits_descriptors_when_project_listing_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race regression: provenance not yet visible → listing [] → still submit."""
    submissions: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submissions.append(submission)
            return "understand-job-1"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    monkeypatch.setattr(
        ka, "_list_project_artifact_sources", lambda _pid: []
    )  # provenance not yet visible

    job = _job()
    job_id = ka._enqueue_understand_job("default", "alice", job, list(_DEDUPED))

    assert job_id == "understand-job-1"
    assert len(submissions) == 1
    spec = submissions[0].spec  # type: ignore[attr-defined]
    assert spec["sources"] == _DEDUPED
    assert spec["project_id"] == "default"
    assert spec["parent_job_id"] == "job-x"


def test_understand_descriptors_skip_listing_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When descriptors are given, the project-listing fallback is NOT consulted."""
    submissions: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submissions.append(submission)
            return "understand-job-2"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())

    def _fail_if_called(_pid: str) -> list:
        raise AssertionError("fallback listing must not be consulted")

    monkeypatch.setattr(ka, "_list_project_artifact_sources", _fail_if_called)

    job = _job()
    job_id = ka._enqueue_understand_job("default", "alice", job, list(_DEDUPED))
    assert job_id == "understand-job-2"


def test_understand_falls_back_to_listing_when_no_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal path preserved: callers without descriptors use the project listing."""
    submissions: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submissions.append(submission)
            return "understand-job-3"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    fallback = list(_DEDUPED)
    consulted: list[str] = []
    monkeypatch.setattr(
        ka,
        "_list_project_artifact_sources",
        lambda pid: consulted.append(pid) or fallback,
    )

    job = _job()
    job_id = ka._enqueue_understand_job("default", "alice", job)
    assert job_id == "understand-job-3"
    assert consulted == ["default"]
    spec = submissions[0].spec  # type: ignore[attr-defined]
    assert spec["sources"] == fallback


def test_understand_skips_when_no_descriptors_and_empty_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ka, "_list_project_artifact_sources", lambda _pid: [])
    assert ka._enqueue_understand_job("default", "alice", _job()) is None
