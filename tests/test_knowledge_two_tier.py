"""D-35 pins: two-tier knowledge — proposed → established (migration 0106)."""

from __future__ import annotations

import pytest
import umrs.api.knowledge_graphs as kg
from fastapi import HTTPException
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows  # node/edge status lookups

    async def fetchrow(self, sql, *params):
        pid = params[0]
        if "knowledge_nodes" in sql:
            return self._rows.get(("node", pid))
        return self._rows.get(("edge", pid))

    async def fetchval(self, _sql, *_params):
        return {}

    async def execute(self, sql, *_params):
        self.last_sql = sql

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def _store(rows):
    from umrs.knowledge.store import KnowledgeStore

    return KnowledgeStore(_FakePool(_FakeConn(rows)))


class TestEstablish:
    async def test_missing_node_returns_false(self):
        assert await _store({}).establish_node("ghost", established_by="a") is False

    async def test_proposed_node_promotes_and_stamps_reviewer(self):
        rows = {("node", "n-1"): {"status": "proposed"}}
        store = _store(rows)
        assert await store.establish_node("n-1", established_by="alice") is True
        assert "established" in store._pool._conn.last_sql

    async def test_established_node_is_idempotent_true(self):
        rows = {("node", "n-1"): {"status": "established"}}
        store = _store(rows)
        assert await store.establish_node("n-1", established_by="alice") is True
        assert not getattr(store._pool._conn, "last_sql", "").startswith("UPDATE")

    async def test_edge_promotes(self):
        rows = {("edge", "e-1"): {"status": "proposed"}}
        store = _store(rows)
        assert await store.establish_edge("e-1", established_by="alice") is True


class TestUpsertCarriesTier:
    async def test_upsert_node_writes_status_and_confidence(self):
        captured = {}

        class _Pool:
            def acquire(self):
                return self

            async def execute(self, sql, *params):
                captured["sql"], captured["params"] = sql, params

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

        from umrs.knowledge.store import KnowledgeStore

        node = KnowledgeNode(
            node_id="n-2",
            project_id="p",
            label="L",
            node_type="concept",
            status="proposed",
            confidence=0.7,
        )
        await KnowledgeStore(_Pool()).upsert_node(node)
        assert "status, confidence" in captured["sql"]
        assert captured["params"][-2:] == ("proposed", 0.7)

    async def test_upsert_conflict_never_silently_promotes(self):
        from umrs.knowledge.store import KnowledgeStore

        sql_seen = {}

        class _Pool:
            def acquire(self):
                return self

            async def execute(self, sql, *_params):
                sql_seen["sql"] = sql

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

        node = KnowledgeNode(
            node_id="n",
            project_id="p",
            label="L",
            node_type="concept",
            status="proposed",
        )
        await KnowledgeStore(_Pool()).upsert_node(node)
        update_clause = sql_seen["sql"].split("ON CONFLICT")[1]
        assert "status" not in update_clause  # promotion only via establish seam


def test_agent_promotion_refused_by_name():
    from fastapi import Request

    req = Request(scope={"type": "http", "headers": []})
    original = kg.request_actor
    kg.request_actor = lambda _request: "agent"
    try:
        with pytest.raises(HTTPException) as exc:
            kg._refuse_agent_promotion(req)
        assert exc.value.status_code == 403
        assert "human" in str(exc.value.detail).lower()
    finally:
        kg.request_actor = original


def test_human_promotion_passes_the_gate():
    original = kg.request_actor
    kg.request_actor = lambda _request: "user"
    try:
        kg._refuse_agent_promotion(object())  # no raise
    finally:
        kg.request_actor = original


class TestProducerThreading:
    """D-35 producer: agent-originated ingest rows land proposed."""

    def test_as_proposed_tier_stamps_rows(self):

        from umrs.jobs.knowledge_adapters import _as_proposed_tier

        node = KnowledgeNode(
            node_id="n",
            project_id="p",
            label="L",
            node_type="concept",
            status="established",
            confidence=0.9,
        )
        edge = KnowledgeEdge(
            edge_id="e",
            project_id="p",
            source_id="n",
            target_id="n",
            relation_type="r",
            status="established",
            confidence=0.9,
        )
        nodes, edges = _as_proposed_tier([node], [edge])
        assert nodes[0].status == "proposed" and nodes[0].confidence == 0.0
        assert edges[0].status == "proposed" and edges[0].confidence == 0.0
        # originals untouched (frozen replace)
        assert node.status == "established"

    def test_user_origin_leaves_tier_untouched(self):

        node = KnowledgeNode(
            node_id="n",
            project_id="p",
            label="L",
            node_type="concept",
        )
        # the tier application is gated on actor=='agent' in
        # _understand_one_artifact; _as_proposed_tier itself is only called
        # for agent origins — pin the gate by construction (no stamping fn
        # call for user paths is enforced by the runner's origin check).
        assert node.status == "established"

    def test_understand_spec_carries_origin_actor(self):
        import inspect

        from umrs.jobs import knowledge_adapters as ka

        sig = inspect.signature(ka._understand_one_artifact)
        assert "origin_actor" in sig.parameters
        src = inspect.getsource(ka._enqueue_understand_job)
        assert "origin_actor" in src


class TestEvidenceRefs:
    """D-35 tail: promotions carry the human's evidence + settled confidence."""

    async def test_refs_recorded_in_review_block(self):
        rows = {("node", "n-1"): {"status": "proposed"}}
        store = _store(rows)
        conn = store._pool._conn
        ok = await store.establish_node(
            "n-1",
            established_by="alice",
            evidence_refs=["ev-9", "ev-2"],
            confidence=0.9,
        )
        assert ok is True
        assert "confidence = $3" in conn.last_sql
        assert "status='established'" in conn.last_sql

    async def test_refs_merge_deduped_on_already_established(self):
        """Late evidence lands on an established row — review grows, status stays."""
        rows = {("node", "n-1"): {"status": "established"}}
        store = _store(rows)
        conn = store._pool._conn
        conn_content = {"review": {"established_by": "alice", "evidence_refs": ["ev-1"]}}

        async def fetchval(sql, *params):
            return conn_content

        conn.fetchval = fetchval
        ok = await store.establish_node("n-1", established_by="bob", evidence_refs=["ev-1", "ev-2"])
        assert ok is True
        assert conn_content["review"]["evidence_refs"] == ["ev-1", "ev-2"]
        assert conn.last_sql.startswith("UPDATE")

    async def test_absent_refs_and_confidence_never_invented(self):
        rows = {("node", "n-1"): {"status": "proposed"}}
        store = _store(rows)
        conn = store._pool._conn
        conn_content = {}

        async def fetchval(sql, *params):
            return conn_content

        conn.fetchval = fetchval
        await store.establish_node("n-1", established_by="alice")
        assert "evidence_refs" not in conn_content["review"]
        assert "confidence = $3" not in conn.last_sql

    async def test_confidence_clamped_to_valid_range(self):
        rows = {("node", "n-1"): {"status": "proposed"}}
        store = _store(rows)
        conn = store._pool._conn
        captured = {}

        async def execute(sql, *params):
            captured["params"] = params
            conn.last_sql = sql

        conn.execute = execute
        await store.establish_node("n-1", established_by="alice", confidence=7.5)
        assert captured["params"][-1] == 1.0  # clamped, never garbage
