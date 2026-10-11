"""Phase 2G regression tests: storage write contract per live schema.

Live audit backed by information_schema:
  - knowledge_artifacts.created_at/updated_at : TEXT           (drifted shape)
  - knowledge_nodes  .created_at/updated_at : timestamptz    (native)
  - knowledge_edges/ir/events time columns  : timestamptz    (native)

The fix: insert_artifact serializes timestamps as ISO strings; all other
write paths keep native datetime parameters.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from umrs.knowledge.store import KnowledgeNode, KnowledgeStore


class _Conn:
    """Captures execute params; fetchval returns None (no existing row)."""

    def __init__(self) -> None:
        self.executed: list[tuple] = []

    async def fetchval(self, *_args: object) -> None:
        return None

    async def execute(self, sql: str, *params: object) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"


class _AcquireCtx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *_exc: object) -> None:
        pass


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


def _iso_ok(value: object) -> bool:
    """True when value is an ISO-8601 string parseable by fromisoformat."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


# ── Test A: insert_artifact writes TEXT-shaped ISO time params ───────────────


def test_insert_artifact_time_params_are_iso_strings() -> None:
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))

    ok = asyncio.run(
        store.insert_artifact(
            artifact_id="wfart-c1",
            content_hash="abc123",
            kind="s3_uri",
            project_id="default",
            source_ref="s3://r4-data/k",
            metadata={"workflow": "demo08_yield_eval"},
        )
    )
    assert ok == "wfart-c1"
    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert "INSERT INTO knowledge_artifacts" in sql
    # $9 and $10 are the created_at/updated_at parameters (0-based idx 8, 9).
    assert _iso_ok(params[8]), f"$9 not ISO string: {params[8]!r}"
    assert _iso_ok(params[9]), f"$10 not ISO string: {params[9]!r}"
    # Both are fresh UTC timestamps (close to now).
    parsed = datetime.fromisoformat(params[8])
    delta = abs((datetime.now(timezone.utc) - parsed).total_seconds())
    assert delta < 60


# ── Test B: upsert_node keeps native datetime params (timestamptz) ───────────


def test_upsert_node_keeps_native_datetimes() -> None:
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))
    node = KnowledgeNode(
        node_id="n-1",
        project_id="default",
        label="job:job-x",
        node_type="job",
        content={"wf": "demo08"},
    )

    asyncio.run(store.upsert_node(node))
    sql, params = conn.executed[-1]
    assert "INSERT INTO knowledge_nodes" in sql
    # $8 created_at fallback, $9 updated_at — native datetime values.
    assert isinstance(params[7], datetime), f"$8 not datetime: {params[7]!r}"
    assert isinstance(params[8], datetime), f"$9 not datetime: {params[8]!r}"


def test_upsert_node_explicit_created_at_preserved() -> None:
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))
    explicit = datetime(2026, 8, 11, 1, 2, 3)
    node = KnowledgeNode(
        node_id="n-2",
        project_id="default",
        label="job:job-y",
        node_type="job",
        content={},
        created_at=explicit,
    )
    asyncio.run(store.upsert_node(node))
    _, params = conn.executed[-1]
    assert params[7] is explicit  # explicit created_at wins over fallback


# ── Test C: full TEXT write boundary — insert → param-ready ISO read ─────────


def test_insert_then_iso_column_readback_matches() -> None:
    """Simulated round trip: written ISO strings are valid ISO datetimes."""
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))
    asyncio.run(
        store.insert_artifact(
            artifact_id="wfart-c2",
            content_hash="def456",
            kind="s3_uri",
            project_id="default",
        )
    )
    _, params = conn.executed[-1]
    created_at_text = params[8]
    updated_at_text = params[9]
    # TEXT columns store exactly these strings; readers parse them back.
    assert datetime.fromisoformat(created_at_text)
    assert datetime.fromisoformat(updated_at_text)
    assert datetime.fromisoformat(created_at_text) == datetime.fromisoformat(updated_at_text)
