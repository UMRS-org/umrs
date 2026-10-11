"""Phase 2F regression tests: JSON/JSONB codecs on the ad-hoc pool.

Live failure that motivated this: knowledge writes passed raw dicts into
jsonb parameters through the ad-hoc pool (no codecs) and asyncpg rejected
them with DataError — silently killing provenance/graph writes ("expected
str, got dict").
"""

from __future__ import annotations

import json

import pytest
from umrs.jobs import knowledge_adapters as ka


class _RecordingConn:
    """Captures set_type_codec calls like an asyncpg connection."""

    def __init__(self) -> None:
        self.codecs: list[dict] = []

    async def set_type_codec(self, typename, **kwargs):
        self.codecs.append({"typename": typename, **kwargs})


class _RecordingAcquireConn:
    """Records execute params; fetchval returns None (no existing artifact)."""

    def __init__(self) -> None:
        self.executed: list[tuple] = []

    async def fetchval(self, *_args: object) -> None:
        return None

    async def execute(self, sql: str, *params: object) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"

    async def fetch(self, *_a: object) -> list:
        return []


class _AcquireCtx:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, *_exc: object) -> None:
        pass


class _StorePoolStub:
    """Pool whose acquire() yields the recording connection."""

    def __init__(self, conn: object) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


# ── Test A: codecs registered for json AND jsonb ─────────────────────────────


def test_open_pool_registers_json_and_jsonb_codecs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    captured: dict[str, object] = {}

    def _factory(*, dsn: str = "", init=None, **_kw):
        captured["init"] = init

        class _Awaitable:
            def __await__(self):  # type: ignore[no-untyped-def]
                async def _v() -> object:
                    return object()

                return _v().__await__()

        return _Awaitable()

    monkeypatch.setattr("asyncpg.create_pool", _factory)

    async def _run() -> None:
        async with ka._open_pool():
            pass

    import asyncio

    asyncio.run(_run())

    assert captured.get("init") is ka._init_json_codecs


def test_init_json_codecs_covers_json_and_jsonb() -> None:
    conn = _RecordingConn()

    import asyncio

    asyncio.run(ka._init_json_codecs(conn))

    names = [c["typename"] for c in conn.codecs]
    assert "json" in names
    assert "jsonb" in names
    assert names.index("json") < names.index("jsonb")
    for entry in conn.codecs:
        assert entry["schema"] == "pg_catalog"
        assert entry["encoder"] is json.dumps
        assert entry["decoder"] is json.loads


# ── Test B: encoder yields JSON serializable output ─────────────────────────


def test_codec_encoder_produces_json_string() -> None:
    payload = {
        "artifact_id": "wfart-95c825952512cddb",
        "source_ref": "s3://r4-data/jobs/job-x/attempt_1/out.json",
        "metadata": {"workflow": "demo08_yield_eval"},
    }
    encoded = json.dumps(payload)
    assert isinstance(encoded, str)
    json.loads(encoded)  # must be JSON-serializable
    assert json.loads(encoded) == payload


# ── Test C: decoder yields a dict (symmetric READ behavior preserved) ────────


def test_codec_decoder_returns_dict() -> None:
    decoded = json.loads('{"workflow": "demo08_yield_eval"}')
    assert isinstance(decoded, dict)
    assert decoded["workflow"] == "demo08_yield_eval"


def test_parse_metrics_jsonb_still_tolerates_both_shapes() -> None:
    """Phase 2C defense stays: dict AND str accepted regardless of codecs."""
    assert ka._parse_metrics_jsonb('{"a": 1}') == {"a": 1}
    assert ka._parse_metrics_jsonb({"a": 1}) == {"a": 1}
    assert ka._parse_metrics_jsonb(None) == {}


# ── Test D: provenance-shaped write accepts dict metadata ────────────────────


def test_knowledge_store_insert_artifact_accepts_metadata_dict() -> None:
    from umrs.knowledge.store import KnowledgeStore

    conn = _RecordingAcquireConn()
    pool = _StorePoolStub(conn)

    store = KnowledgeStore(pool)
    metadata = {
        "produced_by_job_id": "job-x",
        "produced_by_attempt_id": "att-1",
        "workflow": "demo08_yield_eval",
        "step": "*",
    }

    import asyncio

    ok = asyncio.run(
        store.insert_artifact(
            artifact_id="wfart-95c825952512cddb",
            content_hash="95c825952512cddb",
            kind="s3_uri",
            project_id="default",
            source_ref="s3://r4-data/k",
            metadata=metadata,
        )
    )
    assert ok == "wfart-95c825952512cddb"
    assert conn.executed, "store must execute the INSERT"
    sql, params = conn.executed[-1]
    assert "INSERT INTO knowledge_artifacts" in sql
    assert params[7] == metadata  # dict reaches the jsonb param slot
    assert isinstance(params[7], dict)
    json.dumps(params[7])  # value handed to codec is JSON-serializable


def test_ir_node_edge_writes_carry_dict_content_params() -> None:
    """The same codec boundary covers insert_ir / upsert_node / upsert_edge."""
    import asyncio

    from umrs.knowledge.store import (
        KnowledgeEdge,
        KnowledgeNode,
        KnowledgeStore,
    )

    conn = _RecordingAcquireConn()
    store = KnowledgeStore(_StorePoolStub(conn))

    node_content = {"job_id": "job-x", "workflow": "demo08_yield_eval"}
    edge_content = {"workflow": "demo08_yield_eval", "step": "*", "attempt_id": "att-1"}
    ir_content = {"modules": ["m1"], "functions": ["f1"]}

    asyncio.run(
        store.insert_ir(
            ir_id="ir-1",
            artifact_id="wfart-x",
            ir_type="code",
            content=ir_content,
            content_hash="h1",
        )
    )
    asyncio.run(
        store.upsert_node(
            KnowledgeNode(
                node_id="n-1",
                project_id="default",
                label="job:job-x",
                node_type="job",
                content=node_content,
            )
        )
    )
    asyncio.run(
        store.upsert_edge(
            KnowledgeEdge(
                edge_id="e-1",
                project_id="default",
                source_id="a",
                target_id="b",
                relation_type="produced_by",
                content=edge_content,
            )
        )
    )

    assert len(conn.executed) == 3
    for sql, params in conn.executed:
        dict_params = [p for p in params if isinstance(p, dict)]
        assert dict_params, f"no dict param in write: {sql[:60]}"
        for value in dict_params:
            json.dumps(value)  # must be JSON-serializable for the codec
