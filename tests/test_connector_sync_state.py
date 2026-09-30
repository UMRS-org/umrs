"""P2 conformance tests: connector sync state store + sync runner.

Dual-backend (memory + PG migration 0063) store conformance plus runner
behavior: cursor durability, content-hash dedup, DLQ isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.connectors import (
    ConnectorConfig,
    ConnectorType,
    Document,
    DocumentStatus,
)
from umrs.connectors.sync_runner import ConnectorSyncRunner
from umrs.connectors.sync_state import (
    InMemorySyncStateStore,
    SyncStateRecord,
)

# Interface-parity signatures: pytest fakes intentionally ignore args.

_MIGRATION = (
    Path(__file__).resolve().parents[1] / "src/umrs/migrations/0063_connector_sync_state.sql"
)


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_MIGRATION.read_text())
            cur.execute("TRUNCATE connector_seen_hashes")
            cur.execute("TRUNCATE connector_sync_state CASCADE")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def store(request):
    if request.param == "memory":
        return InMemorySyncStateStore()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.connectors.sync_state import PgSyncStateStore

    return PgSyncStateStore(pool)


# ── store conformance ────────────────────────────────────────────────────


def test_put_get_round_trip(store):
    rec = SyncStateRecord(
        name="gh-main",
        connector_type="github",
        cursor="page:3",
        last_sync_at="2026-08-01T00:00:00+00:00",
        last_status="ok",
        docs_synced=40,
        errors_count=1,
        dlq=[{"source_id": "repo/x", "error": "boom"}],
        checkpoint_json={"etag": "abc"},
    )
    store.put(rec)
    loaded = store.get("gh-main")
    assert loaded is not None
    assert loaded.cursor == "page:3"
    assert loaded.docs_synced == 40
    assert loaded.dlq == [{"source_id": "repo/x", "error": "boom"}]
    assert loaded.checkpoint_json == {"etag": "abc"}
    assert loaded.updated_at
    assert store.get("missing") is None


def test_put_upserts_latest(store):
    store.put(SyncStateRecord(name="c1", connector_type="gdrive", docs_synced=1))
    store.put(SyncStateRecord(name="c1", connector_type="gdrive", docs_synced=10, cursor="t"))
    loaded = store.get("c1")
    assert loaded.docs_synced == 10
    assert loaded.cursor == "t"


def test_list_sorted(store):
    store.put(SyncStateRecord(name="zz", connector_type="slack"))
    store.put(SyncStateRecord(name="aa", connector_type="gdrive"))
    assert [r.name for r in store.list()] == ["aa", "zz"]


def test_seen_hash_partition_and_clear(store):
    store.put(SyncStateRecord(name="c1", connector_type="gdrive"))
    assert store.seen("c1", "h1") is False
    store.mark_seen("c1", "h1")
    assert store.seen("c1", "h1") is True
    assert store.seen("c2", "h1") is False
    store.clear_seen("c1")
    assert store.seen("c1", "h1") is False


# ── runner behavior ──────────────────────────────────────────────────────


def _config(name="demo"):
    return ConnectorConfig(connector_type=ConnectorType.CUSTOM, name=name)


def _doc(source_id, content):
    return Document(
        source_id=source_id,
        content=content,
        status=DocumentStatus.PENDING,
    )


class _DummyConnector:
    name = "dummy"

    @property
    def connector_type(self):
        return ConnectorType.CUSTOM

    def validate_config(self, config):
        return []

    def fetch(self, config, checkpoint=None):
        return iter([])


def _runner(store):
    return ConnectorSyncRunner(store)


def test_runner_full_sync_persists_cursor_and_stats(store):
    cfg = _config()
    ingested = []
    report = _runner(store).run(
        _DummyConnector(),
        cfg,
        ingest=ingested.append,
        docs_iter=iter([_doc("a", "alpha"), _doc("b", "beta")]),
    )
    assert report.status == "ok"
    assert report.docs_synced == 2
    assert len(ingested) == 2
    state = store.get("demo")
    assert state.docs_synced == 2
    assert state.last_status == "ok"
    assert state.last_sync_at


def test_runner_dedup_skips_seen_documents(store):
    cfg = _config()
    runner = _runner(store)
    runner.run(_DummyConnector(), cfg, docs_iter=iter([_doc("a", "alpha")]))
    report = runner.run(
        _DummyConnector(), cfg, docs_iter=iter([_doc("a", "alpha"), _doc("b", "beta")])
    )
    assert report.docs_skipped_dup == 1
    assert report.docs_synced == 1
    assert store.get("demo").docs_synced == 2


def test_runner_dead_letters_replayable(store):
    cfg = _config()

    def _boom(doc):
        raise ValueError("bad payload")

    report = _runner(store).run(
        _DummyConnector(),
        cfg,
        ingest=_boom,
        docs_iter=iter([_doc("bad", "x"), _doc("ugly", "y")]),
    )
    assert report.docs_failed == 2
    state = store.get("demo")
    assert state.errors_count == 2
    assert {d["source_id"] for d in state.dlq} == {"bad", "ugly"}
    assert state.last_status == "ok"  # doc-level failures ≠ run failure


def test_runner_connector_crash_isolated_and_reported(store):
    class _Crash(_DummyConnector):
        def fetch(self, config, checkpoint=None):
            raise RuntimeError("offline")

    cfg = _config()
    runner = _runner(store)
    report = runner.run(_Crash(), cfg)
    assert report.status == "error"
    assert "offline" in report.error
    state = store.get("demo")
    assert state.last_status == "error"
    assert "offline" in state.last_error
    # Other connector unaffected
    report2 = runner.run(_DummyConnector(), _config("other"), docs_iter=iter([_doc("z", "z")]))
    assert report2.status == "ok"


def test_runner_checkpoint_rounds_to_fetch(store):
    seen = {}

    class _Echo(_DummyConnector):
        def fetch(self, config, checkpoint=None):
            seen["cursor"] = checkpoint.cursor if checkpoint else None
            seen["docs_synced"] = checkpoint.docs_synced if checkpoint else 0
            return iter([])

    store.put(
        SyncStateRecord(
            name="demo",
            connector_type="custom",
            cursor="p:9",
            docs_synced=9,
            errors_count=1,
            checkpoint_json={"etag": "x"},
        )
    )
    cfg = _config()
    _runner(store).run(_Echo(), cfg)
    assert seen["cursor"] == "p:9"
    assert seen["docs_synced"] == 9


def test_runner_disabled_config_skipped(store):
    cfg = _config()
    cfg.enabled = False
    report = _runner(store).run(_DummyConnector(), cfg, docs_iter=iter([_doc("a", "a")]))
    assert report.status == "skipped_disabled"
    assert store.get("demo") is None
