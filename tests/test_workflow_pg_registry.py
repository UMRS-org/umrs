"""Backend conformance tests for the durable workflow execution registry (W1).

Every behavioral test runs against the in-memory reference and the Postgres
implementation (migration 0062). PG tests skip when no database is reachable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from umrs.server.workflow_execution_registry import (
    ExecutionRecord,
    WorkflowExecutionRegistry,
)

_MIGRATION = (
    Path(__file__).resolve().parents[1] / "src/umrs/migrations/0062_workflow_executions.sql"
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
            cur.execute("TRUNCATE workflow_executions")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def registry(request):
    if request.param == "memory":
        return WorkflowExecutionRegistry()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.workflow.pg_registry import PgWorkflowExecutionRegistry

    return PgWorkflowExecutionRegistry(pool)


def test_create_get_round_trip(registry):
    rec = registry.create("demo-wf", "alice", params={"k": 1})
    assert rec.execution_id.startswith("wf-")
    loaded = registry.get(rec.execution_id)
    assert loaded is not None
    assert loaded.workflow_name == "demo-wf"
    assert loaded.user_id == "alice"
    assert loaded.params == {"k": 1}
    assert registry.get("wf-missing") is None


def test_update_persists_mutation(registry):
    rec = registry.create("demo-wf", "alice")
    rec.update(status="running", started_at=111.0)
    rec.set_step("extract", status="running", started_at=112.0)
    loaded = registry.get(rec.execution_id)
    assert loaded.status == "running"
    assert loaded.steps["extract"].status == "running"
    rec.set_step("extract", status="succeeded", finished_at=113.0)
    rec.update(status="succeeded", finished_at=114.0, progress=1.0)
    loaded = registry.get(rec.execution_id)
    assert loaded.is_terminal()
    assert loaded.progress == 1.0
    assert loaded.steps["extract"].status == "succeeded"


def test_add_artifact_dedup(registry):
    rec = registry.create("demo-wf", "alice")
    rec.add_artifact("s3://a")
    rec.add_artifact("s3://a")
    rec.add_artifact("s3://b")
    loaded = registry.get(rec.execution_id)
    assert loaded.artifacts == ["s3://a", "s3://b"]


def test_list_newest_first_and_user_filter(registry):
    r1 = registry.create("wf-a", "alice")
    r2 = registry.create("wf-a", "bob")
    registry.create("wf-b", "alice")
    names = {r.execution_id for r in registry.list()}
    assert {r1.execution_id, r2.execution_id} <= names
    alice = registry.list(user_id="alice")
    assert all(r.user_id == "alice" for r in alice)
    assert len(alice) == 2


def test_cancel_moves_to_terminal(registry):
    rec = registry.create("wf-a", "alice")
    cancelled = registry.cancel(rec.execution_id)
    assert cancelled is not None
    assert cancelled.is_terminal()
    assert cancelled.cancel_requested
    assert registry.cancel("wf-missing") is None


def test_run_in_thread_terminal_transition(registry):
    rec = registry.create("wf-a", "alice")

    def _runner(r: ExecutionRecord) -> None:
        r.update(status="succeeded", finished_at=time.time())

    registry.launch(rec, _runner)

    deadline = time.time() + 3
    while time.time() < deadline and not rec.is_terminal():
        time.sleep(0.05)
    assert rec.status == "succeeded"


def test_cancel_before_run_yields_cancelled(registry):
    rec = registry.create("wf-a", "alice")
    registry.cancel(rec.execution_id)
    started = []

    def _runner(r: ExecutionRecord) -> None:
        started.append(r.execution_id)

    registry.launch(rec, _runner)

    deadline = time.time() + 2
    while time.time() < deadline and rec.status != "cancelled":
        time.sleep(0.05)
    assert started == []
    assert rec.status == "cancelled"


def test_read_through_after_simulated_restart(registry):
    from umrs.workflow.pg_registry import PgWorkflowExecutionRegistry

    if not isinstance(registry, PgWorkflowExecutionRegistry):
        pytest.skip("restart semantics only apply to PG mirror")
    rec = registry.create("wf-restore", "alice")
    rec.update(status="running")
    fresh = PgWorkflowExecutionRegistry(registry._pool)
    loaded = fresh.get(rec.execution_id)
    assert loaded is not None
    assert loaded.workflow_name == "wf-restore"
    assert loaded.status == "running"


def test_factory_env_contract(monkeypatch):
    from umrs.workflow.pg_registry import build_execution_registry

    monkeypatch.setenv("UMRS_WORKFLOW_EXECUTION_REGISTRY", "memory")
    reg = build_execution_registry()
    assert type(reg).__name__ == "WorkflowExecutionRegistry"


def test_factory_pg_requires_pool(monkeypatch):
    import umrs.shared.pg as shared_pg
    from umrs.workflow.pg_registry import build_execution_registry

    monkeypatch.setenv("UMRS_WORKFLOW_EXECUTION_REGISTRY", "pg")
    monkeypatch.setattr(
        shared_pg,
        "resolve_satellite_pool",
        _no_pool,
    )
    with pytest.raises(RuntimeError):
        build_execution_registry()


def _no_pool(*_args, **_kwargs):
    return None
