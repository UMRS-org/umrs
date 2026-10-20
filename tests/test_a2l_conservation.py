"""A2L conservation domain pins (ADR-034).

- FSM transitions come from the r4-contracts truth (no standalone re-impl).
- Deterministic artifact identity is content+context hashed.
- Derive creates lineage edges atomically; parentless goes through create.
- Recorder degrades silently when the store raises (spine never blocks).
- Router exposes the parallel-run /umrs/a2l/ledger/* surface.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.a2l.models import ArtifactDerive, ArtifactRecord, DecisionCreate
from umrs.a2l.recorder import NullA2lRecorder, StoreA2lRecorder
from umrs.a2l.store import ConservationStore, compute_artifact_id

from r4_contracts.a2l import (
    A2lArtifactStatus,
    a2l_artifact_transition_ok,
)

# ── FSM contract pins ────────────────────────────────────────────────────────


def test_fsm_legal_paths() -> None:
    assert a2l_artifact_transition_ok("pending", "running")
    assert a2l_artifact_transition_ok("running", "completed")
    assert a2l_artifact_transition_ok("running", "failed")


def test_fsm_forbidden_rewinds() -> None:
    assert not a2l_artifact_transition_ok("completed", "running")
    assert not a2l_artifact_transition_ok("failed", "running")
    assert not a2l_artifact_transition_ok("pending", "completed")
    assert not a2l_artifact_transition_ok("bogus", "running")


def test_fsm_terminal_helper() -> None:
    assert ArtifactRecord(
        id="x",
        type="mask",
        uri="s3://b/m",
        user_id="u",
        status=A2lArtifactStatus.COMPLETED.value,
    ).is_terminal()


# ── Deterministic identity ───────────────────────────────────────────────────


def test_artifact_id_deterministic_and_context_bound() -> None:
    a = compute_artifact_id("hash1", "seg", "mask")
    b = compute_artifact_id("hash1", "seg", "mask")
    assert a == b and len(a) == 64
    # context (operation/type) participates in identity
    assert a != compute_artifact_id("hash1", "seg", "embedding")
    assert a != compute_artifact_id("hash1", "classify", "mask")
    assert a != compute_artifact_id("hash2", "seg", "mask")


# ── Store behavior against a fake pool ───────────────────────────────────────


class _FakeTx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeTx:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, pool: _FakePool) -> None:
        self._pool = pool

    def transaction(self) -> _FakeTx:
        return _FakeTx(self)

    async def execute(self, sql: str, *params: Any) -> None:
        await self._pool.execute(sql, *params)

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    """Records SQL; tracks artifact/edge inserts for assertions."""

    def __init__(self) -> None:
        self.sql: list[tuple[str, tuple[Any, ...]]] = []
        self.artifact_rows: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str]] = []

    def acquire(self) -> _FakeConn:
        return _FakeConn(self)

    async def execute(self, sql: str, *params: Any) -> str:
        self.sql.append((sql, params))
        collapsed = re.sub(r"\s+", " ", sql).upper()
        if "INSERT INTO A2L_ARTIFACT_EDGES" in collapsed:
            self.edges.append((params[0], params[1]))
        elif "INSERT INTO A2L_ARTIFACTS" in collapsed:
            self.artifact_rows[params[0]] = {
                "id": params[0],
                "type": params[1],
                "uri": params[3],
                "user_id": params[6],
                "job_id": params[10],
                "attempt_id": params[11],
                "status": params[5],
                "created_at": params[12],
            }
        return "OK"

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        self.sql.append((sql, params))
        if "FROM a2l_artifacts" in sql and params[0] in self.artifact_rows:
            row = dict(self.artifact_rows[params[0]])
            row.setdefault("type_schema", {})
            row.setdefault("metadata", {})
            row.setdefault("project_id", None)
            row.setdefault("hash_version", 2)
            row.setdefault("artifact_class", None)
            row.setdefault("completed_at", "")
            row.setdefault("error", "")
            return row
        return None

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.sql.append((sql, params))
        if "FROM a2l_decisions" in sql:
            return []
        rows = list(self.artifact_rows.values())
        return [
            {
                **r,
                "type_schema": {},
                "metadata": {},
                "project_id": None,
                "hash_version": 2,
                "artifact_class": None,
                "completed_at": "",
                "error": "",
            }
            for r in rows
        ]


@pytest.mark.asyncio
async def test_derive_creates_child_and_edges_atomically() -> None:
    pool = _FakePool()
    store = ConservationStore(pool)
    req = ArtifactDerive.from_dict(
        {
            "parents": ["p1", "p2"],
            "operation": "union",
            "type": "mask",
            "uri": "s3://bucket/m",
            "user_id": "alice",
            "input_data_hash": "deadbeef",
            "job_id": "job-7",
            "attempt_id": "3",
        }
    )
    rec = await store.derive_artifact(req)
    assert rec.id == compute_artifact_id("deadbeef", "union", "mask")
    assert rec.job_id == "job-7" and rec.attempt_id == "3"
    # both parent edges recorded in one call set
    assert ("p1", rec.id) in pool.edges
    assert ("p2", rec.id) in pool.edges


@pytest.mark.asyncio
async def test_derive_rejects_parentless() -> None:
    store = ConservationStore(_FakePool())
    req = ArtifactDerive.from_dict(
        {
            "parents": [],
            "operation": "x",
            "type": "mask",
            "uri": "s3://b/m",
            "user_id": "u",
            "input_data_hash": "h",
        }
    )
    with pytest.raises(ValueError, match="parent"):
        await store.derive_artifact(req)


@pytest.mark.asyncio
async def test_status_update_rejects_illegal_transition() -> None:
    pool = _FakePool()
    store = ConservationStore(pool)
    rec = ArtifactRecord(
        id=compute_artifact_id("h", "op", "mask"),
        type="mask",
        uri="s3://b/m",
        user_id="u",
        status=A2lArtifactStatus.COMPLETED.value,
    )
    await store.create_artifact(rec)
    with pytest.raises(ValueError, match="illegal transition"):
        await store.update_artifact_status(rec.id, "running")


# ── Recorder degradation contract ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_null_recorder_never_raises() -> None:
    rec = NullA2lRecorder()
    await rec.record_attempt_artifact(job_id="j", attempt_id="1")
    await rec.record_decision(decided_by="u", decision="approved")


@pytest.mark.asyncio
async def test_store_recorder_degrades_silently_on_store_failure() -> None:
    broken = MagicMock()
    broken.derive_artifact = AsyncMock(side_effect=RuntimeError("db down"))
    broken.create_artifact = AsyncMock(side_effect=RuntimeError("db down"))
    rec = StoreA2lRecorder(broken)
    # parentless path → create_artifact raises → swallowed, returns None
    result = await rec.record_attempt_artifact(
        job_id="j1",
        attempt_id="1",
        artifact_uri="s3://b/x",
        artifact_type="mask",
        user_id="u",
        content_hash="h",
    )
    assert result is None
    # parent path → derive_artifact raises → swallowed too
    result = await rec.record_attempt_artifact(
        job_id="j2",
        attempt_id="1",
        artifact_uri="s3://b/y",
        artifact_type="mask",
        user_id="u",
        content_hash="h",
        parents=["p1"],
    )
    assert result is None


# ── Migration pins ───────────────────────────────────────────────────────────


_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "migrations"
    / "0077_a2l_conservation.sql"
).read_text()


def test_migration_additive_spine_keys() -> None:
    """Conservation tables gain job_id/attempt_id without touching job tables."""
    assert "a2l_agent_executions" in _MIGRATION
    assert re.search(r"a2l_agent_executions ADD COLUMN IF NOT EXISTS job_id", _MIGRATION)
    assert re.search(r"a2l_agent_executions ADD COLUMN IF NOT EXISTS attempt_id", _MIGRATION)
    assert "a2l_decisions" in _MIGRATION
    # never mutate the authoritative spine tables
    assert "ALTER TABLE jobs" not in _MIGRATION
    assert "ALTER TABLE attempts" not in _MIGRATION


def test_migration_idempotent() -> None:
    assert _MIGRATION.count("IF NOT EXISTS") >= 10
    assert "duplicate_object" in _MIGRATION


# ── Router surface pins ──────────────────────────────────────────────────────


def test_ledger_router_parallel_run_prefix() -> None:
    from umrs.api.a2l_ledger import router

    paths = {r.path for r in router.routes}
    assert all(p.startswith("/umrs/a2l/ledger") for p in paths)
    # compositional read model exists
    assert any(p.endswith("/jobs/{job_id}/view") for p in paths)
    # standalone service surface (/gw/a2l/*) untouched by this router
    assert not any("/a2l/artifacts/derive" in p and "ledger" not in p for p in paths)


def test_recorder_uses_json_serializable_payloads() -> None:
    # decisions must remain JSON-serializable for the jsonb column
    req = DecisionCreate(decided_by="op", decision="approved", job_id="j")
    json.dumps(req.payload)
    json.dumps(req.rationale)


# ── Phase-2 emission wiring ──────────────────────────────────────────────────


class _CaptureRecorder:
    """Records all emissions for assertions (not silent)."""

    def __init__(self) -> None:
        self.artifacts: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []

    async def record_attempt_artifact(self, **kwargs: Any) -> str:
        self.artifacts.append(dict(kwargs))
        return "a2l-x"

    async def record_decision(self, **kwargs: Any) -> str:
        self.decisions.append(dict(kwargs))
        return "dec-x"


def test_recorder_global_defaults_to_null_and_is_configurable() -> None:
    import umrs.a2l.recorder as recorder_mod

    original = recorder_mod.get_recorder()
    assert isinstance(original, NullA2lRecorder)
    capture = _CaptureRecorder()
    recorder_mod.configure_recorder(capture)
    try:
        assert recorder_mod.get_recorder() is capture
    finally:
        recorder_mod.configure_recorder(NullA2lRecorder())


@pytest.mark.asyncio
async def test_workflow_run_emission_maps_artifacts_to_recorder() -> None:
    import umrs.a2l.recorder as recorder_mod
    from umrs.jobs.workflow_run import _record_run_artifacts

    capture = _CaptureRecorder()
    recorder_mod.configure_recorder(capture)
    try:
        job = MagicMock()
        job.job_id = "job-emit-1"
        job.user_id = "alice"
        job.project_id = "proj-a"
        attempt = MagicMock()
        attempt.attempt_id = "att-7"
        record = MagicMock()
        record.user_id = "alice"
        artifacts = [
            {
                "source_ref": "s3://bucket/jobs/job-emit-1/attempt_7/out.nii",
                "content_hash": "ab" * 32,
                "name": "out.nii",
                "size_bytes": 42,
                "registry_id": "reg-1",
            },
            {"source_ref": "", "content_hash": "cd" * 32, "name": "skipped"},  # no uri
            {"name": "no-hash"},  # no digest
        ]
        await _record_run_artifacts(job=job, attempt=attempt, artifacts=artifacts, record=record)
    finally:
        recorder_mod.configure_recorder(NullA2lRecorder())

    assert len(capture.artifacts) == 1
    emitted = capture.artifacts[0]
    assert emitted["job_id"] == "job-emit-1"
    assert emitted["attempt_id"] == "att-7"
    assert emitted["artifact_uri"].endswith("attempt_7/out.nii")
    assert emitted["user_id"] == "alice"
    assert emitted["project_id"] == "proj-a"
    assert emitted["metadata"]["name"] == "out.nii"


@pytest.mark.asyncio
async def test_workflow_run_emission_null_recorder_never_raises() -> None:
    import umrs.a2l.recorder as recorder_mod
    from umrs.jobs.workflow_run import _record_run_artifacts

    recorder_mod.configure_recorder(NullA2lRecorder())
    job = MagicMock(job_id="j", user_id="u", project_id=None)
    attempt = MagicMock(attempt_id="1")
    await _record_run_artifacts(
        job=job,
        attempt=attempt,
        artifacts=[{"source_ref": "s3://b/x", "content_hash": "h"}],
        record=MagicMock(),
    )


# ── Loop affinity (2026-10-20) ───────────────────────────────────────────


class _LoopRecordingStore:
    """Records which OS thread its coroutines execute on."""

    def __init__(self) -> None:
        self.ran_on: list[int] = []

    async def create_artifact(self, _record: Any) -> Any:
        import threading

        self.ran_on.append(threading.get_ident())

        class _R:
            id = "a2l-loop-1"

        return _R()

    async def derive_artifact(self, _derive: Any) -> Any:  # pragma: no cover
        raise AssertionError("parentless path expected")


def _artifact_kwargs() -> dict:
    return {
        "job_id": "j1",
        "attempt_id": "1",
        "artifact_uri": "s3://b/x",
        "artifact_type": "mask",
        "user_id": "u",
        "content_hash": "h",
    }


@pytest.mark.asyncio
async def test_store_recorder_same_loop_runs_inline() -> None:
    import threading

    store = _LoopRecordingStore()
    rec = StoreA2lRecorder(store, owner_loop=asyncio.get_running_loop())

    result = await rec.record_attempt_artifact(**_artifact_kwargs())

    assert result == "a2l-loop-1"
    assert store.ran_on == [threading.get_ident()]


@pytest.mark.asyncio
async def test_store_recorder_bounces_store_calls_to_the_owner_loop() -> None:
    """The durable runtime's loop must never touch the sidecar loop's pool:
    with a foreign owner loop the store coroutine executes on the owner's
    thread (ops-watch 2026-10-17: "Future attached to a different loop" and
    kin on bdrt)."""
    import threading

    started = threading.Event()
    stop = threading.Event()
    loops: list = []

    def _drive_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loops.append(loop)
        started.set()
        try:
            while not stop.is_set():
                loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            loop.close()

    driver = threading.Thread(target=_drive_loop, daemon=True)
    driver.start()
    assert started.wait(5)
    owner = loops[0]
    store = _LoopRecordingStore()
    rec = StoreA2lRecorder(store, owner_loop=owner)
    try:
        result = await rec.record_attempt_artifact(**_artifact_kwargs())
        assert result == "a2l-loop-1"
        assert store.ran_on and store.ran_on[0] != threading.get_ident()
    finally:
        stop.set()
        driver.join(5)


@pytest.mark.asyncio
async def test_store_recorder_without_owner_loop_awaits_inline() -> None:
    store = _LoopRecordingStore()
    rec = StoreA2lRecorder(store)  # legacy wiring: no owner recorded

    import threading

    result = await rec.record_attempt_artifact(**_artifact_kwargs())

    assert result == "a2l-loop-1"
    assert store.ran_on == [threading.get_ident()]
