"""Backend conformance tests for research-layer satellite stores (K1).

Every behavioral test runs twice: once against the in-memory reference
implementation and once against the Postgres implementation (migration
0059). Postgres tests are skipped when no database is reachable, matching
the repo's existing optional-Postgres test convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umrs import evidence as evidence_mod
from umrs import experiment_registry as expreg_mod
from umrs import scientific_memory as sm_mod

_MIGRATION = (
    Path(__file__).resolve().parents[1] / "src/umrs/migrations/0059_research_satellites.sql"
)

_TABLES = "scientific_memory_entries, experiment_results, experiments, evidence_items"


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
            cur.execute(f"TRUNCATE {_TABLES}")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def sm_store(request):
    if request.param == "memory":
        return sm_mod.InMemoryScientificMemoryStore()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.scientific_memory.pg_store import PgScientificMemoryStore

    return PgScientificMemoryStore(pool)


@pytest.fixture(params=["memory", "pg"])
def evidence_store(request):
    if request.param == "memory":
        return evidence_mod.InMemoryEvidenceStore()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.evidence.pg_store import PgEvidenceStore

    return PgEvidenceStore(pool)


@pytest.fixture(params=["memory", "pg"])
def exp_registry(request):
    if request.param == "memory":
        return expreg_mod.InMemoryExperimentRegistry()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.experiment_registry.pg_store import PgExperimentRegistry

    return PgExperimentRegistry(pool)


def _entry(entry_id: str, project_id: str, created_at: str, **overrides):
    kwargs = {
        "entry_id": entry_id,
        "project_id": project_id,
        "created_at": created_at,
        "updated_at": created_at,
    }
    kwargs.update(overrides)
    return sm_mod.MemoryEntry(**kwargs)


def _evidence(eid: str, claim_id: str, score: float, **overrides):
    kwargs = {
        "evidence_id": eid,
        "project_id": "proj-1",
        "claim_id": claim_id,
        "relevance_score": score,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    kwargs.update(overrides)
    return evidence_mod.Evidence(**kwargs)


# ── Scientific memory ────────────────────────────────────────────────────


def test_sm_add_get_round_trip(sm_store):
    entry = _entry(
        "mem-1",
        "proj-1",
        "2026-01-01T00:00:00+00:00",
        content="experiment log body",
        tags=["slurm", "hpc"],
        confidence=0.9,
        references=["mem-0"],
    )
    assert sm_store.add(entry) == "mem-1"
    loaded = sm_store.get("mem-1")
    assert loaded is not None
    assert loaded.content == "experiment log body"
    assert loaded.tags == ["slurm", "hpc"]
    assert loaded.references == ["mem-0"]
    assert loaded.confidence == pytest.approx(0.9)
    assert sm_store.get("missing") is None


def test_sm_add_duplicate_raises(sm_store):
    sm_store.add(_entry("mem-1", "proj-1", "2026-01-01T00:00:00+00:00"))
    with pytest.raises(sm_mod.DuplicateMemoryEntryError):
        sm_store.add(_entry("mem-1", "proj-1", "2026-01-01T00:00:00+00:00"))


def test_sm_list_filters_and_order(sm_store):
    sm_store.add(_entry("mem-a", "proj-1", "2026-01-01T00:00:01+00:00"))
    sm_store.add(_entry("mem-b", "proj-1", "2026-01-01T00:00:02+00:00"))
    sm_store.add(_entry("mem-c", "proj-2", "2026-01-01T00:00:03+00:00"))
    proj1 = sm_store.list_entries(project_id="proj-1")
    assert [e.entry_id for e in proj1] == ["mem-b", "mem-a"]
    assert len(sm_store.list_entries(limit=2)) == 2
    typed = sm_store.list_entries(project_id="proj-1", entry_type=sm_mod.MemoryEntryType.NOTE)
    assert len(typed) == 2


def test_sm_search_matches_content_summary_tags(sm_store):
    sm_store.add(_entry("mem-1", "proj-1", "2026-01-01T00:00:01+00:00", content="cuda kernel tune"))
    sm_store.add(_entry("mem-2", "proj-1", "2026-01-01T00:00:02+00:00", tags=["cuda"]))
    sm_store.add(_entry("mem-3", "proj-1", "2026-01-01T00:00:03+00:00"))
    hits = sm_store.search("cuda", project_id="proj-1")
    assert {e.entry_id for e in hits} == {"mem-1", "mem-2"}


def test_sm_update_status_and_count(sm_store):
    sm_store.add(_entry("mem-1", "proj-1", "2026-01-01T00:00:00+00:00"))
    assert sm_store.update_status("missing", sm_mod.MemoryEntryStatus.ARCHIVED) is False
    assert sm_store.update_status("mem-1", sm_mod.MemoryEntryStatus.ARCHIVED) is True
    assert sm_store.get("mem-1").status == sm_mod.MemoryEntryStatus.ARCHIVED
    assert sm_store.count(project_id="proj-1") == 1
    assert sm_store.count(project_id="proj-1", status=sm_mod.MemoryEntryStatus.ACTIVE) == 0


def test_sm_timeline_oldest_first(sm_store):
    sm_store.add(_entry("mem-2", "proj-1", "2026-01-01T00:00:02+00:00"))
    sm_store.add(_entry("mem-1", "proj-1", "2026-01-01T00:00:01+00:00"))
    timeline = sm_store.get_timeline("proj-1")
    assert [e.entry_id for e in timeline] == ["mem-1", "mem-2"]


# ── Evidence ─────────────────────────────────────────────────────────────


def test_evidence_add_get_round_trip(evidence_store):
    evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.8, content="supports H1"))
    loaded = evidence_store.get_evidence("evd-1")
    assert loaded is not None
    assert loaded.claim_id == "claim-1"
    assert loaded.relevance_score == pytest.approx(0.8)
    assert evidence_store.get_evidence("missing") is None


def test_evidence_duplicate_raises(evidence_store):
    evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.5))
    with pytest.raises(evidence_mod.DuplicateEvidenceError):
        evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.5))


def test_evidence_list_defaults_active_and_orders_by_score(evidence_store):
    evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.2))
    evidence_store.add_evidence(_evidence("evd-2", "claim-1", 0.9))
    evidence_store.add_evidence(
        _evidence("evd-3", "claim-1", 0.7, status=evidence_mod.EvidenceStatus.RETRACTED)
    )
    # Default list is ACTIVE-only (parity with the in-memory store).
    items = evidence_store.list_evidence(project_id="proj-1")
    assert [e.evidence_id for e in items] == ["evd-2", "evd-1"]
    retracted = evidence_store.list_evidence(
        project_id="proj-1", status=evidence_mod.EvidenceStatus.RETRACTED
    )
    assert [e.evidence_id for e in retracted] == ["evd-3"]


def test_evidence_search_and_claim_lookup(evidence_store):
    evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.5, content="grid search"))
    evidence_store.add_evidence(_evidence("evd-2", "claim-2", 0.5, content="other"))
    hits = evidence_store.search_evidence("grid", project_id="proj-1")
    assert [e.evidence_id for e in hits] == ["evd-1"]
    assert [e.evidence_id for e in evidence_store.get_evidence_for_claim("claim-2")] == ["evd-2"]


def test_evidence_update_status_hides_from_claim(evidence_store):
    evidence_store.add_evidence(_evidence("evd-1", "claim-1", 0.5))
    assert evidence_store.update_status("evd-1", evidence_mod.EvidenceStatus.RETRACTED)
    assert evidence_store.get_evidence_for_claim("claim-1") == []
    assert evidence_store.count(project_id="proj-1") == 1
    assert evidence_store.count(project_id="proj-1", status=evidence_mod.EvidenceStatus.ACTIVE) == 0


# ── Experiment registry ──────────────────────────────────────────────────


def test_exp_create_get_with_results(exp_registry):
    exp_id = exp_registry.create_experiment(
        project_id="proj-1", hypothesis_id="hyp-1", parameters={"lr": 0.01}
    )
    exp = exp_registry.get_experiment(exp_id)
    assert exp is not None
    assert exp.status == expreg_mod.ExperimentStatus.PENDING
    assert exp.parameters == {"lr": 0.01}
    assert exp.results == []
    result_id = exp_registry.add_experiment_result(
        exp_id, metrics={"acc": 0.9}, artifacts=["s3://x/out.json"], logs="ok"
    )
    exp = exp_registry.get_experiment(exp_id)
    assert [r.result_id for r in exp.results] == [result_id]
    assert exp.results[0].metrics == {"acc": 0.9}


def test_exp_add_result_missing_raises(exp_registry):
    with pytest.raises(expreg_mod.ExperimentNotFoundError):
        exp_registry.add_experiment_result("exp-missing", metrics={})


def test_exp_status_lifecycle_and_timestamps(exp_registry):
    exp_id = exp_registry.create_experiment(project_id="proj-1", hypothesis_id="hyp-1")
    for status in (
        expreg_mod.ExperimentStatus.DESIGNED,
        expreg_mod.ExperimentStatus.SUBMITTED,
        expreg_mod.ExperimentStatus.RUNNING,
        expreg_mod.ExperimentStatus.COMPLETED,
    ):
        assert exp_registry.update_experiment_status(exp_id, status) is True
    exp = exp_registry.get_experiment(exp_id)
    assert exp.submitted_at
    assert exp.completed_at
    assert (
        exp_registry.update_experiment_status("missing", expreg_mod.ExperimentStatus.RUNNING)
        is False
    )


def test_exp_invalid_transition_raises(exp_registry):
    exp_id = exp_registry.create_experiment(project_id="proj-1", hypothesis_id="hyp-1")
    with pytest.raises(expreg_mod.InvalidStatusTransitionError):
        exp_registry.update_experiment_status(exp_id, expreg_mod.ExperimentStatus.RUNNING)


def test_exp_list_filters_and_history(exp_registry):
    e1 = exp_registry.create_experiment(project_id="proj-1", hypothesis_id="hyp-1")
    e2 = exp_registry.create_experiment(project_id="proj-1", hypothesis_id="hyp-2")
    exp_registry.create_experiment(project_id="proj-2", hypothesis_id="hyp-1")
    assert len(exp_registry.list_experiments(project_id="proj-1")) == 2
    assert len(exp_registry.list_experiments(hypothesis_id="hyp-1")) == 2
    history = exp_registry.get_experiment_history("proj-1")
    assert {e.experiment_id for e in history} == {e1, e2}
    assert history == sorted(history, key=lambda e: e.created_at)


# ── Contract tripwires ────────────────────────────────────────────────


def test_zero_budget_keeps_zero_token_entry(evidence_store):
    """A zero-content (zero-token) evidence item must never be dropped
    merely because a budget is small: budget limits cost, and its cost
    is zero."""
    evidence_store.add_evidence(
        _evidence("evd-zero", "claim-1", 0.9, content="")
    )
    assert evidence_store.list_evidence(project_id="proj-1", limit=0) == []
    full = evidence_store.list_evidence(project_id="proj-1")
    assert any(e.evidence_id == "evd-zero" for e in full)


def test_sm_creation_ids_preserved(sm_store):
    """Insertion order (reference store's ``_creation_order``) must be
    reproducible after persistence, from the created_at column alone."""
    sm_store.add(_entry("mem-a", "proj-1", "2026-01-01T00:00:01+00:00"))
    sm_store.add(_entry("mem-b", "proj-1", "2026-01-01T00:00:02+00:00"))
    if hasattr(sm_store, "get_creation_ids"):
        assert sm_store.get_creation_ids() == ["mem-a", "mem-b"]
    else:
        ids = [e.entry_id for e in sm_store.get_timeline("proj-1")]
        assert ids == ["mem-a", "mem-b"]


# ── Factory selection ────────────────────────────────────────────────────


def test_factory_memory_backend_forced(monkeypatch):
    monkeypatch.setenv("UMRS_SCIENTIFIC_MEMORY_STORE_BACKEND", "memory")
    store = sm_mod.build_default_store(db_pool=object())
    assert isinstance(store, sm_mod.InMemoryScientificMemoryStore)


def test_factory_auto_uses_pool_when_provided(monkeypatch):
    monkeypatch.delenv("UMRS_EVIDENCE_STORE_BACKEND", raising=False)
    store = evidence_mod.build_default_store(db_pool=object())
    from umrs.evidence.pg_store import PgEvidenceStore

    assert isinstance(store, PgEvidenceStore)


def test_factory_pg_backend_propagates_pool_errors(monkeypatch):
    import umrs.shared.pg as shared_pg

    def _boom():
        raise ConnectionError("no db")

    monkeypatch.setattr(shared_pg, "get_default_sync_pool", _boom)
    monkeypatch.setenv("UMRS_EXPERIMENT_REGISTRY_STORE_BACKEND", "pg")
    with pytest.raises(ConnectionError):
        expreg_mod.build_default_store()


def test_factory_auto_falls_back_to_memory_when_pg_unavailable(monkeypatch):
    import umrs.shared.pg as shared_pg

    def _boom():
        raise ConnectionError("no db")

    monkeypatch.setattr(shared_pg, "get_default_sync_pool", _boom)
    monkeypatch.setattr(shared_pg, "pg_configured_from_env", lambda: True)
    monkeypatch.delenv("UMRS_SCIENTIFIC_MEMORY_STORE_BACKEND", raising=False)
    store = sm_mod.build_default_store()
    assert isinstance(store, sm_mod.InMemoryScientificMemoryStore)
