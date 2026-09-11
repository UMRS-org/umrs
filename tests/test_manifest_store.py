"""JobStore manifest persistence + read-back SQL shapes (P7a)."""

from __future__ import annotations

import json

from umrs.jobs.store import JobStore


class _FakeCursor:
    def __init__(self, store):
        self._rows = store
        self._one = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        lowered = sql.strip().lower()
        if lowered.startswith("update job_attempts") and "metrics" in lowered:
            payload = json.loads(params[0])
            self._rows[params[1]] = payload.get("manifest")
            return
        if "metrics ->> 'manifest'" in lowered:
            manifest = self._rows.get(params[0])
            self._one = {"manifest": manifest}

    def fetchone(self):
        return self._one


class _FakeConn:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self):
        return None


class _FakePool:
    def __init__(self):
        self.rows = {}
        self.conn = _FakeConn(self.rows)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        return None


class _TestStore(JobStore):
    """JobStore over the fake pool; schema ensure is not this test's subject."""

    def _ensure_schema(self):
        return None


def test_persist_and_read_back_manifest(monkeypatch):
    pool = _FakePool()
    store = _TestStore(pool)
    monkeypatch.setattr(
        store,
        "get_job",
        lambda _id: type(
            "Job",
            (),
            {"spec": {"workflow_name": "demo08"}, "commit_attempt_id": "att-1"},
        )(),
    )
    monkeypatch.setattr(store, "get_attempts_for_job", lambda _id: [])
    import umrs.jobs.provenance

    monkeypatch.setattr(
        umrs.jobs.provenance,
        "build_provenance",
        lambda job, attempts: type(
            "Prov", (), {"to_dict": lambda self: {"job_id": "job-1"}}
        )(),
    )

    store._persist_attempt_manifest("att-1", "job-1")
    manifest = store.get_attempt_manifest("att-1")
    assert manifest is not None
    assert manifest["environment"]["python_version"]
    assert manifest["trace_ids"] == []
    assert "provenance" in manifest


def test_manifest_absent_returns_none():
    pool = _FakePool()
    store = _TestStore(pool)
    assert store.get_attempt_manifest("att-missing") is None


def test_persist_is_best_effort_under_failure(monkeypatch):
    pool = _FakePool()
    store = _TestStore(pool)

    def boom(_id):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "get_job", boom)
    # must not raise — manifest failure never blocks the terminal state
    store._persist_attempt_manifest("att-2", "job-2")
    assert store.get_attempt_manifest("att-2") is None


def test_venue_enrichment_attaches_only_via_wired_store(monkeypatch):
    """WP-10 r2: venue active_version lands on spec-pinned names.

    Without a wired VectorVersionStore the manifest carries spec pins only
    (never fabricated availability). With one pinned at boot, its active
    version attaches to the matching pin name.
    """

    class _VVS:
        def get_active_version(self, name):
            if name == "bge-m3":
                return type("V", (), {"model_version": "v2026-08"})()
            return None

    def _store(monkeypatch, vvs):
        pool = _FakePool()
        store = _TestStore(pool)
        store.vector_version_store = vvs
        monkeypatch.setattr(
            store,
            "get_job",
            lambda _id: type(
                "Job", (), {"spec": {"embedding_model": "bge-m3"}, "commit_attempt_id": "a1"}
            )(),
        )
        monkeypatch.setattr(store, "get_attempts_for_job", lambda _id: [])
        import umrs.jobs.provenance

        monkeypatch.setattr(
            umrs.jobs.provenance,
            "build_provenance",
            lambda job, attempts: type("Prov", (), {"to_dict": lambda self: {"job_id": "j"}})(),
        )
        return store

    store_novir = _store(monkeypatch, None)
    store_novir._persist_attempt_manifest("att-a", "job-x")
    manifest_a = store_novir.get_attempt_manifest("att-a")
    assert manifest_a["model_pins"][0]["name"] == "bge-m3"
    assert "venue_active_version" not in manifest_a["model_pins"][0]

    store_wired = _store(monkeypatch, _VVS())
    store_wired._persist_attempt_manifest("att-b", "job-y")
    manifest_b = store_wired.get_attempt_manifest("att-b")
    assert manifest_b["model_pins"][0]["venue_active_version"] == "v2026-08"
