"""D-43 pins: durable jobs auto-pin onto the project's single active catalog."""

from __future__ import annotations

import json
from contextlib import contextmanager

from umrs.api import jobs as jobs_mod


class _FakeJobStore:
    def __init__(self, conn):
        self._conn = conn

    @contextmanager
    def _get_conn(self):
        yield self._conn


class _FakeCatConn:
    """psycopg-shaped conn emulating scientific_catalog list+pin."""

    def __init__(self, catalogs):
        self.catalogs = catalogs
        self.pins: list[tuple] = []

    def commit(self):
        pass

    def cursor(self):
        return _Cursor(self)


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        if "jsonb_set" in sql:
            pin_arr, cid = params
            self._conn.pins.append((cid, pin_arr))
            self._result = []
        else:
            self._result = self._select(sql, params)

    def _select(self, sql, params):
        if "WHERE catalog_id = %s" in sql:
            return self._rows("catalog_id", params[0])[:1]
        if "FROM scientific_catalogs" in sql:
            return self._rows("project_id", params[0] if params else None)
        return []

    def _rows(self, key, value):
        return [_row_tuple(c) for c in self._conn.catalogs if value is None or c[key] == value]

    def fetchone(self):
        return self._result.pop(0) if self._result else None

    def fetchall(self):
        out, self._result = self._result[:], []
        return out

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _row_tuple(c: dict) -> tuple:
    import json as _json

    return (
        c["catalog_id"],
        c["project_id"],
        c["name"],
        c["version"],
        "",
        "",
        "[]",
        "[]",
        "[]",
        "[]",
        "[]",
        "",
        _json.dumps(c.get("provenance") or {}),
        "",
        "[]",
        "owner",
        "active",
        None,
        None,
    )


def _catalog(cid: str, pid: str = "p1", name: str = "sc") -> dict:
    return {
        "catalog_id": cid,
        "project_id": pid,
        "name": name,
        "version": "0.1.0",
        "provenance": {"pins": []},
    }


def test_single_active_catalog_gets_the_pin() -> None:
    conn = _FakeCatConn([_catalog("sc-1")])
    jobs_mod._autopin_catalog(_FakeJobStore(conn), "p1", "j-9")
    assert conn.pins == [("sc-1", '[{"kind": "job", "ref": "j-9", "note": ""}]')]


def test_zero_catalogs_is_a_no_op() -> None:
    conn = _FakeCatConn([])
    jobs_mod._autopin_catalog(_FakeJobStore(conn), "p1", "j-9")
    assert conn.pins == []


def test_ambiguity_is_never_silently_guessed() -> None:
    conn = _FakeCatConn([_catalog("sc-1", pid="p1"), _catalog("sc-2", pid="p1")])
    jobs_mod._autopin_catalog(_FakeJobStore(conn), "p1", "j-9")
    assert conn.pins == []


def test_store_error_never_blocks(caplog) -> None:
    class _Boom(_FakeJobStore):
        @contextmanager
        def _get_conn(self):
            raise RuntimeError("pool down")
            yield

    jobs_mod._autopin_catalog(_Boom(None), "p1", "j-9")
    assert any("autopin skipped" in r.getMessage() for r in caplog.records)


# ── D-43 result-kind auto-pins (monitor, on attempt success) ──────────


class _FakeJob:
    project_id = "p1"


class _FakeMonitorStore:
    """JobStore double: project lookup + jobs-DB conn."""

    def __init__(self, conn):
        self._conn = conn

    def get_job(self, _job_id):
        return _FakeJob()

    @contextmanager
    def _get_conn(self):
        yield self._conn


class _Artifact:
    def __init__(self, art_id):
        self.id = art_id


def _monitor(conn):
    from umrs.jobs.monitor import AttemptMonitor

    return AttemptMonitor(_FakeMonitorStore(conn), status_checker=None)


def test_result_autopin_pins_all_artifacts() -> None:
    conn = _FakeCatConn([_catalog("sc-1")])
    mon = _monitor(conn)
    mon._result_artifacts = lambda _job_id: [_Artifact("a-1"), _Artifact("a-2")]
    mon._autopin_result_artifacts(type("A", (), {"job_id": "j-9"})())
    kinds_refs = [(json.loads(arr)[0]["kind"], json.loads(arr)[0]["ref"]) for _, arr in conn.pins]
    assert kinds_refs == [("result", "a-1"), ("result", "a-2")]
    assert all(json.loads(arr)[0]["note"] == "job j-9" for _, arr in conn.pins)


def test_result_autopin_skips_already_pinned_refs() -> None:
    conn = _FakeCatConn([_catalog("sc-1")])
    conn.catalogs[0]["provenance"] = {"pins": [{"kind": "result", "ref": "a-1", "note": "old"}]}
    mon = _monitor(conn)
    mon._result_artifacts = lambda _job_id: [_Artifact("a-1"), _Artifact("a-2")]
    mon._autopin_result_artifacts(type("A", (), {"job_id": "j-9"})())
    refs = [json.loads(arr)[0]["ref"] for _, arr in conn.pins]
    assert refs == ["a-2"]  # a-1 deduped


def test_result_autopin_never_guesses_ambiguous_catalogs() -> None:
    conn = _FakeCatConn([_catalog("sc-1", pid="p1"), _catalog("sc-2", pid="p1")])
    mon = _monitor(conn)
    mon._result_artifacts = lambda _job_id: [_Artifact("a-1")]
    mon._autopin_result_artifacts(type("A", (), {"job_id": "j-9"})())
    assert conn.pins == []


def test_result_autopin_no_artifacts_is_a_no_op() -> None:
    conn = _FakeCatConn([_catalog("sc-1")])
    mon = _monitor(conn)
    mon._result_artifacts = lambda _job_id: []
    mon._autopin_result_artifacts(type("A", (), {"job_id": "j-9"})())
    assert conn.pins == []


def test_result_autopin_registry_failure_never_blocks(caplog) -> None:
    conn = _FakeCatConn([_catalog("sc-1")])

    def _boom(_job_id):
        raise RuntimeError("registry down")

    mon = _monitor(conn)
    mon._result_artifacts = _boom
    mon._autopin_result_artifacts(type("A", (), {"job_id": "j-9"})())
    assert conn.pins == []
    assert any("result autopin skipped" in r.getMessage() for r in caplog.records)


def test_artifact_query_supports_job_id_filter() -> None:
    """The additive ArtifactQuery.job_id filter reaches the SQL conditions."""
    from umrs.artifact_registry.models import ArtifactQuery
    from umrs.artifact_registry.store import ArtifactStore

    captured = {}

    class _Cur:
        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params

        def __enter__(self):
            return self

        def fetchall(self):
            return []

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

    store = ArtifactStore(lambda: _Conn())
    store.list(ArtifactQuery(artifact_type="result", job_id="j-9"))
    assert "job_id = %s" in captured["sql"]
    assert "j-9" in captured["params"]
