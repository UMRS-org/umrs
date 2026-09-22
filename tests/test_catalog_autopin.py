"""D-43 pins: durable jobs auto-pin onto the project's single active catalog."""

from __future__ import annotations

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
        elif "WHERE catalog_id = %s" in sql:
            row = [c for c in self._conn.catalogs if c["catalog_id"] == params[0]]
            self._result = [_row_tuple(row[0])] if row else []
        elif "FROM scientific_catalogs" in sql:
            rows = self._conn.catalogs
            if params:
                rows = [c for c in rows if c["project_id"] == params[0]]
            self._result = [_row_tuple(c) for c in rows]
        else:
            self._result = []

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
        c["catalog_id"], c["project_id"], c["name"], c["version"],
        "", "", "[]", "[]", "[]", "[]", "[]", "",
        _json.dumps(c.get("provenance") or {}), "", "[]", "owner", "active",
        None, None,
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
