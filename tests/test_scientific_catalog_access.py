"""Scientific Catalog access scopes (ADR-080) + the read-guard fix.

Before this wave ``GET /{catalog_id}`` and ``GET /versions/list`` had NO
scope check — any verified caller read any project's catalog rows. Now:
owning project, admin, or the row's access list (user:/project:/group:/
system, several entries; multi-membership admitted); anything else is 404
(no disclosure). Writes stay project-scoped.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.scientific_catalogs import ScientificCatalogStore, router


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        self._conn._handle(" ".join(str(sql).split()), params, self)

    def fetchone(self):
        return self._result.pop(0) if self._result else None

    def fetchall(self):
        out = self._result[:]
        self._result.clear()
        return out


def _tuple_from_dict(d: dict) -> tuple:
    return (
        d["catalog_id"], d["project_id"], d["name"], d["version"],
        d["description"], d["ontology"], d["member_graph_ids"], d["skill_ids"],
        d["mcp_server_ids"], d["embedding_model_ids"], d.get("mcp_trust_classes", {}),
        d["contributors"], d["licence"], d["provenance"], d["change_note"],
        d["redaction_rules"], d["owner_id"], d["status"],
        d["created_at"], d["updated_at"], d.get("access", []),
    )


class _Conn:
    """psycopg-shaped fake over the 0116 column set (access included)."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.groups: list[str] = []
        self.record: list[str] = []

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    # -- SQL routing ------------------------------------------------------
    # (predicate on the SQL text, handler) in match order; anything else is
    # the active list (project-scoped or all).
    _ROUTES = (
        (lambda sql: sql.startswith("INSERT"), "_insert"),
        (lambda sql: sql.startswith("UPDATE"), "_update_access"),
        (lambda sql: "FROM group_members" in sql, "_group_rows"),
        (lambda sql: "WHERE catalog_id = %s" in sql, "_one_row"),
        (lambda sql: "access != '[]'::jsonb" in sql, "_shared_rows"),
        (lambda sql: "ORDER BY created_at, catalog_id" in sql, "_name_versions"),
    )

    def _handle(self, sql, params, cur: _Cursor) -> None:
        self.record.append(sql)
        for matches, handler in self._ROUTES:
            if matches(sql):
                getattr(self, handler)(params, cur)
                return
        self._active_rows(params, cur)

    def _update_access(self, params, cur: _Cursor) -> None:
        self.rows[params[1]]["access"] = json.loads(params[0])
        cur._result = []

    def _group_rows(self, _params, cur: _Cursor) -> None:
        cur._result = [(g,) for g in self.groups]

    def _one_row(self, params, cur: _Cursor) -> None:
        row = self.rows.get(params[0])
        cur._result = [_tuple_from_dict(row)] if row else []

    def _shared_rows(self, _params, cur: _Cursor) -> None:
        cur._result = [
            _tuple_from_dict(v) for v in self.rows.values()
            if v["status"] != "retired" and v["access"]
        ]

    def _name_versions(self, params, cur: _Cursor) -> None:
        rows = [
            v for v in self.rows.values()
            if v["project_id"] == params[0] and v["name"] == params[1]
        ]
        cur._result = [_tuple_from_dict(v) for v in rows]

    def _active_rows(self, params, cur: _Cursor) -> None:
        rows = [v for v in self.rows.values() if v["status"] != "retired"]
        if params:
            rows = [v for v in rows if v["project_id"] == params[0]]
        cur._result = [_tuple_from_dict(v) for v in rows]

    def _insert(self, params, cur: _Cursor) -> None:
        now = datetime.now(timezone.utc)
        self.rows[params[0]] = {
            "catalog_id": params[0], "project_id": params[1], "name": params[2],
            "version": params[3], "description": params[4], "ontology": params[5],
            "member_graph_ids": params[6], "skill_ids": params[7],
            "mcp_server_ids": params[8], "embedding_model_ids": params[9],
            "mcp_trust_classes": params[10], "contributors": params[11],
            "licence": params[12], "provenance": params[13], "change_note": params[14],
            "redaction_rules": params[15], "owner_id": params[16],
            "status": "active", "created_at": now, "updated_at": now, "access": [],
        }
        cur._result = []


def _row(conn: _Conn, catalog_id: str, project: str, name: str = "mof-defects",
         version: str = "0.1.0", access: list | None = None) -> dict:
    now = datetime.now(timezone.utc)
    row = {
        "catalog_id": catalog_id, "project_id": project, "name": name,
        "version": version, "description": "", "ontology": "",
        "member_graph_ids": [], "skill_ids": [], "mcp_server_ids": [],
        "embedding_model_ids": [], "mcp_trust_classes": {}, "contributors": [],
        "licence": "", "provenance": {}, "change_note": "",
        "redaction_rules": [], "owner_id": "alice",
        "status": "active", "created_at": now, "updated_at": now,
        "access": access or [],
    }
    conn.rows[catalog_id] = row
    return row


def _client(conn: _Conn) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    pool = MagicMock()
    pool.getconn.return_value = conn
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _headers(user: str, project: str = "proj-a", admin: bool = False) -> dict[str, str]:
    headers = {"X-User-ID": user, "X-Verified-Project-ID": project}
    if admin:
        headers["X-Verified-Is-Admin"] = "true"
    return headers


class TestCrossProjectReads:
    def test_plain_row_is_invisible_across_projects(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-b")
        r = _client(conn).get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("alice")
        )
        assert r.status_code == 404  # the leak fix: was an unguarded 200

    def test_own_project_still_reads(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-a")
        r = _client(conn).get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("alice")
        )
        assert r.status_code == 200

    def test_user_scope_opens_one_named_user(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-b", access=["user:bob"])
        client = _client(conn)
        assert client.get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("bob", "proj-b")
        ).status_code == 200
        assert client.get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("heidi", "proj-c")
        ).status_code == 404

    def test_group_scope_via_active_membership(self):
        conn = _Conn()
        conn.groups = ["lab"]
        _row(conn, "scat-x", "proj-b", access=["group:lab"])
        client = _client(conn)
        assert client.get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("carol", "proj-c")
        ).status_code == 200
        assert any("FROM group_members" in sql for sql in conn.record)

    def test_system_scope_reads_for_everyone(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-b", access=["system"])
        assert _client(conn).get(
            "/umrs/scientific-catalogs/scat-x", headers=_headers("heidi", "proj-c")
        ).status_code == 200

    def test_list_shows_own_plus_shared_only(self):
        conn = _Conn()
        _row(conn, "scat-own", "proj-b", name="own")
        _row(conn, "scat-shared", "proj-a", name="shared", access=["user:bob"])
        _row(conn, "scat-hidden", "proj-a", name="hidden")
        r = _client(conn).get(
            "/umrs/scientific-catalogs", headers=_headers("bob", "proj-b")
        )
        names = sorted(row["name"] for row in r.json()["catalogs"])
        assert names == ["own", "shared"]

    def test_versions_chain_is_filtered(self):
        conn = _Conn()
        _row(conn, "scat-v1", "proj-b", version="0.1.0")
        _row(conn, "scat-v2", "proj-b", version="0.2.0", access=["user:alice"])
        r = _client(conn).get(
            "/umrs/scientific-catalogs/versions/list",
            params={"project_id": "proj-b", "name": "mof-defects"},
            headers=_headers("alice"),
        )
        assert [v["version"] for v in r.json()["versions"]] == ["0.2.0"]


class TestSetCatalogAccess:
    def test_project_member_sets_users_and_groups(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-a")
        client = _client(conn)
        r = client.put(
            "/umrs/scientific-catalogs/scat-x/access",
            json={"access": ["user:bob", "group:lab"]},
            headers=_headers("alice"),
        )
        assert r.status_code == 200, r.text
        assert r.json()["access"] == ["user:bob", "group:lab"]
        assert conn.rows["scat-x"]["access"] == ["user:bob", "group:lab"]

    def test_foreign_project_cannot_set(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-b")
        r = _client(conn).put(
            "/umrs/scientific-catalogs/scat-x/access",
            json={"access": ["user:bob"]},
            headers=_headers("alice"),
        )
        assert r.status_code == 403

    def test_system_is_admin_only(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-a")
        client = _client(conn)
        assert client.put(
            "/umrs/scientific-catalogs/scat-x/access",
            json={"access": ["system"]}, headers=_headers("alice"),
        ).status_code == 403
        assert client.put(
            "/umrs/scientific-catalogs/scat-x/access",
            json={"access": ["system"]}, headers=_headers("ops", admin=True),
        ).status_code == 200

    def test_access_route_visible_to_shared_readers(self):
        conn = _Conn()
        _row(conn, "scat-x", "proj-b", access=["user:bob"])
        r = _client(conn).get(
            "/umrs/scientific-catalogs/scat-x/access", headers=_headers("bob", "proj-b")
        )
        assert r.status_code == 200
        assert r.json()["access"] == ["user:bob"]


def test_store_set_access_sql():
    conn = _Conn()
    _row(conn, "scat-x", "proj-a")
    store = ScientificCatalogStore(conn)
    out = store.set_access("scat-x", ["system"])
    assert out["access"] == ["system"]
    assert any("SET access" in sql for sql in conn.record)
