"""MCP registration ownership and server-pinned tool calls (2026-11-23 hunt).

#26: names are registry-unique and the upsert rebound an existing row to the
registering project. Any verified project caller could take over the global
``default-rag`` or another project's server (new URL, new owner) by
registering the same name. A non-admin registration now updates only a row
its project already owns; admins keep the rebind.

#27: ``POST /mcp/invoke`` resolved a tool with a global ``LIMIT 1`` and then
filtered visibility, so another project's later server with the same tool
name made the caller's own tool read as "not found", and ``mcp.<A>.<tool>``
could not name server A. The call now takes an optional ``server`` and the
pick runs among the caller's visible servers.

The route tests use a fake store with the real store's semantics; the store
tests check the SQL guard (fake cursor) and, where the CI job's Postgres is
configured, the real upsert.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.mcp.models import MCPServer, MCPTool
from umrs.mcp.store import MCPServerOwnershipError, MCPServerStore

from umrs_protocol.handshake_hmac import sign_claims

SECRET = "mcp-owner-secret"
PREFIX = "/umrs/mcp"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _server(name: str, project: str | None, tool: str, seq: int) -> MCPServer:
    return MCPServer(
        server_id=f"mcp-{name}",
        name=name,
        url=f"http://{name}.example.org",
        project_id=project,
        tools=[MCPTool(tool_id=f"t-{name}", server_id=f"mcp-{name}", name=tool)],
        metadata={"seq": seq},
    )


def _visible(srv: MCPServer, project: str | None) -> bool:
    return project is None or (srv.project_id or "") in ("", project)


class _Store:
    """In-memory store with the real store's ownership and lookup rules."""

    def __init__(self, servers: list[MCPServer]):
        self.rows = {s.name: s for s in servers}

    def register_server(self, *, name, url, project_id=None, owner_guard=False, **_kw):
        old = self.rows.get(name)
        if old and owner_guard and (old.project_id or "") != (project_id or ""):
            raise MCPServerOwnershipError(name)
        self.rows[name] = _server(name, project_id, "t", seq=len(self.rows) + 1)
        self.rows[name].url = url
        return self.rows[name]

    def get_server_for_tool(self, tool, server_name=None, visible_project=None):
        found = [
            s
            for s in self.rows.values()
            if any(t.name == tool for t in s.tools)
            and server_name in (None, s.name)
            and _visible(s, visible_project)
        ]
        return max(found, key=lambda s: s.metadata["seq"], default=None)


@pytest.fixture
def api(monkeypatch):
    from umrs.api import mcp

    store = _Store(
        [
            _server("default-rag", None, "search", 1),
            _server("a-tools", "proj-a", "search", 2),
            _server("b-tools", "proj-b", "search", 3),
        ]
    )
    monkeypatch.setattr(mcp, "_get_store", lambda: store)
    proxied: list[str] = []

    async def _fake_proxy(url, *_args, **_kwargs):
        proxied.append(url)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(mcp, "_proxy_tool_call", _fake_proxy)
    app = FastAPI()
    app.include_router(mcp.router)
    return SimpleNamespace(client=TestClient(app), store=store, proxied=proxied)


def _register(api, name: str, headers: dict[str, str]):
    body = {"name": name, "url": "http://attacker.example.org:9", "tools": []}
    return api.client.post(f"{PREFIX}/servers", json=body, headers=headers)


# ── #26 registration ownership ──────────────────────────────────────────────


@pytest.mark.parametrize("name", ["default-rag", "b-tools"])
def test_project_cannot_take_over_a_name_it_does_not_own(api, name):
    before = api.store.rows[name]
    resp = _register(api, name, _headers("alice", "proj-a"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "scope_denied"
    assert api.store.rows[name] is before


def test_project_may_update_its_own_server(api):
    resp = _register(api, "a-tools", _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert api.store.rows["a-tools"].url == "http://attacker.example.org:9"


def test_project_may_register_a_new_name(api):
    assert _register(api, "a-new", _headers("alice", "proj-a")).status_code == 201
    assert api.store.rows["a-new"].project_id == "proj-a"


def test_admin_keeps_the_rebind_path(api):
    resp = _register(api, "default-rag", _headers("root", "ops", admin=True))
    assert resp.status_code == 201


# ── #27 tool calls pinned to a server, picked among visible servers ─────────


def _call(api, headers: dict[str, str], server: str | None = None):
    body: dict = {"tool": "search", "arguments": {}}
    if server is not None:
        body["server"] = server
    return api.client.post(f"{PREFIX}/invoke", json=body, headers=headers)


def test_foreign_later_registration_does_not_shadow_own_tool(api):
    resp = _call(api, _headers("alice", "proj-a"))
    assert resp.status_code == 200
    assert resp.json()["server"] == "a-tools"


def test_named_server_is_the_one_invoked(api):
    resp = _call(api, _headers("alice", "proj-a"), server="default-rag")
    assert resp.status_code == 200
    assert resp.json()["server"] == "default-rag"
    assert api.proxied == ["http://default-rag.example.org"]


def test_naming_a_foreign_server_reads_as_not_found(api):
    resp = _call(api, _headers("alice", "proj-a"), server="b-tools")
    assert resp.status_code == 404
    assert api.proxied == []


# ── store: the guard lives in the upsert itself ─────────────────────────────


class _Cursor:
    def __init__(self, returned):
        self.sql: list[str] = []
        self.params: list = []
        self._returned = returned

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))
        self.params.append(params)

    def fetchone(self):
        return self._returned

    def fetchall(self):
        return []


class _Conn:
    closed = 0

    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def get_transaction_status(self):
        return 0  # TRANSACTION_STATUS_IDLE

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def test_store_guarded_upsert_refused_raises_and_commits_nothing():
    cur = _Cursor(returned=None)  # ON CONFLICT ... WHERE false returns no row
    conn = _Conn(cur)
    with pytest.raises(MCPServerOwnershipError):
        MCPServerStore(conn).register_server(
            "default-rag", "http://x.example.org", project_id="proj-a", owner_guard=True
        )
    assert "WHERE COALESCE(TRIM(mcp_servers.project_id), '')" in cur.sql[0]
    assert not conn.committed
    assert len(cur.sql) == 1  # no tool rows touched


def test_store_unguarded_upsert_keeps_the_admin_rebind():
    cur = _Cursor(returned=("mcp-1", None, None))
    MCPServerStore(_Conn(cur)).register_server("g", "http://g.example.org")
    assert "mcp_servers.project_id" not in cur.sql[0]


def test_store_tool_lookup_filters_before_the_pick():
    cur = _Cursor(returned=None)
    MCPServerStore(_Conn(cur)).get_server_for_tool(
        "search", server_name="a-tools", visible_project="proj-a"
    )
    sql = cur.sql[0]
    assert "s.name = %s" in sql
    assert "COALESCE(TRIM(s.project_id), '') IN ('', %s)" in sql
    assert sql.index("IN ('', %s)") < sql.index("LIMIT 1")
    assert cur.params[0] == ["search", "a-tools", "proj-a"]


# ── real Postgres (CI job's POSTGRES_* or UMRS_TEST_DB_URL) ─────────────────

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_NAMES = ("owner-test-global", "owner-test-b")


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


def _delete_test_rows(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM mcp_servers WHERE name = ANY(%s)", (list(_NAMES),))
    conn.commit()


@pytest.fixture
def pg_store():
    import psycopg2

    conn = psycopg2.connect(_db_url())
    with conn.cursor() as cur:
        for name in ("0001_mcp_registry.sql", "0105_mcp_placement_trust.sql"):
            cur.execute((_MIGRATIONS / name).read_text())
    conn.commit()
    _delete_test_rows(conn)
    yield MCPServerStore(conn)
    _delete_test_rows(conn)
    conn.close()


def _pg_register(store, name, project, *, guard, tool="t"):
    return store.register_server(
        name,
        f"http://{project or 'global'}.example.org",
        project_id=project,
        tools=[{"name": tool}],
        owner_guard=guard,
    )


@pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")
def test_pg_guarded_upsert_leaves_foreign_rows_untouched(pg_store):
    _pg_register(pg_store, "owner-test-global", None, guard=False)
    _pg_register(pg_store, "owner-test-b", "proj-b", guard=True)
    for name in _NAMES:
        with pytest.raises(MCPServerOwnershipError):
            _pg_register(pg_store, name, "proj-a", guard=True)
    assert pg_store.get_server_by_name("owner-test-global").project_id is None
    kept = pg_store.get_server_by_name("owner-test-b")
    assert (kept.project_id, kept.url) == ("proj-b", "http://proj-b.example.org")
    assert [t.name for t in kept.tools] == ["t"]
    assert _pg_register(pg_store, "owner-test-b", "proj-b", guard=True).project_id == "proj-b"


@pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")
def test_pg_tool_lookup_ignores_foreign_later_registration(pg_store):
    _pg_register(pg_store, "owner-test-global", None, guard=False, tool="owner-search")
    _pg_register(pg_store, "owner-test-b", "proj-b", guard=True, tool="owner-search")
    found = pg_store.get_server_for_tool("owner-search", visible_project="proj-a")
    assert found.name == "owner-test-global"
    pinned = pg_store.get_server_for_tool("owner-search", server_name="owner-test-b")
    assert pinned.name == "owner-test-b"
