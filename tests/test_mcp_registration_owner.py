"""MCP registration ownership and server-pinned tool calls (2026-11-23 hunt).

#26: names were registry-unique and the upsert rebound an existing row to the
registering project. Any verified project caller could take over the global
``default-rag`` or another project's server (new URL, new owner) by
registering the same name. A first fix refused such writes with a 403, which
left one name space: the first project to register ``fetch`` blocked every
other project from it. Names are now unique per project (migration 0140):
a project writes, reads and deletes its own row, falls back to the global one
for reads, and never touches another owner's row.

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
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.mcp.models import MCPServer, MCPTool
from umrs.mcp.store import (
    MCPServerNotFoundError,
    MCPServerOwnershipError,
    MCPServerStore,
)

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
        server_id=f"mcp-{project or 'global'}-{name}",
        name=name,
        url=f"http://{name}.example.org",
        project_id=project,
        tools=[MCPTool(tool_id=f"t-{name}", server_id=f"mcp-{name}", name=tool)],
        metadata={"seq": seq},
    )


def _visible(srv: MCPServer, project: str | None) -> bool:
    return project is None or (srv.project_id or "") in ("", project)


def _rank(srv: MCPServer, project: str | None) -> tuple[bool, int]:
    """Own project's row first, then the most recent (the store's ORDER BY)."""
    return ((srv.project_id or "") == (project or ""), srv.metadata["seq"])


class _Store:
    """In-memory store with the real store's (project, name) key and lookups."""

    def __init__(self, servers: list[MCPServer]):
        self.rows = {(s.project_id or "", s.name): s for s in servers}

    def register_server(self, *, name, url, project_id=None, **_kw):
        row = _server(name, project_id, "t", seq=len(self.rows) + 1)
        row.url = url
        self.rows[(project_id or "", name)] = row
        return row

    def _named(self, name, project_id):
        found = [s for s in self.rows.values() if s.name == name and _visible(s, project_id)]
        return max(found, key=lambda s: _rank(s, project_id), default=None)

    def get_server_by_name(self, name, project_id=None):
        found = self._named(name, project_id)
        if found is None:
            raise MCPServerNotFoundError(name)
        return found

    def unregister_server(self, name, project_id=None):
        found = self._named(name, project_id)
        if found is None or (project_id is not None and found.project_id != project_id):
            return False
        del self.rows[(found.project_id or "", name)]
        return True

    def list_servers(self, **_filters):
        return list(self.rows.values())

    def list_tools(self, server_name=None):
        return [
            {"name": t.name, "description": "", "inputSchema": {}, "server": s.name}
            | {"server_id": s.server_id}
            for s in self.rows.values()
            if server_name in (None, s.name)
            for t in s.tools
        ]

    def get_server_for_tool(self, tool, server_name=None, visible_project=None):
        found = [
            s
            for s in self.rows.values()
            if any(t.name == tool for t in s.tools)
            and server_name in (None, s.name)
            and _visible(s, visible_project)
        ]
        return max(found, key=lambda s: _rank(s, visible_project), default=None)


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


@pytest.mark.parametrize("owner, name", [("", "default-rag"), ("proj-b", "b-tools")])
def test_a_taken_name_gives_the_project_its_own_row(api, owner, name):
    """Neither a takeover nor a refusal: proj-a gets its own ``name``."""
    before = api.store.rows[(owner, name)]
    resp = _register(api, name, _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert api.store.rows[(owner, name)] is before
    assert api.store.rows[("proj-a", name)].url == "http://attacker.example.org:9"


def test_project_may_update_its_own_server(api):
    resp = _register(api, "a-tools", _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert api.store.rows[("proj-a", "a-tools")].url == "http://attacker.example.org:9"


def test_project_may_register_a_new_name(api):
    assert _register(api, "a-new", _headers("alice", "proj-a")).status_code == 201
    assert api.store.rows[("proj-a", "a-new")].project_id == "proj-a"


def test_admin_updates_the_global_row(api):
    resp = _register(api, "default-rag", _headers("root", "ops", admin=True))
    assert resp.status_code == 201
    assert api.store.rows[("", "default-rag")].url == "http://attacker.example.org:9"


def _get(api, name: str, headers: dict[str, str]):
    return api.client.get(f"{PREFIX}/servers/{name}", headers=headers)


def test_by_name_reads_the_own_row_then_the_global_one(api):
    alice, bob = _headers("alice", "proj-a"), _headers("bob", "proj-b")
    assert _get(api, "default-rag", alice).json()["project_id"] is None
    _register(api, "default-rag", alice)
    assert _get(api, "default-rag", alice).json()["project_id"] == "proj-a"
    assert _get(api, "default-rag", bob).json()["project_id"] is None


def test_unregister_by_name_deletes_only_the_own_row(api):
    alice = _headers("alice", "proj-a")
    _register(api, "b-tools", alice)
    assert api.client.delete(f"{PREFIX}/servers/b-tools", headers=alice).status_code == 200
    assert ("proj-b", "b-tools") in api.store.rows
    assert ("proj-a", "b-tools") not in api.store.rows


def test_unregister_by_name_never_reaches_the_global_row(api):
    alice = _headers("alice", "proj-a")
    assert api.client.delete(f"{PREFIX}/servers/default-rag", headers=alice).status_code == 403
    assert ("", "default-rag") in api.store.rows


def test_tool_listing_hides_a_foreign_row_sharing_a_visible_name(api):
    """A visible global ``default-rag`` used to vouch, by name, for proj-b's
    ``default-rag`` tools too."""
    _register(api, "default-rag", _headers("bob", "proj-b"))
    resp = api.client.get(f"{PREFIX}/tools", headers=_headers("alice", "proj-a"))
    assert sorted(t["name"] for t in resp.json() if t["server"] == "default-rag") == ["search"]


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


def test_named_server_resolves_to_the_callers_own_row(api):
    _register(api, "b-tools", _headers("alice", "proj-a"))
    api.store.rows[("proj-a", "b-tools")].tools[0].name = "search"
    resp = _call(api, _headers("alice", "proj-a"), server="b-tools")
    assert resp.status_code == 200
    assert api.proxied == ["http://attacker.example.org:9"]


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


def test_store_upsert_conflicts_on_the_project_and_name_key():
    cur = _Cursor(returned=("mcp-1", None, None))
    MCPServerStore(_Conn(cur)).register_server("g", "http://g.example.org")
    assert "ON CONFLICT ((COALESCE(TRIM(project_id), '')), name)" in cur.sql[0]
    assert "mcp_servers.project_id" not in cur.sql[0]  # unguarded


def test_store_tool_lookup_filters_before_the_pick_and_prefers_the_own_row():
    cur = _Cursor(returned=None)
    MCPServerStore(_Conn(cur)).get_server_for_tool(
        "search", server_name="a-tools", visible_project="proj-a"
    )
    sql = cur.sql[0]
    assert "s.name = %s" in sql
    assert "COALESCE(TRIM(s.project_id), '') IN ('', %s)" in sql
    assert sql.index("IN ('', %s)") < sql.index("ORDER BY (COALESCE") < sql.index("LIMIT 1")
    assert cur.params[0] == ["search", "a-tools", "proj-a", "proj-a"]


# ── real Postgres (CI job's POSTGRES_* or UMRS_TEST_DB_URL) ─────────────────

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_REGISTRY = ("0001_mcp_registry.sql", "0105_mcp_placement_trust.sql")
_KEY_MIGRATION = "0140_mcp_servers_project_name.sql"
_NAME = "owner-test"


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


def _scratch_schema(conn, migrations: tuple[str, ...]) -> str:
    """A private schema with the MCP registry tables: the shared test DB's
    own mcp_servers (and other sessions using it) are never touched."""
    schema = f"mcp_owner_{uuid.uuid4().hex[:12]}"
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
        cur.execute(f'SET search_path TO "{schema}"')
        for name in migrations:
            cur.execute((_MIGRATIONS / name).read_text())
    conn.commit()
    return schema


@contextmanager
def _pg_conn(migrations: tuple[str, ...]):
    import psycopg2

    conn = psycopg2.connect(_db_url())
    schema = _scratch_schema(conn, migrations)
    try:
        yield conn
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
        conn.commit()
        conn.close()


@pytest.fixture
def pg_store():
    with _pg_conn((*_REGISTRY, _KEY_MIGRATION)) as conn:
        yield MCPServerStore(conn)


def _pg_register(store, project, *, name=_NAME, tool="t", guard=True):
    return store.register_server(
        name,
        f"http://{(project or 'global').strip()}.example.org",
        project_id=project,
        tools=[{"name": tool}],
        owner_guard=guard and project is not None,
    )


def _url_for(store, project) -> str:
    return store.get_server_by_name(_NAME, project_id=project).url


pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pg
def test_pg_projects_share_a_name_and_never_touch_each_others_row(pg_store):
    """The reviewer's case: proj-a holds ``fetch``; proj-b registers it too."""
    _pg_register(pg_store, None)
    _pg_register(pg_store, "proj-a")
    _pg_register(pg_store, "proj-b")
    _pg_register(pg_store, " proj-b ")  # padded: proj-b's row, not a fourth
    assert len(pg_store.list_servers()) == 3
    assert _url_for(pg_store, "proj-a") == "http://proj-a.example.org"
    assert _url_for(pg_store, "proj-b") == "http://proj-b.example.org"
    assert _url_for(pg_store, "proj-c") == "http://global.example.org"
    assert _url_for(pg_store, None) == "http://global.example.org"
    assert [t.name for t in pg_store.get_server_by_name(_NAME, project_id="proj-a").tools] == ["t"]


@pg
def test_pg_unregister_by_name_deletes_only_the_callers_row(pg_store):
    _pg_register(pg_store, None)
    _pg_register(pg_store, "proj-b")
    assert pg_store.unregister_server(_NAME, project_id="proj-a") is False
    assert pg_store.unregister_server(_NAME, project_id="proj-b") is True
    assert _url_for(pg_store, "proj-b") == "http://global.example.org"
    assert pg_store.unregister_server(_NAME) is True
    with pytest.raises(MCPServerNotFoundError):
        pg_store.get_server_by_name(_NAME, project_id="proj-b")


@pg
def test_pg_tool_lookup_picks_among_visible_rows_own_first(pg_store):
    _pg_register(pg_store, "proj-a", tool="owner-search")
    _pg_register(pg_store, None, tool="owner-search")  # later, global
    _pg_register(pg_store, "proj-b", tool="owner-search")  # later still, foreign
    found = pg_store.get_server_for_tool(
        "owner-search", server_name=_NAME, visible_project="proj-a"
    )
    assert found.project_id == "proj-a"
    found = pg_store.get_server_for_tool("owner-search", visible_project="proj-c")
    assert found.project_id is None


@pg
def test_pg_0140_moves_the_key_on_an_existing_registry():
    """Upgrade path: a registry migrated before 0140 (names registry-unique)."""
    with _pg_conn(_REGISTRY) as conn:
        store = MCPServerStore(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mcp_servers (server_id, name, url, project_id) "
                "VALUES ('mcp-old', %s, 'http://global.example.org', NULL)",
                (_NAME,),
            )
            cur.execute((_MIGRATIONS / _KEY_MIGRATION).read_text())
            cur.execute((_MIGRATIONS / _KEY_MIGRATION).read_text())  # idempotent
        conn.commit()
        _pg_register(store, "proj-a")
        assert _pg_register(store, None).server_id == "mcp-old"
        assert len(store.list_servers()) == 2
