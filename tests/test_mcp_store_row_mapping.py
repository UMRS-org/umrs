"""MCPServerStore row mapping matches every SELECT (2026-09-25).

Live defect: `_row_to_server` read created_at from the placement slot, so
GET /umrs/mcp/servers answered 500 ('str' has no .isoformat) once a server
existed, and `get_server_for_tool` selected no placement/trust_class at all —
the D-19 call-time refusal of trusted-host-only servers could never fire.
The fake cursor below builds each row from the executed SELECT's own column
list, so a query and the mapper can no longer drift apart silently.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from umrs.mcp.store import MCPServerStore

CREATED = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)

SERVER = {
    "server_id": "srv-1",
    "name": "lab-scope",
    "version": "1.0.0",
    "url": "http://127.0.0.1:9901/mcp",
    "transport": "http",
    "status": "healthy",
    "description": "scope control",
    "auth_type": "none",
    "auth_config": "{}",
    "metadata": "{}",
    "registered_by": "admin",
    "project_id": None,
    "placement": "trusted:lab-scope-1",
    "trust_class": "trusted-host-only",
    "created_at": CREATED,
    "updated_at": UPDATED,
}


class _Cursor:
    def __init__(self):
        self._rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, _params=None):
        text = " ".join(sql.split())
        if "FROM mcp_tools WHERE server_id" in text:
            self._rows = []  # no tools needed for the mapping check
            return
        select = re.search(r"SELECT (.*?) FROM mcp_servers", text)
        assert select, text
        cols = [c.strip().split(".")[-1] for c in select.group(1).split(",")]
        self._rows = [tuple(SERVER[c] for c in cols)]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _Conn:
    closed = 0

    def get_transaction_status(self):
        return 0  # TRANSACTION_STATUS_IDLE

    def cursor(self):
        return _Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture
def store():
    return MCPServerStore(_Conn())


def _assert_mapped(server):
    assert server.placement == "trusted:lab-scope-1"
    assert server.trust_class == "trusted-host-only"
    assert server.created_at == CREATED
    assert server.updated_at == UPDATED


def test_get_server_maps_every_column(store):
    _assert_mapped(store.get_server("srv-1"))


def test_get_server_by_name_maps_every_column(store):
    _assert_mapped(store.get_server_by_name("lab-scope"))


def test_list_servers_maps_every_column(store):
    (server,) = store.list_servers()
    _assert_mapped(server)


def test_call_time_lookup_carries_the_trust_class(store):
    _assert_mapped(store.get_server_for_tool("focus"))


def test_list_servers_response_serialises(store):
    from umrs.api.mcp import _server_to_response

    (server,) = store.list_servers()
    resp = _server_to_response(server)
    assert resp.created_at == CREATED.isoformat()


@pytest.mark.asyncio
async def test_call_tool_refuses_trusted_host_only_through_the_real_store(store, monkeypatch):
    import umrs.api.mcp as api
    from fastapi import HTTPException
    from umrs.knowledge.access_scope import CallerScope

    monkeypatch.setattr(api, "_store", store)
    # call/invoke resolve the caller (H-1); an admin sees every server.
    monkeypatch.setattr(api, "_caller_scope", lambda _req: CallerScope(is_admin=True))
    request = api.ToolCallRequest(tool="focus", arguments={})
    with pytest.raises(HTTPException) as excinfo:
        await api.call_tool(request, object())
    # Refused by name before any proxy attempt (the old mapping proxied it).
    assert excinfo.value.status_code == 403
    assert "trusted-host-only server" in str(excinfo.value.detail)


def test_call_time_lookup_prefers_a_healthy_recent_server():
    captured = {}

    class _RecordingCursor(_Cursor):
        def execute(self, sql, _params=None):
            if "JOIN mcp_tools" in sql:
                captured["sql"] = " ".join(sql.split())
            super().execute(sql, _params)

    class _Conn2(_Conn):
        def cursor(self):
            return _RecordingCursor()

    MCPServerStore(_Conn2()).get_server_for_tool("focus")
    assert "ORDER BY (s.status = 'healthy') DESC, s.updated_at DESC" in captured["sql"]
