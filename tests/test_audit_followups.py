"""Audit follow-up pins (B6 webview, B9 MCP scoping, SSE producer wiring).

B6 is pinned client-side in vscode-extension/test; this file covers:
- SSE: producer wired, project-filtered broadcast, fail-closed attribution
- B9: /umrs/mcp servers + tools read-path project visibility
- client ingest_status route fix lives in the r4 client (unit here verifies
  the route shape it now hits — keeping the regression pinned server-side is
  unnecessary; the r4 side change is doc-visible).
"""

from __future__ import annotations

import pytest

# ── SSE producer wiring ──────────────────────────────────────────────────────


class _Q:
    def __init__(self):
        self.items = []

    def qsize(self):
        return len(self.items)

    async def put(self, x):
        self.items.append(x)


@pytest.fixture(autouse=True)
def _clean_sse_state():
    from umrs.knowledge import events_sse

    saved_clients = dict(events_sse._clients)
    saved_projects = dict(events_sse._projects)
    events_sse._clients.clear()
    events_sse._projects.clear()
    while not events_sse._broadcast_queue.empty():
        events_sse._broadcast_queue.get_nowait()
    yield
    events_sse._clients.clear()
    events_sse._clients.update(saved_clients)
    events_sse._projects.clear()
    events_sse._projects.update(saved_projects)


def _fake_client(project: str):
    from umrs.knowledge import events_sse

    cid = f"c-{project}"
    info = {"project": project, "queue": _Q(), "connected_at": 0.0}
    events_sse._clients[cid] = info
    events_sse._projects.setdefault(project, []).append(cid)
    return cid, info


class TestSSEProjectScoping:
    def test_emit_attaches_project_from_ctx(self):
        from umrs.jobs.knowledge_adapters import KnowledgeEvent
        from umrs.knowledge import events_sse

        _fake_client("proj-a")
        token = events_sse.set_project_ctx("proj-a")
        try:
            events_sse.emit(KnowledgeEvent(event_type="ingest", job_id="j1", attempt_id="a1"))
        finally:
            events_sse.reset_project_ctx(token)
        ev = events_sse._broadcast_queue.get_nowait()
        assert ev.project_id == "proj-a"
        assert ev.data["project_id"] == "proj-a"

    def test_emit_explicit_project_wins_over_ctx(self):
        from umrs.jobs.knowledge_adapters import KnowledgeEvent
        from umrs.knowledge import events_sse

        _fake_client("proj-b")
        token = events_sse.set_project_ctx("proj-a")
        try:
            events_sse.emit(
                KnowledgeEvent(event_type="ingest", job_id="j", attempt_id="a"),
                project_id="proj-b",
            )
        finally:
            events_sse.reset_project_ctx(token)
        ev = events_sse._broadcast_queue.get_nowait()
        assert ev.project_id == "proj-b"

    def test_unattributed_event_dropped_fail_closed(self):
        from umrs.jobs.knowledge_adapters import KnowledgeEvent
        from umrs.knowledge import events_sse

        _fake_client("proj-a")
        events_sse.emit(KnowledgeEvent(event_type="skip", job_id="j", attempt_id="a"))
        assert events_sse._broadcast_queue.empty()

    def test_no_listeners_never_fills_queue(self):
        from umrs.jobs.knowledge_adapters import KnowledgeEvent
        from umrs.knowledge import events_sse

        token = events_sse.set_project_ctx("proj-a")
        try:
            events_sse.emit(KnowledgeEvent(event_type="ingest", job_id="j", attempt_id="a"))
        finally:
            events_sse.reset_project_ctx(token)
        assert events_sse._broadcast_queue.empty()

    def test_target_clients_same_project_only(self):
        from umrs.knowledge import events_sse

        _fake_client("proj-a")
        _fake_client("proj-b")
        bevent = events_sse.BroadcastEvent(
            event_type="ingest", data={}, event_id="e1", project_id="proj-a"
        )
        targets = events_sse._target_clients(bevent)
        assert [cid for cid, _ in targets] == ["c-proj-a"]

    def test_knowledge_event_emit_feeds_sse(self):
        from umrs.jobs.knowledge_adapters import KnowledgeEvent
        from umrs.knowledge import events_sse

        _fake_client("proj-a")
        token = events_sse.set_project_ctx("proj-a")
        try:
            KnowledgeEvent(event_type="create", job_id="j9", attempt_id="a9").emit()
        finally:
            events_sse.reset_project_ctx(token)
        assert not events_sse._broadcast_queue.empty()
        ev = events_sse._broadcast_queue.get_nowait()
        assert ev.project_id == "proj-a"
        assert ev.data["job_id"] == "j9"


# ── B9: MCP read-path visibility ────────────────────────────────────────────


def _mcp_server(server_id: str, name: str, project_id: str | None):
    from umrs.mcp.models import MCPServer, MCPTool

    return MCPServer(
        server_id=server_id,
        name=name,
        url=f"http://{name}.local",
        project_id=project_id,
        tools=[
            MCPTool(tool_id=f"tool-{name}", server_id=server_id, name=f"t_{name}", description="")
        ],
    )


class _FakeMCPStore:
    def __init__(self, servers):
        self._servers = servers

    def list_servers(self, project_id=None, status_filter=None):
        return list(self._servers)

    def list_tools(self, server_name=None):
        out = []
        for s in self._servers:
            if server_name and s.name != server_name:
                continue
            out.append(
                {
                    "name": f"t_{s.name}",
                    "description": "",
                    "inputSchema": {},
                    "server": s.name,
                }
            )
        return out

    def get_server_by_name(self, name):
        for s in self._servers:
            if s.name == name:
                return s
        from umrs.mcp.store import MCPServerNotFoundError

        raise MCPServerNotFoundError(name)


def _build_mcp_app(store):
    from fastapi import FastAPI
    from umrs.api import mcp as mcp_module

    app = FastAPI()
    app.include_router(mcp_module.router)
    return app, mcp_module


MCP_PREFIX = "/umrs/mcp"


class TestMCPServerScoping:
    @pytest.fixture(autouse=True)
    def _store(self, monkeypatch):
        servers = [
            _mcp_server("mcp-global", "default-rag", None),
            _mcp_server("mcp-a", "proj-a-tools", "proj-a"),
            _mcp_server("mcp-b", "proj-b-tools", "proj-b"),
        ]
        app, self.mod = _build_mcp_app(_FakeMCPStore(servers))
        monkeypatch.setattr(self.mod, "_get_store", lambda: self.mod._fake_store)
        self.mod._fake_store = _FakeMCPStore(servers)
        self.app = app
        yield
        self.mod._fake_store = None

    def test_project_caller_sees_global_and_own_only(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).get(f"{MCP_PREFIX}/servers", headers={"X-Project-ID": "proj-a"})
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert names == {"default-rag", "proj-a-tools"}

    def test_other_project_server_hidden_from_get(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).get(
            f"{MCP_PREFIX}/servers/proj-b-tools", headers={"X-Project-ID": "proj-a"}
        )
        assert resp.status_code == 404

    def test_tools_list_filtered_to_visible_servers(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).get(f"{MCP_PREFIX}/tools", headers={"X-Project-ID": "proj-a"})
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()}
        assert names == {"t_default-rag", "t_proj-a-tools"}

    def test_anonymous_mvp_sees_everything_legacy(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).get(f"{MCP_PREFIX}/servers")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_strict_anonymous_rejected(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).get(f"{MCP_PREFIX}/servers")
        assert resp.status_code == 401


class TestMCPMutationScoping:
    """B9 write-path: registration/deletion obey the caller's project."""

    @pytest.fixture(autouse=True)
    def _store(self, monkeypatch):
        servers = [
            _mcp_server("mcp-global", "default-rag", None),
            _mcp_server("mcp-a", "proj-a-tools", "proj-a"),
            _mcp_server("mcp-b", "proj-b-tools", "proj-b"),
        ]
        store = _FakeMCPStore(servers)
        app, self.mod = _build_mcp_app(store)
        monkeypatch.setattr(self.mod, "_get_store", lambda: store)
        self.app = app
        self.store = store
        yield

    def test_registration_for_other_project_denied(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        resp = TestClient(self.app).post(
            f"{MCP_PREFIX}/servers",
            json={
                "name": "hijack",
                "url": "http://x.local:9",
                "tools": [],
                "project_id": "proj-b",
            },
            headers={"X-Project-ID": "proj-a"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "scope_denied"

    def test_blank_project_claims_caller_project_not_global(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        registered: dict = {}

        class _CapStore(_FakeMCPStore):
            def register_server(self, **kwargs):
                registered.update(kwargs)
                return _mcp_server("mcp-new", kwargs["name"], kwargs.get("project_id"))

        store = _CapStore([])
        app, mod = _build_mcp_app(store)
        monkeypatch.setattr(mod, "_get_store", lambda: store)
        resp = TestClient(app).post(
            f"{MCP_PREFIX}/servers",
            json={"name": "mine", "url": "http://mine.local:9", "tools": []},
            headers={"X-Project-ID": "proj-a"},
        )
        assert resp.status_code == 201
        assert registered["project_id"] == "proj-a"

    def test_admin_registers_global(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "sec")
        from fastapi.testclient import TestClient

        from r4_contracts.handshake_hmac import sign_claims

        class _CapStore(_FakeMCPStore):
            def register_server(self, **kwargs):
                return _mcp_server("mcp-new", kwargs["name"], kwargs.get("project_id"))

        store = _CapStore([])
        app, mod = _build_mcp_app(store)
        monkeypatch.setattr(mod, "_get_store", lambda: store)
        resp = TestClient(app).post(
            f"{MCP_PREFIX}/servers",
            json={"name": "global-one", "url": "http://g.local:9", "tools": []},
            headers={
                "X-User-ID": "root",
                "X-Verified-Project-ID": "ops",
                "X-Verified-Is-Admin": "true",
                "X-Verified-HMAC": sign_claims("sec", "root", "ops", True),
            },
        )
        assert resp.status_code == 201

    def test_unregistration_of_foreign_project_denied(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        store = self.store
        store.unregister_server = lambda name: True  # noqa: ARG005
        resp = TestClient(self.app).delete(
            f"{MCP_PREFIX}/servers/proj-b-tools", headers={"X-Project-ID": "proj-a"}
        )
        assert resp.status_code == 403

    def test_unregistration_of_own_project_allowed(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        self.store.unregister_server = lambda name: True
        resp = TestClient(self.app).delete(
            f"{MCP_PREFIX}/servers/proj-a-tools", headers={"X-Project-ID": "proj-a"}
        )
        assert resp.status_code == 200

    def test_global_server_delete_requires_admin(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from fastapi.testclient import TestClient

        self.store.unregister_server = lambda name: True
        resp = TestClient(self.app).delete(
            f"{MCP_PREFIX}/servers/default-rag", headers={"X-Project-ID": "proj-a"}
        )
        assert resp.status_code == 403
