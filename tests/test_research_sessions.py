"""Research Sessions (ADR-025): store CRUD, router, loop auto-register."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.research_sessions import router, set_session_store
from umrs.research_sessions import ResearchSession, SessionLoopLink
from umrs.research_sessions.store import SessionStore


def _store() -> SessionStore:
    return SessionStore()


class TestSessionStore:
    def test_create_get_list(self):
        s = _store()
        s.create(
            ResearchSession(
                session_id="sess-a",
                project_id="p1",
                title="MERMAID filter",
                research_goal="digital filter for seismic peaks",
                metadata={"origin": "companion"},
            )
        )
        stored = s.get("sess-a")
        assert stored is not None
        assert stored.title == "MERMAID filter"
        assert stored.metadata == {"origin": "companion"}
        assert len(s.list(project_id="p1")) == 1
        assert s.list(project_id="other") == []

    def test_update_title_status_and_archive_marker(self):
        s = _store()
        s.create(ResearchSession(session_id="sess-b", project_id="p1"))
        upd = s.update("sess-b", title="Renamed", status="paused")
        assert upd.title == "Renamed" and upd.status == "paused"
        assert upd.archived_at is None
        arc = s.update("sess-b", status="archived")
        assert arc.status == "archived" and arc.archived_at is not None

    def test_update_unknown_returns_none(self):
        assert _store().update("nope", title="x") is None

    def test_attach_detach_loop_and_reverse_lookup(self):
        s = _store()
        s.create(ResearchSession(session_id="sess-c", project_id="p1"))
        s.attach_loop(SessionLoopLink(session_id="sess-c", loop_id="loop-1", venue="local"))
        sess = s.get("sess-c")
        assert [lnk.loop_id for lnk in sess.loops] == ["loop-1"]
        assert sess.loops[0].venue == "local"
        # idempotent attach
        s.attach_loop(SessionLoopLink(session_id="sess-c", loop_id="loop-1", venue="local"))
        assert len(s.get("sess-c").loops) == 1
        assert s.session_for_loop("loop-1").session_id == "sess-c"
        s.detach_loop("sess-c", "loop-1")
        assert s.get("sess-c").loops == []
        assert s.session_for_loop("loop-1") is None

    def test_tuple_row_access_matches_psycopg_cursors(self):
        """Postgres default cursor returns plain tuples — verify hydration works."""
        from umrs.research_sessions.store import _row_to_link, _row_to_session

        session_row = (
            "sess-pg",
            "proj-x",
            "T",
            "goal",
            "active",
            "umrs",
            '{"origin": "api"}',
            "2026-08-06",
            "2026-08-06",
            None,
        )
        sess = _row_to_session(session_row)
        assert sess.session_id == "sess-pg"
        assert sess.project_id == "proj-x"
        assert sess.title == "T"
        assert sess.metadata == {"origin": "api"}

        link_row = ("sess-pg", "loop-9", "umrs", "auto", "2026-08-06")
        link = _row_to_link(link_row)
        assert link.session_id == "sess-pg"
        assert link.loop_id == "loop-9"
        assert link.venue == "umrs"

    def test_map_row_still_works(self):
        """sqlite3.Row rows still hydrate via key access."""
        s = _store()
        s.create(ResearchSession(session_id="sess-map", project_id="p1"))
        stored = s.get("sess-map")
        assert stored.session_id == "sess-map"
        assert stored.project_id == "p1"


def _client():
    set_session_store(_store())
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestSessionRouter:
    def test_crud_lifecycle(self):
        c = _client()
        r = c.post(
            "/umrs/research/sessions",
            json={"project_id": "p9", "research_goal": "MERMAID digital filter for peaks"},
        )
        assert r.status_code == 201, r.text
        sess = r.json()["session"]
        assert sess["title"].startswith("MERMAID digital filter")
        assert sess["status"] == "active"
        sid = sess["session_id"]

        # idempotent create with pinned id
        r2 = c.post("/umrs/research/sessions", json={"project_id": "pX", "session_id": sid})
        assert r2.status_code == 201 and r2.json()["session"]["project_id"] == "p9"

        assert c.get(f"/umrs/research/sessions/{sid}").status_code == 200
        assert c.get("/umrs/research/sessions/sess-nope").status_code == 404

        r4 = c.patch(f"/umrs/research/sessions/{sid}", json={"status": "paused"})
        assert r4.json()["session"]["status"] == "paused"

        listed = c.get("/umrs/research/sessions", params={"project_id": "p9"})
        assert listed.json()["count"] == 1
        filtered = c.get("/umrs/research/sessions", params={"project_id": "p9", "status": "active"})
        assert filtered.json()["count"] == 0

    def test_loop_attach_detach(self):
        c = _client()
        sid = c.post("/umrs/research/sessions", json={"project_id": "p1"}).json()["session"][
            "session_id"
        ]
        r = c.post(f"/umrs/research/sessions/{sid}/loops", json={"loop_id": "loop-9"})
        assert r.status_code == 201
        assert (
            c.post("/umrs/research/sessions/sess-nope/loops", json={"loop_id": "x"}).status_code
            == 404
        )
        loops = c.get(f"/umrs/research/sessions/{sid}/loops")
        assert loops.json()["count"] == 1 and loops.json()["loops"][0]["venue"] == "umrs"
        assert c.get("/umrs/research/sessions/sess-nope/loops").status_code == 404
        d = c.delete(f"/umrs/research/sessions/{sid}/loops/loop-9")
        assert d.json()["detached"] is True
        assert c.get(f"/umrs/research/sessions/{sid}/loops").json()["count"] == 0


class TestLoopAutoRegister:
    def test_start_loop_registers_under_session(self):
        import contextlib

        from umrs.api.loop import _loop_store
        from umrs.api.loop import router as loops_router

        set_session_store(_store())
        # Fresh loop store per test
        with contextlib.suppress(Exception):
            _loop_store.clear()

        app = FastAPI()
        app.include_router(router)
        app.include_router(loops_router)
        c = TestClient(app)

        sid = c.post("/umrs/research/sessions", json={"project_id": "proj-r"}).json()["session"][
            "session_id"
        ]
        r = c.post(
            "/umrs/loops",
            json={
                "project_id": "proj-r",
                "research_goal": "goal with session",
                "max_iterations": 2,
                "session_id": sid,
            },
        )
        assert r.status_code == 201, r.text
        loop_id = r.json()["loop_id"]

        membership = c.get(f"/umrs/research/sessions/{sid}/loops").json()
        assert [lnk["loop_id"] for lnk in membership["loops"]] == [loop_id]

        status = c.get(f"/umrs/loops/{loop_id}").json()
        assert status["metadata"]["session_id"] == sid

    def test_start_loop_unknown_session_404(self):
        from umrs.api.loop import router as loops_router

        app = FastAPI()
        app.include_router(loops_router)
        c = TestClient(app)
        r = c.post(
            "/umrs/loops",
            json={
                "project_id": "proj-r",
                "research_goal": "goal",
                "max_iterations": 2,
                "session_id": "sess-does-not-exist",
            },
        )
        assert r.status_code in (404, 503), r.text
