"""ADR-051 D-1 pins: self-service durable projects registry.

Contracts locked:

- creation is ensuring-shape (existing row returned; never a silent
  no-op flagged as new); every row carries the creator as owner.
- admin mints for another user; non-admin is typed-forbidden on behalf of
  someone else.
- retire is a tombstone (state flips, row kept); 404 for unknown project
  ids — never a smudged empty; lane family is BETA per the wave decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from umrs.api.projects import ProjectNotFound, ProjectStore, router


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        self._conn._handle(sql, params, self)

    def fetchone(self):
        if not self._result:
            return None
        return self._result.pop(0)

    def fetchall(self):
        out = self._result[:]
        self._result.clear()
        return out


# The projects columns in table order; the fake stores full rows and returns
# only what a SELECT names (TV-1: a fake that returned full rows let a
# reverted 7-column listing SELECT pass).
_COLUMNS = (
    "project_id",
    "name",
    "owner_id",
    "status",
    "retired_at",
    "created_at",
    "updated_at",
    "data_class",
    "ask_mode",
    "pause_triggers",
)


def _project(sql: str, row: tuple) -> tuple:
    """The stored row reduced to the SELECT's column list, in its order."""
    selected = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    names = [c.strip() for c in selected.split(",") if c.strip()]
    return tuple(row[_COLUMNS.index(name)] for name in names)


class _Conn:
    """psycopg-shaped fake — rows flow from _project_id/PK read."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def cursor(self):
        return _Cursor(self)

    def _handle_insert(self, sql, params, cur: _Cursor):
        pid, name, owner = params[:3]
        # INSERT may carry data_class as a 4th param (ADR-062 amendment:
        # the deployment's default class for NEW rows).
        data_class = params[3] if len(params) > 3 else "sensitive"
        if pid in self.rows:
            # Postgres semantics on a primary-key conflict (TV-3).
            if "DO NOTHING" in sql:
                pass
            elif "DO UPDATE" in sql:
                old = self.rows[pid]
                self.rows[pid] = old[:7] + (data_class,) + old[8:]
            else:
                raise RuntimeError("duplicate key value violates unique constraint")
        else:
            self.rows[pid] = (pid, name, owner, "active", None, None, None, data_class, "", "[]")
        cur._result = []

    def _handle_update(self, sql, params, cur: _Cursor):
        now = datetime.now(timezone.utc)
        base = self.rows[params[-1]]
        # row shape: (pid, name, owner, status, retired, created, updated,
        #             data_class, ask_mode, pause_triggers)
        pid, name, owner, status, r1, r2, r3 = base[:7]
        dc = base[7] if len(base) > 7 else "sensitive"
        am = base[8] if len(base) > 8 else ""
        pt = base[9] if len(base) > 9 else "[]"
        if "data_class" in sql:
            self.rows[params[-1]] = (pid, name, owner, status, r1, r2, now, params[0], am, pt)
        elif "ask_mode" in sql:
            self.rows[params[-1]] = (
                pid,
                name,
                owner,
                status,
                r1,
                r2,
                now,
                dc,
                params[0],
                params[1],
            )
        else:
            self.rows[params[-1]] = (pid, name, owner, "retired", now, r2, now, dc, am, pt)
        cur._result = []

    def _handle_select_by_id(self, sql, params, cur: _Cursor):
        pid = params[0]
        row = self.rows.get(pid)
        cur._result = [_project(sql, row)] if row else []

    def _handle_select_ordered(self, sql, params, cur: _Cursor):
        owner_filter = next((params[0] for row in params if isinstance(row, str)), None)
        rows = [v for v in self.rows.values() if v[3] != "retired"]
        if owner_filter:
            rows = [v for v in rows if v[2] == owner_filter]
        cur._result = [_project(sql, v) for v in rows]

    def _handle(self, sql, params, cur: _Cursor):
        verb = sql.lstrip().upper()
        if verb.startswith("INSERT"):
            self._handle_insert(sql, params, cur)
        elif verb.startswith("UPDATE"):
            self._handle_update(sql, params, cur)
        elif "WHERE project_id = %s" in sql:
            self._handle_select_by_id(sql, params, cur)
        elif "ORDER BY created_at" in sql:
            self._handle_select_ordered(sql, params, cur)
        else:
            cur._result = []

    def commit(self):
        pass


def _client(store) -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    pool = MagicMock()
    pool.getconn.return_value = store._conn
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _headers(user: str, admin: bool = False) -> dict[str, str]:
    headers = {"X-User-ID": user}
    if admin:
        headers["X-Verified-Is-Admin"] = "true"
    headers["X-Verified-Project-ID"] = "proj-call"
    return headers


class TestStoreLifecycle:
    def test_create_insert_row_and_owner(self) -> None:
        store = ProjectStore(_Conn())
        row = store.create("lab-alpha", name="Lab", owner_id="alice")
        assert row["owner_id"] == "alice"
        again = store.create("lab-alpha", name="Lab B", owner_id="bob")
        # ensuring-shape: existing returned as-is, never stomped
        assert again["owner_id"] == "alice"
        assert again["name"] == "Lab"

    def test_get_unknown_raises_not_found(self) -> None:
        store = ProjectStore(_Conn())
        with pytest.raises(ProjectNotFound):
            store.get("ghost")

    def test_retire_tombstones_never_deletes(self) -> None:
        store = ProjectStore(_Conn())
        store.create("p-1", owner_id="alice")
        dead = store.retire("p-1")
        assert dead["status"] == "retired"
        # row kept, not deleted
        store.get("p-1")

    def test_list_owner_and_retired_filter(self) -> None:
        store = ProjectStore(_Conn())
        store.create("p-1", owner_id="alice")
        store.create("p-2", owner_id="bob")
        mine = store.list(owner="alice")
        assert [r["project_id"] for r in mine] == ["p-1"]

    def test_list_carries_the_real_data_class(self) -> None:
        """S2 (2026-10-15): the listing selected 7 columns, so every listed
        project reported the `_row_response` default 'sensitive'. The fake
        returns only the SELECTed columns, so reverting the SELECT fails."""
        store = ProjectStore(_Conn())
        store.create("p-1", owner_id="alice")
        store.set_data_class("p-1", "non_sensitive")
        store.set_ask_policy("p-1", "profile", ["cost"])
        mine = store.list(owner="alice")
        assert mine[0]["data_class"] == "non_sensitive"
        assert (mine[0]["ask_mode"], mine[0]["pause_triggers"]) == ("profile", ["cost"])

    def test_create_honours_the_deployment_default_class(self, monkeypatch) -> None:
        """ADR-062 amendment D-4: stand-alone sets
        UMRS_PROJECT_DEFAULT_DATA_CLASS=non_sensitive; sites keep sensitive.
        Only NEW rows take it — the value never widens an existing row."""
        monkeypatch.setenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", "non_sensitive")
        store = ProjectStore(_Conn())
        row = store.create("p-1", owner_id="alice")
        assert row["data_class"] == "non_sensitive"

    def test_deployment_default_never_widens_an_existing_row(self, monkeypatch) -> None:
        """TV-3 (D-4): a row created before the stand-alone default was set
        keeps its class when create() is called again with the default."""
        monkeypatch.delenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", raising=False)
        store = ProjectStore(_Conn())
        assert store.create("p-old", owner_id="alice")["data_class"] == "sensitive"
        monkeypatch.setenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", "non_sensitive")
        assert store.create("p-old", owner_id="alice")["data_class"] == "sensitive"
        assert store.get("p-old")["data_class"] == "sensitive"
        # ... while a NEW row in the same deployment takes the default.
        assert store.create("p-new", owner_id="alice")["data_class"] == "non_sensitive"

    def test_deployment_default_never_widens_a_concurrently_created_row(self, monkeypatch) -> None:
        """TV-3: the row appears between create()'s existence check and its
        INSERT (another request won the race); the INSERT must not rewrite
        that row's class (ON CONFLICT DO NOTHING, never DO UPDATE)."""
        monkeypatch.setenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", "non_sensitive")

        class _RacingConn(_Conn):
            def _handle(self, sql, params, cur):
                if sql.lstrip().upper().startswith("INSERT"):
                    self.rows.setdefault(
                        params[0],
                        (params[0], "", "bob", "active", None, None, None, "sensitive", "", "[]"),
                    )
                super()._handle(sql, params, cur)

        store = ProjectStore(_RacingConn())
        row = store.create("p-race", owner_id="alice")
        assert row["data_class"] == "sensitive"
        assert row["owner_id"] == "bob"  # the winner's row, untouched

    def test_create_default_class_stays_sensitive_on_sites(self, monkeypatch) -> None:
        monkeypatch.delenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", raising=False)
        store = ProjectStore(_Conn())
        assert store.create("p-1", owner_id="alice")["data_class"] == "sensitive"

    def test_create_invalid_default_class_falls_back_to_sensitive(self, monkeypatch) -> None:
        monkeypatch.setenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", "public")
        store = ProjectStore(_Conn())
        assert store.create("p-1", owner_id="alice")["data_class"] == "sensitive"


class TestRoutes:
    def test_create_binds_caller_as_owner(self) -> None:
        client = _client(ProjectStore(_Conn()))
        r = client.post(
            "/umrs/projects", json={"project_id": "lab-c", "name": "C"}, headers=_headers("alice")
        )
        assert r.status_code == 201
        assert r.json()["project"]["owner_id"] == "alice"

    def test_for_user_mint_requires_admin(self) -> None:
        client = _client(ProjectStore(_Conn()))
        unauth = client.post(
            "/umrs/projects",
            json={"project_id": "lab-d", "for_user": "alice"},
            headers=_headers("bob"),
        )
        assert unauth.status_code == 403
        admin = client.post(
            "/umrs/projects",
            json={"project_id": "lab-d", "for_user": "alice"},
            headers=_headers("bob", admin=True),
        )
        assert admin.status_code == 201
        assert admin.json()["project"]["owner_id"] == "alice"

    def test_get_unknown_project_404(self) -> None:
        client = _client(ProjectStore(_Conn()))
        r = client.get("/umrs/projects/ghost-x", headers=_headers("alice"))
        assert r.status_code == 404

    def test_retire_requires_owner_or_admin(self) -> None:
        store = ProjectStore(_Conn())
        store.create("p-own", owner_id="alice")
        client = _client(store)
        bad = client.request(
            "PATCH",
            "/umrs/projects/p-own",
            json={"action": "retire"},
            headers=_headers("mallory"),
        )
        assert bad.status_code == 403
        ok = client.request(
            "PATCH",
            "/umrs/projects/p-own",
            json={"action": "retire"},
            headers=_headers("alice"),
        )
        assert ok.status_code == 200
        assert ok.json()["project"]["status"] == "retired"


class TestServiceLanes:
    def test_projects_family_declared_beta(self) -> None:
        from umrs.shared.lanes import FAMILY_LANES, SurfaceLane

        assert FAMILY_LANES["projects"] == SurfaceLane.BETA


# ── ADR-067: loop budgets ────────────────────────────────────────────────────


class _BudgetConn(_Conn):
    """Adds project_budgets table emulation beside the projects fake."""

    def __init__(self):
        super().__init__()
        self.budgets: dict[str, dict] = {}

    def _handle(self, sql, params, cur: _Cursor):
        if "project_budgets" in sql:
            self._handle_budget(sql, params, cur)
            return
        super()._handle(sql, params, cur)

    def _handle_budget(self, sql, params, cur: _Cursor):
        verb = sql.lstrip().upper()
        if verb.startswith("INSERT"):
            pid, gpu, wall = params
            self.budgets[pid] = (pid, float(gpu), float(wall), None)
            cur._result = []
        else:
            row = self.budgets.get(params[0])
            cur._result = [row] if row else []


class TestLoopBudget:
    def test_default_uncapped_when_absent(self) -> None:
        store = ProjectStore(_BudgetConn())
        budget = store.get_budget("nope")
        assert budget["max_gpu_hours"] == 0.0
        assert budget["max_wall_clock_hours"] == 0.0

    def test_set_and_read_back(self) -> None:
        store = ProjectStore(_BudgetConn())
        store.create("p-b", owner_id="alice")
        set_row = store.set_budget("p-b", max_gpu_hours=12.5, max_wall_clock_hours=4.0)
        assert set_row["max_gpu_hours"] == 12.5
        assert set_row["max_wall_clock_hours"] == 4.0
        again = store.get_budget("p-b")
        assert again["max_wall_clock_hours"] == 4.0

    def test_put_requires_owner_or_admin(self) -> None:
        store = ProjectStore(_BudgetConn())
        store.create("p-own", owner_id="alice")
        client = _client(store)
        # H-1 (2026-10-18): bob cannot read p-own (not owner, not verified
        # into it, no circle), so the PUT hides it as GET does: 404, not 403.
        # A reader who is not owner/admin still gets 403 (test_h1r_projects_*).
        r = client.put(
            "/umrs/projects/p-own/budget",
            json={"max_gpu_hours": 1.0, "max_wall_clock_hours": 2.0},
            headers=_headers("bob"),
        )
        assert r.status_code == 404
        ok = client.put(
            "/umrs/projects/p-own/budget",
            json={"max_gpu_hours": 1.0, "max_wall_clock_hours": 2.0},
            headers=_headers("alice"),
        )
        assert ok.status_code == 200
        assert ok.json()["budget"]["max_wall_clock_hours"] == 2.0


# ── ADR-062 (D-51): per-project data class ──────────────────────────────


class TestDataClass:
    def test_create_defaults_to_sensitive(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        assert store.get("proj-dc")["data_class"] == "sensitive"

    def test_owner_widens_and_narrows(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        client = _client(store)
        r = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "non_sensitive"},
            headers=_headers("alice"),
        )
        assert r.status_code == 200
        assert r.json()["project"]["data_class"] == "non_sensitive"
        r2 = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "sensitive"},
            headers=_headers("alice"),
        )
        assert r2.json()["project"]["data_class"] == "sensitive"

    def test_non_owner_refused_even_admin_false(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        client = _client(store)
        r = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "non_sensitive"},
            headers=_headers("mallory"),
        )
        assert r.status_code == 403

    def test_bad_class_value_refused(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        client = _client(store)
        r = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "public"},
            headers=_headers("alice"),
        )
        assert r.status_code == 400

    @pytest.fixture
    def audit_events(self, monkeypatch) -> list[tuple[str, dict]]:
        """The audit emitter, captured (projects.py imports it per call)."""
        from umrs.api import audit_events

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            audit_events, "emit_audit", lambda event, **kw: events.append((event, kw))
        )
        return events

    def test_widening_is_audited_with_the_actor(self, audit_events) -> None:
        """R4-4 / spec F3: ADR-062 asks that widening be attributable; the
        CLI tells the owner the act is recorded with their identity."""
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        r = _client(store).patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "non_sensitive"},
            headers=_headers("alice"),
        )
        assert r.status_code == 200
        ((event, kw),) = audit_events
        assert event == "project.data_class_changed"
        assert (kw["actor"], kw["project_id"], kw["outcome"]) == ("alice", "proj-dc", "success")
        assert kw["details"] == {
            "from_class": "sensitive",
            "to_class": "non_sensitive",
            "widening": True,
            "owner_id": "alice",
            "by_admin": False,
        }

    def test_admin_change_names_the_admin_not_the_owner(self, audit_events) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        store.set_data_class("proj-dc", "non_sensitive")
        r = _client(store).patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "sensitive"},
            headers=_headers("root", admin=True),
        )
        assert r.status_code == 200
        ((_event, kw),) = audit_events
        assert kw["actor"] == "root"
        assert kw["details"] == {
            "from_class": "non_sensitive",
            "to_class": "sensitive",
            "widening": False,
            "owner_id": "alice",
            "by_admin": True,
        }

    def test_refused_change_emits_no_event(self, audit_events) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        client = _client(store)
        forbidden = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "non_sensitive"},
            headers=_headers("mallory"),
        )
        bad_value = client.patch(
            "/umrs/projects/proj-dc",
            json={"action": "set_data_class", "data_class": "public"},
            headers=_headers("alice"),
        )
        assert (forbidden.status_code, bad_value.status_code) == (403, 400)
        assert audit_events == []
        assert store.get("proj-dc")["data_class"] == "sensitive"

    def test_audit_record_reaches_the_audit_log(self, monkeypatch) -> None:
        """Through the real emitter: the record lands in the audit trail
        (the ring buffer here; the sidecar adds a persistent store)."""
        from umrs.api import audit_events
        from umrs.api.audit_events import query_audit_events

        monkeypatch.setattr(audit_events, "_audit_store", None)
        store = ProjectStore(_Conn())
        store.create("proj-audit", owner_id="carol-audit")
        _client(store).patch(
            "/umrs/projects/proj-audit",
            json={"action": "set_data_class", "data_class": "non_sensitive"},
            headers=_headers("carol-audit"),
        )
        found = query_audit_events(event="project.data_class_changed", actor="carol-audit")
        assert found and found[0]["project_id"] == "proj-audit"
        assert found[0]["details"]["to_class"] == "non_sensitive"

    def test_get_carries_the_class(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-dc", owner_id="alice")
        client = _client(store)
        r = client.get("/umrs/projects/proj-dc", headers=_headers("alice"))
        assert r.json()["project"]["data_class"] == "sensitive"


# ── D-33 tail: the persisted per-project ask-policy default (0111) ────


class TestAskPolicy:
    def test_set_and_clear_round_trip(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-ask", owner_id="alice")
        row = store.set_ask_policy("proj-ask", "profile", ["cost"])
        assert (row["ask_mode"], row["pause_triggers"]) == ("profile", ["cost"])
        cleared = store.set_ask_policy("proj-ask", "", [])
        assert cleared["ask_mode"] == ""
        assert list(cleared["pause_triggers"]) == []

    def test_invalid_mode_refused(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-ask", owner_id="alice")
        with pytest.raises(ValueError, match="invalid ask_mode"):
            store.set_ask_policy("proj-ask", "always", [])

    def test_route_owner_can_set_and_non_owner_refused(self) -> None:
        store = ProjectStore(_Conn())
        store.create("proj-ask", owner_id="alice")
        client = _client(store)
        ok = client.patch(
            "/umrs/projects/proj-ask",
            json={
                "action": "set_ask_policy",
                "ask_mode": "profile",
                "pause_triggers": ["cost", "competing_hypotheses"],
            },
            headers=_headers("alice"),
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["project"]["ask_mode"] == "profile"
        assert ok.json()["project"]["pause_triggers"] == [
            "cost",
            "competing_hypotheses",
        ]
        bad = client.patch(
            "/umrs/projects/proj-ask",
            json={"action": "set_ask_policy", "ask_mode": "ask"},
            headers=_headers("mallory"),
        )
        assert bad.status_code == 403


class TestLiveMemberships:
    """ADR-097 D-2: GET /umrs/projects carries the caller's live project ids
    — the venue user record plus owned projects — so an admin's
    `add-to-project` takes effect without minting a new token."""

    @pytest.fixture
    def users(self, tmp_path, monkeypatch):
        from umrs.admin.user_store import UserStore

        monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
        return UserStore(tmp_path)

    def test_record_projects_and_owned_projects(self, users) -> None:
        users.create("ryoshimi2", projects=["default", "opossum"])
        store = ProjectStore(_Conn())
        store.create("lab-own", owner_id="ryoshimi2")
        r = _client(store).get("/umrs/projects", headers=_headers("ryoshimi2"))
        assert r.status_code == 200
        assert r.json()["memberships"] == ["default", "lab-own", "opossum"]

    def test_add_to_project_shows_on_the_next_request(self, users) -> None:
        users.create("alice", projects=["default"])
        client = _client(ProjectStore(_Conn()))
        assert client.get("/umrs/projects", headers=_headers("alice")).json()["memberships"] == [
            "default"
        ]
        users.add_project("alice", "opossum")
        after = client.get("/umrs/projects", headers=_headers("alice")).json()
        assert after["memberships"] == ["default", "opossum"]

    def test_retired_project_and_disabled_user_grant_nothing(self, users) -> None:
        users.create("bob", projects=["gone", "kept"])
        store = ProjectStore(_Conn())
        store.create("gone", owner_id="carol")
        store.retire("gone")
        assert _client(store).get("/umrs/projects", headers=_headers("bob")).json()[
            "memberships"
        ] == ["kept"]
        users.set_disabled("bob", True)
        assert (
            _client(store).get("/umrs/projects", headers=_headers("bob")).json()["memberships"]
            == []
        )

    @pytest.mark.usefixtures("users")
    def test_unknown_user_has_no_memberships(self) -> None:
        r = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("nobody"))
        assert r.json()["memberships"] == []

    @pytest.mark.usefixtures("users")
    def test_unreadable_record_is_null_not_empty(self, monkeypatch) -> None:
        from umrs.api import projects

        def _boom(user: str) -> list[str]:
            raise OSError("user files unreadable")

        monkeypatch.setattr(projects, "_record_projects", _boom)
        r = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("alice"))
        assert r.status_code == 200
        assert r.json()["memberships"] is None


class TestDisabledStanding:
    """ADR-097 Amendment D-8 (2026-10-23): GET /umrs/projects reports the
    caller's standing, so the gateway refuses a disabled account's tokens
    within its membership cache window instead of at token expiry."""

    @pytest.fixture
    def users(self, tmp_path, monkeypatch):
        from umrs.admin.user_store import UserStore

        monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
        return UserStore(tmp_path)

    def test_an_enabled_caller_is_not_disabled(self, users) -> None:
        users.create("alice", projects=["default"])
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("alice"))
        assert body.json()["disabled"] is False
        assert body.json()["memberships"] == ["default"]

    def test_a_disabled_caller_is_reported_and_owns_nothing(self, users) -> None:
        users.create("bob", projects=["default"])
        store = ProjectStore(_Conn())
        store.create("bob-lab", owner_id="bob")
        users.set_disabled("bob", True)
        body = _client(store).get("/umrs/projects", headers=_headers("bob")).json()
        assert body["disabled"] is True
        # an owned project no longer counts: an older gateway reading only
        # `memberships` then refuses every project but the token's own
        assert body["memberships"] == []

    def test_re_enabling_restores_the_standing(self, users) -> None:
        users.create("carol", projects=["default"])
        client = _client(ProjectStore(_Conn()))
        users.set_disabled("carol", True)
        assert client.get("/umrs/projects", headers=_headers("carol")).json()["disabled"]
        users.set_disabled("carol", False)
        assert client.get("/umrs/projects", headers=_headers("carol")).json()["disabled"] is False

    @pytest.mark.usefixtures("users")
    def test_an_unknown_caller_is_not_disabled(self) -> None:
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("nobody"))
        assert body.json()["disabled"] is False

    @pytest.mark.usefixtures("users")
    def test_an_unreadable_record_is_null_not_false(self, monkeypatch) -> None:
        from umrs.admin import account_cascade

        def _boom(user: str) -> None:
            raise OSError("user files unreadable")

        monkeypatch.setattr(account_cascade, "read_record", _boom)
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("alice"))
        assert body.json()["disabled"] is None
        assert body.json()["memberships"] is None


class TestDisabledStandingReviewFixes:
    """ADR-097 Amendment D-8 review fixes (2026-10-23): an IdP subject
    resolves through ``oidc_subject``, a JSON-lane disable holds while the
    ``users`` row cannot be read, and the gateway's lookup can skip the
    project listing."""

    @pytest.fixture
    def users(self, tmp_path, monkeypatch):
        from umrs.admin.user_store import UserStore

        monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
        return UserStore(tmp_path)

    @staticmethod
    def _postgres_down(monkeypatch) -> None:
        from umrs.admin import user_store

        pool = MagicMock()
        pool.getconn.side_effect = RuntimeError("users table unreachable")
        monkeypatch.setattr(user_store, "_PG_POOL", pool)

    def test_an_idp_subject_is_resolved_through_oidc_subject(self, users) -> None:
        # under JWKS/SAML the token sub is the IdP's id (a numeric RIKEN ID),
        # which is no valid username: it read as "unknown, not disabled"
        users.create("alice", projects=["default"], oidc_subject="241723")
        client = _client(ProjectStore(_Conn()))
        body = client.get("/umrs/projects", headers=_headers("241723")).json()
        assert body["disabled"] is False
        assert body["memberships"] == ["default"]
        users.set_disabled("alice", True)
        body = client.get("/umrs/projects", headers=_headers("241723")).json()
        assert body["disabled"] is True
        assert body["memberships"] == []

    def test_a_mixed_case_subject_is_resolved_too(self, users) -> None:
        users.create("bob", oidc_subject="Bob.Tanaka")
        users.set_disabled("bob", True)
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("Bob.Tanaka"))
        assert body.json()["disabled"] is True

    def test_a_json_disable_holds_while_postgres_is_down(self, users, monkeypatch) -> None:
        users.create("carol", projects=["default"])
        users.set_disabled("carol", True)
        self._postgres_down(monkeypatch)
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("carol"))
        assert body.json()["disabled"] is True
        assert body.json()["memberships"] == []

    def test_postgres_down_without_a_json_disable_is_unknown(self, users, monkeypatch) -> None:
        users.create("dave", projects=["default"])
        self._postgres_down(monkeypatch)
        body = _client(ProjectStore(_Conn())).get("/umrs/projects", headers=_headers("dave"))
        assert body.json()["disabled"] is None
        assert body.json()["memberships"] is None

    def test_standing_only_skips_the_listing(self, users) -> None:
        users.create("root", role="admin", projects=["ops"])
        store = ProjectStore(_Conn())
        store.create("lab", owner_id="zed")
        owners: list[object] = []
        listing = store.list

        def _spy(owner=None, **kwargs):
            owners.append(owner)
            return listing(owner=owner, **kwargs)

        store.list = _spy  # type: ignore[method-assign]
        r = _client(store).get(
            "/umrs/projects?standing_only=true", headers=_headers("root", admin=True)
        )
        body = r.json()
        assert r.status_code == 200
        assert body["standing_only"] is True
        assert body["disabled"] is False
        assert body["memberships"] == ["ops"]
        assert "projects" not in body
        # the admin's every-row listing (owner=None) is never run
        assert None not in owners

    def test_projects_owned_under_any_of_the_accounts_ids_count(self, users) -> None:
        # ADR-097 Amendment D-8 re-review (2026-10-23): a gateway-created
        # project is owned by the token sub (the IdP subject), a CLI-made
        # one by the username; either id is the same account
        users.create("tanaka", oidc_subject="241723")
        store = ProjectStore(_Conn())
        store.create("by-name", owner_id="tanaka")
        store.create("by-sub", owner_id="241723")
        client = _client(store)
        for caller in ("241723", "tanaka"):
            body = client.get("/umrs/projects?standing_only=true", headers=_headers(caller))
            assert body.json()["memberships"] == ["by-name", "by-sub"], caller

    def test_another_accounts_username_as_subject_owns_nothing(self, users, tmp_path) -> None:
        # 2026-10-23, round 3: bob.oidc_subject == "alice" (a record written
        # before UserStore refused it) made alice's project bob's membership
        import json

        from umrs.admin.user_store import UserRecord

        users.create("alice")
        bob = UserRecord(username="bob", oidc_subject="alice")
        (tmp_path / "bob.json").write_text(json.dumps(bob.to_json()))
        store = ProjectStore(_Conn())
        store.create("p-alice", owner_id="alice")
        client = _client(store)
        body = client.get("/umrs/projects?standing_only=true", headers=_headers("bob"))
        assert body.json()["memberships"] == []
        body = client.get("/umrs/projects?standing_only=true", headers=_headers("alice"))
        assert body.json()["memberships"] == ["p-alice"]

    def test_without_standing_only_the_listing_is_unchanged(self, users) -> None:
        users.create("root", role="admin")
        store = ProjectStore(_Conn())
        store.create("lab", owner_id="zed")
        body = _client(store).get("/umrs/projects", headers=_headers("root", admin=True)).json()
        assert [p["project_id"] for p in body["projects"]] == ["lab"]
        assert body["count"] == 1
        assert body["disabled"] is False
