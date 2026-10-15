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


class _Conn:
    """psycopg-shaped fake — rows flow from _project_id/PK read."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def cursor(self):
        return _Cursor(self)

    def _handle_insert(self, params, cur: _Cursor):
        pid, name, owner = params[:3]
        # INSERT may carry data_class as a 4th param (ADR-062 amendment:
        # the deployment's default class for NEW rows).
        data_class = params[3] if len(params) > 3 else "sensitive"
        if pid not in self.rows:
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

    def _handle_select_by_id(self, params, cur: _Cursor):
        pid = params[0]
        row = self.rows.get(pid)
        cur._result = [row] if row else []

    def _handle_select_ordered(self, params, cur: _Cursor):
        owner_filter = next((params[0] for row in params if isinstance(row, str)), None)
        rows = [v for v in self.rows.values() if v[3] != "retired"]
        if owner_filter:
            rows = [v for v in rows if v[2] == owner_filter]
        cur._result = rows

    def _handle(self, sql, params, cur: _Cursor):
        verb = sql.lstrip().upper()
        if verb.startswith("INSERT"):
            self._handle_insert(params, cur)
        elif verb.startswith("UPDATE"):
            self._handle_update(sql, params, cur)
        elif "WHERE project_id = %s" in sql:
            self._handle_select_by_id(params, cur)
        elif "ORDER BY created_at" in sql:
            self._handle_select_ordered(params, cur)
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
        project reported the `_row_response` default 'sensitive'."""
        store = ProjectStore(_Conn())
        store.create("p-1", owner_id="alice")
        store.set_data_class("p-1", "non_sensitive")
        mine = store.list(owner="alice")
        assert mine[0]["data_class"] == "non_sensitive"

    def test_create_honours_the_deployment_default_class(self, monkeypatch) -> None:
        """ADR-062 amendment D-4: stand-alone sets
        UMRS_PROJECT_DEFAULT_DATA_CLASS=non_sensitive; sites keep sensitive.
        Only NEW rows take it — the value never widens an existing row."""
        monkeypatch.setenv("UMRS_PROJECT_DEFAULT_DATA_CLASS", "non_sensitive")
        store = ProjectStore(_Conn())
        row = store.create("p-1", owner_id="alice")
        assert row["data_class"] == "non_sensitive"

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
        r = client.put(
            "/umrs/projects/p-own/budget",
            json={"max_gpu_hours": 1.0, "max_wall_clock_hours": 2.0},
            headers=_headers("bob"),
        )
        assert r.status_code == 403
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
