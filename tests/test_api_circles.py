"""Circles authority pins (ADR-086 wave, 2026-10-05).

Store + route invariants: ensure-shape creation with creator-as-first-member
(role admin), project-scoped keys, personal-circle casual entry, named-circle
affiliation gate, roster ops owner/circle-admin/project-owner/platform-admin
only (four-rung ladder, ADR-086 amendment 2026-10-19 — the project-owner rung
covers the roster, never lifecycle), personal cells never gaining members,
owners never removed, retired circles refusing growth and vanishing from
resolution, mine listing deriving projects, unknown ids 404, and the pool conn
always returned (generator dependency).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.circles import (
    CircleNotFound,
    CirclesStore,
    circles_for_user,
    router,
)


def _in(haystack: str, *needles: str) -> bool:
    return all(n in haystack for n in needles)


class _Cursor:
    """In-memory projects + circles + circle_members swallowing the SQL
    CirclesStore emits."""

    def __init__(self, conn):
        self._conn = conn
        self._result: list = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        low = sql.strip().lower()
        for verb, handler in self._WRITES().items():
            if low.startswith(verb):
                handler(params)
                self._result = []
                return
        self._read(low, params)

    def fetchone(self):
        return self._result.pop(0) if self._result else None

    def fetchall(self):
        out = self._result[:]
        self._result = []
        return out

    def _WRITES(self):
        conn = self._conn
        return {
            "insert into circles": lambda p: conn.circles.setdefault(
                (p[0], p[1]), (p[0], p[1], p[2], p[3], "active", None, None, None)
            ),
            # owner insert carries 'admin' as a SQL literal (4 params);
            # member inserts carry role as p[3] (5 params)
            "insert into circle_members": lambda p: conn.members_add(
                p[0],
                p[1],
                p[2],
                p[3] if len(p) > 4 else "admin",
                p[4] if len(p) > 4 else p[3],
            ),
            "delete from circle_members": lambda p: conn.members.discard((p[0], p[1], p[2])),
            "update circles": self._write_retire,
        }

    def _write_retire(self, params):
        key = (params[0], params[1])
        conn = self._conn
        row = conn.circles.get(key)
        if row and row[4] != "retired":
            conn.circles[key] = (
                row[0],
                row[1],
                row[2],
                row[3],
                "retired",
                datetime.now(timezone.utc),
                None,
                None,
            )
            self.rowcount = 1

    def _read(self, low, params):
        # First matching marker set wins (order matters: the joins and the
        # roster query also contain the shorter markers further down).
        for markers, handler in _READS:
            if _in(low, *markers):
                self._result = handler(self, self._conn, low, params)
                return
        self._result = []

    def _mine(self, conn, low, params):
        user = params[0]
        owned = {k for k, r in conn.circles.items() if r[3] == user}
        joined = {(p, c) for (p, c, u) in conn.members if u == params[1]}
        rows = [conn.circles[k] for k in sorted((owned | joined) & set(conn.circles))]
        if _in(low, "status <> 'retired'"):
            rows = [r for r in rows if r[4] != "retired"]
        return rows


def _read_role(_cur, conn, _low, params):
    key = (params[0], params[1], params[2])
    return [(conn._role[key],)] if key in conn._role else []


def _read_affiliation(_cur, conn, _low, params):
    return [(1,)] if _affiliated(conn, params[0], params[1]) else []


def _read_refs(_cur, conn, _low, params):
    return [
        (f"{p}/{c}",)
        for (p, c, u) in sorted(conn.members)
        if u == params[0] and conn.circles[(p, c)][4] == "active"
    ]


def _read_project(_cur, conn, _low, params):
    row = conn.projects.get(params[0])
    return [row] if row else []


def _read_roster(_cur, conn, _low, params):
    return [
        (u, conn._role[(params[0], params[1], u)], conn._by[(params[0], params[1], u)], None)
        for (p, c, u) in sorted(conn.members)
        if p == params[0] and c == params[1]
    ]


def _read_circle(_cur, conn, _low, params):
    row = conn.circles.get((params[0], params[1]))
    return [row] if row else []


def _read_project_circles(_cur, conn, low, params):
    rows = [r for (p, _c), r in conn.circles.items() if p == params[0]]
    if _in(low, "status <> 'retired'"):
        rows = [r for r in rows if r[4] != "retired"]
    return rows


def _read_mine(cur, conn, low, params):
    return cur._mine(conn, low, params)


_READS = (
    (("select role from circle_members",), _read_role),
    (("from circle_members cm", "join circles", "limit 1"), _read_affiliation),
    (("cm.project_id || '/' || cm.circle_id",), _read_refs),
    (("from projects where project_id = %s",), _read_project),
    (("from circle_members", "where project_id = %s"), _read_roster),
    (("where project_id = %s and circle_id = %s",), _read_circle),
    (("from circles where project_id = %s",), _read_project_circles),
    (("left join circle_members",), _read_mine),
)


def _affiliated(conn, project_id, user_id) -> bool:
    return any(
        p == project_id and u == user_id and conn.circles[(p, c)][4] == "active"
        for (p, c, u) in conn.members
    )


class _Conn:
    def __init__(self):
        self.projects: dict[str, tuple] = {}
        self.circles: dict[tuple[str, str], tuple] = {}
        self.members: set[tuple[str, str, str]] = set()
        self._role: dict[tuple[str, str, str], str] = {}
        self._by: dict[tuple[str, str, str], str] = {}

    def members_add(self, pid, cid, uid, role, by):
        key = (pid, cid, uid)
        self.members.add(key)
        self._role[key] = role
        self._by[key] = by

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass


class _Pool:
    def __init__(self):
        self._conn = _Conn()
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _c):
        self.returned += 1


def _client(pool=None, projects=("proj-a",)) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    pool = pool or _Pool()
    for pid in projects:
        pool._conn.projects[pid] = (pid, "owner-a", "active")
    app.state.umrs_db_pool = pool
    return TestClient(app)


_HDR = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}
_HDR_ADMIN = {**_HDR, "X-Verified-Is-Admin": "true"}


class TestStoreAndResolution:
    def test_create_ensures_and_creator_is_first_member_admin(self):
        store = CirclesStore(_Conn())
        row = store.create("proj-a", "team-x", name="Team X", owner_id="alice")
        assert row["owner_id"] == "alice"
        again = store.create("proj-a", "team-x", name="Other", owner_id="bob")
        assert again["owner_id"] == "alice" and again["name"] == "Team X"
        members = store.members("proj-a", "team-x")
        assert members == [
            {"user_id": "alice", "role": "admin", "added_by": "alice", "added_at": None}
        ]

    def test_circle_refs_for_user_filters_retired(self):
        conn = _Conn()
        conn.circles[("proj-a", "c1")] = ("proj-a", "c1", "", "x", "active", None, None, None)
        conn.circles[("proj-a", "c2")] = ("proj-a", "c2", "", "x", "retired", None, None, None)
        conn.members_add("proj-a", "c1", "alice", "member", "x")
        conn.members_add("proj-a", "c2", "alice", "member", "x")
        assert circles_for_user(conn, "alice") == ["proj-a/c1"]

    def test_unknown_circle_raises(self):
        with pytest.raises(CircleNotFound):
            CirclesStore(_Conn()).get("proj-a", "c-ghost")


class TestCreationGates:
    def test_unknown_and_retired_projects_refuse(self):
        client = _client()
        resp = client.post(
            "/umrs/circles",
            json={"project_id": "proj-ghost", "circle_id": "alice"},
            headers=_HDR,
        )
        assert resp.status_code == 404
        pool = _Pool()
        pool._conn.projects["proj-dead"] = ("proj-dead", "owner-a", "retired")
        client = _client(pool)
        resp = client.post(
            "/umrs/circles",
            json={"project_id": "proj-dead", "circle_id": "alice"},
            headers=_HDR,
        )
        assert resp.status_code == 409

    def test_personal_circle_is_casual_named_circle_needs_affiliation(self):
        client = _client()
        # personal: always allowed (entering the project)
        resp = client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "alice"},
            headers=_HDR,
        )
        assert resp.status_code == 201, resp.text
        # bob cannot mint a named circle from outside
        forbidden = client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "team-x"},
            headers={**_HDR, "X-User-ID": "bob"},
        )
        assert forbidden.status_code == 403
        # bob enters via his personal circle, then may create named circles
        assert (
            client.post(
                "/umrs/circles",
                json={"project_id": "proj-a", "circle_id": "bob"},
                headers={**_HDR, "X-User-ID": "bob"},
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/umrs/circles",
                json={"project_id": "proj-a", "circle_id": "team-x"},
                headers={**_HDR, "X-User-ID": "bob"},
            ).status_code
            == 201
        )
        # for_user minting is admin-only
        denied = client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "zoe", "for_user": "zoe"},
            headers=_HDR,
        )
        assert denied.status_code == 403
        assert (
            client.post(
                "/umrs/circles",
                json={"project_id": "proj-a", "circle_id": "zoe", "for_user": "zoe"},
                headers=_HDR_ADMIN,
            ).status_code
            == 201
        )


class TestListing:
    def test_mine_derives_projects(self):
        client = _client(projects=("proj-a", "proj-b"))
        for pid in ("proj-a", "proj-b"):
            assert (
                client.post(
                    "/umrs/circles",
                    json={"project_id": pid, "circle_id": "alice"},
                    headers=_HDR,
                ).status_code
                == 201
            )
        mine = client.get("/umrs/circles/mine", headers=_HDR)
        assert mine.status_code == 200
        assert mine.json()["projects"] == ["proj-a", "proj-b"]

    def test_project_listing_visibility(self):
        client = _client()
        client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "alice"},
            headers=_HDR,
        )
        client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "team-x"},
            headers=_HDR,
        )
        # affiliated alice sees the project roster
        listing = client.get("/umrs/circles/proj-a", headers=_HDR)
        assert {c["circle_id"] for c in listing.json()["circles"]} == {
            "alice",
            "team-x",
        }
        # unaffiliated bob sees only his own rows (none yet)
        bob = client.get("/umrs/circles/proj-a", headers={**_HDR, "X-User-ID": "bob"})
        assert bob.json()["circles"] == []
        # the project owner sees the roster without membership
        owner = client.get("/umrs/circles/proj-a", headers={**_HDR, "X-User-ID": "owner-a"})
        assert len(owner.json()["circles"]) == 2

    def test_get_unknown_404s(self):
        client = _client()
        resp = client.get("/umrs/circles/proj-a/c-ghost", headers=_HDR)
        assert resp.status_code == 404


class TestMembership:
    def _mk_team(self, client) -> None:
        # enter the project (personal circle), then create the named circle
        client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "alice"},
            headers=_HDR,
        )
        resp = client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "team-x"},
            headers=_HDR,
        )
        assert resp.status_code == 201, resp.text

    def test_roster_guard_ladder(self):
        client = _client()
        self._mk_team(client)
        hdr_bob = {**_HDR, "X-User-ID": "bob"}
        # member cannot manage
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "bob"},
                headers=_HDR,
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "eve"},
                headers=hdr_bob,
            ).status_code
            == 403
        )
        # circle admin can manage
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "bob", "role": "admin"},
                headers=_HDR_ADMIN,
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "eve"},
                headers=hdr_bob,
            ).status_code
            == 201
        )
        # retire stays owner-or-platform-admin (circle admin is not enough)
        assert (
            client.patch(
                "/umrs/circles/proj-a/team-x",
                json={"action": "retire"},
                headers=hdr_bob,
            ).status_code
            == 403
        )
        assert (
            client.patch(
                "/umrs/circles/proj-a/team-x",
                json={"action": "retire"},
                headers=_HDR,
            ).status_code
            == 200
        )
        # retired refuses growth
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "zoe"},
                headers=_HDR,
            ).status_code
            == 409
        )

    def test_project_owner_run_governs_roster(self):
        """ADR-086 amendment 2026-10-19: the containing project's owner holds a
        roster rung without owning or joining the circle."""
        client = _client()
        self._mk_team(client)
        hdr_owner = {**_HDR, "X-User-ID": "owner-a"}  # fixture proj-a owner
        # in: the project owner adds a member to alice's circle
        added = client.post(
            "/umrs/circles/proj-a/team-x/members",
            json={"user_id": "zoe"},
            headers=hdr_owner,
        )
        assert added.status_code == 201, added.text
        assert any(m["user_id"] == "zoe" for m in added.json()["members"])
        # out: the project owner removes that member again
        removed = client.delete(
            "/umrs/circles/proj-a/team-x/members/zoe", headers=hdr_owner
        )
        assert removed.status_code == 200, removed.text
        assert all(m["user_id"] != "zoe" for m in removed.json()["members"])

    def test_project_owner_rung_never_covers_lifecycle(self):
        """The amendment is roster-only: retire stays circle-owner/platform-admin."""
        client = _client()
        self._mk_team(client)
        hdr_owner = {**_HDR, "X-User-ID": "owner-a"}
        assert (
            client.patch(
                "/umrs/circles/proj-a/team-x",
                json={"action": "retire"},
                headers=hdr_owner,
            ).status_code
            == 403
        )
        # and the personal-circle guard still fires before the ladder
        assert (
            client.post(
                "/umrs/circles/proj-a/alice/members",
                json={"user_id": "zoe"},
                headers=hdr_owner,
            ).status_code
            == 409
        )

    def test_foreign_project_owner_has_no_rung(self):
        """Owning ANOTHER project grants nothing here (rungs are per-project)."""
        pool = _Pool()
        pool._conn.projects["proj-a"] = ("proj-a", "owner-a", "active")
        pool._conn.projects["proj-b"] = ("proj-b", "owner-b", "active")
        client = _client(pool=pool, projects=())
        self._mk_team(client)
        # owner-b owns proj-b, holds no standing in proj-a
        hdr_b = {
            "X-User-ID": "owner-b",
            "X-Verified-Project-ID": "proj-b",
        }
        assert (
            client.post(
                "/umrs/circles/proj-a/team-x/members",
                json={"user_id": "zoe"},
                headers=hdr_b,
            ).status_code
            == 403
        )

    def test_personal_circle_never_grows_and_owner_stays(self):
        client = _client()
        client.post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "alice"},
            headers=_HDR,
        )
        resp = client.post(
            "/umrs/circles/proj-a/alice/members",
            json={"user_id": "bob"},
            headers=_HDR_ADMIN,
        )
        assert resp.status_code == 409
        resp = client.delete("/umrs/circles/proj-a/alice/members/alice", headers=_HDR_ADMIN)
        assert resp.status_code == 409

    def test_member_remove_and_conn_return(self):
        pool = _Pool()
        client = _client(pool)
        self._mk_team(client)
        client.post(
            "/umrs/circles/proj-a/team-x/members",
            json={"user_id": "bob"},
            headers=_HDR,
        )
        gone = client.delete("/umrs/circles/proj-a/team-x/members/bob", headers=_HDR)
        assert gone.status_code == 200
        remaining = {m["user_id"] for m in gone.json()["members"]}
        assert remaining == {"alice"}
        assert pool.returned >= 3  # every call's generator-dep putconn ran

    def test_bad_role_is_400(self):
        client = _client()
        self._mk_team(client)
        resp = client.post(
            "/umrs/circles/proj-a/team-x/members",
            json={"user_id": "bob", "role": "boss"},
            headers=_HDR,
        )
        assert resp.status_code == 400


class TestStoreGuards:
    def test_member_role_and_affiliated(self):
        conn = _Conn()
        store = CirclesStore(conn)
        store.create("proj-a", "team-x", owner_id="alice")
        assert store.member_role("proj-a", "team-x", "alice") == "admin"
        store.add_member("proj-a", "team-x", "bob", role="member", added_by="alice")
        assert store.member_role("proj-a", "team-x", "bob") == "member"
        assert store.member_role("proj-a", "team-x", "eve") is None
        assert store.affiliated("proj-a", "bob")
        assert not store.affiliated("proj-b", "bob")

    def test_pool_absent_is_typed_503(self):
        app = FastAPI()
        app.include_router(router)
        resp = TestClient(app).post(
            "/umrs/circles",
            json={"project_id": "proj-a", "circle_id": "c-x"},
            headers=_HDR,
        )
        assert resp.status_code == 503
        assert resp.headers.get("x-platform-error-code") == "circles.unavailable"
