"""Groups + membership authority pins (ADR-050 §3 groundwork wave, 2026-09-16).

Store + route invariants: ensure-shape creation with creator-as-first-member,
mine-list crossing owner+membership, owner-or-admin membership ops, retired
groups refusing growth and vanishing from list_mine, unknown ids 404, and the
pool conn always returned (generator dependency).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.groups import GroupNotFound, GroupsStore, groups_for_user, router


def _in(haystack: str, *needles: str) -> bool:
    return all(n in haystack for n in needles)


class _Cursor:
    """In-memory groups + group_members swallowing the SQL GroupsStore emits."""

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
            "insert into groups": lambda p: conn.groups.setdefault(
                p[0], (p[0], p[1], p[2], "active", None, None, None)
            ),
            "insert into group_members": lambda p: conn.members_add(p[0], p[1], p[2]),
            "delete from group_members": lambda p: conn.members.discard((p[0], p[1])),
            "update groups": self._write_retire,
        }

    def _write_retire(self, params):
        gid = params[0]
        conn = self._conn
        if gid in conn.groups and conn.groups[gid][3] != "retired":
            g = conn.groups[gid]
            conn.groups[gid] = (g[0], g[1], g[2], "retired", datetime.now(timezone.utc), None, None)
            self.rowcount = 1

    def _read(self, low, params):
        conn = self._conn
        if _in(low, "where group_id = %s", "from group_members"):
            self._result = [
                (m[1], "member", conn._by.get((m[0], m[1]), ""), None)
                for m in sorted(conn.members)
                if m[0] == params[0]
            ]
        elif _in(low, "from groups where group_id = %s"):
            row = conn.groups.get(params[0])
            self._result = [row] if row else []
        elif _in(low, "from group_members gm", "join groups"):
            self._result = [(g,) for g in _active_groups_of(conn, params[0])]
        elif self._is_groups_list(low):
            self._result = self._read_groups_list(low, params)
        else:
            self._result = []

    def _is_groups_list(self, low) -> bool:
        return _in(low, "order by g.created_at") or _in(low, "order by created_at")

    def _read_groups_list(self, low, params):
        conn = self._conn
        if _in(low, "left join group_members"):
            return self._mine(conn, low, params)
        rows = list(conn.groups.values())
        if _in(low, "owner_id = %s"):
            rows = [r for r in rows if r[2] == params[0]]
        if _in(low, "status <> 'retired'"):
            rows = [r for r in rows if r[3] != "retired"]
        return rows

    def _mine(self, conn, low, params):
        owned = {g for g in conn.groups if conn.groups[g][2] == params[0]}
        joined = {gid for (gid, uid) in conn.members if uid == params[1]}
        rows = [conn.groups[g] for g in sorted((owned | joined) & set(conn.groups))]
        if _in(low, "status <> 'retired'"):
            rows = [r for r in rows if r[3] != "retired"]
        return rows


class _Conn:
    def __init__(self):
        self.groups: dict[str, tuple] = {}
        self.members: set[tuple[str, str]] = set()
        self._by: dict[tuple[str, str], str] = {}

    def members_add(self, gid, uid, by):
        self.members.add((gid, uid))
        self._by[(gid, uid)] = by

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


def _active_groups_of(conn, user_id):
    joined = {m[0] for m in conn.members if m[1] == user_id}
    live = {g for g, r in conn.groups.items() if r[3] == "active"}
    return sorted(joined & live)


def _client(pool=None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool or _Pool()
    return TestClient(app)


_HDR = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}
_HDR_ADMIN = {**_HDR, "X-Verified-Is-Admin": "true"}


class TestStoreAndResolution:
    def test_create_ensures_and_creator_is_first_member(self):
        store = GroupsStore(_Conn())
        row = store.create("g-lab", name="Lab", owner_id="alice")
        assert row["owner_id"] == "alice"
        again = store.create("g-lab", name="Lab B", owner_id="bob")
        assert again["owner_id"] == "alice" and again["name"] == "Lab"
        assert any(m["user_id"] == "alice" for m in store.members("g-lab"))

    def test_groups_for_user_filters_retired(self):
        conn = _Conn()
        conn.groups["g1"] = ("g1", "", "x", "active", None, None, None)
        conn.groups["g2"] = ("g2", "", "x", "retired", None, None, None)
        conn.members |= {("g1", "alice"), ("g2", "alice")}
        assert groups_for_user(conn, "alice") == ["g1"]

    def test_unknown_group_raises(self):
        with pytest.raises(GroupNotFound):
            GroupsStore(_Conn()).get("g-ghost")


class TestRoutes:
    def test_create_get_mine_flow(self):
        client = _client()
        resp = client.post("/umrs/groups", json={"group_id": "g-lab", "name": "Lab"}, headers=_HDR)
        assert resp.status_code == 201, resp.text
        one = client.get("/umrs/groups/g-lab", headers=_HDR)
        assert one.status_code == 200 and one.json()["group"]["owner_id"] == "alice"
        mine = client.get("/umrs/groups", headers=_HDR)
        ids = {g["group_id"] for g in mine.json()["groups"]}
        assert "g-lab" in ids
        # another user's number stays invisible to me
        other = client.get("/umrs/groups", headers={**_HDR, "X-User-ID": "bob"})
        assert "g-lab" not in {g["group_id"] for g in other.json()["groups"]}
        admin = client.get("/umrs/groups", headers=_HDR_ADMIN)
        assert admin.json()["count"] >= 1

    def test_unknown_id_404s(self):
        client = _client()
        resp = client.get("/umrs/groups/g-ghost", headers=_HDR)
        assert resp.status_code == 404

    def test_membership_owner_or_admin_only_and_retire_blocks_growth(self):
        client = _client()
        client.post("/umrs/groups", json={"group_id": "g-lab"}, headers=_HDR)
        # owner can add
        ok = client.post("/umrs/groups/g-lab/members", json={"user_id": "bob"}, headers=_HDR)
        assert ok.status_code == 201, ok.text
        # bob (non-owner, non-admin) may not
        forbidden = client.post(
            "/umrs/groups/g-lab/members",
            json={"user_id": "eve"},
            headers={**_HDR, "X-User-ID": "bob"},
        )
        assert forbidden.status_code == 403
        # admin can
        adm = client.post("/umrs/groups/g-lab/members", json={"user_id": "eve"}, headers=_HDR_ADMIN)
        assert adm.status_code == 201
        # retire only by owner (or admin)
        not_allowed = client.patch(
            "/umrs/groups/g-lab",
            json={"action": "retire"},
            headers={**_HDR, "X-User-ID": "bob"},
        )
        assert not_allowed.status_code == 403
        retired = client.patch("/umrs/groups/g-lab", json={"action": "retire"}, headers=_HDR)
        assert retired.status_code == 200
        assert retired.json()["group"]["status"] == "retired"
        # growth of a retired group refused
        add = client.post("/umrs/groups/g-lab/members", json={"user_id": "zoe"}, headers=_HDR)
        assert add.status_code == 409

    def test_member_remove_and_conn_return(self):
        pool = _Pool()
        client = _client(pool)
        client.post("/umrs/groups", json={"group_id": "g-lab"}, headers=_HDR)
        client.post("/umrs/groups/g-lab/members", json={"user_id": "bob"}, headers=_HDR)
        gone = client.delete("/umrs/groups/g-lab/members/bob", headers=_HDR)
        assert gone.status_code == 200
        remaining = {m["user_id"] for m in gone.json()["members"]}
        assert remaining == {"alice"}
        assert pool.returned >= 3  # every call's generator-dep putconn ran

    def test_pool_absent_is_typed_503(self):
        app = FastAPI()
        app.include_router(router)
        resp = TestClient(app).post("/umrs/groups", json={"group_id": "g-x"}, headers=_HDR)
        assert resp.status_code == 503
        assert resp.headers.get("x-platform-error-code") == "groups.unavailable"
