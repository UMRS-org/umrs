"""One access guard for shareable resources (ADR-080).

Every family parses the same stored list and validates by the same rules:
``user:<id>`` | ``project:<id>`` | ``group:<id>`` | ``system``, several
entries of each kind allowed; a caller may hold many group memberships —
any match admits. Absent/malformed lists grant nothing.
"""

from __future__ import annotations

import pytest
from umrs.shared.access_scopes import (
    MAX_ACCESS_ENTRIES,
    AccessListError,
    ResourceCaller,
    access_admits,
    stored_access_scopes,
    validate_access,
)


def _caller(user="alice", project="proj-a", groups=(), admin=False, calls=None):
    def resolve():
        if calls is not None:
            calls.append(user)
        return groups

    return ResourceCaller(user, project, is_admin=admin, groups_resolver=resolve)


# ── admission: multiple users AND multiple groups on one element ──────────


@pytest.mark.parametrize(
    ("user", "groups", "expected"),
    [
        ("alice", (), True),  # direct user entry
        ("brenda", (), True),  # second direct user entry
        ("carol", ("lab",), True),  # first group entry via membership
        ("dan", ("ops",), True),  # second group entry via membership
        ("erin", ("lab", "ops"), True),  # multi-membership: either admits
        ("frank", ("lab", "other"), True),  # multi-membership: one match is enough
        ("grace", ("other",), False),  # wrong group only
        ("heidi", (), False),  # nobody named
    ],
)
def test_one_list_admits_many_users_and_groups(user, groups, expected):
    access = ["user:alice", "user:brenda", "group:lab", "group:ops"]
    assert access_admits(access, _caller(user=user, groups=groups)) is expected


def test_project_and_system_scopes():
    access = ["project:proj-a"]
    assert access_admits(access, _caller(project="proj-a"))
    assert not access_admits(access, _caller(project="proj-b"))
    assert access_admits(["system"], _caller(project="proj-b"))


def test_absent_and_malformed_grant_nothing():
    assert not access_admits(None, _caller())
    assert not access_admits([], _caller())
    assert not access_admits("user:alice", _caller(user="alice"))  # not a list
    assert not access_admits(["garbage", "session:s1"], _caller())
    malformed = stored_access_scopes(["garbage", "user:alice"])
    assert [str(s) for s in malformed] == ["user:alice"]  # good entries kept


def test_group_lookup_is_lazy_and_memoized():
    calls: list[str] = []
    caller = _caller(groups=("lab",), calls=calls)
    access_admits(["user:someone-else"], caller)
    assert calls == []  # no group scope → no lookup
    assert not access_admits(["group:nope"], caller)
    assert not access_admits(["group:alsonope"], caller)
    assert calls == ["alice"]  # one lookup, memoized


# ── validation: who may set what ─────────────────────────────────────────


def test_validate_normalizes_and_dedupes():
    caller = _caller()
    assert validate_access(None, caller) == []
    assert validate_access("user:bob", caller) == ["user:bob"]
    assert validate_access(["user:bob", "user:bob", "group:lab"], caller) == [
        "user:bob",
        "group:lab",
    ]


def test_validate_refuses_system_to_non_admins():
    with pytest.raises(AccessListError):
        validate_access(["system"], _caller(admin=False))
    assert validate_access(["system"], _caller(admin=True)) == ["system"]


def test_validate_refuses_bad_kinds_and_caps():
    with pytest.raises(AccessListError):
        validate_access(["session:s1"], _caller())
    with pytest.raises(AccessListError):
        validate_access(["garbage"], _caller())
    with pytest.raises(AccessListError):
        validate_access([f"user:u{i}" for i in range(MAX_ACCESS_ENTRIES + 1)], _caller())


def test_multi_membership_comes_from_one_authority():
    """The SQL authority joins active memberships — several rows per user."""
    from umrs.shared.group_membership import group_ids_for_user

    class _Cur:
        def __init__(self):
            self.sql = ""

        def execute(self, sql, params):
            self.sql = " ".join(sql.split())
            self.params = params

        def fetchall(self):
            return [("lab",), ("ops",), ("fieldwork",)]

    cur = _Cur()
    conn = type("C", (), {"cursor": lambda self: cur})()
    assert group_ids_for_user(conn, "alice") == ["lab", "ops", "fieldwork"]
    assert "g.status = 'active'" in cur.sql  # retired groups never resolve
    assert cur.params == ("alice",)
