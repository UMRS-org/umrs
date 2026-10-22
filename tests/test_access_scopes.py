"""One access guard for shareable resources (ADR-080; circles per ADR-086).

Every family parses the same stored list and validates by the same rules:
``user:<id>`` | ``project:<id>`` | ``circle:<proj>/<id>`` | ``system``,
several entries of each kind allowed; a caller may hold many circle
memberships — any match admits. Absent/malformed lists grant nothing, and
superseded ``group:`` entries admit no one and are rejected on writes.
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


def _caller(user="alice", project="proj-a", circles=(), admin=False, calls=None):
    def resolve():
        if calls is not None:
            calls.append(user)
        return circles

    return ResourceCaller(user, project, is_admin=admin, circles_resolver=resolve)


# ── admission: multiple users AND multiple circles on one element ─────────


@pytest.mark.parametrize(
    ("user", "circles", "expected"),
    [
        ("alice", (), True),  # direct user entry
        ("brenda", (), True),  # second direct user entry
        ("carol", ("proj-a/lab",), True),  # first circle entry via membership
        ("dan", ("proj-a/ops",), True),  # second circle entry via membership
        ("erin", ("proj-a/lab", "proj-a/ops"), True),  # multi: either admits
        ("frank", ("proj-a/lab", "proj-b/other"), True),  # one match is enough
        ("grace", ("proj-a/other",), False),  # wrong circle only
        ("heidi", (), False),  # nobody named
    ],
)
def test_one_list_admits_many_users_and_circles(user, circles, expected):
    access = [
        "user:alice",
        "user:brenda",
        "circle:proj-a/lab",
        "circle:proj-a/ops",
    ]
    assert access_admits(access, _caller(user=user, circles=circles)) is expected


def test_project_and_system_scopes():
    access = ["project:proj-a"]
    assert access_admits(access, _caller(project="proj-a"))
    assert not access_admits(access, _caller(project="proj-b"))
    assert access_admits(["system"], _caller(project="proj-b"))


def test_group_entries_admit_no_one_and_are_rejected_on_writes():
    # ADR-086: the superseded platform-global audience kind is fail-closed.
    assert not access_admits(["group:lab"], _caller(circles=("proj-a/lab",)))
    with pytest.raises(AccessListError):
        validate_access(["group:lab"], _caller())


def test_absent_and_malformed_grant_nothing():
    assert not access_admits(None, _caller())
    assert not access_admits([], _caller())
    assert not access_admits("user:alice", _caller(user="alice"))  # not a list
    assert not access_admits(["garbage", "session:s1"], _caller())
    malformed = stored_access_scopes(["garbage", "user:alice"])
    assert [str(s) for s in malformed] == ["user:alice"]  # good entries kept


def test_circle_lookup_is_lazy_and_memoized():
    calls: list[str] = []
    caller = _caller(circles=("proj-a/lab",), calls=calls)
    access_admits(["user:someone-else"], caller)
    assert calls == []  # no circle scope → no lookup
    assert not access_admits(["circle:proj-a/nope"], caller)
    assert not access_admits(["circle:proj-a/alsonope"], caller)
    assert calls == ["alice"]  # one lookup, memoized


def test_circle_refs_need_the_project_shape():
    with pytest.raises(AccessListError):
        validate_access(["circle:noslash"], _caller())
    assert validate_access(["circle:proj-a/lab"], _caller()) == ["circle:proj-a/lab"]


# ── validation: who may set what ─────────────────────────────────────────


def test_validate_normalizes_and_dedupes():
    caller = _caller()
    assert validate_access(None, caller) == []
    assert validate_access("user:bob", caller) == ["user:bob"]
    assert validate_access(["user:bob", "user:bob", "circle:proj-a/lab"], caller) == [
        "user:bob",
        "circle:proj-a/lab",
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
    from umrs.shared.circle_membership import circle_refs_for_user

    class _Cur:
        def __init__(self):
            self.sql = ""

        def execute(self, sql, params):
            self.sql = " ".join(sql.split())
            self.params = params

        def fetchall(self):
            return [("proj-a/lab",), ("proj-a/ops",), ("proj-b/fieldwork",)]

    cur = _Cur()
    conn = type("C", (), {"cursor": lambda self: cur})()  # noqa: ARG005 — test double mirrors the real signature
    assert circle_refs_for_user(conn, "alice") == [
        "proj-a/lab",
        "proj-a/ops",
        "proj-b/fieldwork",
    ]
    assert "c.status = 'active'" in cur.sql  # retired circles never resolve
    assert cur.params == ("alice",)


# ── a circle entry must name an existing circle (2026-10-22) ─────────────
# `circle:BDR/opossum` was stored for two workflows while the circle is
# `bdr/opossum`; matching is exact, so the share admitted no one, silently.


def _finder(*active: str):
    asked: list[str] = []

    def find(ref: str) -> str | None:
        asked.append(ref)
        return next((a for a in active if a == ref), None) or next(
            (a for a in active if a.lower() == ref.lower()), None
        )

    return find, asked


def _caller_with(finder) -> ResourceCaller:
    return ResourceCaller("alice", "proj-a", circle_finder=finder)


def test_an_existing_circle_is_accepted():
    find, _ = _finder("bdr/opossum")
    assert validate_access(["circle:bdr/opossum"], _caller_with(find)) == ["circle:bdr/opossum"]


def test_a_case_only_mismatch_is_refused_with_the_circle_it_meant():
    find, _ = _finder("bdr/opossum")
    with pytest.raises(AccessListError, match=r"no active circle BDR/opossum .*circle:bdr/opossum"):
        validate_access(["circle:BDR/opossum"], _caller_with(find))


def test_an_unknown_circle_is_refused():
    find, _ = _finder("bdr/opossum")
    with pytest.raises(AccessListError, match="no active circle lab/none$"):
        validate_access(["circle:lab/none"], _caller_with(find))


def test_only_circle_entries_are_looked_up():
    find, asked = _finder("bdr/opossum")
    validate_access(["user:bob", "project:p", "circle:bdr/opossum"], _caller_with(find))
    assert asked == ["bdr/opossum"]


def test_without_a_finder_the_entry_is_kept_as_before():
    assert validate_access(["circle:BDR/opossum"], _caller()) == ["circle:BDR/opossum"]

