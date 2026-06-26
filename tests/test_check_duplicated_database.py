"""Regression tests for per-user database namespacing + non-fatal duplicates.

With Option B, the Postgres database name includes the platform username, so
two *different* users that pick the same ``(xp_name, sql_username)`` pair get
isolated databases (no collision). The only remaining collision is a single
user listing the same pair twice — which must be quarantined, NOT crash the
whole server (previously ``check_duplicated_database`` called ``sys.exit``).
"""

from umrs.server.maintenance import check_duplicated_database


def _user(username, experiences):
    return {"username": username, "experiences": experiences}


def _xp(xp_name, sql_username):
    return {"xp_name": xp_name, "sql_username": sql_username}


def test_cross_user_same_names_are_now_isolated():
    """Two users sharing (xp_name, sql_username) keep BOTH experiences.

    Per-user namespacing means their databases differ, so there is no
    collision to quarantine.
    """
    users = [
        _user("mnagaso", [_xp("Opossum Analysis", "demo_opossum")]),
        _user("mnagaso3", [_xp("Opossum Analysis", "demo_opossum")]),
    ]
    check_duplicated_database(users)
    assert len(users[0]["experiences"]) == 1
    assert len(users[1]["experiences"]) == 1


def test_does_not_exit_on_intra_user_collision():
    """A single user listing the same pair twice must not abort startup."""
    users = [
        _user(
            "alice",
            [_xp("Shared", "shared_db"), _xp("Shared", "shared_db")],
        ),
    ]
    # Must return normally (no SystemExit).
    check_duplicated_database(users)


def test_intra_user_duplicate_is_quarantined():
    users = [
        _user(
            "alice",
            [_xp("Shared", "shared_db"), _xp("Shared", "shared_db")],
        ),
    ]
    check_duplicated_database(users)
    # The duplicate second copy is dropped; the first is kept.
    assert len(users[0]["experiences"]) == 1


def test_intra_user_collision_keeps_other_experiences():
    users = [
        _user(
            "bob",
            [
                _xp("Shared", "shared_db"),
                _xp("Shared", "shared_db"),
                _xp("Bob Only", "bob_priv"),
            ],
        ),
    ]
    check_duplicated_database(users)
    kept = [xp["xp_name"] for xp in users[0]["experiences"]]
    assert kept == ["Shared", "Bob Only"]


def test_distinct_experiences_are_preserved():
    users = [
        _user("alice", [_xp("default", "alice"), _xp("Project X", "alice_x")]),
        _user("bob", [_xp("default", "bob")]),
    ]
    check_duplicated_database(users)
    assert len(users[0]["experiences"]) == 2
    assert len(users[1]["experiences"]) == 1


def test_case_and_space_insensitive_intra_user_collision():
    """create_db_name lowercases and replaces spaces, so these collide."""
    users = [
        _user("alice", [_xp("My XP", "shared"), _xp("my xp", "Shared")]),
    ]
    check_duplicated_database(users)
    assert len(users[0]["experiences"]) == 1
