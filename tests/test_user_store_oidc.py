"""Tests for UserStore OIDC subject mapping (oidc_subject field)."""

import json

import pytest
from umrs.admin.user_store import UserRecord, UserStore, UserStoreError


@pytest.fixture()
def store(tmp_path, monkeypatch):
    # JSON lane only: the PG lane is a module global, and in CI an earlier
    # test can leave it wired to the shared test database, whose ``users``
    # rows (an "alice" among them) then answer this file's identity checks
    # (2026-10-23, PR #553 Test UMRS). Tests that want the PG lane patch it in.
    from umrs.admin import user_store

    monkeypatch.setattr(user_store, "_PG_POOL", None)
    monkeypatch.setattr(user_store, "_PG_VENUE", "")
    return UserStore(user_files_dir=tmp_path)


def _write_legacy(store, record):
    """A record as an older store could write it, past today's checks."""
    (store.dir / f"{record.username}.json").write_text(json.dumps(record.to_json()))


# ---------------------------------------------------------------------------
# UserRecord serialization
# ---------------------------------------------------------------------------


def test_oidc_subject_round_trip_via_json():
    rec = UserRecord(
        username="alice",
        oidc_subject="241723",
        created_at="2026-01-01T00:00:00Z",
    )
    data = rec.to_json()
    assert data["oidc_subject"] == "241723"
    restored = UserRecord.from_json(data)
    assert restored.oidc_subject == "241723"


def test_oidc_subject_omitted_when_empty():
    rec = UserRecord(username="bob")
    data = rec.to_json()
    assert "oidc_subject" not in data


def test_from_json_tolerates_missing_oidc_subject():
    data = {"username": "carol", "role": "user"}
    rec = UserRecord.from_json(data)
    assert rec.oidc_subject == ""


# ---------------------------------------------------------------------------
# UserStore.create with oidc_subject
# ---------------------------------------------------------------------------


def test_create_with_oidc_subject(store):
    rec = store.create("alice", oidc_subject="241723")
    assert rec.oidc_subject == "241723"
    # Verify persisted
    reloaded = store.get("alice")
    assert reloaded.oidc_subject == "241723"


def test_create_rejects_duplicate_oidc_subject(store):
    store.create("alice", oidc_subject="241723")
    with pytest.raises(UserStoreError, match="already mapped"):
        store.create("bob", oidc_subject="241723")


def test_create_allows_empty_oidc_subject_without_conflict(store):
    store.create("alice")
    store.create("bob")  # Both have empty oidc_subject — no conflict


# ---------------------------------------------------------------------------
# UserStore.set_oidc_subject
# ---------------------------------------------------------------------------


def test_set_oidc_subject(store):
    store.create("alice")
    rec = store.set_oidc_subject("alice", "241723")
    assert rec.oidc_subject == "241723"


def test_set_oidc_subject_rejects_duplicate(store):
    store.create("alice", oidc_subject="241723")
    store.create("bob")
    with pytest.raises(UserStoreError, match="already mapped"):
        store.set_oidc_subject("bob", "241723")


def test_set_oidc_subject_same_user_is_idempotent(store):
    store.create("alice", oidc_subject="241723")
    rec = store.set_oidc_subject("alice", "241723")
    assert rec.oidc_subject == "241723"


def test_clear_oidc_subject(store):
    store.create("alice", oidc_subject="241723")
    rec = store.set_oidc_subject("alice", "")
    assert rec.oidc_subject == ""


# ---------------------------------------------------------------------------
# UserStore.lookup_by_oidc_subject
# ---------------------------------------------------------------------------


def test_lookup_by_oidc_subject_found(store):
    store.create("alice", oidc_subject="241723")
    rec = store.lookup_by_oidc_subject("241723")
    assert rec is not None
    assert rec.username == "alice"


def test_lookup_by_oidc_subject_not_found(store):
    store.create("alice")
    assert store.lookup_by_oidc_subject("999999") is None


def test_lookup_by_oidc_subject_empty_returns_none(store):
    assert store.lookup_by_oidc_subject("") is None


# ---------------------------------------------------------------------------
# UserStore.resolve_user
# ---------------------------------------------------------------------------


def test_resolve_user_by_username(store):
    store.create("alice")
    rec = store.resolve_user("alice")
    assert rec.username == "alice"


def test_resolve_user_by_oidc_subject(store):
    store.create("alice", oidc_subject="241723")
    rec = store.resolve_user("241723")
    assert rec.username == "alice"


def test_resolve_user_prefers_direct_username(store):
    """If both a username and an oidc_subject match, username wins (a
    record written before the store refused such a subject)."""
    store.create("bob")
    _write_legacy(store, UserRecord(username="alice", oidc_subject="bob"))
    rec = store.resolve_user("bob")
    assert rec.username == "bob"


def test_resolve_user_not_found_raises(store):
    store.create("alice")
    with pytest.raises(UserStoreError, match="No user found"):
        store.resolve_user("999999")


# ---------------------------------------------------------------------------
# A subject never names another account (ADR-097 Amendment D-8, 2026-10-23)
# ---------------------------------------------------------------------------


class TestSubjectNamingAnotherAccount:
    """``bob.oidc_subject == "alice"`` made one identity name two accounts:
    the standing check counted the projects ``alice`` owns as bob's."""

    def test_create_refuses_a_subject_that_is_another_username(self, store):
        store.create("alice")
        with pytest.raises(UserStoreError, match="username of another user"):
            store.create("bob", oidc_subject="alice")
        assert not store.exists("bob")

    def test_create_refuses_a_subject_naming_a_row_only_account(self, store, monkeypatch):
        from umrs.admin import user_store

        rows = {"alice": UserRecord(username="alice")}
        monkeypatch.setattr(user_store, "_PG_POOL", object())
        monkeypatch.setattr(user_store, "pg_get", rows.get)
        with pytest.raises(UserStoreError, match="username of another user"):
            store.create("bob", oidc_subject="alice")

    def test_create_refuses_a_username_that_is_another_subject(self, store):
        store.create("bob", oidc_subject="alice")  # no alice yet: bob's IdP name
        with pytest.raises(UserStoreError, match="OIDC subject of user 'bob'"):
            store.create("alice")

    def test_set_oidc_subject_refuses_another_username(self, store):
        store.create("alice")
        store.create("bob")
        with pytest.raises(UserStoreError, match="username of another user"):
            store.set_oidc_subject("bob", "alice")
        assert store.get("bob").oidc_subject == ""

    def test_the_accounts_own_name_and_idp_ids_stay_allowed(self, store):
        store.create("alice", oidc_subject="alice")
        store.create("bob")
        assert store.set_oidc_subject("bob", "Bob.Tanaka").oidc_subject == "Bob.Tanaka"
        assert store.set_oidc_subject("bob", "carol").oidc_subject == "carol"
        assert store.set_oidc_subject("alice", "").oidc_subject == ""

    def test_names_account(self, store):
        store.create("alice")
        assert store.names_account("alice") is True
        assert store.names_account("ghost") is False
        assert store.names_account("241723") is False
