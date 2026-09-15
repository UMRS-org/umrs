"""Tests for UserStore OIDC subject mapping (oidc_subject field)."""

import json
import tempfile
from pathlib import Path

import pytest

from umrs.admin.user_store import UserRecord, UserStore, UserStoreError


@pytest.fixture()
def store(tmp_path):
    return UserStore(user_files_dir=tmp_path)


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
    """If both a username and an oidc_subject match, username wins."""
    store.create("alice", oidc_subject="bob")
    store.create("bob")
    rec = store.resolve_user("bob")
    assert rec.username == "bob"


def test_resolve_user_not_found_raises(store):
    store.create("alice")
    with pytest.raises(UserStoreError, match="No user found"):
        store.resolve_user("999999")
