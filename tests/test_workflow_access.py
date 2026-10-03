"""Workflow access scopes (ADR-080).

Catalog entries carry ``access`` (user:/project:/group:/system, several
entries; multi-membership admitted). Run resolution order: own → system →
access-shared → admin-foreign. ``project:`` entries admit only a
gateway-verified caller project; a durable re-dispatch never re-resolves
(the job spec carries the source_path from the first, guarded resolution).
"""

from __future__ import annotations

import json

import pytest
from umrs.server import workflow_registry_handler as wrh


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path, monkeypatch):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    monkeypatch.setattr(wrh, "_catalog_path", lambda: catalog_file)
    wrh.set_workflow_groups_resolver(None)
    yield catalog_file
    wrh.set_workflow_groups_resolver(lambda _user: [])  # no groups off-test


@pytest.fixture
def alice_wf(tmp_catalog):
    wrh.register_workflow(
        workflow_name="cnn_track",
        version="1.0",
        user_id="alice",
        source_path="/tmp/cnn_track",
        workflow_json={"steps": [{"name": "s1", "entrypoint": "m:S"}], "version": "1.0"},
    )
    return "cnn_track"


def _groups(mapping):
    return lambda user: list(mapping.get(user, []))


class TestRunResolution:
    def test_other_owner_cannot_resolve_by_default(self, alice_wf):
        assert wrh.resolve_workflow_for_caller(alice_wf, "bob") is None
        assert wrh.resolve_workflow_for_caller(alice_wf, "alice") is not None

    def test_user_scope_admits_the_named_user(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["user:bob"])
        entry = wrh.resolve_workflow_for_caller(alice_wf, "bob")
        assert entry is not None and entry["user_id"] == "alice"
        assert wrh.resolve_workflow_for_caller(alice_wf, "carol") is None

    def test_multiple_users_and_groups_admit(self, alice_wf):
        wrh.set_workflow_access(
            alice_wf, "alice", ["user:bob", "user:brenda", "group:lab", "group:ops"]
        )
        wrh.set_workflow_groups_resolver(
            _groups({"carol": ["lab"], "dan": ["ops"], "erin": ["lab", "ops"]})
        )
        assert wrh.resolve_workflow_for_caller(alice_wf, "bob") is not None
        assert wrh.resolve_workflow_for_caller(alice_wf, "brenda") is not None
        assert wrh.resolve_workflow_for_caller(alice_wf, "carol") is not None
        assert wrh.resolve_workflow_for_caller(alice_wf, "dan") is not None
        assert wrh.resolve_workflow_for_caller(alice_wf, "erin") is not None  # multi-membership
        assert wrh.resolve_workflow_for_caller(alice_wf, "frank") is None

    def test_group_lookup_is_lazy(self, alice_wf):
        calls: list[str] = []
        wrh.set_workflow_groups_resolver(lambda u: calls.append(u) or [])
        wrh.set_workflow_access(alice_wf, "alice", ["user:bob"])
        assert wrh.resolve_workflow_for_caller(alice_wf, "bob") is not None
        assert calls == []  # a user: entry never triggers the group lookup

    def test_project_scope_needs_a_verified_project(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["project:proj-b"])
        assert wrh.resolve_workflow_for_caller(
            alice_wf, "bob", caller_project="proj-b"
        ) is not None
        assert wrh.resolve_workflow_for_caller(alice_wf, "bob") is None
        assert wrh.resolve_workflow_for_caller(
            alice_wf, "bob", caller_project="proj-c"
        ) is None

    def test_system_access_entry_shares_with_everyone(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["system"])
        assert wrh.resolve_workflow_for_caller(alice_wf, "anyone") is not None

    def test_own_entry_beats_a_shared_foreign_one(self, alice_wf):
        wrh.register_workflow(
            workflow_name="cnn_track",
            version="2.0",
            user_id="bob",
            source_path="/tmp/bob_cnn",
            workflow_json={"steps": [], "version": "2.0"},
        )
        wrh.set_workflow_access(alice_wf, "alice", ["user:bob"])
        entry = wrh.resolve_workflow_for_caller(alice_wf, "bob")
        assert entry["user_id"] == "bob"  # his own, not the shared one

    def test_admin_foreign_fallback_unchanged(self, alice_wf):
        entry = wrh.resolve_workflow_for_caller(alice_wf, "ops", is_admin=True)
        assert entry is not None and entry["user_id"] == "alice"


class TestAccessMutation:
    def test_set_workflow_access_covers_all_versions(self, tmp_catalog):  # noqa: ARG002 — the fixture points the registry at a temp catalog
        wrh.register_workflow(
            workflow_name="wf", version="1.0", user_id="alice",
            source_path="/tmp/a", workflow_json={"steps": []},
        )
        wrh.register_workflow(
            workflow_name="wf", version="2.0", user_id="alice",
            source_path="/tmp/b", workflow_json={"steps": []},
        )
        updated = wrh.set_workflow_access("wf", "alice", ["user:bob"])
        assert len(updated) == 2
        assert all(e["access"] == ["user:bob"] for e in updated)
        assert wrh.workflow_access_entries("wf", "alice")
        assert wrh.set_workflow_access("wf", "carol", ["user:bob"]) == []

    def test_reupload_preserves_the_access_list(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["group:lab"])
        wrh.register_workflow(  # re-upload of the same (name, version, scope, user)
            workflow_name=alice_wf, version="1.0", user_id="alice",
            source_path="/tmp/cnn_track_v2",
            workflow_json={"steps": [], "version": "1.0"},
        )
        entry = wrh.resolve_workflow_for_caller(alice_wf, "alice")
        assert entry["access"] == ["group:lab"]

    def test_fresh_upload_starts_private(self, alice_wf):
        entry = wrh.resolve_workflow_for_caller(alice_wf, "alice")
        assert entry["access"] == []


class TestListing:
    def test_shared_entries_marked_for_admitted_callers(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["user:bob"])
        out = json.loads(
            wrh._handle_list_workflows(
                "bob", json.dumps({"user_id": "bob", "include_shared_for": "bob"})
            )
        )
        shared = [w for w in out["workflows"] if w.get("shared")]
        assert [w["name"] for w in shared] == [alice_wf]
        assert shared[0]["user_id"] == "alice"

    def test_unshared_entries_stay_hidden(self, alice_wf):  # noqa: ARG002 — the fixture registers alice's shared workflow
        out = json.loads(
            wrh._handle_list_workflows(
                "carol", json.dumps({"user_id": "carol", "include_shared_for": "carol"})
            )
        )
        assert out["workflows"] == []


class TestRunCommandGuard:
    def test_unverified_project_does_not_admit_project_scope(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["project:proj-b"])
        payload = json.dumps(
            {
                "workflow_name": alice_wf,
                "params": {"project_id": "proj-b"},
                "_identity_verified": False,  # forged claim: never trusted
            }
        )
        out = json.loads(wrh._handle_run_workflow("bob", payload, None, trusted_identity=False))
        assert out["status"] == "error"
        assert "not found in registry" in out["message"]

    def test_verified_project_admits_project_scope(self, alice_wf):
        wrh.set_workflow_access(alice_wf, "alice", ["project:proj-b"])
        payload = json.dumps(
            {
                "workflow_name": alice_wf,
                "params": {"project_id": "proj-b"},
                "_identity_verified": True,
            }
        )
        out = json.loads(wrh._handle_run_workflow("bob", payload, None, trusted_identity=True))
        # past the guard: fails honestly on the missing source dir instead
        assert "not found in registry" not in out["message"]
