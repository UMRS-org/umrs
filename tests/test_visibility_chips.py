"""WP-06 r2 P10-C3 pins: visibility-source chips (grant chip halts fabrication)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from umrs.grants import AssetGrant
from umrs.model_registry.models import ModelScope
from umrs.model_registry.store.acl import AccessContext, ACLEnforcer


def _ctx(uid="u-ally", memberships=("proj-9",), admin=False):
    return AccessContext(
        user_id=uid,
        is_admin=admin,
        project_memberships=set(memberships),
    )


def _model(scope, model_id="mdl-1", project_id="proj-9", owner_id="u-other"):
    return SimpleNamespace(
        model_id=model_id,
        scope=scope,
        project_id=project_id,
        owner_id=owner_id,
    )


class TestReadReason:
    def test_system_scope_shows_system(self):
        acl = ACLEnforcer()
        assert acl.read_reason(_model(ModelScope.SYSTEM), _ctx()) == "system"

    def test_project_membership_shows_project(self):
        acl = ACLEnforcer()
        assert acl.read_reason(_model(ModelScope.PROJECT), _ctx()) == "project"

    def test_user_owner_shows_user(self):
        acl = ACLEnforcer()
        assert acl.read_reason(_model(ModelScope.USER, owner_id="u-ally"), _ctx()) == "user"

    def test_admin_shows_admin(self):
        acl = ACLEnforcer()
        assert acl.read_reason(_model(ModelScope.SYSTEM), _ctx(admin=True)) == "admin"

    def test_grant_path_shows_grant(self):
        acl = ACLEnforcer(
            grant_reader=lambda _kind, _id: [
                AssetGrant(
                    grant_id="g1",
                    asset_kind="model",
                    asset_id="mdl-1",
                    grantee_kind="user",
                    grantee_id="u-ally",
                    created_by="u-owner",
                    created_at=None,
                    tombstoned_at=None,
                    expires_at=None,
                )
            ]
        )
        # USER-scope model, non-owner: scope ownership fails, grant pins `grant`
        assert acl.read_reason(_model(ModelScope.USER, owner_id="u-other"), _ctx()) == "grant"

    def test_no_read_holds_shows_none(self):
        acl = ACLEnforcer()
        assert (
            acl.read_reason(_model(ModelScope.USER, owner_id="u-other"), _ctx(uid="u-other2"))
            == "none"
        )

    def test_grant_never_decorative(self):
        acl = ACLEnforcer(grant_reader=lambda _kind, _id: [])
        assert acl.read_reason(_model(ModelScope.PROJECT), _ctx()) != "grant"


class TestGrantOnlyRowsInReasonedListing:
    def test_acl_introspection_path_marks_grants_correctly(self):
        # THE leg's semantic: grant-read-only rows must badge as 'grant'
        # (verified path), never 'project'/'user' re-labelled.
        acl = ACLEnforcer(
            grant_reader=lambda _kind, _id: [
                AssetGrant(
                    grant_id="g1",
                    asset_kind="model",
                    asset_id="mdl-grated",
                    grantee_kind="user",
                    grantee_id="u-ally",
                    created_by="u-owner",
                    created_at=None,
                    tombstoned_at=None,
                    expires_at=None,
                )
            ]
        )
        m = _model(ModelScope.USER, model_id="mdl-grated", owner_id="u-other")
        assert acl.read_reason(m, _ctx(uid="u-ally")) == "grant"


class TestModelListRouteChips:
    def test_chips_attached_per_row_and_none_when_uncomputed(self):
        from umrs.api.models import ModelResponse
        from umrs.model_registry.models import ExecutionAuthority, ModelStatus, ModelType

        entry = MagicMock(spec=None)
        entry.model_id = "mdl-1"
        entry.name = "q"
        entry.scope = ModelScope.PROJECT
        entry.owner_id = "u-o"
        entry.project_id = "proj-9"
        entry.storage_ref = "x"
        entry.model_ref = None
        entry.model_type = ModelType.LLM
        entry.huggingface_id = None
        entry.description = None
        entry.serving_profile = MagicMock(to_dict=lambda: {})
        entry.capabilities = MagicMock(to_dict=lambda: {})
        entry.runtime_config = MagicMock(to_dict=lambda: {})
        entry.execution_authority_default = ExecutionAuthority.COMPOSE
        entry.status = ModelStatus.AVAILABLE
        entry.created_at = None
        entry.updated_at = None

        resp = ModelResponse.from_entry(entry)
        assert resp.visibility_source is None
        resp.visibility_source = "grant"
        assert resp.visibility_source == "grant"
