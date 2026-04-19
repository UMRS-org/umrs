"""
Unit tests for Model Registry.

Tests:
- ModelEntry dataclass and enums
- PathValidator
- ACLEnforcer
- ModelStore CRUD operations
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from umrs.model_registry.models import (
    ModelEntry,
    ModelScope,
    ModelStatus,
    ModelType,
    ServingProfile,
    generate_model_id,
)
from umrs.model_registry.store import (
    AccessContext,
    ACLEnforcer,
    DuplicateModelError,
    ModelNotFoundError,
    ModelStore,
    PathNotAllowedError,
    PathValidator,
    PermissionDeniedError,
)


# =============================================================================
# Model Tests
# =============================================================================


class TestModelEntry:
    """Tests for ModelEntry dataclass."""

    def test_create_system_model(self):
        """Test creating a system-scoped model."""
        model = ModelEntry(
            model_id="mdl-test123456",
            name="gpt-4-turbo",
            scope=ModelScope.SYSTEM,
            storage_ref="/shared/models/system/gpt-4-turbo",
            model_type=ModelType.LLM,
        )
        assert model.model_id == "mdl-test123456"
        assert model.scope == ModelScope.SYSTEM
        assert model.owner_id is None
        assert model.project_id is None
        assert model.display_name == "system/gpt-4-turbo"

    def test_create_project_model(self):
        """Test creating a project-scoped model."""
        model = ModelEntry(
            model_id="mdl-proj123456",
            name="team-finetune",
            scope=ModelScope.PROJECT,
            project_id="proj-123",
            storage_ref="/shared/projects/proj-123/models/team-finetune",
            model_type=ModelType.LLM,
        )
        assert model.scope == ModelScope.PROJECT
        assert model.project_id == "proj-123"
        assert "project:proj-123/team-finetune" in model.display_name

    def test_create_user_model(self):
        """Test creating a user-scoped model."""
        model = ModelEntry(
            model_id="mdl-user123456",
            name="my-experiment",
            scope=ModelScope.USER,
            owner_id="user-456",
            storage_ref="/shared/users/user-456/models/my-experiment",
            model_type=ModelType.LLM,
        )
        assert model.scope == ModelScope.USER
        assert model.owner_id == "user-456"
        assert "user:user-456/my-experiment" in model.display_name

    def test_system_model_rejects_owner(self):
        """System models should not have owner_id."""
        with pytest.raises(ValueError, match="System models must have NULL"):
            ModelEntry(
                model_id="mdl-bad",
                name="bad",
                scope=ModelScope.SYSTEM,
                owner_id="should-not-exist",
                storage_ref="/shared/models/system/bad",
            )

    def test_project_model_requires_project_id(self):
        """Project models must have project_id."""
        with pytest.raises(ValueError, match="Project models must have"):
            ModelEntry(
                model_id="mdl-bad",
                name="bad",
                scope=ModelScope.PROJECT,
                storage_ref="/shared/projects/x/models/bad",
            )

    def test_user_model_requires_owner(self):
        """User models must have owner_id."""
        with pytest.raises(ValueError, match="User models must have"):
            ModelEntry(
                model_id="mdl-bad",
                name="bad",
                scope=ModelScope.USER,
                storage_ref="/shared/users/x/models/bad",
            )

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = ModelEntry(
            model_id="mdl-test123456",
            name="test-model",
            scope=ModelScope.PROJECT,
            project_id="proj-123",
            storage_ref="/shared/projects/proj-123/models/test",
            model_type=ModelType.LLM,
            huggingface_id="org/model",
            description="Test description",
            serving_profile=ServingProfile(tensor_parallel_size=2),
            status=ModelStatus.AVAILABLE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        data = original.to_dict()
        restored = ModelEntry.from_dict(data)

        assert restored.model_id == original.model_id
        assert restored.name == original.name
        assert restored.scope == original.scope
        assert restored.serving_profile.tensor_parallel_size == 2


class TestServingProfile:
    """Tests for ServingProfile."""

    def test_default_values(self):
        """Test default serving profile values."""
        profile = ServingProfile()
        assert profile.tensor_parallel_size == 1
        assert profile.gpu_memory_utilization == 0.9
        assert profile.dtype == "auto"
        assert profile.quantization is None

    def test_to_dict_from_dict(self):
        """Test serialization."""
        profile = ServingProfile(
            tensor_parallel_size=4,
            max_model_len=8192,
            quantization="awq",
        )
        data = profile.to_dict()
        restored = ServingProfile.from_dict(data)

        assert restored.tensor_parallel_size == 4
        assert restored.max_model_len == 8192
        assert restored.quantization == "awq"


class TestGenerateModelId:
    """Tests for model ID generation."""

    def test_format(self):
        """Test ID format."""
        model_id = generate_model_id()
        assert model_id.startswith("mdl-")
        assert len(model_id) == 16  # "mdl-" + 12 hex chars

    def test_uniqueness(self):
        """Test IDs are unique."""
        ids = {generate_model_id() for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Path Validator Tests
# =============================================================================


class TestPathValidator:
    """Tests for PathValidator."""

    def test_default_allowed_paths(self):
        """Test default path patterns are set."""
        validator = PathValidator()
        assert len(validator.allowed_patterns) > 0

    def test_system_path_allowed(self):
        """Test system model paths are allowed."""
        validator = PathValidator()
        assert validator.is_allowed("/shared/models/system/gpt-4")
        assert validator.is_allowed("/models/my-model")

    def test_project_path_allowed(self):
        """Test project model paths are allowed."""
        validator = PathValidator()
        assert validator.is_allowed("/shared/projects/proj-123/models/fine-tune")

    def test_user_path_allowed(self):
        """Test user model paths are allowed."""
        validator = PathValidator()
        assert validator.is_allowed("/shared/users/user-456/models/experiment")
        assert validator.is_allowed("/home/john/models/test")

    def test_arbitrary_path_rejected(self):
        """Test arbitrary paths are rejected."""
        validator = PathValidator()
        assert not validator.is_allowed("/etc/passwd")
        assert not validator.is_allowed("/tmp/models/hack")
        assert not validator.is_allowed("/root/secret")

    def test_validate_raises_on_bad_path(self):
        """Test validate() raises PathNotAllowedError."""
        validator = PathValidator()
        with pytest.raises(PathNotAllowedError) as exc:
            validator.validate("/etc/passwd")
        assert "/etc/passwd" in str(exc.value)

    def test_custom_patterns(self):
        """Test custom allowed patterns."""
        validator = PathValidator(allowed_patterns=["/custom/*"])
        assert validator.is_allowed("/custom/model")
        assert not validator.is_allowed("/shared/models/system/test")

    def test_scope_path_validation_system(self):
        """Test system scope path validation."""
        validator = PathValidator()
        # Valid system path
        validator.validate_scope_path(
            "/shared/models/system/gpt-4",
            ModelScope.SYSTEM,
        )
        # Invalid system path (in user directory)
        with pytest.raises(PathNotAllowedError):
            validator.validate_scope_path(
                "/shared/users/john/models/fake-system",
                ModelScope.SYSTEM,
            )

    def test_scope_path_validation_project(self):
        """Test project scope path validation."""
        validator = PathValidator()
        # Valid project path
        validator.validate_scope_path(
            "/shared/projects/proj-123/models/fine-tune",
            ModelScope.PROJECT,
            project_id="proj-123",
        )

    def test_scope_path_validation_project_rejects_wrong_project(self):
        """Test project scope path validation rejects a different project root."""
        validator = PathValidator()
        with pytest.raises(PathNotAllowedError):
            validator.validate_scope_path(
                "/shared/projects/proj-999/models/fine-tune",
                ModelScope.PROJECT,
                project_id="proj-123",
            )

    def test_scope_path_validation_user(self):
        """Test user scope path validation."""
        validator = PathValidator()
        # Valid user path
        validator.validate_scope_path(
            "/shared/users/user-456/models/experiment",
            ModelScope.USER,
            owner_id="user-456",
        )

    def test_scope_path_validation_user_rejects_other_users_experimental_root(self):
        """Test user scope path validation rejects another user's custom root."""
        validator = PathValidator(
            allowed_patterns=[
                "/shared/users/{owner_id}/models/*",
                "/scratch/experiments/{owner_id}/*",
            ]
        )
        validator.validate_scope_path(
            "/scratch/experiments/user-456/model-a",
            ModelScope.USER,
            owner_id="user-456",
        )
        with pytest.raises(PathNotAllowedError):
            validator.validate_scope_path(
                "/scratch/experiments/other-user/model-a",
                ModelScope.USER,
                owner_id="user-456",
            )


# =============================================================================
# ACL Enforcer Tests
# =============================================================================


class TestACLEnforcer:
    """Tests for ACLEnforcer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.acl = ACLEnforcer()

        # Create test models
        self.system_model = ModelEntry(
            model_id="mdl-system",
            name="system-model",
            scope=ModelScope.SYSTEM,
            storage_ref="/shared/models/system/test",
        )
        self.project_model = ModelEntry(
            model_id="mdl-project",
            name="project-model",
            scope=ModelScope.PROJECT,
            project_id="proj-123",
            storage_ref="/shared/projects/proj-123/models/test",
        )
        self.user_model = ModelEntry(
            model_id="mdl-user",
            name="user-model",
            scope=ModelScope.USER,
            owner_id="user-456",
            storage_ref="/shared/users/user-456/models/test",
        )

    def test_admin_can_read_all(self):
        """Admin can read any model."""
        ctx = AccessContext(user_id="admin", is_admin=True)
        assert self.acl.can_read(self.system_model, ctx)
        assert self.acl.can_read(self.project_model, ctx)
        assert self.acl.can_read(self.user_model, ctx)

    def test_admin_can_write_all(self):
        """Admin can write any model."""
        ctx = AccessContext(user_id="admin", is_admin=True)
        assert self.acl.can_write(self.system_model, ctx)
        assert self.acl.can_write(self.project_model, ctx)
        assert self.acl.can_write(self.user_model, ctx)

    def test_user_can_read_system_models(self):
        """Any user can read system models."""
        ctx = AccessContext(user_id="regular-user", is_admin=False)
        assert self.acl.can_read(self.system_model, ctx)

    def test_user_cannot_write_system_models(self):
        """Regular users cannot write system models."""
        ctx = AccessContext(user_id="regular-user", is_admin=False)
        assert not self.acl.can_write(self.system_model, ctx)

    def test_project_member_can_read_project_model(self):
        """Project members can read project models."""
        ctx = AccessContext(
            user_id="member",
            is_admin=False,
            project_memberships={"proj-123"},
        )
        assert self.acl.can_read(self.project_model, ctx)

    def test_non_member_cannot_read_project_model(self):
        """Non-members cannot read project models."""
        ctx = AccessContext(
            user_id="outsider",
            is_admin=False,
            project_memberships={"other-project"},
        )
        assert not self.acl.can_read(self.project_model, ctx)

    def test_owner_can_read_write_user_model(self):
        """Model owner can read/write their models."""
        ctx = AccessContext(user_id="user-456", is_admin=False)
        assert self.acl.can_read(self.user_model, ctx)
        assert self.acl.can_write(self.user_model, ctx)

    def test_other_user_cannot_access_user_model(self):
        """Other users cannot access user models."""
        ctx = AccessContext(user_id="other-user", is_admin=False)
        assert not self.acl.can_read(self.user_model, ctx)
        assert not self.acl.can_write(self.user_model, ctx)

    def test_enforce_read_raises(self):
        """enforce_read raises PermissionDeniedError."""
        ctx = AccessContext(user_id="outsider", is_admin=False)
        with pytest.raises(PermissionDeniedError):
            self.acl.enforce_read(self.user_model, ctx)

    def test_can_create_in_user_scope(self):
        """Any user can create in user scope."""
        ctx = AccessContext(user_id="anyone", is_admin=False)
        assert self.acl.can_create(ModelScope.USER, ctx)

    def test_cannot_create_in_system_scope(self):
        """Regular users cannot create system models."""
        ctx = AccessContext(user_id="regular", is_admin=False)
        assert not self.acl.can_create(ModelScope.SYSTEM, ctx)

    def test_can_create_in_project_scope_if_member(self):
        """Project members can create project models."""
        ctx = AccessContext(
            user_id="member",
            is_admin=False,
            project_memberships={"proj-123"},
        )
        assert self.acl.can_create(ModelScope.PROJECT, ctx, project_id="proj-123")
        assert not self.acl.can_create(ModelScope.PROJECT, ctx, project_id="other")


# =============================================================================
# Model Store Tests (with mocked DB)
# =============================================================================


class TestModelStore:
    """Tests for ModelStore with mocked database."""

    def setup_method(self):
        """Set up test fixtures with mocked connection."""
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value.__enter__ = MagicMock(
            return_value=self.mock_cursor
        )
        self.mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Use permissive path validator for tests
        # Pattern "/*/*" matches paths like "/models/test"
        self.validator = PathValidator(allowed_patterns=["/*/*"])
        self.store = ModelStore(
            self.mock_conn,
            path_validator=self.validator,
        )

        # Test context
        self.admin_ctx = AccessContext(user_id="admin", is_admin=True)
        self.user_ctx = AccessContext(
            user_id="user-123",
            is_admin=False,
            project_memberships={"proj-123"},
        )

    def test_create_model_generates_id(self):
        """Test that create_model generates a model ID."""
        self.mock_cursor.fetchone.return_value = (datetime.now(), datetime.now())

        model = self.store.create_model(
            name="test-model",
            scope=ModelScope.SYSTEM,
            storage_ref="/models/test",
            ctx=self.admin_ctx,
        )

        assert model.model_id.startswith("mdl-")
        assert model.name == "test-model"
        assert self.mock_cursor.execute.called
        self.mock_conn.commit.assert_called_once()

    def test_create_model_checks_acl(self):
        """Test that create_model enforces ACL."""
        # User trying to create system model
        with pytest.raises(PermissionDeniedError):
            self.store.create_model(
                name="system-hack",
                scope=ModelScope.SYSTEM,
                storage_ref="/models/test",
                ctx=self.user_ctx,
            )

    def test_get_model_returns_entry(self):
        """Test get_model returns ModelEntry."""
        # Mock database row
        # Column order: model_id, name, scope, owner_id, project_id, storage_ref,
        #   model_type, huggingface_id, description, serving_profile, status,
        #   model_ref_json, execution_authority_default, created_at, updated_at
        self.mock_cursor.fetchone.return_value = (
            "mdl-123",
            "test-model",
            "system",
            None,
            None,
            "/models/test",
            "llm",
            "org/model",
            "Description",
            {},
            "available",
            None,  # model_ref_json
            "compose",  # execution_authority_default
            datetime.now(),
            datetime.now(),
        )

        model = self.store.get_model("mdl-123", self.admin_ctx)

        assert model.model_id == "mdl-123"
        assert model.name == "test-model"
        assert model.scope == ModelScope.SYSTEM

    def test_get_model_not_found(self):
        """Test get_model raises ModelNotFoundError."""
        self.mock_cursor.fetchone.return_value = None

        with pytest.raises(ModelNotFoundError):
            self.store.get_model("mdl-nonexistent", self.admin_ctx)

    def test_list_models_filters_by_visibility(self):
        """Test list_models returns only visible models."""
        # Mock multiple rows - includes model_ref_json, execution_authority_default
        self.mock_cursor.fetchall.return_value = [
            (
                "mdl-1", "system-model", "system", None, None,
                "/models/sys", "llm", None, None, {}, "available",
                None, "compose",  # model_ref_json, execution_authority_default
                datetime.now(), datetime.now(),
            ),
            (
                "mdl-2", "project-model", "project", None, "proj-123",
                "/models/proj", "llm", None, None, {}, "available",
                None, "compose",  # model_ref_json, execution_authority_default
                datetime.now(), datetime.now(),
            ),
        ]

        models = self.store.list_models(self.user_ctx)

        assert len(models) == 2
        # Check that visibility filter was applied in query
        call_args = self.mock_cursor.execute.call_args
        query = call_args[0][0]
        assert "scope = 'system'" in query or "proj-123" in str(call_args)

    def test_delete_model_removes_entry(self):
        """Test delete_model removes the catalog entry."""
        # Mock get_model - includes model_ref_json, execution_authority_default
        self.mock_cursor.fetchone.return_value = (
            "mdl-123", "test", "user", "user-123", None,
            "/models/test", "llm", None, None, {}, "available",
            None, "compose",  # model_ref_json, execution_authority_default
            datetime.now(), datetime.now(),
        )

        ctx = AccessContext(user_id="user-123", is_admin=False)
        self.store.delete_model("mdl-123", ctx)

        # Check DELETE was executed
        delete_calls = [
            call for call in self.mock_cursor.execute.call_args_list
            if "DELETE" in str(call)
        ]
        assert len(delete_calls) == 1
        self.mock_conn.commit.assert_called()

    def test_resolve_model_path_returns_storage_ref(self):
        """Test resolve_model_path returns validated path."""
        # Includes model_ref_json, execution_authority_default
        self.mock_cursor.fetchone.return_value = (
            "mdl-123", "test", "system", None, None,
            "/models/gpt-4", "llm", None, None, {}, "available",
            None, "compose",  # model_ref_json, execution_authority_default
            datetime.now(), datetime.now(),
        )

        path = self.store.resolve_model_path("mdl-123", self.admin_ctx)

        assert path == "/models/gpt-4"

    def test_resolve_model_path_rejects_staging(self):
        """Test resolve_model_path rejects models in staging status."""
        # Includes model_ref_json, execution_authority_default
        self.mock_cursor.fetchone.return_value = (
            "mdl-123", "test", "system", None, None,
            "/models/downloading", "llm", None, None, {}, "staging",
            None, "compose",  # model_ref_json, execution_authority_default
            datetime.now(), datetime.now(),
        )

        with pytest.raises(PermissionDeniedError, match="staging"):
            self.store.resolve_model_path("mdl-123", self.admin_ctx)


# =============================================================================
# Integration Tests (require real database)
# =============================================================================


@pytest.mark.integration
class TestModelStoreIntegration:
    """
    Integration tests requiring a real PostgreSQL connection.

    These tests are skipped by default. Run with:
        pytest -m integration --db-url=postgresql://...
    """

    @pytest.fixture(autouse=True)
    def setup_db(self, db_connection):
        """Set up database schema before each test."""
        from umrs.model_registry.schema import ensure_schema, drop_schema

        drop_schema(db_connection)
        ensure_schema(db_connection)
        self.conn = db_connection
        self.store = ModelStore(db_connection)
        yield
        drop_schema(db_connection)

    def test_create_and_get_model(self):
        """Test full create/get cycle."""
        ctx = AccessContext(user_id="admin", is_admin=True)

        model = self.store.create_model(
            name="integration-test",
            scope=ModelScope.SYSTEM,
            storage_ref="/shared/models/system/test",
            model_type=ModelType.LLM,
            description="Integration test model",
            ctx=ctx,
        )

        retrieved = self.store.get_model(model.model_id, ctx)
        assert retrieved.name == "integration-test"
        assert retrieved.description == "Integration test model"

    def test_duplicate_name_rejected(self):
        """Test that duplicate names in same scope are rejected."""
        ctx = AccessContext(user_id="admin", is_admin=True)

        self.store.create_model(
            name="unique-name",
            scope=ModelScope.SYSTEM,
            storage_ref="/shared/models/system/first",
            ctx=ctx,
        )

        with pytest.raises(DuplicateModelError):
            self.store.create_model(
                name="unique-name",
                scope=ModelScope.SYSTEM,
                storage_ref="/shared/models/system/second",
                ctx=ctx,
            )
