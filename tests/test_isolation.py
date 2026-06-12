"""
Unit tests for multi-user isolation (U-005).
"""

import pytest

from umrs.shared.isolation import (
    AccessControl,
    AccessLevel,
    ConfigIsolation,
    IsolationValidator,
    ResourceOwnership,
    ResourceType,
    UserContext,
    UserRole,
)


class TestUserContext:
    """Tests for UserContext."""

    def test_anonymous_context(self):
        """Create anonymous context."""
        ctx = UserContext.anonymous()
        assert ctx.user_id == "anonymous"
        assert ctx.project_id == "default"
        assert ctx.role == UserRole.ANONYMOUS
        assert not ctx.is_authenticated
        assert not ctx.is_admin

    def test_from_headers_full(self):
        """Create context from headers with all fields."""
        ctx = UserContext.from_headers(
            x_user_id="user1",
            x_project_id="project1",
            x_user_role="admin",
            x_user_groups="group1,group2",
        )
        assert ctx.user_id == "user1"
        assert ctx.project_id == "project1"
        assert ctx.role == UserRole.ADMIN
        assert ctx.groups == ["group1", "group2"]
        assert ctx.is_authenticated
        assert ctx.is_admin

    def test_from_headers_minimal(self):
        """Create context from headers with minimal fields."""
        ctx = UserContext.from_headers(None, None)
        assert ctx.user_id == "anonymous"
        assert ctx.project_id == "default"
        assert not ctx.is_authenticated

    def test_project_access_same_project(self):
        """User can access their own project."""
        ctx = UserContext(user_id="user1", project_id="project1")
        assert ctx.can_access_project("project1")

    def test_project_access_different_project(self):
        """User cannot access other projects."""
        ctx = UserContext(user_id="user1", project_id="project1")
        assert not ctx.can_access_project("project2")

    def test_project_access_admin(self):
        """Admin can access any project."""
        ctx = UserContext(
            user_id="admin1",
            project_id="admin-project",
            role=UserRole.ADMIN,
        )
        assert ctx.can_access_project("project1")
        assert ctx.can_access_project("project2")

    def test_user_data_access_own(self):
        """User can access their own data."""
        ctx = UserContext(user_id="user1", project_id="project1")
        assert ctx.can_access_user_data("user1")

    def test_user_data_access_other(self):
        """User cannot access other user's data."""
        ctx = UserContext(user_id="user1", project_id="project1")
        assert not ctx.can_access_user_data("user2")


class TestResourceOwnership:
    """Tests for ResourceOwnership."""

    def test_owner_access(self):
        """Owner can access their own resource."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        ctx = UserContext(user_id="user1", project_id="project1")
        assert ownership.is_accessible_by(ctx)

    def test_non_owner_no_access(self):
        """Non-owner cannot access private resource."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        ctx = UserContext(user_id="user2", project_id="project1")
        assert not ownership.is_accessible_by(ctx)

    def test_shared_resource_access(self):
        """Users in shared_with can access resource."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.DOCUMENT,
            resource_id="doc-xyz",
            owner_id="user1",
            project_id="project1",
            is_shared=True,
            shared_with={"user2", "user3"},
        )
        ctx = UserContext(user_id="user2", project_id="project1")
        assert ownership.is_accessible_by(ctx)

    def test_project_shared_access(self):
        """Project-level sharing works."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.PIPELINE,
            resource_id="pipe-123",
            owner_id="user1",
            project_id="project1",
            shared_with={"project:project1"},
        )
        # Any user in project1 can access
        ctx = UserContext(user_id="user2", project_id="project1")
        assert ownership.is_accessible_by(ctx)

    def test_cross_project_no_access(self):
        """Users from other projects cannot access."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        ctx = UserContext(user_id="user2", project_id="project2")
        assert not ownership.is_accessible_by(ctx)

    def test_admin_access(self):
        """Admin can access any resource."""
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        ctx = UserContext(
            user_id="admin1",
            project_id="admin-project",
            role=UserRole.ADMIN,
        )
        assert ownership.is_accessible_by(ctx)


class TestAccessControl:
    """Tests for AccessControl."""

    def test_validate_safe_path(self):
        """Safe paths pass validation."""
        ac = AccessControl()
        ctx = UserContext.anonymous()
        assert ac.validate_path("files/document.pdf", ctx)
        assert ac.validate_path("data/2024/report.xlsx", ctx)

    def test_validate_path_traversal_blocked(self):
        """Path traversal attempts are blocked."""
        ac = AccessControl()
        ctx = UserContext.anonymous()
        assert not ac.validate_path("../etc/passwd", ctx)
        assert not ac.validate_path("files/../secret.txt", ctx)

    def test_validate_absolute_path_blocked(self):
        """Absolute paths are blocked."""
        ac = AccessControl()
        ctx = UserContext.anonymous()
        assert not ac.validate_path("/etc/passwd", ctx)

    def test_validate_shell_metacharacters_blocked(self):
        """Shell metacharacters are blocked."""
        ac = AccessControl()
        ctx = UserContext.anonymous()
        assert not ac.validate_path("file|cat", ctx)
        assert not ac.validate_path("file<>out", ctx)

    def test_validate_resource_access_allowed(self):
        """Resource access validation for owner."""
        ac = AccessControl()
        ctx = UserContext(user_id="user1", project_id="project1")
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        assert ac.validate_resource_access(ctx, ownership)

    def test_validate_resource_access_denied(self):
        """Resource access validation for non-owner."""
        ac = AccessControl()
        ctx = UserContext(user_id="user2", project_id="project1")
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
        )
        assert not ac.validate_resource_access(ctx, ownership)

    def test_write_access_owner_only(self):
        """Write access requires owner or admin."""
        ac = AccessControl()
        ownership = ResourceOwnership(
            resource_type=ResourceType.FILE,
            resource_id="file-abc",
            owner_id="user1",
            project_id="project1",
            shared_with={"user2"},  # user2 has read access
        )

        # Owner can write
        owner_ctx = UserContext(user_id="user1", project_id="project1")
        assert ac.validate_resource_access(owner_ctx, ownership, AccessLevel.WRITE)

        # Shared user cannot write
        shared_ctx = UserContext(user_id="user2", project_id="project1")
        assert not ac.validate_resource_access(shared_ctx, ownership, AccessLevel.WRITE)


class TestIsolationValidator:
    """Tests for IsolationValidator."""

    def test_validate_request_isolation_clean(self):
        """Clean request with matching project."""
        validator = IsolationValidator()
        ctx = UserContext(user_id="user1", project_id="project1")
        request = {"project_id": "project1", "data": "test"}
        violations = validator.validate_request_isolation(ctx, request)
        assert len(violations) == 0

    def test_validate_request_project_mismatch(self):
        """Detect project_id mismatch."""
        validator = IsolationValidator()
        ctx = UserContext(user_id="user1", project_id="project1")
        request = {"project_id": "project2", "data": "test"}
        violations = validator.validate_request_isolation(ctx, request)
        assert len(violations) == 1
        assert "mismatch" in violations[0].lower()

    def test_validate_request_user_override(self):
        """Detect user_id override attempt."""
        validator = IsolationValidator()
        ctx = UserContext(user_id="user1", project_id="project1")
        request = {"user_id": "user2", "data": "test"}
        violations = validator.validate_request_isolation(ctx, request)
        assert len(violations) == 1
        assert "override" in violations[0].lower()

    def test_validate_request_path_traversal(self):
        """Detect path traversal in request."""
        validator = IsolationValidator()
        ctx = UserContext(user_id="user1", project_id="project1")
        request = {"path": "../secret/data.txt"}
        violations = validator.validate_request_isolation(ctx, request)
        assert len(violations) == 1
        assert "suspicious" in violations[0].lower()

    def test_response_scoping_filters_other_projects(self):
        """Response scoping filters out other projects."""
        validator = IsolationValidator()
        ctx = UserContext(user_id="user1", project_id="project1")
        response = [
            {"id": 1, "project_id": "project1", "data": "visible"},
            {"id": 2, "project_id": "project2", "data": "hidden"},
            {"id": 3, "project_id": "project1", "data": "visible"},
        ]
        scoped = validator.validate_response_scoping(ctx, response)
        assert len(scoped) == 2
        assert all(item["project_id"] == "project1" for item in scoped)

    def test_response_scoping_admin_sees_all(self):
        """Admin can see all projects."""
        validator = IsolationValidator()
        ctx = UserContext(
            user_id="admin1",
            project_id="admin-project",
            role=UserRole.ADMIN,
        )
        response = [
            {"id": 1, "project_id": "project1"},
            {"id": 2, "project_id": "project2"},
        ]
        scoped = validator.validate_response_scoping(ctx, response)
        assert len(scoped) == 2


class TestConfigIsolation:
    """Tests for ConfigIsolation."""

    def test_register_and_get_config_path(self):
        """Register and retrieve user config path."""
        isolation = ConfigIsolation()
        isolation.register_user_config("user1", "/path/to/user1/config.json")

        ctx = UserContext(user_id="user1", project_id="project1")
        path = isolation.get_user_config_path(ctx)
        assert path == "/path/to/user1/config.json"

    def test_get_config_path_unknown_user(self):
        """Unknown user returns None."""
        isolation = ConfigIsolation()

        ctx = UserContext(user_id="unknown", project_id="project1")
        path = isolation.get_user_config_path(ctx)
        assert path is None

    def test_user_cannot_access_other_config(self):
        """User cannot access another user's config."""
        isolation = ConfigIsolation(
            {
                "user1": "/path/to/user1/config.json",
                "user2": "/path/to/user2/config.json",
            }
        )

        # user1 gets user1's config
        ctx1 = UserContext(user_id="user1", project_id="project1")
        assert isolation.get_user_config_path(ctx1) == "/path/to/user1/config.json"

        # user2 gets user2's config (not user1's)
        ctx2 = UserContext(user_id="user2", project_id="project1")
        assert isolation.get_user_config_path(ctx2) == "/path/to/user2/config.json"

    def test_admin_can_access_target_user_config(self):
        """Admin can access any user's config via metadata."""
        isolation = ConfigIsolation(
            {
                "user1": "/path/to/user1/config.json",
            }
        )

        ctx = UserContext(
            user_id="admin1",
            project_id="admin-project",
            role=UserRole.ADMIN,
            metadata={"target_user": "user1"},
        )
        path = isolation.get_user_config_path(ctx)
        assert path == "/path/to/user1/config.json"
