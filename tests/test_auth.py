"""
Tests for UMRS Authentication and Authorization module.

Tests for:
- U-501: Admin role enforcement
- U-502: Per-user resource isolation
- U-503: UUID-based user identification
- U-504: Config identity validation
- U-505: XDG config directory standard
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from umrs.shared.auth import (
    ADMIN_ONLY_COMMANDS,
    USER_ALLOWED_COMMANDS,
    AdminTokenManager,
    AuthContext,
    AuthorizationError,
    ConfigDirectory,
    UserIdentityStore,
    UserRole,
    create_auth_context_from_request,
    create_service_auth_context,
    generate_user_uuid,
    require_admin,
    require_ownership,
    sanitize_username,
    validate_config_identity,
)


# =============================================================================
# U-501: Admin Role Enforcement Tests
# =============================================================================


class TestAdminRoleEnforcement:
    """Test admin role enforcement (U-501)."""

    def test_admin_can_execute_all_commands(self):
        """Admin users can execute any command."""
        auth = AuthContext(
            user_id="admin-uuid",
            username="admin",
            role=UserRole.ADMIN,
        )

        for cmd in ADMIN_ONLY_COMMANDS:
            assert auth.can_execute_command(cmd) is True

        for cmd in USER_ALLOWED_COMMANDS:
            assert auth.can_execute_command(cmd) is True

    def test_user_cannot_execute_admin_commands(self):
        """Regular users cannot execute admin commands."""
        auth = AuthContext(
            user_id="user-uuid",
            username="user1",
            role=UserRole.USER,
        )

        for cmd in ADMIN_ONLY_COMMANDS:
            assert auth.can_execute_command(cmd) is False

    def test_user_admin_probe_can_skip_denial_logging(self, caplog):
        """Capability probes can suppress denial warnings without changing auth."""
        auth = AuthContext(
            user_id="user-uuid",
            username="user1",
            role=UserRole.USER,
        )

        with caplog.at_level("WARNING"):
            assert auth.can_execute_command("RELOAD", log_denial=False) is False

        assert "attempted admin command" not in caplog.text

    def test_user_can_execute_allowed_commands(self):
        """Regular users can execute user-allowed commands."""
        auth = AuthContext(
            user_id="user-uuid",
            username="user1",
            role=UserRole.USER,
        )

        for cmd in USER_ALLOWED_COMMANDS:
            assert auth.can_execute_command(cmd) is True

    def test_service_can_execute_all_commands(self):
        """Service accounts can execute any command."""
        auth = AuthContext(
            user_id="service:gateway",
            username="gateway",
            role=UserRole.SERVICE,
        )

        for cmd in ADMIN_ONLY_COMMANDS:
            assert auth.can_execute_command(cmd) is True

        for cmd in USER_ALLOWED_COMMANDS:
            assert auth.can_execute_command(cmd) is True

    def test_require_admin_raises_for_user(self):
        """require_admin raises for non-admin users."""
        auth = AuthContext(
            user_id="user-uuid",
            username="user1",
            role=UserRole.USER,
        )

        with pytest.raises(AuthorizationError) as exc_info:
            require_admin(auth, "RELOAD")

        assert "requires admin role" in str(exc_info.value)

    def test_require_admin_passes_for_admin(self):
        """require_admin passes for admin users."""
        auth = AuthContext(
            user_id="admin-uuid",
            username="admin",
            role=UserRole.ADMIN,
        )

        # Should not raise
        require_admin(auth, "RELOAD")

    def test_is_admin_property(self):
        """Test is_admin property."""
        admin = AuthContext(user_id="1", username="a", role=UserRole.ADMIN)
        user = AuthContext(user_id="2", username="b", role=UserRole.USER)

        assert admin.is_admin is True
        assert user.is_admin is False


# =============================================================================
# U-502: Per-user Resource Isolation Tests
# =============================================================================


class TestPerUserIsolation:
    """Test per-user resource isolation (U-502)."""

    def test_user_cannot_access_other_users_data(self):
        """Users cannot access other users' data."""
        auth = AuthContext(
            user_id="user1-uuid",
            username="user1",
            role=UserRole.USER,
        )

        # Cannot execute commands targeting other users
        assert auth.can_execute_command("PROCESS_OBJECTS", target_user="user2") is False

    def test_user_can_access_own_data(self):
        """Users can access their own data."""
        auth = AuthContext(
            user_id="user1-uuid",
            username="user1",
            role=UserRole.USER,
        )

        assert auth.can_execute_command("PROCESS_OBJECTS", target_user="user1") is True

    def test_admin_can_access_any_data(self):
        """Admins can access any user's data."""
        auth = AuthContext(
            user_id="admin-uuid",
            username="admin",
            role=UserRole.ADMIN,
        )

        assert auth.can_execute_command("PROCESS_OBJECTS", target_user="user1") is True
        assert auth.can_execute_command("PROCESS_OBJECTS", target_user="user2") is True

    def test_filter_by_ownership_user(self):
        """filter_by_ownership filters data for regular users."""
        auth = AuthContext(
            user_id="user1-uuid",
            username="user1",
            role=UserRole.USER,
        )

        data = {
            "bucket1": {"s3_username": "user1", "data": "user1_data"},
            "bucket2": {"s3_username": "user2", "data": "user2_data"},
            "bucket3": {"s3_username": "user1", "data": "more_user1_data"},
        }

        filtered = auth.filter_by_ownership(data)

        assert len(filtered) == 2
        assert "bucket1" in filtered
        assert "bucket3" in filtered
        assert "bucket2" not in filtered

    def test_filter_by_ownership_admin(self):
        """filter_by_ownership returns all data for admins."""
        auth = AuthContext(
            user_id="admin-uuid",
            username="admin",
            role=UserRole.ADMIN,
        )

        data = {
            "bucket1": {"s3_username": "user1", "data": "user1_data"},
            "bucket2": {"s3_username": "user2", "data": "user2_data"},
        }

        filtered = auth.filter_by_ownership(data)

        assert len(filtered) == 2
        assert "bucket1" in filtered
        assert "bucket2" in filtered

    def test_require_ownership_raises_for_other_user(self):
        """require_ownership raises when accessing other user's data."""
        auth = AuthContext(
            user_id="user1-uuid",
            username="user1",
            role=UserRole.USER,
        )

        with pytest.raises(AuthorizationError):
            require_ownership(auth, "user2")

    def test_require_ownership_passes_for_own_data(self):
        """require_ownership passes for own data."""
        auth = AuthContext(
            user_id="user1-uuid",
            username="user1",
            role=UserRole.USER,
        )

        # Should not raise
        require_ownership(auth, "user1")


# =============================================================================
# U-503: UUID-based User Identification Tests
# =============================================================================


class TestUUIDIdentification:
    """Test UUID-based user identification (U-503)."""

    def test_generate_user_uuid_is_deterministic(self):
        """Same username always generates same UUID."""
        uuid1 = generate_user_uuid("testuser")
        uuid2 = generate_user_uuid("testuser")

        assert uuid1 == uuid2

    def test_generate_user_uuid_different_for_different_users(self):
        """Different usernames generate different UUIDs."""
        uuid1 = generate_user_uuid("user1")
        uuid2 = generate_user_uuid("user2")

        assert uuid1 != uuid2

    def test_generate_user_uuid_different_namespaces(self):
        """Different namespaces generate different UUIDs."""
        uuid1 = generate_user_uuid("testuser", namespace="umrs")
        uuid2 = generate_user_uuid("testuser", namespace="aisci")

        assert uuid1 != uuid2

    def test_uuid_format(self):
        """Generated UUID has correct format."""
        user_uuid = generate_user_uuid("testuser")

        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, user_uuid)


class TestUserIdentityStore:
    """Test UserIdentityStore."""

    def test_get_or_create_user_id(self, tmp_path):
        """get_or_create_user_id creates and retrieves user IDs."""
        config = ConfigDirectory()
        config._data_dir = tmp_path

        store = UserIdentityStore(config)

        # First call creates
        user_id1 = store.get_or_create_user_id("testuser")
        assert user_id1 is not None

        # Second call retrieves same ID
        user_id2 = store.get_or_create_user_id("testuser")
        assert user_id1 == user_id2

    def test_get_user_id_returns_none_for_unknown(self, tmp_path):
        """get_user_id returns None for unknown users."""
        config = ConfigDirectory()
        config._data_dir = tmp_path

        store = UserIdentityStore(config)

        assert store.get_user_id("nonexistent") is None

    def test_get_username_from_user_id(self, tmp_path):
        """get_username retrieves username from user ID."""
        config = ConfigDirectory()
        config._data_dir = tmp_path

        store = UserIdentityStore(config)

        user_id = store.get_or_create_user_id("testuser")
        username = store.get_username(user_id)

        assert username == "testuser"


# =============================================================================
# U-504: Config Identity Validation Tests
# =============================================================================


class TestConfigIdentityValidation:
    """Test config identity validation (U-504)."""

    def test_valid_config(self):
        """Valid config passes validation."""
        config = {"username": "user1", "data": "some_data"}
        result = validate_config_identity(config, "user1")

        assert result.valid is True
        assert len(result.errors) == 0

    def test_mismatched_username(self):
        """Mismatched username fails validation."""
        config = {"username": "user2", "data": "some_data"}
        result = validate_config_identity(config, "user1")

        assert result.valid is False
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_mismatched_user_id(self):
        """Mismatched user_id fails validation."""
        config = {"username": "user1", "user_id": "wrong-uuid"}
        result = validate_config_identity(
            config,
            expected_username="user1",
            expected_user_id="correct-uuid",
        )

        assert result.valid is False
        assert any("user id mismatch" in e.lower() for e in result.errors)

    def test_admin_claim_warning(self):
        """Admin claim in config generates warning."""
        config = {"username": "user1", "is_admin": True}
        result = validate_config_identity(config, "user1")

        assert result.valid is True  # Still valid
        assert len(result.warnings) > 0
        assert any("admin" in w.lower() for w in result.warnings)

    def test_s3_username_field(self):
        """s3_username field is also validated."""
        config = {"s3_username": "wrong_user"}
        result = validate_config_identity(config, "correct_user")

        assert result.valid is False


class TestSanitizeUsername:
    """Test username sanitization."""

    def test_basic_username(self):
        """Basic alphanumeric username passes through."""
        assert sanitize_username("user123") == "user123"

    def test_special_characters_replaced(self):
        """Special characters are replaced with underscore."""
        assert sanitize_username("user@domain.com") == "user_domain_com"

    def test_starts_with_number(self):
        """Usernames starting with numbers get prefix."""
        assert sanitize_username("123user").startswith("u_")

    def test_length_limited(self):
        """Very long usernames are truncated."""
        long_name = "a" * 100
        assert len(sanitize_username(long_name)) == 64


# =============================================================================
# U-505: XDG Config Directory Tests
# =============================================================================


class TestXDGConfigDirectory:
    """Test XDG config directory standard (U-505)."""

    def test_umrs_config_env_priority(self, tmp_path):
        """UMRS_CONFIG env var has highest priority."""
        with patch.dict(os.environ, {"UMRS_CONFIG": str(tmp_path)}):
            config = ConfigDirectory()
            assert config.config_dir == tmp_path

    def test_xdg_config_home_fallback(self, tmp_path):
        """XDG_CONFIG_HOME is used if UMRS_CONFIG not set."""
        env = {
            "XDG_CONFIG_HOME": str(tmp_path),
        }
        # Remove UMRS_CONFIG if set
        env_copy = os.environ.copy()
        env_copy.pop("UMRS_CONFIG", None)
        env_copy.update(env)

        with patch.dict(os.environ, env_copy, clear=True):
            config = ConfigDirectory()
            config._config_dir = None  # Reset cache
            assert config.config_dir == tmp_path / "umrs"

    def test_home_fallback(self, tmp_path):
        """~/.config/umrs is fallback."""
        env = {}  # No XDG or UMRS vars
        with patch.dict(os.environ, env, clear=True):
            with patch.object(Path, "home", return_value=tmp_path):
                config = ConfigDirectory()
                config._config_dir = None
                expected = tmp_path / ".config" / "umrs"
                assert config.config_dir == expected

    def test_data_dir_xdg(self, tmp_path):
        """XDG_DATA_HOME is respected for data directory."""
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
            config = ConfigDirectory()
            config._data_dir = None
            assert config.data_dir == tmp_path / "umrs"

    def test_cache_dir_xdg(self, tmp_path):
        """XDG_CACHE_HOME is respected for cache directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            config = ConfigDirectory()
            config._cache_dir = None
            assert config.cache_dir == tmp_path / "umrs"

    def test_ensure_dirs_creates_directories(self, tmp_path):
        """ensure_dirs creates all directories."""
        with patch.dict(os.environ, {
            "UMRS_CONFIG": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        }):
            config = ConfigDirectory()
            config._config_dir = None
            config._data_dir = None
            config._cache_dir = None

            config.ensure_dirs()

            assert config.config_dir.exists()
            assert config.data_dir.exists()
            assert config.cache_dir.exists()


# =============================================================================
# Admin Token Management Tests
# =============================================================================


class TestAdminTokenManager:
    """Test admin token management."""

    def test_generate_and_validate_token(self, tmp_path):
        """Generated tokens can be validated."""
        config = ConfigDirectory()
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)

        token = manager.generate_token("test-token", "Test description")
        assert token.startswith("umrs-admin-")

        name = manager.validate_token(token)
        assert name == "test-token"

    def test_invalid_token_returns_none(self, tmp_path):
        """Invalid tokens return None."""
        config = ConfigDirectory()
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)

        result = manager.validate_token("invalid-token")
        assert result is None

    def test_weak_admin_token_literals_are_rejected(self, tmp_path):
        """Known weak admin token literals never grant admin access."""
        config = ConfigDirectory()
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)
        token_hash, token_salt = manager._hash_token("admin")
        manager._save_tokens(
            {
                "weak-token": {
                    "token_hash": token_hash,
                    "token_salt": token_salt,
                    "description": "weak fixture",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            }
        )

        assert manager.validate_token("admin") is None

    def test_revoke_token(self, tmp_path):
        """Revoked tokens become invalid."""
        config = ConfigDirectory()
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)

        token = manager.generate_token("to-revoke", "Will be revoked")
        assert manager.validate_token(token) == "to-revoke"

        manager.revoke_token("to-revoke")
        assert manager.validate_token(token) is None

    def test_list_tokens(self, tmp_path):
        """list_tokens shows all tokens without hashes."""
        config = ConfigDirectory()
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)

        manager.generate_token("token1", "First token")
        manager.generate_token("token2", "Second token")

        tokens = manager.list_tokens()
        assert len(tokens) == 2

        names = {t["name"] for t in tokens}
        assert "token1" in names
        assert "token2" in names

        # Should not expose hashes
        for t in tokens:
            assert "token_hash" not in t


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for creating auth contexts."""

    def test_create_auth_context_from_request_user(self, tmp_path):
        """create_auth_context_from_request creates user context."""
        config = ConfigDirectory()
        config._data_dir = tmp_path
        config._config_dir = tmp_path

        with patch("umrs.shared.auth.get_config_directory", return_value=config):
            auth = create_auth_context_from_request("testuser")

        assert auth.username == "testuser"
        assert auth.role == UserRole.USER
        assert auth.user_id is not None

    def test_create_auth_context_with_admin_token(self, tmp_path):
        """create_auth_context_from_request with valid admin token."""
        config = ConfigDirectory()
        config._data_dir = tmp_path
        config._config_dir = tmp_path

        manager = AdminTokenManager(config)
        token = manager.generate_token("admin-token", "Admin access")

        with patch("umrs.shared.auth.get_config_directory", return_value=config):
            with patch("umrs.shared.auth.AdminTokenManager", return_value=manager):
                auth = create_auth_context_from_request("admin", token=token)

        assert auth.role == UserRole.ADMIN

    def test_create_service_auth_context(self):
        """create_service_auth_context creates service context."""
        auth = create_service_auth_context("gateway")

        assert auth.username == "gateway"
        assert auth.role == UserRole.SERVICE
        assert auth.user_id == "service:gateway"
        assert auth.is_service is True
