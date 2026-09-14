"""Tests for admin enrollment hardening and CLI subcommands.

Covers:
- persist_uploaded_user_config with allow_create=False rejects unknown users
- persist_uploaded_user_config with allow_create=True allows new users
- sanitize_username neutralises path-traversal characters
- ENROLL_USER is in ADMIN_ONLY_COMMANDS
- admin user CLI argparse structure
"""

from __future__ import annotations

import json
import os

import pytest
from umrs.shared.auth import (
    ADMIN_ONLY_COMMANDS,
    AuthContext,
    UserRole,
    sanitize_username,
    validate_config_identity,
)

# ---------------------------------------------------------------------------
# sanitize_username
# ---------------------------------------------------------------------------


class TestSanitizeUsername:
    def test_clean_username_unchanged(self):
        assert sanitize_username("alice") == "alice"

    def test_path_traversal_neutralised(self):
        result = sanitize_username("../../etc/shadow")
        assert "/" not in result
        assert ".." not in result

    def test_leading_digit_prefixed(self):
        result = sanitize_username("123user")
        assert result.startswith("u_")

    def test_special_chars_replaced(self):
        result = sanitize_username("user@domain.com")
        assert "@" not in result
        assert "." not in result

    def test_length_limited(self):
        long_name = "a" * 200
        result = sanitize_username(long_name)
        assert len(result) <= 64


# ---------------------------------------------------------------------------
# validate_config_identity
# ---------------------------------------------------------------------------


class TestValidateConfigIdentity:
    def test_matching_username(self):
        config = {"username": "alice"}
        result = validate_config_identity(config, "alice")
        assert result.valid

    def test_mismatched_username(self):
        config = {"username": "bob"}
        result = validate_config_identity(config, "alice")
        assert not result.valid
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_admin_claim_warned(self):
        config = {"username": "alice", "role": "admin"}
        result = validate_config_identity(config, "alice")
        assert result.valid  # still valid, but warned
        assert any("admin" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# ENROLL_USER in ADMIN_ONLY_COMMANDS
# ---------------------------------------------------------------------------


class TestEnrollUserAuth:
    def test_enroll_user_is_admin_only(self):
        assert "ENROLL_USER" in ADMIN_ONLY_COMMANDS

    def test_user_cannot_enroll(self):
        ctx = AuthContext(user_id="u1", username="user1", role=UserRole.USER)
        assert not ctx.can_execute_command("ENROLL_USER", log_denial=False)

    def test_admin_can_enroll(self):
        ctx = AuthContext(user_id="a1", username="admin1", role=UserRole.ADMIN)
        assert ctx.can_execute_command("ENROLL_USER")


# ---------------------------------------------------------------------------
# persist_uploaded_user_config hardening
# ---------------------------------------------------------------------------


class TestPersistUploadedUserConfig:
    """Test allow_create gating without touching real server state."""

    @pytest.fixture()
    def tmp_env(self, tmp_path):
        """Set up a minimal conf_umrs.json and temp user config."""
        conf_dir = tmp_path / "server"
        conf_dir.mkdir()
        conf_file = conf_dir / "conf_umrs.json"
        conf_file.write_text(json.dumps({"users": []}))
        return conf_dir, str(conf_file)

    def _write_user_json(self, tmp_path, username="alice"):
        """Write a minimal valid-looking user config JSON."""
        user_file = tmp_path / f"{username}_upload.json"
        user_file.write_text(
            json.dumps(
                {
                    "username": username,
                    "fullname": "Alice",
                    "s3_configuration_file": "/dev/null",
                    "experiences": [],
                }
            )
        )
        return str(user_file)

    def test_allow_create_false_rejects_unknown_user(self, tmp_env, tmp_path):
        from umrs.server.file_handling import persist_uploaded_user_config

        conf_dir, conf_file = tmp_env
        user_config_map = {}  # no users enrolled

        upload = self._write_user_json(tmp_path, "newuser")
        ok, uname, dest = persist_uploaded_user_config(
            upload,
            conf_file,
            user_config_map,
            expected_username=None,
            allow_create=False,
        )
        assert not ok
        assert uname == "newuser"
        assert dest is None

    def test_allow_create_true_permits_new_user(self, tmp_env, tmp_path):
        from umrs.server.file_handling import persist_uploaded_user_config

        conf_dir, conf_file = tmp_env
        user_config_map = {}

        upload = self._write_user_json(tmp_path, "newuser")
        ok, uname, dest = persist_uploaded_user_config(
            upload,
            conf_file,
            user_config_map,
            expected_username=None,
            allow_create=True,
        )
        assert ok
        assert uname == "newuser"
        assert dest is not None
        assert os.path.isfile(dest)

    def test_existing_user_update_allowed_without_allow_create(self, tmp_env, tmp_path):
        from umrs.server.file_handling import persist_uploaded_user_config

        conf_dir, conf_file = tmp_env

        # Pre-populate: user already enrolled
        existing_path = str(conf_dir / "user_files" / "alice.json")
        os.makedirs(os.path.dirname(existing_path), exist_ok=True)
        with open(existing_path, "w") as f:
            json.dump({"username": "alice"}, f)

        user_config_map = {"alice": existing_path}

        upload = self._write_user_json(tmp_path, "alice")
        ok, uname, dest = persist_uploaded_user_config(
            upload,
            conf_file,
            user_config_map,
            expected_username="alice",
            allow_create=False,
        )
        assert ok
        assert uname == "alice"
        assert dest == existing_path


# ---------------------------------------------------------------------------
# CLI argparse structure for admin user
# ---------------------------------------------------------------------------


class TestAdminUserCLI:
    def test_admin_user_add_parser(self):
        """Ensure 'admin user add <path>' is parseable."""
        import argparse

        from umrs.client.control_commands import (
            configure_admin_token_parser,
            configure_admin_user_parser,
        )

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="action")
        admin_p = sub.add_parser("admin")
        admin_sub = admin_p.add_subparsers(dest="admin_action")
        token_p = admin_sub.add_parser("token")
        configure_admin_token_parser(token_p)
        user_p = admin_sub.add_parser("user")
        configure_admin_user_parser(user_p)

        args = parser.parse_args(["admin", "user", "add", "/tmp/user.json"])
        assert args.admin_action == "user"
        assert args.user_action == "add"
        assert args.config_path == "/tmp/user.json"

    def test_admin_user_list_parser(self):
        import argparse

        from umrs.client.control_commands import configure_admin_user_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="action")
        admin_p = sub.add_parser("admin")
        admin_sub = admin_p.add_subparsers(dest="admin_action")
        user_p = admin_sub.add_parser("user")
        configure_admin_user_parser(user_p)

        args = parser.parse_args(["admin", "user", "list"])
        assert args.user_action == "list"

    def test_admin_user_remove_parser(self):
        import argparse

        from umrs.client.control_commands import configure_admin_user_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="action")
        admin_p = sub.add_parser("admin")
        admin_sub = admin_p.add_subparsers(dest="admin_action")
        user_p = admin_sub.add_parser("user")
        configure_admin_user_parser(user_p)

        args = parser.parse_args(["admin", "user", "remove", "bob"])
        assert args.user_action == "remove"
        assert args.username == "bob"


# ---------------------------------------------------------------------------
# Unsafe username rejection in persist_uploaded_user_config (Fix 2)
# ---------------------------------------------------------------------------


class TestUnsafeUsernameRejection:
    """persist_uploaded_user_config must REJECT configs whose username
    contains path-traversal or special characters, even with allow_create."""

    @pytest.fixture()
    def tmp_env(self, tmp_path):
        conf_dir = tmp_path / "server"
        conf_dir.mkdir()
        conf_file = conf_dir / "conf_umrs.json"
        conf_file.write_text(json.dumps({"users": []}))
        return conf_dir, str(conf_file)

    def _write_user_json(self, tmp_path, username):
        user_file = tmp_path / "upload.json"
        user_file.write_text(
            json.dumps(
                {
                    "username": username,
                    "fullname": "Test",
                    "s3_configuration_file": "/dev/null",
                    "experiences": [],
                }
            )
        )
        return str(user_file)

    def test_path_traversal_rejected(self, tmp_env, tmp_path):
        from umrs.server.file_handling import persist_uploaded_user_config

        conf_dir, conf_file = tmp_env
        upload = self._write_user_json(tmp_path, "../../etc/shadow")
        ok, uname, dest = persist_uploaded_user_config(
            upload,
            conf_file,
            {},
            expected_username=None,
            allow_create=True,
        )
        assert not ok
        assert dest is None

    def test_special_chars_rejected(self, tmp_env, tmp_path):
        from umrs.server.file_handling import persist_uploaded_user_config

        conf_dir, conf_file = tmp_env
        upload = self._write_user_json(tmp_path, "user@evil.com")
        ok, uname, dest = persist_uploaded_user_config(
            upload,
            conf_file,
            {},
            expected_username=None,
            allow_create=True,
        )
        assert not ok
        assert dest is None


# ---------------------------------------------------------------------------
# Salted token hashing (Fix 5)
# ---------------------------------------------------------------------------


class TestSaltedTokenHashing:
    """AdminTokenManager must use salted PBKDF2 and support legacy upgrade."""

    def _make_config(self, tmp_path):
        from pathlib import Path

        class FakeConfig:
            config_dir = tmp_path

            def get_config_file(self, filename):
                return Path(tmp_path) / filename

        return FakeConfig()

    def test_new_token_has_salt(self, tmp_path):
        from umrs.shared.auth import AdminTokenManager

        mgr = AdminTokenManager(self._make_config(tmp_path))
        token = mgr.generate_token("test-key")
        assert token.startswith("umrs-admin-")

        tokens = mgr._load_tokens()
        entry = tokens["test-key"]
        assert "token_salt" in entry
        assert len(entry["token_salt"]) == 32  # 16 bytes hex

    def test_salted_token_validates(self, tmp_path):
        from umrs.shared.auth import AdminTokenManager

        mgr = AdminTokenManager(self._make_config(tmp_path))
        token = mgr.generate_token("mytoken")
        assert mgr.validate_token(token) == "mytoken"

    def test_wrong_token_fails(self, tmp_path):
        from umrs.shared.auth import AdminTokenManager

        mgr = AdminTokenManager(self._make_config(tmp_path))
        mgr.generate_token("mytoken")
        assert mgr.validate_token("umrs-admin-WRONG") is None

    def test_legacy_unsalted_upgraded(self, tmp_path):
        """Simulate a legacy unsalted hash and verify it validates + upgrades."""
        import hashlib

        from umrs.shared.auth import AdminTokenManager

        mgr = AdminTokenManager(self._make_config(tmp_path))
        # Manually write a legacy (unsalted SHA-256) entry
        raw_token = "umrs-admin-legacytoken123"
        legacy_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        tokens = {
            "legacy-key": {
                "token_hash": legacy_hash,
                "description": "old format",
                "created_at": "2025-01-01T00:00:00Z",
            }
        }
        mgr._save_tokens(tokens)

        # Validate should succeed and upgrade in-place
        result = mgr.validate_token(raw_token)
        assert result == "legacy-key"

        # After validation the entry should now have a salt
        upgraded = mgr._load_tokens()
        assert "token_salt" in upgraded["legacy-key"]
        assert upgraded["legacy-key"]["token_hash"] != legacy_hash
