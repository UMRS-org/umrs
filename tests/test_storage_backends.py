"""ADR-055 — object-store backend registry + SeaweedFS adapter pins."""

from __future__ import annotations

import configparser
import json
import os
import stat
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from umrs.jobs import sts_federation
from umrs.shared.storage import backends
from umrs.shared.storage.backends.base import AdminEndpoint
from umrs.shared.storage.backends.minio import MinioBackend
from umrs.shared.storage.backends.seaweedfs import (
    BUCKET_POLICY_PREFIX,
    LIST_POLICY_NAME,
    SeaweedFSBackend,
)


def _client_error(code: str, op: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


def _admin_cfg(tmp_path, alias="rivault"):
    path = tmp_path / "admin.s3cfg"
    path.write_text(
        "[default]\n"
        "host_base = 127.0.0.1:9000\n"
        "bucket_location = us-east-1\n"
        "use_https = False\n"
        "check_ssl_certificate = False\n"
        "access_key = rootkey\n"
        "secret_key = rootsecret\n"
        f"alias = {alias}\n",
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------- registry --
class TestRegistry:
    def test_default_backend_is_minio(self, monkeypatch):
        monkeypatch.delenv("S3_BACKEND", raising=False)
        assert backends.configured_backend_name() == "minio"
        assert isinstance(backends.get_backend(), MinioBackend)

    def test_seaweedfs_selected_by_env(self, monkeypatch):
        monkeypatch.setenv("S3_BACKEND", "SeaweedFS")
        assert backends.configured_backend_name() == "seaweedfs"
        assert isinstance(backends.get_backend(), SeaweedFSBackend)

    def test_unknown_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("S3_BACKEND", "ceph")
        with pytest.raises(ValueError, match="Unknown S3_BACKEND"):
            backends.configured_backend_name()

    @pytest.mark.parametrize(
        ("header", "expected"),
        [("MinIO", "minio"), ("SeaweedFS 30GB 4.47", "seaweedfs"), ("nginx", None), (None, None)],
    )
    def test_server_header_detection(self, header, expected):
        assert backends.backend_name_from_server_header(header) == expected

    def test_resolve_header_decides_without_env(self, monkeypatch):
        monkeypatch.delenv("S3_BACKEND", raising=False)
        assert backends.resolve_backend_name("SeaweedFS 30GB 4.47") == "seaweedfs"
        with pytest.raises(RuntimeError, match="Do not know how to handle"):
            backends.resolve_backend_name("nginx")

    def test_resolve_env_wins_but_mismatch_is_an_error(self, monkeypatch):
        monkeypatch.setenv("S3_BACKEND", "seaweedfs")
        assert backends.resolve_backend_name("SeaweedFS 30GB 4.47") == "seaweedfs"
        assert backends.resolve_backend_name("unknown-server") == "seaweedfs"
        with pytest.raises(RuntimeError, match="S3_BACKEND=seaweedfs"):
            backends.resolve_backend_name("MinIO")

    def test_sts_role_arn_follows_backend_and_env_override(self, monkeypatch):
        monkeypatch.delenv("S3_STS_ROLE_ARN", raising=False)
        monkeypatch.setenv("S3_BACKEND", "minio")
        assert sts_federation.default_role_arn() == "arn:aws:iam:::umrs-attempt"
        monkeypatch.setenv("S3_BACKEND", "seaweedfs")
        assert sts_federation.default_role_arn() == "arn:aws:iam::role/umrs-attempt"
        monkeypatch.setenv("S3_STS_ROLE_ARN", "arn:aws:iam::role/custom")
        assert sts_federation.default_role_arn() == "arn:aws:iam::role/custom"

    def test_assume_role_uses_backend_role_arn(self, monkeypatch):
        monkeypatch.setenv("S3_BACKEND", "seaweedfs")
        monkeypatch.delenv("S3_STS_ROLE_ARN", raising=False)
        from umrs.jobs.attempt_credentials import mint_attempt_credentials

        client = MagicMock()
        client.assume_role.return_value = {
            "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "b", "SessionToken": "c"}
        }
        sts_federation.assume_role_for_attempt(
            mint_attempt_credentials("job-1", "attempt-1", 1, ttl_sec=600),
            bucket="b",
            endpoint_url="e",
            access_key="k",
            secret_key="s",
            sts_client=client,
        )
        assert client.assume_role.call_args.kwargs["RoleArn"] == "arn:aws:iam::role/umrs-attempt"

    def test_profiles_record_health_and_iam_facts(self):
        assert backends.SEAWEEDFS_PROFILE.health_paths[0] == "/status"
        assert backends.SEAWEEDFS_PROFILE.iam_default_port == 8111
        assert backends.MINIO_PROFILE.health_paths == ("/minio/health/live",)
        assert backends.SEAWEEDFS_PROFILE.sts_list_prefix_condition_enforced is False


# --------------------------------------------------------- admin endpoint --
class TestAdminEndpoint:
    def test_parses_s3cfg_and_derives_iam_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.delenv("S3_IAM_ENDPOINT_URL", raising=False)
        admin = AdminEndpoint.from_s3cfg(_admin_cfg(tmp_path), iam_default_port=8111)
        assert admin.endpoint_url == "http://127.0.0.1:9000"
        assert admin.iam_endpoint_url == "http://127.0.0.1:8111"
        assert (admin.access_key, admin.secret_key, admin.alias) == (
            "rootkey",
            "rootsecret",
            "rivault",
        )
        assert admin.verify_ssl is False

    def test_env_iam_endpoint_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("S3_IAM_ENDPOINT_URL", "http://iam.internal:8111")
        admin = AdminEndpoint.from_s3cfg(_admin_cfg(tmp_path), iam_default_port=8111)
        assert admin.iam_endpoint_url == "http://iam.internal:8111"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AdminEndpoint.from_s3cfg(str(tmp_path / "nope.s3cfg"))


# ------------------------------------------------------- seaweedfs adapter --
class _FakeIam:
    def __init__(self, *, existing_users=()):
        self.users = set(existing_users)
        self.policies: dict[str, dict[str, dict]] = {}
        self.keys: dict[str, list[str]] = {}
        self.deleted_keys: list[str] = []
        self.calls: list[str] = []
        self._n = 0

    def create_user(self, UserName):
        self.calls.append(f"create_user:{UserName}")
        if UserName in self.users:
            raise _client_error("EntityAlreadyExists", "CreateUser")
        self.users.add(UserName)

    def put_user_policy(self, UserName, PolicyName, PolicyDocument):
        self.calls.append(f"put_user_policy:{UserName}:{PolicyName}")
        self.policies.setdefault(UserName, {})[PolicyName] = json.loads(PolicyDocument)

    def create_access_key(self, UserName):
        self._n += 1
        key = f"AKIA{self._n:04d}"
        self.keys.setdefault(UserName, []).append(key)
        return {"AccessKey": {"AccessKeyId": key, "SecretAccessKey": f"secret-{self._n}"}}

    def list_user_policies(self, UserName):
        if UserName not in self.users:
            raise _client_error("NoSuchEntity", "ListUserPolicies")
        return {"PolicyNames": list(self.policies.get(UserName, {}))}

    def delete_user_policy(self, UserName, PolicyName):
        self.policies.get(UserName, {}).pop(PolicyName, None)

    def list_access_keys(self, UserName):
        return {"AccessKeyMetadata": [{"AccessKeyId": k} for k in self.keys.get(UserName, [])]}

    def delete_access_key(self, UserName, AccessKeyId):  # noqa: ARG002 — boto-shaped fake
        self.deleted_keys.append(AccessKeyId)

    def delete_user(self, UserName):
        if UserName not in self.users:
            raise _client_error("NoSuchEntity", "DeleteUser")
        self.users.discard(UserName)


def _backend(iam, *, user_key_valid=True, s3=None):
    def s3_factory(endpoint_url, access_key, secret_key, verify):
        if s3 is not None:
            return s3
        client = MagicMock()
        if not user_key_valid:
            client.list_buckets.side_effect = _client_error("InvalidAccessKeyId", "ListBuckets")
        return client

    return SeaweedFSBackend(iam_client_factory=lambda _admin: iam, s3_client_factory=s3_factory)


class TestSeaweedFSBackend:
    def test_first_provision_issues_key_and_writes_s3cfg(self, tmp_path):
        iam = _FakeIam()
        user_cfg = tmp_path / "cfg" / "s3cfg_alice"
        alias, content = _backend(iam).create_user("alice", str(user_cfg), _admin_cfg(tmp_path))

        assert alias == "rivault"
        assert "create_user:alice" in iam.calls
        assert LIST_POLICY_NAME in iam.policies["alice"]
        cfg = configparser.ConfigParser()
        cfg.read_string(content)
        assert cfg.get("default", "access_key") == "AKIA0001"  # issued, not the username
        assert cfg.get("default", "secret_key") == "secret-1"
        assert cfg.get("default", "host_base") == "127.0.0.1:9000"
        assert user_cfg.read_text() == content
        assert stat.S_IMODE(os.stat(user_cfg).st_mode) == 0o600

    def test_existing_valid_s3cfg_is_kept_not_rekeyed(self, tmp_path):
        iam = _FakeIam(existing_users=("alice",))
        user_cfg = tmp_path / "s3cfg_alice"
        user_cfg.write_text("[default]\naccess_key = OLDKEY\nsecret_key = oldsecret\n")
        _, content = _backend(iam, user_key_valid=True).create_user(
            "alice", str(user_cfg), _admin_cfg(tmp_path)
        )
        assert "OLDKEY" in content
        assert iam.keys == {}  # no new key issued
        assert LIST_POLICY_NAME in iam.policies["alice"]  # baseline still reconciled

    def test_existing_s3cfg_with_dead_key_is_reissued(self, tmp_path):
        iam = _FakeIam(existing_users=("alice",))
        user_cfg = tmp_path / "s3cfg_alice"
        user_cfg.write_text("[default]\naccess_key = DEADKEY\nsecret_key = dead\n")
        _, content = _backend(iam, user_key_valid=False).create_user(
            "alice", str(user_cfg), _admin_cfg(tmp_path)
        )
        assert "DEADKEY" not in content and "AKIA0001" in content
        assert "AKIA0001" in user_cfg.read_text()

    def test_grant_bucket_access_puts_scoped_policy(self, tmp_path):
        iam = _FakeIam(existing_users=("alice",))
        _backend(iam).grant_bucket_access("alice", "alice.xp.texts", _admin_cfg(tmp_path))
        doc = iam.policies["alice"][f"{BUCKET_POLICY_PREFIX}alice.xp.texts"]
        resources = {r for s in doc["Statement"] for r in s["Resource"]}
        assert resources == {"arn:aws:s3:::alice.xp.texts", "arn:aws:s3:::alice.xp.texts/*"}
        actions = {a for s in doc["Statement"] for a in s["Action"]}
        assert {"s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"} <= actions

    def test_delete_user_cascades_and_is_idempotent(self, tmp_path):
        iam = _FakeIam(existing_users=("alice",))
        iam.policies["alice"] = {LIST_POLICY_NAME: {}, "allow-b": {}}
        iam.keys["alice"] = ["AKIA0009"]
        backend = _backend(iam)
        backend.delete_user("alice", _admin_cfg(tmp_path))
        assert "alice" not in iam.users
        assert iam.deleted_keys == ["AKIA0009"]
        assert iam.policies["alice"] == {}
        backend.delete_user("alice", _admin_cfg(tmp_path))  # second call: NoSuchEntity swallowed

    def test_remove_bucket_empties_then_deletes_and_tolerates_missing(self, tmp_path):
        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "a"}, {"Key": "d/b"}]},
            {},
        ]
        _backend(_FakeIam(), s3=s3).remove_bucket("gone", _admin_cfg(tmp_path))
        s3.delete_objects.assert_called_once()
        assert s3.delete_objects.call_args.kwargs["Delete"]["Objects"] == [
            {"Key": "a"},
            {"Key": "d/b"},
        ]
        s3.delete_bucket.assert_called_once_with(Bucket="gone")

        missing = MagicMock()
        missing.get_paginator.return_value.paginate.side_effect = _client_error("NoSuchBucket")
        _backend(_FakeIam(), s3=missing).remove_bucket("gone", _admin_cfg(tmp_path))  # no raise

    def test_iam_endpoint_required(self, tmp_path, monkeypatch):
        monkeypatch.delenv("S3_IAM_ENDPOINT_URL", raising=False)
        admin = AdminEndpoint.from_s3cfg(_admin_cfg(tmp_path))  # no default port -> None
        with pytest.raises(RuntimeError, match="S3_IAM_ENDPOINT_URL"):
            SeaweedFSBackend._default_iam_client(admin)


# ------------------------------------------------------------ minio adapter --
class TestMinioBackend:
    def test_delegates_to_mc_helpers_with_alias_from_s3cfg(self, tmp_path, monkeypatch):
        from umrs.shared.storage import minio as minio_storage

        seen = {}
        monkeypatch.setattr(minio_storage, "minio_s3_create_user", lambda *_a: ("rivault", "cfg"))
        monkeypatch.setattr(
            minio_storage,
            "minio_grant_user_access",
            lambda u, b, t: seen.setdefault("grant", (u, b, t)),
        )
        monkeypatch.setattr(
            minio_storage, "minio_delete_user", lambda u, t: seen.setdefault("delete", (u, t))
        )
        monkeypatch.setattr(
            minio_storage, "minio_remove_bucket", lambda b, t: seen.setdefault("rb", (b, t))
        )
        admin = _admin_cfg(tmp_path)
        backend = MinioBackend()
        assert backend.create_user("alice", "/tmp/x", admin) == ("rivault", "cfg")
        backend.grant_bucket_access("alice", "b1", admin)
        backend.delete_user("alice", admin)
        backend.remove_bucket("b1", admin)
        assert seen == {
            "grant": ("alice", "b1", "rivault"),
            "delete": ("alice", "rivault"),
            "rb": ("b1", "rivault"),
        }

    def test_historical_helper_names_still_importable(self):
        from umrs.shared.storage import minio as minio_storage

        assert callable(minio_storage._mirror_s3cfg_to_user_files)
        assert callable(minio_storage._prune_stale_user_files_s3cfg)


# ------------------------------------------------- login-time self-healing --
class TestUserS3cfgCurrency:
    def test_seaweedfs_reports_dead_key_and_missing_file(self, tmp_path):
        iam = _FakeIam(existing_users=("alice",))
        admin = _admin_cfg(tmp_path)
        user_cfg = tmp_path / "s3cfg_alice"
        assert _backend(iam).user_s3cfg_is_current("alice", str(user_cfg), admin) is False
        user_cfg.write_text("[default]\naccess_key = DEADKEY\nsecret_key = dead\n")
        assert (
            _backend(iam, user_key_valid=False).user_s3cfg_is_current("alice", str(user_cfg), admin)
            is False
        )
        assert (
            _backend(iam, user_key_valid=True).user_s3cfg_is_current("alice", str(user_cfg), admin)
            is True
        )

    def test_minio_trusts_the_file(self, tmp_path):
        admin = _admin_cfg(tmp_path)
        user_cfg = tmp_path / "s3cfg_alice"
        assert MinioBackend().user_s3cfg_is_current("alice", str(user_cfg), admin) is False
        user_cfg.write_text("[default]\naccess_key = alice\nsecret_key = s\n")
        assert MinioBackend().user_s3cfg_is_current("alice", str(user_cfg), admin) is True


class TestLoginS3cfgDelivery:
    """ops._get_or_create_user_s3cfg re-provisions a stale s3cfg (ADR-055 cutover)."""

    def _ops(self, monkeypatch, tmp_path, *, existing, admin_cfg, backend):
        from umrs.api import ops

        monkeypatch.setattr(ops, "_read_user_s3cfg", lambda _username: existing)
        monkeypatch.setenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", admin_cfg)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("umrs.shared.storage.backends.get_backend", lambda _name=None: backend)
        return ops

    def test_without_admin_config_returns_existing_unverified(self, monkeypatch, tmp_path):
        ops = self._ops(monkeypatch, tmp_path, existing="OLD", admin_cfg="", backend=None)
        monkeypatch.delenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", raising=False)
        assert ops._get_or_create_user_s3cfg("alice") == "OLD"

    def test_current_s3cfg_is_delivered_as_is(self, monkeypatch, tmp_path):
        backend = MagicMock()
        backend.user_s3cfg_is_current.return_value = True
        ops = self._ops(monkeypatch, tmp_path, existing="OLD", admin_cfg="/adm", backend=backend)
        assert ops._get_or_create_user_s3cfg("alice") == "OLD"
        backend.create_user.assert_not_called()

    def test_stale_s3cfg_is_reprovisioned(self, monkeypatch, tmp_path):
        backend = MagicMock()
        backend.user_s3cfg_is_current.return_value = False
        backend.create_user.return_value = ("alias", "NEW")
        backend.profile.display_name = "SeaweedFS"
        ops = self._ops(monkeypatch, tmp_path, existing="OLD", admin_cfg="/adm", backend=backend)
        assert ops._get_or_create_user_s3cfg("alice") == "NEW"
        args = backend.create_user.call_args.args
        assert (
            args[0] == "alice"
            and args[1].endswith("/.config/umrs/s3cfg_alice")
            and args[2] == "/adm"
        )

    def test_missing_s3cfg_is_provisioned(self, monkeypatch, tmp_path):
        backend = MagicMock()
        backend.user_s3cfg_is_current.return_value = False
        backend.create_user.return_value = ("alias", "NEW")
        ops = self._ops(monkeypatch, tmp_path, existing=None, admin_cfg="/adm", backend=backend)
        assert ops._get_or_create_user_s3cfg("alice") == "NEW"

    def test_store_failure_keeps_login_working_with_existing_file(self, monkeypatch, tmp_path):
        backend = MagicMock()
        backend.user_s3cfg_is_current.side_effect = RuntimeError("store down")
        ops = self._ops(monkeypatch, tmp_path, existing="OLD", admin_cfg="/adm", backend=backend)
        assert ops._get_or_create_user_s3cfg("alice") == "OLD"
