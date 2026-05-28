import configparser

import pytest

from umrs.shared.storage import minio as minio_storage
from umrs.shared.storage.s3 import s3_strong_password


def test_s3_strong_password_never_starts_with_dash():
    for _ in range(64):
        secret = s3_strong_password()
        assert secret
        assert not secret.startswith("-")


def test_minio_create_user_with_policy_rejects_unsafe_leading_dash_secret():
    with pytest.raises(ValueError, match="must not start with '-'"):
        minio_storage.minio_create_user_with_policy(
            "user1",
            "-unsafe-secret",
            "myminio",
            "listonly",
        )


def test_minio_s3_create_user_rewrites_unsafe_existing_secret(tmp_path, monkeypatch):
    admin_cfg_path = tmp_path / "admin.s3cfg"
    user_cfg_path = tmp_path / "user1.s3cfg"

    admin_cfg_path.write_text(
        "[default]\n"
        "host_base = 127.0.0.1:9000\n"
        "bucket_location = us-east-1\n"
        "use_https = False\n"
        "check_ssl_certificate = False\n"
        "access_key = minioadmin\n"
        "secret_key = minioadmin\n"
        "alias = myminio\n",
        encoding="utf-8",
    )
    user_cfg_path.write_text(
        "[default]\n"
        "host_base = 127.0.0.1:9000\n"
        "bucket_location = us-east-1\n"
        "use_https = False\n"
        "check_ssl_certificate = False\n"
        "access_key = user1\n"
        "secret_key = -unsafe-secret\n"
        "alias = myminio\n",
        encoding="utf-8",
    )

    captured = {}

    monkeypatch.setattr(minio_storage, "minio_check_create_alias", lambda *args, **kwargs: True)
    monkeypatch.setattr(minio_storage, "minio_check_create_list_only_policy", lambda alias: "listonly")
    monkeypatch.setattr(
        "umrs.shared.storage.s3.s3_strong_password",
        lambda length=24: "safeSecret123",
    )

    def fake_create_user(user, password, target, policy_name):
        captured["user"] = user
        captured["password"] = password
        captured["target"] = target
        captured["policy_name"] = policy_name

    monkeypatch.setattr(minio_storage, "minio_create_user_with_policy", fake_create_user)

    alias, s3cfg_content = minio_storage.minio_s3_create_user(
        "user1",
        s3_connection=None,
        user_s3_configuration_file=str(user_cfg_path),
        admin_s3_configuration_file=str(admin_cfg_path),
    )

    assert alias == "myminio"
    assert s3cfg_content is not None
    assert captured == {
        "user": "user1",
        "password": "safeSecret123",
        "target": "myminio",
        "policy_name": "listonly",
    }

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(user_cfg_path)
    assert cfg.get("default", "secret_key") == "safeSecret123"