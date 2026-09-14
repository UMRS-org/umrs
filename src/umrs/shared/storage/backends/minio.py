"""MinIO adapter — thin wrapper over the historical ``mc``-based helpers.

Behaviour is unchanged from pre-ADR-055: every operation shells out to the
MinIO client through :mod:`umrs.shared.storage.minio`. This adapter exists so
callers dispatch through one contract whichever backend a profile runs.
"""

from __future__ import annotations

import configparser
import os

from .base import MINIO_PROFILE, BackendProfile


def _alias_from_s3cfg(admin_s3cfg_path: str) -> str:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(os.path.expanduser(admin_s3cfg_path))
    return cfg.get("default", "alias", fallback="myminio")


class MinioBackend:
    """Admin plane via ``mc admin`` (MinIO community edition, frozen 2025-10)."""

    profile: BackendProfile = MINIO_PROFILE

    def create_user(
        self, username: str, user_s3cfg_path: str, admin_s3cfg_path: str
    ) -> tuple[str, str]:
        from umrs.shared.storage.minio import minio_s3_create_user

        return minio_s3_create_user(username, None, user_s3cfg_path, admin_s3cfg_path)

    def user_s3cfg_is_current(
        self, username: str, user_s3cfg_path: str, admin_s3cfg_path: str
    ) -> bool:
        # Status quo for MinIO: identities are username-keyed and the s3cfg is
        # the durable source of the secret; the reload path reconciles it.
        return os.path.isfile(os.path.expanduser(user_s3cfg_path))

    def grant_bucket_access(
        self, username: str, bucket: str, admin_s3cfg_path: str
    ) -> None:
        from umrs.shared.storage.minio import minio_grant_user_access

        minio_grant_user_access(username, bucket, _alias_from_s3cfg(admin_s3cfg_path))

    def delete_user(self, username: str, admin_s3cfg_path: str) -> None:
        from umrs.shared.storage.minio import minio_delete_user

        minio_delete_user(username, _alias_from_s3cfg(admin_s3cfg_path))

    def remove_bucket(self, bucket: str, admin_s3cfg_path: str) -> None:
        from umrs.shared.storage.minio import minio_remove_bucket

        minio_remove_bucket(bucket, _alias_from_s3cfg(admin_s3cfg_path))
