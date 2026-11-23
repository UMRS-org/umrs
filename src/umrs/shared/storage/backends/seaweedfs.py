"""SeaweedFS adapter — admin plane through the IAM HTTP API (ADR-055).

SeaweedFS runs its identity store in the filer; ``weed server -iam`` exposes
an AWS-IAM-shaped HTTP API for it (CreateUser / CreateAccessKey /
PutUserPolicy / DeleteUser …). This adapter drives that API with boto3's
``iam`` client — no extra binary in the server image, unlike ``mc``.

Semantics mirrored from the MinIO adapter, with one deliberate difference:
SeaweedFS issues the access-key id (it cannot be chosen), so a user's
``access_key`` is **not** their username. The IAM *user name* is the
platform username; the s3cfg carries whatever key the store issued.

Verified against SeaweedFS 4.47 on 2026-09-14 (see ADR-055 § Verification):
inline user policies union across names, ``ListAllMyBuckets`` lists only
buckets the identity can reach, deleting an access key invalidates it
immediately, identities persist across restarts.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
from typing import Any

from umrs.shared.storage.user_s3cfg import (
    mirror_to_user_files,
    prune_stale_user_files,
    render_user_s3cfg,
    write_user_s3cfg,
)

from .base import SEAWEEDFS_PROFILE, AdminEndpoint, BackendProfile

logger = logging.getLogger(__name__)

LIST_POLICY_NAME = "umrs-list"
BUCKET_POLICY_PREFIX = "allow-"
_INVALID_KEY_CODES = {"InvalidAccessKeyId", "SignatureDoesNotMatch"}


def list_only_policy() -> dict[str, Any]:
    """Baseline policy: see bucket names you can reach, resolve regions."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListAllMyBuckets", "s3:GetBucketLocation"],
                "Resource": ["arn:aws:s3:::*"],
            }
        ],
    }


def bucket_access_policy(bucket: str) -> dict[str, Any]:
    """Per-bucket grant: list the bucket, get/put/delete its objects."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [f"arn:aws:s3:::{bucket}"],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"],
            },
        ],
    }


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str((response.get("Error") or {}).get("Code") or "")


def _boto_config():
    from botocore.config import Config as BotoConfig

    return BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"})


class SeaweedFSBackend:
    """Admin plane for SeaweedFS via its IAM API + plain S3 for buckets."""

    profile: BackendProfile = SEAWEEDFS_PROFILE

    def __init__(self, *, iam_client_factory=None, s3_client_factory=None) -> None:
        # Injectable for tests; defaults build boto3 clients lazily.
        self._iam_factory = iam_client_factory or self._default_iam_client
        self._s3_factory = s3_client_factory or self._default_s3_client

    # ----------------------------------------------------------- clients --
    @staticmethod
    def _default_iam_client(admin: AdminEndpoint):
        import boto3

        if not admin.iam_endpoint_url:
            raise RuntimeError(
                "SeaweedFS IAM endpoint unknown: set S3_IAM_ENDPOINT_URL "
                "(e.g. http://127.0.0.1:8111)"
            )
        return boto3.client(
            "iam",
            endpoint_url=admin.iam_endpoint_url,
            aws_access_key_id=admin.access_key,
            aws_secret_access_key=admin.secret_key,
            region_name="us-east-1",
            verify=admin.verify_ssl,
        )

    @staticmethod
    def _default_s3_client(
        endpoint_url: str, access_key: str, secret_key: str, verify: bool
    ):
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=_boto_config(),
            verify=verify,
        )

    def _admin(self, admin_s3cfg_path: str) -> AdminEndpoint:
        return AdminEndpoint.from_s3cfg(
            admin_s3cfg_path, iam_default_port=self.profile.iam_default_port
        )

    # -------------------------------------------------------- operations --
    def create_user(
        self, username: str, user_s3cfg_path: str, admin_s3cfg_path: str
    ) -> tuple[str, str]:
        admin = self._admin(admin_s3cfg_path)
        iam = self._iam_factory(admin)
        self._ensure_user(iam, username)
        self._put_policy(iam, username, LIST_POLICY_NAME, list_only_policy())

        existing = self._existing_valid_s3cfg(username, user_s3cfg_path, admin)
        if existing is not None:
            prune_stale_user_files(username, existing)
            return admin.alias, existing

        key = iam.create_access_key(UserName=username)["AccessKey"]
        content = render_user_s3cfg(
            host_base=admin.host_base,
            bucket_location=admin.bucket_location,
            use_https=admin.use_https,
            verify_ssl=admin.verify_ssl,
            access_key=key["AccessKeyId"],
            secret_key=key["SecretAccessKey"],
            alias=admin.alias,
        )
        written = write_user_s3cfg(user_s3cfg_path, content)
        mirror_to_user_files(username, content)
        logger.info("SeaweedFS identity issued for %s -> %s", username, written)
        return admin.alias, content

    def grant_bucket_access(
        self, username: str, bucket: str, admin_s3cfg_path: str
    ) -> None:
        iam = self._iam_factory(self._admin(admin_s3cfg_path))
        self._ensure_user(iam, username)
        self._put_policy(
            iam,
            username,
            f"{BUCKET_POLICY_PREFIX}{bucket}",
            bucket_access_policy(bucket),
        )

    def delete_user(self, username: str, admin_s3cfg_path: str) -> None:
        iam = self._iam_factory(self._admin(admin_s3cfg_path))
        for name in self._ignore_missing(
            lambda: iam.list_user_policies(UserName=username)["PolicyNames"], []
        ):
            self._ignore_missing(
                lambda n=name: iam.delete_user_policy(UserName=username, PolicyName=n)
            )
        keys = self._ignore_missing(
            lambda: iam.list_access_keys(UserName=username).get(
                "AccessKeyMetadata", []
            ),
            [],
        )
        for key in keys:
            self._ignore_missing(
                lambda k=key: iam.delete_access_key(
                    UserName=username, AccessKeyId=k["AccessKeyId"]
                )
            )
        if self._ignore_missing(lambda: iam.delete_user(UserName=username)) is None:
            logger.info("SeaweedFS user %s absent, nothing to delete", username)

    def remove_bucket(self, bucket: str, admin_s3cfg_path: str) -> None:
        admin = self._admin(admin_s3cfg_path)
        s3 = self._s3_factory(
            admin.endpoint_url, admin.access_key, admin.secret_key, admin.verify_ssl
        )
        try:
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if objects:
                    s3.delete_objects(
                        Bucket=bucket, Delete={"Objects": objects, "Quiet": True}
                    )
            s3.delete_bucket(Bucket=bucket)
        except Exception as exc:
            if _error_code(exc) in {"NoSuchBucket", "404"}:
                logger.info("bucket %s already absent", bucket)
                return
            raise

    # ----------------------------------------------------------- helpers --
    @staticmethod
    def _ensure_user(iam, username: str) -> None:
        try:
            iam.create_user(UserName=username)
        except Exception as exc:
            if _error_code(exc) != "EntityAlreadyExists":
                raise

    @staticmethod
    def _put_policy(iam, username: str, name: str, document: dict[str, Any]) -> None:
        iam.put_user_policy(
            UserName=username, PolicyName=name, PolicyDocument=json.dumps(document)
        )

    @staticmethod
    def _ignore_missing(fn, default=None):
        try:
            result = fn()
        except Exception as exc:
            if _error_code(exc) not in {"NoSuchEntity", "NoSuchEntityException"}:
                raise
            return default
        return result if result is not None else True

    def _existing_valid_s3cfg(
        self, username: str, user_s3cfg_path: str, admin: AdminEndpoint
    ) -> str | None:
        """Return the current user s3cfg when its key still authenticates."""
        expanded = os.path.expanduser(user_s3cfg_path)
        if not os.path.isfile(expanded):
            return None
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read(expanded)
        access_key = cfg.get("default", "access_key", fallback="")
        secret_key = cfg.get("default", "secret_key", fallback="")
        if not access_key or not secret_key:
            return None
        if not self._credentials_authenticate(admin, access_key, secret_key):
            logger.warning(
                "s3cfg for %s holds a key the store no longer accepts; re-issuing",
                username,
            )
            return None
        with open(expanded, encoding="utf-8") as fh:
            return fh.read()

    def _credentials_authenticate(
        self, admin: AdminEndpoint, access_key: str, secret_key: str
    ) -> bool:
        client = self._s3_factory(
            admin.endpoint_url, access_key, secret_key, admin.verify_ssl
        )
        try:
            client.list_buckets()
        except Exception as exc:  # noqa: BLE001
            # AccessDenied means the key is valid but unprivileged — still ours.
            return _error_code(exc) not in _INVALID_KEY_CODES
        return True
