"""Object-store backend profiles and the admin-plane contract (ADR-055).

The **data plane** (uploads, presigned URLs, multipart, listing) is plain
S3 through boto3 and needs no abstraction — every supported backend speaks
S3. What differs per backend is the **admin plane** (creating per-user
identities, scoping them to buckets, revoking them, removing buckets) and a
handful of operational facts (health path, metrics path, STS role
convention). Those are what :class:`ObjectStoreBackend` and
:class:`BackendProfile` capture.

Selection is by the ``S3_BACKEND`` environment variable (``minio`` is the
default so every pre-ADR-055 profile behaves exactly as before). The
:func:`backend_name_from_server_header` helper lets callers assert that the
configured backend is the one actually answering on the wire.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

MINIO = "minio"
SEAWEEDFS = "seaweedfs"


@dataclass(frozen=True)
class BackendProfile:
    """Operational facts about one object-store implementation."""

    name: str
    display_name: str
    # Substring (lower-cased) looked for in the HTTP ``Server`` header.
    server_header_token: str
    # Health probes on the S3 endpoint, tried in order; any 2xx means live.
    health_paths: tuple[str, ...]
    # Prometheus scrape path on the S3 endpoint, or None when metrics live on
    # a dedicated port (the deployment template then owns the target).
    metrics_path: str | None
    # RoleArn convention for STS AssumeRole (MinIO ignores the ARN body but
    # requires the field; SeaweedFS resolves it to a configured role).
    sts_role_arn: str
    # Hard ceiling for AssumeRole sessions on this backend.
    sts_max_ttl_sec: int = 3600
    # Port of the IAM management API when it is separate from S3 (else None).
    iam_default_port: int | None = None
    # Whether ``s3:ListBucket`` prefix conditions in STS session policies are
    # honoured. Recorded from the ADR-055 conformance run; drives doc + tests.
    sts_list_prefix_condition_enforced: bool = True


MINIO_PROFILE = BackendProfile(
    name=MINIO,
    display_name="MinIO",
    server_header_token="minio",
    health_paths=("/minio/health/live",),
    metrics_path="/minio/v2/metrics/cluster",
    sts_role_arn="arn:aws:iam:::umrs-attempt",
)

SEAWEEDFS_PROFILE = BackendProfile(
    name=SEAWEEDFS,
    display_name="SeaweedFS",
    server_header_token="seaweedfs",
    health_paths=("/status", "/healthz"),
    metrics_path=None,  # dedicated -metricsPort, scraped at /metrics
    sts_role_arn="arn:aws:iam::role/umrs-attempt",
    iam_default_port=8111,
    # SeaweedFS 4.47 ignored the s3:prefix condition on ListBucket in the
    # session policy (ADR-055 conformance, 2026-09-14); the deployment role
    # therefore omits ListBucket from the STS role's base policy.
    sts_list_prefix_condition_enforced=False,
)

PROFILES: dict[str, BackendProfile] = {
    MINIO: MINIO_PROFILE,
    SEAWEEDFS: SEAWEEDFS_PROFILE,
}


def configured_backend_name() -> str:
    """Backend selected by ``S3_BACKEND`` (default ``minio``); validated."""
    name = (os.getenv("S3_BACKEND") or MINIO).strip().lower()
    if name not in PROFILES:
        raise ValueError(
            f"Unknown S3_BACKEND '{name}'; expected one of {sorted(PROFILES)}"
        )
    return name


def get_profile(name: str | None = None) -> BackendProfile:
    """Profile for *name* (default: the configured backend)."""
    return PROFILES[(name or configured_backend_name()).lower()]


def backend_name_from_server_header(server_header: str | None) -> str | None:
    """Map an HTTP ``Server`` header to a backend name, or None if unknown."""
    header = (server_header or "").lower()
    for profile in PROFILES.values():
        if profile.server_header_token in header:
            return profile.name
    return None


@dataclass
class AdminEndpoint:
    """Admin credentials + endpoints parsed from the admin s3cfg file.

    The s3cfg format is the s3cmd one already used platform-wide
    (``host_base``, ``access_key`` …). The IAM endpoint is taken from the
    ``S3_IAM_ENDPOINT_URL`` environment variable when set, otherwise derived
    from ``host_base`` and the backend's default IAM port.
    """

    endpoint_url: str
    access_key: str
    secret_key: str
    host_base: str
    use_https: bool = False
    verify_ssl: bool = True
    alias: str = "myminio"
    bucket_location: str = "us-east-1"
    iam_endpoint_url: str | None = None

    @classmethod
    def from_s3cfg(
        cls, path: str, *, iam_default_port: int | None = None
    ) -> AdminEndpoint:
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        expanded = os.path.expanduser(path)
        if not cfg.read(expanded):
            raise FileNotFoundError(f"Admin S3 config '{path}' not found")
        host_base = cfg.get("default", "host_base")
        use_https = cfg.getboolean("default", "use_https", fallback=False)
        scheme = "https" if use_https else "http"
        return cls(
            endpoint_url=f"{scheme}://{host_base}",
            access_key=cfg.get("default", "access_key"),
            secret_key=cfg.get("default", "secret_key"),
            host_base=host_base,
            use_https=use_https,
            verify_ssl=cfg.getboolean(
                "default", "check_ssl_certificate", fallback=True
            ),
            alias=cfg.get("default", "alias", fallback="myminio"),
            bucket_location=cfg.get("default", "bucket_location", fallback="us-east-1"),
            iam_endpoint_url=_resolve_iam_endpoint(host_base, scheme, iam_default_port),
        )


def _resolve_iam_endpoint(
    host_base: str, scheme: str, default_port: int | None
) -> str | None:
    env = (os.getenv("S3_IAM_ENDPOINT_URL") or "").strip()
    if env:
        return env
    if default_port is None:
        return None
    host = urlsplit(f"{scheme}://{host_base}").hostname or "127.0.0.1"
    return f"{scheme}://{host}:{default_port}"


class ObjectStoreBackend(Protocol):
    """Admin-plane operations the platform performs on the object store.

    Every method takes the **path of the admin s3cfg** (not parsed
    credentials) so adapters stay drop-in for the existing callers in
    ``server/provisioning.py``, ``server/maintenance.py`` and ``api/ops.py``.
    """

    profile: BackendProfile

    def create_user(
        self, username: str, user_s3cfg_path: str, admin_s3cfg_path: str
    ) -> tuple[str, str]:
        """Ensure *username* exists with a list-only baseline policy.

        Returns ``(alias, s3cfg_content)`` — the content is what gets
        delivered to the client. Idempotent: an existing valid user s3cfg is
        reconciled and returned, never silently re-keyed.
        """

    def grant_bucket_access(
        self, username: str, bucket: str, admin_s3cfg_path: str
    ) -> None:
        """Allow list/get/put/delete on *bucket* for *username* (idempotent)."""

    def delete_user(self, username: str, admin_s3cfg_path: str) -> None:
        """Remove *username* and everything attached to it (idempotent)."""

    def remove_bucket(self, bucket: str, admin_s3cfg_path: str) -> None:
        """Delete *bucket* and its contents (idempotent)."""
