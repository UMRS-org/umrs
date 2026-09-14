"""Object-store backend registry (ADR-055).

    >>> from umrs.shared.storage.backends import get_backend
    >>> backend = get_backend()            # honours S3_BACKEND (default minio)
    >>> backend.grant_bucket_access("alice", "alice.xp.texts", admin_s3cfg)

Adapters are imported lazily so importing this package never pulls boto3 or
subprocess-heavy modules at import time.
"""

from __future__ import annotations

import os

from .base import (
    MINIO,
    MINIO_PROFILE,
    PROFILES,
    SEAWEEDFS,
    SEAWEEDFS_PROFILE,
    AdminEndpoint,
    BackendProfile,
    ObjectStoreBackend,
    backend_name_from_server_header,
    configured_backend_name,
    get_profile,
)

__all__ = [
    "MINIO",
    "MINIO_PROFILE",
    "PROFILES",
    "SEAWEEDFS",
    "SEAWEEDFS_PROFILE",
    "AdminEndpoint",
    "BackendProfile",
    "ObjectStoreBackend",
    "backend_name_from_server_header",
    "configured_backend_name",
    "get_backend",
    "get_profile",
    "resolve_backend_name",
]


def get_backend(name: str | None = None) -> ObjectStoreBackend:
    """Instantiate the adapter for *name* (default: ``S3_BACKEND``)."""
    resolved = (name or configured_backend_name()).lower()
    if resolved == SEAWEEDFS:
        from .seaweedfs import SeaweedFSBackend

        return SeaweedFSBackend()
    if resolved == MINIO:
        from .minio import MinioBackend

        return MinioBackend()
    raise ValueError(f"Unknown object-store backend '{resolved}'")


def resolve_backend_name(server_header: str | None) -> str:
    """Pick the backend for a live connection and assert it matches config.

    ``S3_BACKEND`` wins when set; the ``Server`` header is then only a
    consistency check (a mismatch is a deployment error, not something to
    paper over). Without ``S3_BACKEND`` the header decides, so pre-ADR-055
    profiles keep auto-detecting MinIO exactly as before.
    """
    detected = backend_name_from_server_header(server_header)
    configured = configured_backend_name()
    explicitly_configured = bool((os.getenv("S3_BACKEND") or "").strip())
    if explicitly_configured:
        if detected and detected != configured:
            raise RuntimeError(
                f"S3_BACKEND={configured} but the endpoint identifies as "
                f"'{server_header}' ({detected}); fix the deployment before provisioning"
            )
        return configured
    if detected is None:
        raise RuntimeError(f"Do not know how to handle S3 server '{server_header}'")
    return detected
