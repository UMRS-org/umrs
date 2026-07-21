"""Shared presigned URL utilities for UMRS.

Provides reusable functions for generating presigned S3 URLs (GET/PUT)
and performing HTTP operations with them.  Used by:
  - ``api/files.py`` (HTTP API for browser clients)
  - ``embedding_backend.py`` (ZMQ payloads for embedding workers)
  - ``vision_service/bentoml_adapter.py`` (BentoML task payloads)
"""

from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Sequence
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from umrs.shared.config import get_resolved_umrs_subdir

logger = logging.getLogger(__name__)

# Default TTL for presigned URLs: 1 hour (sufficient for typical inference
# jobs). Can be overridden per-call.
DEFAULT_TTL_SEC = 3600

# ---------------------------------------------------------------------------
# External-facing S3 client for presigned URL generation
# ---------------------------------------------------------------------------
# The server's primary S3 client may use an internal Docker hostname
# (e.g. ``minio:9000``) that is not resolvable from compute nodes.
# ``S3_PRESIGN_ENDPOINT`` lets the admin specify the S3 endpoint reachable
# from compute nodes.  When set, a separate client is created lazily
# and used exclusively for generating presigned URLs.
#
# Example: ``S3_PRESIGN_ENDPOINT=http://minio-host:19000``

_presign_clients: dict[tuple[str, str], Any] = {}


def _get_presign_endpoint(endpoint_override: str | None = None):
    """Read the presign endpoint from env or config file.

    Checks (in order):
      0. explicit ``endpoint_override`` argument
      1. ``S3_PRESIGN_ENDPOINT`` environment variable
            2. ``<resolved UMRS config dir>/s3_presign_endpoint`` file (first non-empty line)
    """
    if endpoint_override:
        return endpoint_override

    endpoint = os.environ.get("S3_PRESIGN_ENDPOINT", "")
    if endpoint:
        return endpoint
    # Fallback: read from config file (allows runtime injection without
    # restarting the server process).
    config_file = get_resolved_umrs_subdir("s3_presign_endpoint")
    try:
        with open(config_file) as f:
            endpoint = f.readline().strip()
    except FileNotFoundError:
        pass
    return endpoint


def _get_presign_client(s3_client, endpoint_override: str | None = None):
    """Return a boto3 S3 client configured for external presigned URLs.

    If ``S3_PRESIGN_ENDPOINT`` is set (env or config file), creates a
    new client with that endpoint using the same credentials as
    *s3_client*.  Otherwise returns *s3_client* unchanged.
    """
    endpoint = _get_presign_endpoint(endpoint_override=endpoint_override)
    if not endpoint:
        return s3_client

    # Extract credentials from the existing client's session.
    creds = s3_client._request_signer._credentials  # noqa: SLF001

    # Cache per (endpoint, access_key): presigned URLs embed the signing
    # credentials, so a cache keyed by endpoint alone would reuse the first
    # user's credentials for every other user (403 on their buckets).
    cache_key = (endpoint, creds.access_key)
    if cache_key in _presign_clients:
        return _presign_clients[cache_key]

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )
    _presign_clients[cache_key] = client
    logger.info(
        "Created presign S3 client with external endpoint: %s",
        endpoint,
    )
    return client


def generate_presigned_get(
    s3_client,
    bucket: str,
    key: str,
    ttl: int = DEFAULT_TTL_SEC,
    use_external_endpoint: bool = True,
    endpoint_override: str | None = None,
) -> str:
    """Generate a presigned GET URL for downloading an S3 object.

    Args:
        s3_client: boto3 S3 client with credentials.
        bucket: S3 bucket name.
        key: S3 object key.
        ttl: URL lifetime in seconds.
        use_external_endpoint: When *True* (default), use ``S3_PRESIGN_ENDPOINT``
            so compute nodes can reach the URL.  Set to *False* for
            browser-facing URLs that should use the server's own endpoint.

    Returns:
        Presigned URL string.
    """
    client = (
        _get_presign_client(s3_client, endpoint_override=endpoint_override)
        if use_external_endpoint
        else s3_client
    )
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl,
    )


def generate_presigned_put(
    s3_client,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
    ttl: int = DEFAULT_TTL_SEC,
    use_external_endpoint: bool = True,
    endpoint_override: str | None = None,
) -> str:
    """Generate a presigned PUT URL for uploading an S3 object.

    Args:
        s3_client: boto3 S3 client with credentials.
        bucket: S3 bucket name.
        key: S3 object key.
        content_type: MIME type for the uploaded object.
        ttl: URL lifetime in seconds.
        use_external_endpoint: When *True* (default), use ``S3_PRESIGN_ENDPOINT``
            so compute nodes can reach the URL.  Set to *False* for
            browser-facing URLs that should use the server's own endpoint.

    Returns:
        Presigned URL string.
    """
    client = (
        _get_presign_client(s3_client, endpoint_override=endpoint_override)
        if use_external_endpoint
        else s3_client
    )
    return client.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=ttl,
    )


def generate_presigned_urls_for_keys(
    s3_client,
    bucket: str,
    keys: Sequence[str],
    operation: str = "get",
    content_type: str = "application/octet-stream",
    ttl: int = DEFAULT_TTL_SEC,
    use_external_endpoint: bool = True,
    endpoint_override: str | None = None,
) -> dict[str, str]:
    """Generate presigned URLs for a batch of S3 keys.

    Args:
        s3_client: boto3 S3 client with credentials.
        bucket: S3 bucket name.
        keys: Sequence of S3 object keys.
        operation: ``"get"`` or ``"put"``.
        content_type: MIME type for PUT operations.
        ttl: URL lifetime in seconds.
        use_external_endpoint: When *True* (default), use ``S3_PRESIGN_ENDPOINT``
            so compute nodes can reach the URL.  Set to *False* for
            browser-facing URLs that should use the server's own endpoint.

    Returns:
        Dict mapping each key to its presigned URL.
    """
    gen = generate_presigned_get if operation == "get" else generate_presigned_put
    result: dict[str, str] = {}
    for key in keys:
        if operation == "put":
            result[key] = gen(
                s3_client,
                bucket,
                key,
                content_type=content_type,
                ttl=ttl,
                use_external_endpoint=use_external_endpoint,
                endpoint_override=endpoint_override,
            )
        else:
            result[key] = gen(
                s3_client,
                bucket,
                key,
                ttl=ttl,
                use_external_endpoint=use_external_endpoint,
                endpoint_override=endpoint_override,
            )
    return result


def download_via_presigned_url(url: str, timeout: int = 120) -> bytes:
    """Download an object using a presigned GET URL.

    Uses ``urllib.request`` (stdlib) to avoid adding ``requests`` as a
    dependency on compute workers.

    Args:
        url: Presigned GET URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Raw object bytes.

    Raises:
        urllib.error.URLError: On HTTP errors.
    """
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def upload_via_presigned_url(
    url: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    timeout: int = 120,
) -> int:
    """Upload data using a presigned PUT URL.

    Args:
        url: Presigned PUT URL.
        data: Raw bytes to upload.
        content_type: MIME type matching the presigned URL's ContentType.
        timeout: HTTP timeout in seconds.

    Returns:
        HTTP status code (200/204 on success for most S3 providers).

    Raises:
        urllib.error.URLError: On HTTP errors.
    """
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status
