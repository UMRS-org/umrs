"""Resilient single-file S3 upload (plan U1 A2; port of upstream 6632dfb0 + b951866f).

Two behaviours on top of ``s3.upload_file``, for every ADR-055 backend
(MinIO, SeaweedFS), since both speak plain S3:

* **Skip unchanged.** When the key already holds the same bytes (single-part
  ETag equal to the local MD5, same size, same requested metadata and content
  type), the transfer is skipped. A multipart ETag (``<hex>-<parts>``) or an
  SSE-KMS / SSE-C ETag is not an MD5, so those always re-upload. Any HEAD
  failure also falls through to the upload, which reports real errors itself.
* **Bounded retry, one layer per error.** Transient failures (connection
  errors, timeouts, 5xx, throttling, ``RequestTimeout``) are retried by
  botocore in its ``standard`` mode, configured on the client by
  :func:`s3_client_config` (``connect_to_s3`` uses it). Upstream wrapped
  ``upload_file`` in its own loop; on top of botocore's default retries that
  multiplied the attempts per file (review U1). One code of upstream's set
  is not retried by botocore: ``IncompleteBody``, the HTTP 400 that S3, MinIO
  and SeaweedFS answer when the request body arrives short. Only that code
  gets a loop here (:data:`INCOMPLETE_BODY_ATTEMPTS`, 0 s then 1 s apart as
  in upstream), so attempts never multiply.

Knobs: ``UMRS_S3_MAX_ATTEMPTS`` (total attempts per S3 request, first try
included, default 5 = botocore's legacy default, 1..10) and
``UMRS_S3_UPLOAD_SKIP_UNCHANGED`` (default on).
"""

from __future__ import annotations

import logging
import os
import time

from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from umrs.shared.env_knobs import env_flag, env_int
from umrs.shared.utils import md5_of_file

logger = logging.getLogger(__name__)

MAX_ATTEMPTS_ENV = "UMRS_S3_MAX_ATTEMPTS"
SKIP_UNCHANGED_ENV = "UMRS_S3_UPLOAD_SKIP_UNCHANGED"

DEFAULT_MAX_ATTEMPTS = 5
MAX_ATTEMPTS = 10

_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

# The short-body error botocore does not retry (its standard mode retries
# RequestTimeout-like codes, throttling and 5xx only); 6632dfb0 retried it.
INCOMPLETE_BODY = "IncompleteBody"
INCOMPLETE_BODY_ATTEMPTS = 3  # total sends, first included; the only bound


def _incomplete_body_delay_s(retry: int) -> int:
    """Seconds before retry number *retry* (1-based): 0 s, then 1 s (6632dfb0)."""
    return 0 if retry == 1 else 1


def s3_max_attempts() -> int:
    """Total attempts per S3 request (first try included), 1..MAX_ATTEMPTS."""
    return env_int(MAX_ATTEMPTS_ENV, DEFAULT_MAX_ATTEMPTS, lo=1, hi=MAX_ATTEMPTS)


def s3_client_config(**overrides) -> Config:
    """botocore ``Config`` whose standard-mode retries are the only S3 retry layer."""
    return Config(
        retries={"total_max_attempts": s3_max_attempts(), "mode": "standard"},
        **overrides,
    )


def skip_unchanged_enabled() -> bool:
    return env_flag(SKIP_UNCHANGED_ENV, default=True)


def single_part_md5_etag(head: dict) -> str | None:
    """The object's ETag when it is a plain MD5, else None.

    Multipart ETags carry a ``-<parts>`` suffix and SSE-KMS / SSE-C ETags are
    opaque; neither can be compared with a local MD5.
    """
    if (
        head.get("SSECustomerAlgorithm")
        or head.get("ServerSideEncryption") == "aws:kms"
    ):
        return None
    etag = str(head.get("ETag", "")).strip().strip('"').lower()
    if len(etag) != 32 or "-" in etag:
        return None
    return etag


def _normalised_metadata(metadata) -> dict:
    return {str(k).lower(): str(v) for k, v in (metadata or {}).items()}


def _extra_args_match(head: dict, extra_args) -> bool:
    """Requested Metadata (and ContentType, when given) equal the stored ones."""
    wanted = extra_args or {}
    if _normalised_metadata(wanted.get("Metadata")) != _normalised_metadata(
        head.get("Metadata")
    ):
        return False
    content_type = wanted.get("ContentType")
    return content_type is None or content_type == head.get("ContentType")


def _head_or_none(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in _MISSING_CODES:
            logger.debug(
                "head_object s3://%s/%s failed (%s); uploading", bucket, key, code
            )
        return None
    except BotoCoreError as exc:
        logger.debug("head_object s3://%s/%s failed (%s); uploading", bucket, key, exc)
        return None


def remote_copy_is_current(s3, bucket, key, local_path, extra_args=None) -> bool:
    """True when *key* already holds *local_path*'s bytes and requested metadata."""
    head = _head_or_none(s3, bucket, key)
    if head is None:
        return False
    etag = single_part_md5_etag(head)
    if etag is None or head.get("ContentLength") != os.path.getsize(local_path):
        return False
    if not _extra_args_match(head, extra_args):
        return False
    return md5_of_file(local_path) == etag


def upload_local_file(
    s3, bucket, key, local_path, extra_args=None, *, skip_unchanged=None
) -> bool:
    """Upload *local_path* to ``s3://bucket/key``; False when skipped as unchanged.

    Raises FileNotFoundError when *local_path* is missing. Transient errors
    are retried by the client's own botocore config (:func:`s3_client_config`).
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Error: {local_path} not found")
    if skip_unchanged is None:
        skip_unchanged = skip_unchanged_enabled()
    if skip_unchanged and remote_copy_is_current(
        s3, bucket, key, local_path, extra_args
    ):
        logger.debug("skip upload, unchanged: s3://%s/%s", bucket, key)
        return False
    _upload_retrying_incomplete_body(s3, bucket, key, local_path, extra_args)
    return True


def _is_incomplete_body(exc) -> bool:
    """``IncompleteBody`` as a ClientError, or wrapped by s3transfer."""
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") == INCOMPLETE_BODY
    return f"({INCOMPLETE_BODY})" in str(exc)


def _upload_retrying_incomplete_body(s3, bucket, key, local_path, extra_args) -> None:
    """``upload_file``, sent up to :data:`INCOMPLETE_BODY_ATTEMPTS` times on
    ``IncompleteBody`` only; every other error is botocore's to retry."""
    attempts = INCOMPLETE_BODY_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            return
        except (S3UploadFailedError, ClientError) as exc:
            if attempt == attempts or not _is_incomplete_body(exc):
                raise
            delay = _incomplete_body_delay_s(attempt)
            logger.warning("upload s3://%s/%s: %s, retry %d/%d in %ss", bucket, key,
                           INCOMPLETE_BODY, attempt, attempts - 1, delay)
            time.sleep(delay)
