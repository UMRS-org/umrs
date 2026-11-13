"""Resilient single-file S3 upload (plan U1 A2; port of upstream 6632dfb0 + b951866f).

Two behaviours on top of ``s3.upload_file``, for every ADR-055 backend
(MinIO, SeaweedFS), since both speak plain S3:

* **Skip unchanged.** When the key already holds the same bytes (single-part
  ETag equal to the local MD5, same size, same requested metadata and content
  type), the transfer is skipped. A multipart ETag (``<hex>-<parts>``) or an
  SSE-KMS / SSE-C ETag is not an MD5, so those always re-upload. Any HEAD
  failure also falls through to the upload, which reports real errors itself.
* **Bounded retry.** Transient failures (connection, read/connect timeout,
  5xx, throttling, ``RequestTimeout``/``IncompleteBody``) are retried with an
  exponential backoff. Auth and other 4xx errors are raised at once.

Knobs: ``UMRS_S3_UPLOAD_ATTEMPTS`` (total attempts, default 3, 1..10),
``UMRS_S3_UPLOAD_BACKOFF_S`` (first delay in seconds, default 1, doubled per
retry, capped at 30) and ``UMRS_S3_UPLOAD_SKIP_UNCHANGED`` (default on).
"""

from __future__ import annotations

import logging
import os
import time

from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError, SSLError
from botocore.exceptions import ConnectionError as BotoConnectionError

from umrs.shared.utils import md5_of_file

logger = logging.getLogger(__name__)

ATTEMPTS_ENV = "UMRS_S3_UPLOAD_ATTEMPTS"
BACKOFF_ENV = "UMRS_S3_UPLOAD_BACKOFF_S"
SKIP_UNCHANGED_ENV = "UMRS_S3_UPLOAD_SKIP_UNCHANGED"

DEFAULT_ATTEMPTS = 3
MAX_ATTEMPTS = 10
DEFAULT_BACKOFF_S = 1.0
MAX_DELAY_S = 30.0

# Error codes that name a transient condition even when the status is 4xx
# (S3 answers RequestTimeout / IncompleteBody with 400).
TRANSIENT_CODES = frozenset(
    {
        "RequestTimeout",
        "RequestTimeoutException",
        "IncompleteBody",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestLimitExceeded",
        "TooManyRequests",
        "ServiceUnavailable",
        "InternalError",
    }
)
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_OFF_VALUES = frozenset({"0", "false", "no", "off"})

# Indirection so tests can observe the backoff without sleeping.
_sleep = time.sleep


def _env_number(name, default, cast, lo, hi):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r", name, raw)
        return default
    return min(max(value, lo), hi)


def upload_attempts() -> int:
    """Total upload attempts (first try included), bounded to 1..MAX_ATTEMPTS."""
    return _env_number(ATTEMPTS_ENV, DEFAULT_ATTEMPTS, int, 1, MAX_ATTEMPTS)


def backoff_base_s() -> float:
    """Delay before the first retry, in seconds."""
    return _env_number(BACKOFF_ENV, DEFAULT_BACKOFF_S, float, 0.0, MAX_DELAY_S)


def skip_unchanged_enabled() -> bool:
    raw = os.environ.get(SKIP_UNCHANGED_ENV, "").strip().lower()
    return raw not in _OFF_VALUES


def retry_delay_s(failed_attempt: int, base: float) -> float:
    """Exponential backoff: base, 2*base, 4*base, ... capped at MAX_DELAY_S."""
    return min(base * 2 ** (failed_attempt - 1), MAX_DELAY_S)


def _status_is_transient(status: int) -> bool:
    return status == 429 or (500 <= status <= 599 and status != 501)


def _client_error_is_transient(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    if code in TRANSIENT_CODES:
        return True
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if not isinstance(status, int) and code.isdigit():
        status = int(code)
    return isinstance(status, int) and _status_is_transient(status)


def _upload_failed_is_transient(exc: S3UploadFailedError) -> bool:
    # boto3 raises S3UploadFailedError inside ``except ClientError``, so the
    # original error is the implicit context; the message is the fallback.
    inner = exc.__cause__ or exc.__context__
    if inner is not None:
        return is_transient_upload_error(inner)
    message = str(exc)
    return any(f"({code})" in message for code in TRANSIENT_CODES)


def is_transient_upload_error(exc: BaseException) -> bool:
    """True when retrying *exc* may succeed (never for auth or other 4xx)."""
    if isinstance(exc, S3UploadFailedError):
        return _upload_failed_is_transient(exc)
    if isinstance(exc, ClientError):
        return _client_error_is_transient(exc)
    if isinstance(exc, SSLError):
        return False
    return isinstance(exc, (BotoConnectionError, HTTPClientError))


def upload_with_retry(s3, local_path, bucket, key, extra_args=None) -> None:
    """``s3.upload_file`` with bounded retries on transient errors."""
    attempts = upload_attempts()
    base = backoff_base_s()
    for attempt in range(1, attempts + 1):
        try:
            s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            return
        except (S3UploadFailedError, ClientError, BotoCoreError) as exc:
            if attempt == attempts or not is_transient_upload_error(exc):
                raise
            delay = retry_delay_s(attempt, base)
            logger.warning(
                "upload retry %d/%d for s3://%s/%s in %.1fs (%s)",
                attempt,
                attempts - 1,
                bucket,
                key,
                delay,
                exc,
            )
            _sleep(delay)


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

    Raises FileNotFoundError when *local_path* is missing.
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
    upload_with_retry(s3, local_path, bucket, key, extra_args)
    return True
