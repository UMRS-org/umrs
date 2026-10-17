"""
File handling API endpoints (U-402, U-403, U-404).

Provides presigned URL generation for secure file access.

Endpoints:
- POST /umrs/files/presign - Generate presigned upload/download URLs
- GET /umrs/files/{file_id}/metadata - Get file metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from botocore.exceptions import ClientError
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from umrs.shared.storage.s3_config import S3Config, get_s3_config
from umrs.shared.storage.s3_presign import (
    generate_presigned_get,
    generate_presigned_put,
)
from umrs.shared.warnings import (
    UMRSWarningError,
    blocking_download_warning,
    check_download_ready,
)

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/files", tags=["files"])


# =============================================================================
# Configuration
# =============================================================================


# =============================================================================
# Request/Response Models
# =============================================================================


class PresignedOperation(str, Enum):
    """Presigned URL operation type."""

    UPLOAD = "upload"  # PUT object
    DOWNLOAD = "download"  # GET object


class PresignRequest(BaseModel):
    """Request to generate presigned URL (U-402)."""

    operation: PresignedOperation = Field(
        ..., description="Operation type: upload or download"
    )
    key: str = Field(..., description="S3 object key (path within bucket)")
    bucket: str | None = Field(
        None, description="Bucket name (uses default if not specified)"
    )
    ttl_sec: int | None = Field(
        None, ge=60, le=86400, description="TTL in seconds (60-86400)"
    )
    content_type: str | None = Field(
        None, description="Content-Type for upload operations"
    )
    metadata: dict[str, str] | None = Field(
        None, description="Custom metadata for upload operations"
    )
    # Optional job association for download validation
    job_id: str | None = Field(
        None, description="Job ID to validate download readiness"
    )
    blocking: bool = Field(
        False, description="Block until download is ready (requires job_id)"
    )
    block_timeout: int = Field(
        300, ge=0, le=3600, description="Max seconds to block (0=infinite)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "operation": "upload",
                "key": "project-123/documents/report.pdf",
                "ttl_sec": 3600,
                "content_type": "application/pdf",
            }
        }
    )


class PresignResponse(BaseModel):
    """Response with presigned URL."""

    url: str = Field(..., description="Presigned URL")
    method: str = Field(..., description="HTTP method to use (PUT/GET)")
    expires_at: str = Field(..., description="URL expiration time (ISO 8601)")
    ttl_sec: int = Field(..., description="TTL in seconds")
    key: str = Field(..., description="S3 object key")
    bucket: str = Field(..., description="S3 bucket name")
    headers: dict[str, str] | None = Field(
        None, description="Required headers for the request"
    )


class FileMetadata(BaseModel):
    """File metadata response."""

    key: str
    bucket: str
    size: int
    content_type: str | None = None
    last_modified: str | None = None
    etag: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Response for Gateway-mediated file uploads."""

    key: str
    bucket: str
    uri: str
    size: int
    content_type: str | None = None
    filename: str | None = None


class PresignBatchRequest(BaseModel):
    """Batch presigned URL request."""

    requests: list[PresignRequest] = Field(
        ..., min_length=1, max_length=100, description="Batch of presign requests"
    )


class PresignBatchResponse(BaseModel):
    """Batch presigned URL response."""

    results: list[PresignResponse]
    errors: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Dependencies
# =============================================================================


def _header_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _header_bool(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in ("true", "1")


def _fire_post_upload_ingest(
    s3_client: Any,
    bucket: str,
    object_key: str,
    object_etag: str,
    project_id: str,
    x_user_id: str | None,
    filename: str,
) -> None:
    """Fire the post-upload ingest trigger under a safe owner identity.

    Rejects literal '-'/empty owner placeholders leaked by internal
    forwarders so ingest never runs under bogus owners (user_files/-, s3cfg_-).
    """
    from umrs.data_catalog.triggers import trigger_post_upload_ingestion

    owner = _header_str(x_user_id)
    if owner is None or owner == "-":
        owner = project_id

    trigger_post_upload_ingestion(
        s3_client,
        bucket,
        s3_key=object_key,
        object_etag=object_etag,
        project_id=project_id,
        owner=owner,
        filename=filename,
    )


def get_project_id(
    authorization: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.

    Strict mode: honor X-Verified-Project-ID ONLY with a valid gateway
    signature over the full identity tuple (live-spoof class closed.
    """
    verified_project_id = _header_str(x_verified_project_id)
    if (
        # Default strict (audit B12): mvp is an explicit dev opt-in.
        os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"
        and verified_project_id
    ):
        from ._verified_headers import handshake_secret, verified_headers_trusted

        if verified_headers_trusted(
            strict=True,
            secret=handshake_secret(),
            presented_hmac=_header_str(x_verified_hmac),
            user_id=_header_str(x_user_id),
            project_id=verified_project_id,
            is_admin=_header_bool(x_verified_is_admin),
        ):
            return verified_project_id
        raise_typed(
            PlatformError.auth(
                "verified_project_mismatch",
                "Verified project identity rejected (signature mismatch)",
            )
        )
    # Default strict (audit B12): mvp is an explicit dev opt-in.
    if os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict":
        raise_typed(
            PlatformError.auth(
                "verified_project_required",
                (
                    "UMRS_AUTH_MODE=strict requires a verified project claim; "
                    "X-Project-ID and default-project fallbacks are disabled."
                ),
            )
        )

    project_id = _header_str(x_project_id)
    if project_id:
        return project_id

    if authorization and authorization.startswith("Bearer "):
        logger.warning("JWT project_id extraction not implemented, using default")
        return "default-project"

    raise_typed(
        PlatformError.auth(
            "project_required",
            "Missing project_id. Provide X-Project-ID header or valid JWT.",
        )
    )


def get_project_s3_config(
    project_id: str = Depends(get_project_id),
    default: S3Config = Depends(get_s3_config),
) -> S3Config:
    """The caller project's object store (ADR-077): bound, else *default*.

    A project whose binding cannot be read is refused (503) rather than
    served from the default store — its data must not land in the wrong one.
    """
    from umrs.shared.storage.project_storage import (
        ProjectStorageUnavailable,
        project_s3_config,
    )

    try:
        bound = project_s3_config(project_id)
    except ProjectStorageUnavailable as exc:
        raise_typed(PlatformError.projects("unavailable", str(exc)), cause=exc)
    return bound if bound is not None else default


def validate_key(key: str, project_id: str, *, warn_on_rename: bool = False) -> str:
    """
    Validate and normalize S3 key.

    Ensures key is within project namespace and safe from path traversal.
    Keys outside the caller's namespace are silently re-namespaced into it
    (fail-closed: the read/write then targets the caller's own namespace and
    404s instead of leaking another project's object). ``warn_on_rename``
    surfaces that re-namespacing on client-supplied read paths, where a
    foreign-looking key almost always means client misconfiguration —
    the error handling on those paths would otherwise chase a phantom 404.
    """
    # Normalize key
    key = key.strip("/")

    # Reject path traversal attempts
    if ".." in key.split("/"):
        raise_typed(
            PlatformError.files("traversal", "Invalid key: path traversal not allowed")
        )

    # Check if key is already namespaced
    if key.startswith(f"{project_id}/"):
        return key

    if warn_on_rename and "/" in key:
        logger.warning(
            "validate_key re-namespacing foreign-looking key into project "
            "%r (fail-closed namespacing; likely client misconfiguration): %r",
            project_id,
            key[:120],
        )

    # Namespace the key
    return f"{project_id}/{key}"


def _safe_filename(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload.bin"


def _upload_key(filename: str, purpose: str | None) -> str:
    safe_purpose = re.sub(r"[^A-Za-z0-9._/-]+", "_", (purpose or "uploads")).strip("/")
    if not safe_purpose:
        safe_purpose = "uploads"
    return f"{safe_purpose}/{uuid.uuid4().hex}_{_safe_filename(filename)}"


# =============================================================================
# Endpoints
# =============================================================================


def _charge_upload_quota(
    request: Request,
    s3_client: Any,
    bucket: str,
    object_key: str,
    content: bytes | tuple[str, int],
    x_user_id: str | None,
) -> None:
    """Disk budget (ADR-051 D-3, remaining-lanes leg 2026-09-16): catalog-bound
    bytes charged once per unique content hash; the hash needs no crypto —
    dedup is the only consumer, so a plain md5 of the staged body matches the
    ETag convention one hop up. Refusal deletes the just-staged bytes (cleanup
    never masks the typed 503 — same shape as the knowledge lane; pool conns
    always returned — the 2026-09-09 exhaustion class)."""
    if not x_user_id:
        return
    from umrs.api.disk_ledger import DiskLedger

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return
    if isinstance(content, tuple):
        content_hash, byte_size = content  # streamed upload: hashed on the way
    else:
        content_hash, byte_size = hashlib.md5(content).hexdigest(), len(content)
    conn = pool.getconn()
    try:
        DiskLedger(conn).charge(
            x_user_id,
            content_hash=content_hash,
            byte_size=byte_size,
        )
    except Exception:
        try:
            s3_client.delete_object(Bucket=bucket, Key=object_key)
        except Exception:  # noqa: BLE001 — sweep must never mask the typed refusal
            logger.warning("upload quota refusal cleanup failed for %s", object_key)
        raise
    finally:
        pool.putconn(conn)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a file through UMRS",
    description="Upload a file into project-scoped object storage without returning presigned URLs.",
)
async def upload_file(
    request: Request,
    file: UploadFile = File(None),
    key: str | None = Form(None),
    purpose: str | None = Form("uploads"),
    bucket: str | None = Form(None),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> UploadResponse:
    """Store an uploaded file under the caller's project namespace, or — with
    ``bucket`` — at ``key`` in a bucket the caller owns (ADR-079) or whose
    ADR-080 access list admits them (object writes; lifecycle stays
    owner-only)."""
    if not file.filename:
        raise_typed(PlatformError.files("filename_required", "Filename required"))
    if bucket:
        return await _upload_to_owned_bucket(
            request, file, bucket, key, x_user_id, project_id
        )

    object_key = validate_key(key or _upload_key(file.filename, purpose), project_id)
    bucket = config.bucket
    content_type = file.content_type or "application/octet-stream"
    content = await file.read()

    try:
        s3_client = config.get_client()
        put_response = s3_client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=content,
            ContentType=content_type,
            Metadata={
                "project_id": project_id,
                "filename": _safe_filename(file.filename),
            },
        )
    except ClientError as e:
        logger.error("S3 error uploading file: %s", e)
        raise_typed(
            PlatformError.files("upload_failed", "Failed to upload file"), cause=e
        )

    object_etag = (put_response.get("ETag") or "").strip('"')

    _charge_upload_quota(request, s3_client, bucket, object_key, content, x_user_id)

    _fire_post_upload_ingest(
        s3_client, bucket, object_key, object_etag, project_id, x_user_id, file.filename
    )
    return UploadResponse(
        key=object_key,
        bucket=bucket,
        uri=f"s3://{bucket}/{object_key}",
        size=len(content),
        content_type=content_type,
        filename=file.filename,
    )


@router.post(
    "/presign",
    response_model=PresignResponse,
    summary="Generate presigned URL",
    description="""
    Generate a presigned URL for secure file upload or download (U-402).

    **Upload workflow:**
    1. Client requests presigned upload URL
    2. Client uploads file directly to S3 using PUT with the presigned URL
    3. S3 validates signature and accepts the upload

    **Download workflow:**
    1. Client requests presigned download URL
    2. Client downloads file directly from S3 using GET
    3. URL expires after TTL

    **Security:**
    - URLs are time-limited (configurable TTL, max 24 hours)
    - Keys are namespaced by project_id
    - Signatures prevent tampering
    """,
)
async def generate_presigned_url(
    request: PresignRequest,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> PresignResponse:
    """Generate presigned URL for upload or download.

    For download operations, if job_id is provided, validates that
    the job's task has completed before generating the URL.
    If blocking=true, waits until the download is ready or timeout.
    """
    return await _generate_presigned_url_impl(
        request, project_id, config, x_user_id=x_user_id
    )


def _download_not_ready_error(job_id: str, attempt_status: str) -> UMRSWarningError:
    """Create HTTP 409 for download not ready."""
    warning = blocking_download_warning(job_id, attempt_status, retry_after=30)
    warning.details.setdefault("operation", "download")
    warning.details.setdefault("reason", "task_not_ready")
    return UMRSWarningError(warning, http_status=status.HTTP_409_CONFLICT)


async def _await_download_readiness(request: PresignRequest) -> None:
    """Validate download readiness for a job_id (loop until ready / deadline / fail)."""
    from umrs.jobs.store import JobNotFoundError

    from .jobs import get_optional_job_store as get_jobs_store

    job_store = get_jobs_store()
    if not job_store:
        raise_typed(
            PlatformError.files(
                "auth_unavailable",
                "Job status is unavailable; download cannot be authorized",
            )
        )

    deadline = asyncio.get_running_loop().time() + request.block_timeout
    while True:
        try:
            job = job_store.get_job(request.job_id)
            attempt_status = job.status.value
            if job.active_attempt_id:
                attempt = job_store.get_attempt(job.active_attempt_id)
                attempt_status = attempt.status.value
        except JobNotFoundError as exc:
            raise_typed(
                PlatformError.files(
                    "job_not_found", f"Job {request.job_id} was not found"
                ),
                cause=exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to read job status for presign download job_id=%s: %s",
                request.job_id,
                exc,
            )
            raise_typed(
                PlatformError.files(
                    "auth_unavailable",
                    "Job status is unavailable; download cannot be authorized",
                ),
                cause=exc,
            )

        readiness = check_download_ready(
            request.job_id, attempt_status, blocking=request.blocking
        )
        if readiness["ready"]:
            break
        if not request.blocking or asyncio.get_running_loop().time() >= deadline:
            raise _download_not_ready_error(request.job_id, attempt_status)
        await asyncio.sleep(2)


async def _presign_upload(
    request: PresignRequest,
    s3_client: Any,
    bucket: str,
    key: str,
    ttl_sec: int,
    endpoint_override: str | None = None,
) -> str:
    """Compute URL for an upload (PUT) presign.

    ``endpoint_override`` pins a project-bound store's own endpoint (ADR-077);
    otherwise the server-wide presign endpoint applies.
    """
    # NOTE: for metadata support we still call boto3 directly here because the
    # shared utility doesn't handle per-object metadata.
    if request.metadata:
        params: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Metadata": request.metadata,
        }
        if request.content_type:
            params["ContentType"] = request.content_type
        return s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=ttl_sec,
        )
    return generate_presigned_put(
        s3_client,
        bucket,
        key,
        content_type=request.content_type or "application/octet-stream",
        ttl=ttl_sec,
        endpoint_override=endpoint_override,
    )


async def _generate_presigned_url_impl(
    request: PresignRequest,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    *,
    x_user_id: str | None = None,
) -> PresignResponse:
    """Core presign implementation split by operation (upload/download)."""
    bucket = _namespace_bucket(request.bucket, config, x_user_id)
    if request.operation == PresignedOperation.DOWNLOAD and request.job_id:
        await _await_download_readiness(request)

    key = validate_key(request.key, project_id, warn_on_rename=True)
    ttl_sec = min(request.ttl_sec or config.default_ttl, config.max_ttl)

    try:
        s3_client = config.get_client()

        if request.operation == PresignedOperation.UPLOAD:
            url = await _presign_upload(
                request,
                s3_client,
                bucket,
                key,
                ttl_sec,
                endpoint_override=getattr(config, "presign_endpoint", None),
            )
            headers = {}
            if request.content_type:
                headers["Content-Type"] = request.content_type
            method = "PUT"
        else:
            url = generate_presigned_get(
                s3_client, bucket, key, ttl=ttl_sec, use_external_endpoint=False
            )
            headers = None
            method = "GET"

        expires_at = datetime.now(timezone.utc).replace(microsecond=0) + __import__(
            "datetime"
        ).timedelta(seconds=ttl_sec)

        logger.info(
            "Generated presigned %s URL",
            request.operation.value,
            extra={"project_id": project_id, "key": key, "ttl_sec": ttl_sec},
        )

        return PresignResponse(
            url=url,
            method=method,
            expires_at=expires_at.isoformat() + "Z",
            ttl_sec=ttl_sec,
            key=key,
            bucket=bucket,
            headers=headers,
        )

    except ClientError as e:
        logger.error(f"S3 error generating presigned URL: {e}")
        raise_typed(
            PlatformError.files(
                "presign_failed",
                f"Failed to generate presigned URL: {e.response['Error']['Message']}",
            ),
            cause=e,
        )
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise_typed(
            PlatformError.files("presign_failed", "Failed to generate presigned URL"),
            cause=e,
        )


@router.post(
    "/presign/batch",
    response_model=PresignBatchResponse,
    summary="Generate multiple presigned URLs",
    description="Generate presigned URLs for multiple files in a single request.",
)
async def generate_presigned_urls_batch(
    request: PresignBatchRequest,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> PresignBatchResponse:
    """Generate multiple presigned URLs."""

    results = []
    errors = []

    for idx, req in enumerate(request.requests):
        try:
            result = await generate_presigned_url(req, project_id, config, x_user_id)
            results.append(result)
        except HTTPException as e:
            errors.append(
                {
                    "index": idx,
                    "key": req.key,
                    "error": e.detail,
                }
            )
        except Exception as e:
            errors.append(
                {
                    "index": idx,
                    "key": req.key,
                    "error": str(e),
                }
            )

    return PresignBatchResponse(results=results, errors=errors)


@router.get(
    "/{file_id}/metadata",
    response_model=FileMetadata,
    summary="Get file metadata",
    description="Get metadata for a file by its ID (key).",
)
async def get_file_metadata(
    file_id: str,
    bucket: str | None = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> FileMetadata:
    """Get file metadata."""

    # Validate and namespace the key
    key = validate_key(file_id, project_id, warn_on_rename=True)
    bucket_name = _namespace_bucket(bucket, config, x_user_id)

    try:
        s3_client = config.get_client()

        response = s3_client.head_object(Bucket=bucket_name, Key=key)

        return FileMetadata(
            key=key,
            bucket=bucket_name,
            size=response.get("ContentLength", 0),
            content_type=response.get("ContentType"),
            last_modified=response.get(
                "LastModified", datetime.now(timezone.utc)
            ).isoformat()
            + "Z",
            etag=response.get("ETag", "").strip('"'),
            metadata=response.get("Metadata", {}),
        )

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise_typed(
                PlatformError.files("not_found", f"File not found: {key}"), cause=e
            )
        logger.error(f"S3 error getting file metadata: {e}")
        raise_typed(
            PlatformError.files("metadata_failed", "Failed to get file metadata"),
            cause=e,
        )


@router.delete(
    "/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file",
    description="Delete a file by its ID (key).",
)
async def delete_file(
    file_id: str,
    bucket: str | None = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> None:
    """Delete a file."""

    key = validate_key(file_id, project_id, warn_on_rename=True)
    bucket_name = _namespace_bucket(bucket, config, x_user_id)

    try:
        s3_client = config.get_client()
        s3_client.delete_object(Bucket=bucket_name, Key=key)

        try:
            from umrs.api.catalog_store import get_optional_catalog_meta_store

            catalog_store = get_optional_catalog_meta_store()
            if catalog_store is not None:
                catalog_store.delete(project_id, key)
        except Exception:
            logger.exception("Failed to delete catalog metadata for %s", key)

        logger.info(
            "Deleted file",
            extra={
                "project_id": project_id,
                "key": key,
                "bucket": bucket_name,
            },
        )

    except ClientError as e:
        logger.error(f"S3 error deleting file: {e}")
        raise_typed(
            PlatformError.files("delete_failed", "Failed to delete file"), cause=e
        )


# -- personal / provisioned buckets the caller owns (ADR-079) -----------------


def _require_owned_bucket(bucket: str, user_id: str | None) -> None:
    """Refuse unless the verified caller owns *bucket* (or its input bucket,
    for an ``-output`` pair) or the bucket's ADR-080 access list admits
    them — the HPC lane's own ownership rule."""
    from umrs.server.step_staging import bucket_owned_by

    if not isinstance(user_id, str) or not user_id or not bucket_owned_by(bucket, user_id):
        raise_typed(
            PlatformError.files(
                "bucket_denied",
                f"Bucket {bucket!r} is not one of yours or shared with you "
                "(create one with `r4 bucket create`; list them with "
                "`r4 bucket list`)",
            )
        )


def _namespace_bucket(bucket: str | None, config: S3Config, user_id: str | None) -> str:
    """Bucket for a project-namespaced key operation (presign, metadata,
    delete, exists): the caller project's store bucket, its ``-output`` pair,
    an ``S3_DOWNLOAD_EXTRA_BUCKETS`` entry or the server default bucket (the
    one the gateway's download ACL names) pass as before; any other bucket
    must be one the caller owns or is shared (``_require_owned_bucket``).

    These four routes honoured any caller-named ``bucket`` (H-1, 2026-10-17);
    the key stays project-prefixed either way. ``user_id`` is the
    ``X-User-ID`` that ``get_project_id`` verified with the identity tuple
    in strict mode.
    """
    if not bucket:
        return config.bucket
    if bucket in _cross_bucket_allowlist(config) or bucket == get_s3_config().bucket:
        return bucket
    _require_owned_bucket(bucket, user_id)
    return bucket


def _raw_key(key: str) -> str:
    """A key in an owned bucket keeps its own layout (no project prefix);
    traversal and empty keys are refused."""
    cleaned = str(key or "").strip().lstrip("/")
    if not cleaned or ".." in cleaned.split("/") or "\x00" in cleaned:
        raise_typed(PlatformError.files("traversal", f"Invalid key: {key!r}"))
    return cleaned


class _HashingReader:
    """File-like view that md5-hashes and counts what boto3 reads (streamed
    uploads charge the disk ledger without holding the body in memory)."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.md5 = hashlib.md5(usedforsecurity=False)
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self.md5.update(chunk)
        self.size += len(chunk)
        return chunk


async def _upload_to_owned_bucket(
    request: Request,
    file: UploadFile,
    bucket: str,
    key: str | None,
    x_user_id: str | None,
    project_id: str | None = None,
) -> UploadResponse:
    _require_owned_bucket(bucket, x_user_id)
    object_key = _raw_key(key or _safe_filename(file.filename or ""))
    content_type = file.content_type or "application/octet-stream"
    config = get_s3_config()
    s3_client = config.get_client()
    reader = _HashingReader(file.file)
    try:
        await asyncio.to_thread(
            s3_client.upload_fileobj,
            reader,
            bucket,
            object_key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {"filename": _safe_filename(file.filename or "")},
            },
        )
    except ClientError as e:
        logger.error("S3 error uploading to owned bucket %s: %s", bucket, e)
        raise_typed(
            PlatformError.files("upload_failed", "Failed to upload file"), cause=e
        )
    _charge_upload_quota(
        request, s3_client, bucket, object_key,
        (reader.md5.hexdigest(), reader.size), x_user_id,
    )
    if project_id:
        # W4: uploads into a personal/shared bucket must fire the same ingest
        # event as the project-namespaced path so `triggers: on: ingest` work.
        _fire_post_upload_ingest(
            s3_client,
            bucket,
            object_key,
            reader.md5.hexdigest(),
            project_id,
            x_user_id,
            file.filename or "",
        )
    return UploadResponse(
        key=object_key,
        bucket=bucket,
        uri=f"s3://{bucket}/{object_key}",
        size=reader.size,
        content_type=content_type,
        filename=file.filename,
    )


class ListedObject(BaseModel):
    key: str
    size: int
    last_modified: str | None = None


class ListObjectsResponse(BaseModel):
    bucket: str
    prefix: str
    objects: list[ListedObject] = Field(default_factory=list)
    next_token: str | None = None


def _cross_bucket_allowlist(config: S3Config) -> set[str]:
    """Buckets a caller may read without project namespace enforcement.

    Upload bucket is per-project namespaced; workflow artifacts live in the
    paired ``<input>-output`` bucket with an internal layout (``runs/…``,
    ``results/…``). Additional allowed buckets via
    ``S3_DOWNLOAD_EXTRA_BUCKETS`` (comma separated).
    """
    allowed = {config.bucket, f"{config.bucket}-output"}
    extra = os.getenv("S3_DOWNLOAD_EXTRA_BUCKETS", "")
    allowed.update(b.strip() for b in extra.split(",") if b.strip())
    return allowed


def _resolve_download_key(
    path: str, bucket_name: str, project_id: str, config: S3Config
) -> str:
    """Resolve caller key (project-namespaced for upload bucket; raw for
    cross-bucket artifact reads) with traversal guard everywhere."""
    if bucket_name == config.bucket:
        return validate_key(path, project_id, warn_on_rename=True)
    # Cross-bucket reads (workflow artifact plane) use the bucket's own
    # layout; still reject traversal but do not re-namespace.
    key = path.strip("/")
    if ".." in key.split("/"):
        raise_typed(
            PlatformError.files("traversal", "Invalid key: path traversal not allowed")
        )
    return key


def _iter_body(obj, chunk_size: int = 8 * 1024 * 1024):
    """Stream S3 body chunks; close when done."""
    stream = obj["Body"]
    try:
        while chunk := stream.read(chunk_size):
            yield chunk
    finally:
        stream.close()


def _stream_object(s3_client, bucket_name: str, key: str):
    """Fetch object header + body; map S3 errors to HTTP codes."""
    from fastapi.responses import StreamingResponse

    try:
        try:
            head = s3_client.head_object(Bucket=bucket_name, Key=key)
        except ClientError as e:
            raise_typed(
                PlatformError.files("not_found", f"File not found: {key}"), cause=e
            )
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    except HTTPException:
        raise
    except ClientError as e:
        logger.error(f"S3 error downloading file: {e}")
        raise_typed(
            PlatformError.files("download_failed", "Failed to download file"), cause=e
        )

    content_type = head.get("ContentType") or "application/octet-stream"
    size = head.get("ContentLength")
    filename = key.rsplit("/", 1)[-1] if "/" in key else key
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if size is not None:
        headers["Content-Length"] = str(size)
    return StreamingResponse(_iter_body(obj), media_type=content_type, headers=headers)


@router.get(
    "/download/{path:path}",
    summary="Stream a file from project storage (Phase-2l: no presigned URL exposure)",
    description=(
        "Stream the contents of an S3 object to the caller. The platform "
        "performs the S3 request internally (Phase 2l); never exposes "
        "presigned URLs to external clients."
    ),
)
async def download_file_stream(
    path: str,
    bucket: str | None = Query(
        None, description="Bucket name (defaults to configured bucket)"
    ),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
):
    """Stream an S3 object stored in the caller's project namespace, or from a
    bucket the caller owns (ADR-079: personal/provisioned + ``-output``)."""
    bucket_name = bucket or config.bucket
    if bucket and bucket_name not in _cross_bucket_allowlist(config):
        _require_owned_bucket(bucket_name, x_user_id)
        default = get_s3_config()
        return _stream_object(default.get_client(), bucket_name, _raw_key(path))
    if bucket_name not in _cross_bucket_allowlist(config):
        raise_typed(
            PlatformError.files(
                "bucket_denied", f"Bucket not in download allowlist: {bucket_name}"
            )
        )
    key = _resolve_download_key(path, bucket_name, project_id, config)
    return _stream_object(config.get_client(), bucket_name, key)


@router.get(
    "/list",
    response_model=ListObjectsResponse,
    summary="List objects in a bucket the caller owns (ADR-079)",
)
async def list_owned_objects(
    bucket: str = Query(..., description="A bucket you own, or its -output pair"),
    prefix: str = Query("", description="Key prefix"),
    token: str | None = Query(None, description="next_token of the previous page"),
    limit: int = Query(1000, ge=1, le=1000),
    project_id: str = Depends(get_project_id),
    x_user_id: str | None = Header(None),
) -> ListObjectsResponse:
    """Keys and sizes under *prefix*; the server lists, the caller never gets
    a credential or a presigned URL."""
    del project_id  # verifies the gateway identity tuple (strict mode)
    _require_owned_bucket(bucket, x_user_id)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix.lstrip("/"), "MaxKeys": limit}
    if token:
        kwargs["ContinuationToken"] = token
    try:
        page = await asyncio.to_thread(get_s3_config().get_client().list_objects_v2, **kwargs)
    except ClientError as e:
        raise_typed(PlatformError.files("list_failed", f"Failed to list {bucket}"), cause=e)
    objects = [
        ListedObject(
            key=str(item.get("Key")),
            size=int(item.get("Size") or 0),
            last_modified=(
                item["LastModified"].isoformat() if item.get("LastModified") else None
            ),
        )
        for item in page.get("Contents") or []
    ]
    return ListObjectsResponse(
        bucket=bucket,
        prefix=kwargs["Prefix"],
        objects=objects,
        next_token=page.get("NextContinuationToken") if page.get("IsTruncated") else None,
    )


@router.post(
    "/exists",
    summary="Check if files exist",
    description="Check if one or more files exist in storage.",
)
async def check_files_exist(
    keys: list[str],
    bucket: str | None = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_project_s3_config),
    x_user_id: str | None = Header(None),
) -> dict[str, bool]:
    """Check if files exist."""

    bucket_name = _namespace_bucket(bucket, config, x_user_id)
    results = {}

    try:
        s3_client = config.get_client()

        for key in keys:
            namespaced_key = validate_key(key, project_id, warn_on_rename=True)
            try:
                s3_client.head_object(Bucket=bucket_name, Key=namespaced_key)
                results[key] = True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    results[key] = False
                else:
                    raise

        return results

    except ClientError as e:
        logger.error(f"S3 error checking files: {e}")
        raise_typed(
            PlatformError.files("bucket_unreachable", "Failed to check files"), cause=e
        )


# =============================================================================
# Download Status & Blocking Mode (Issue #87)
# =============================================================================


class DownloadStatusResponse(BaseModel):
    """Response for download readiness check."""

    ready: bool
    job_id: str
    status: str
    message: str | None = None
    warnings: list[dict[str, Any]] | None = None
    retry_after: int | None = None
    blocking: bool = False


def _job_not_found_for_job_id_response(job_id: str) -> DownloadStatusResponse:
    return DownloadStatusResponse(
        ready=False,
        job_id=job_id,
        status="unknown",
        message=f"Job {job_id} was not found",
        retry_after=10,
    )


def _attempt_ready_state(job_store, job_id: str, job) -> DownloadStatusResponse | None:
    """Return ready response or None when attempt isn't ready (blocking poll only)."""
    try:
        attempt = job_store.get_attempt(job.active_attempt_id)
        status_result = check_download_ready(
            job_id, attempt.status.value, blocking=False
        )
        if status_result["ready"]:
            return DownloadStatusResponse(
                ready=True,
                job_id=job_id,
                status=attempt.status.value,
                message="Download is ready",
            )
    except Exception:
        logger.warning(
            "Failed to read attempt status for job %s during blocking poll",
            job_id,
            exc_info=True,
        )
    return None


async def _poll_job_until_ready(
    job_store, job_id: str, timeout: int
) -> DownloadStatusResponse:
    """Blocking poll until ready or timeout."""
    from umrs.jobs.store import JobNotFoundError

    start = time.time()
    while time.time() - start < timeout:
        try:
            job = job_store.get_job(job_id)
            if job.active_attempt_id:
                ready_state = _attempt_ready_state(job_store, job_id, job)
                if ready_state is not None:
                    return ready_state
        except JobNotFoundError:
            return _job_not_found_for_job_id_response(job_id)
        except Exception:
            logger.warning(
                "Failed to read job status for job %s during blocking poll",
                job_id,
                exc_info=True,
            )
        await asyncio.sleep(2)

    return DownloadStatusResponse(
        ready=False,
        job_id=job_id,
        status="timeout",
        message=f"Download not ready within {timeout}s timeout",
        warnings=[
            {
                "category": "blocking",
                "message": f"Timed out waiting for job {job_id} to complete",
                "suggestion": "Check job status or increase timeout",
            }
        ],
        blocking=True,
    )


def _status_from_attempt(job_id: str, attempt) -> DownloadStatusResponse:
    """Build response from an attempt's readiness evaluation."""
    status_result = check_download_ready(job_id, attempt.status.value, blocking=False)
    not_ready_warning = status_result.get("error")
    return DownloadStatusResponse(
        ready=status_result["ready"],
        job_id=job_id,
        status=attempt.status.value,
        message=(
            None
            if status_result["ready"]
            else (
                not_ready_warning.get("message")
                if isinstance(not_ready_warning, dict)
                else None
            )
        ),
        warnings=(
            [not_ready_warning]
            if not status_result["ready"] and isinstance(not_ready_warning, dict)
            else None
        ),
        blocking=status_result.get("blocking", False),
        retry_after=30 if not status_result["ready"] else None,
    )


def _nonblocking_job_status(job_store, job_id: str) -> DownloadStatusResponse:
    """Single-shot (non-blocking) readiness check."""
    from umrs.jobs.store import JobNotFoundError

    try:
        job = job_store.get_job(job_id)
        if job.active_attempt_id:
            attempt = job_store.get_attempt(job.active_attempt_id)
            return _status_from_attempt(job_id, attempt)
        if job.status.value in ("completed",):
            return DownloadStatusResponse(
                ready=True,
                job_id=job_id,
                status=job.status.value,
                message="Download is ready",
            )
        return DownloadStatusResponse(
            ready=False,
            job_id=job_id,
            status=job.status.value,
            message=f"Task is {job.status.value} — download not available yet",
            retry_after=30,
        )
    except JobNotFoundError:
        return _job_not_found_for_job_id_response(job_id)
    except Exception:
        logger.warning(
            "Failed to determine download status for job %s",
            job_id,
            exc_info=True,
        )
        return DownloadStatusResponse(
            ready=False,
            job_id=job_id,
            status="unknown",
            message=f"Could not determine status for job {job_id}",
            retry_after=10,
        )


@router.get(
    "/download-status",
    response_model=DownloadStatusResponse,
    summary="Check download readiness",
    description=(
        "Check if a job's download is ready. Returns current task status. "
        "Use blocking=true to wait until ready or timeout."
    ),
)
async def check_download_status(
    job_id: str = Query(..., description="Job ID to check"),
    blocking: bool = Query(False, description="Block until download is ready"),
    timeout: int = Query(300, ge=0, le=3600, description="Max seconds to block"),
) -> DownloadStatusResponse:
    """Check if download is ready for a job, optionally blocking.

    Returns:
        - ready=true: Download is available
        - ready=false: Task still running (include status)
        - With blocking=true: Polls until ready or timeout
    """

    from .jobs import get_optional_job_store as get_jobs_store

    job_store = get_jobs_store()
    if not job_store:
        raise_typed(
            PlatformError.files(
                "auth_unavailable",
                "Job status is unavailable; download readiness cannot be determined",
            )
        )

    if blocking and timeout > 0:
        return await _poll_job_until_ready(job_store, job_id, timeout)

    return _nonblocking_job_status(job_store, job_id)
