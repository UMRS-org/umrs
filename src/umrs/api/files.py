"""
File handling API endpoints (U-402, U-403, U-404).

Provides presigned URL generation for secure file access.

Endpoints:
- POST /umrs/files/presign - Generate presigned upload/download URLs
- GET /umrs/files/{file_id}/metadata - Get file metadata
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from umrs.shared.storage.s3_presign import generate_presigned_get, generate_presigned_put

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/files", tags=["files"])


# =============================================================================
# Configuration
# =============================================================================


class S3Config:
    """S3/MinIO configuration from environment."""

    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = os.getenv("S3_ACCESS_KEY", "")
        self.secret_key = os.getenv("S3_SECRET_KEY", "")
        self.region = os.getenv("S3_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "umrs-files")
        self.default_ttl = int(os.getenv("UMRS_PRESIGNED_URL_TTL_SEC", "3600"))
        self.max_ttl = int(os.getenv("UMRS_PRESIGNED_URL_MAX_TTL_SEC", "86400"))
        self.use_ssl = os.getenv("S3_USE_SSL", "false").lower() == "true"

    def get_client(self) -> boto3.client:
        """Create boto3 S3 client."""
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            use_ssl=self.use_ssl,
            verify=self.use_ssl,
        )


_s3_config: S3Config | None = None


def get_s3_config() -> S3Config:
    """Get S3 configuration singleton."""
    global _s3_config
    if _s3_config is None:
        _s3_config = S3Config()
    return _s3_config


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
    key: str = Field(
        ..., description="S3 object key (path within bucket)"
    )
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

    class Config:
        json_schema_extra = {
            "example": {
                "operation": "upload",
                "key": "project-123/documents/report.pdf",
                "ttl_sec": 3600,
                "content_type": "application/pdf",
            }
        }


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


def get_project_id(
    authorization: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.
    """
    verified_project_id = _header_str(x_verified_project_id)
    if os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict" and verified_project_id:
        return verified_project_id
    if os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "UMRS_AUTH_MODE=strict requires a verified project claim; "
                "X-Project-ID and default-project fallbacks are disabled."
            ),
        )

    project_id = _header_str(x_project_id)
    if project_id:
        return project_id

    if authorization and authorization.startswith("Bearer "):
        logger.warning("JWT project_id extraction not implemented, using default")
        return "default-project"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing project_id. Provide X-Project-ID header or valid JWT.",
    )


def validate_key(key: str, project_id: str) -> str:
    """
    Validate and normalize S3 key.

    Ensures key is within project namespace and safe from path traversal.
    """
    # Normalize key
    key = key.strip("/")

    # Reject path traversal attempts
    if ".." in key.split("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key: path traversal not allowed",
        )

    # Check if key is already namespaced
    if key.startswith(f"{project_id}/"):
        return key

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


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a file through UMRS",
    description="Upload a file into project-scoped object storage without returning presigned URLs.",
)
async def upload_file(
    file: UploadFile = File(...),
    key: str | None = Form(None),
    purpose: str | None = Form("uploads"),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> UploadResponse:
    """Store an uploaded file under the caller's project namespace."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required")

    object_key = validate_key(key or _upload_key(file.filename, purpose), project_id)
    bucket = config.bucket
    content_type = file.content_type or "application/octet-stream"
    content = await file.read()

    try:
        s3_client = config.get_client()
        s3_client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=content,
            ContentType=content_type,
            Metadata={"project_id": project_id, "filename": _safe_filename(file.filename)},
        )
    except ClientError as e:
        logger.error("S3 error uploading file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file",
        ) from e

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
    config: S3Config = Depends(get_s3_config),
) -> PresignResponse:
    """Generate presigned URL for upload or download."""

    # Validate and namespace the key
    key = validate_key(request.key, project_id)
    bucket = request.bucket or config.bucket
    ttl_sec = min(request.ttl_sec or config.default_ttl, config.max_ttl)

    try:
        s3_client = config.get_client()

        if request.operation == PresignedOperation.UPLOAD:
            # Generate presigned PUT URL
            # NOTE: for metadata support we still call boto3 directly here
            # because the shared utility doesn't handle per-object metadata.
            if request.metadata:
                params: dict[str, Any] = {
                    "Bucket": bucket,
                    "Key": key,
                    "Metadata": request.metadata,
                }
                if request.content_type:
                    params["ContentType"] = request.content_type
                url = s3_client.generate_presigned_url(
                    ClientMethod="put_object",
                    Params=params,
                    ExpiresIn=ttl_sec,
                )
            else:
                url = generate_presigned_put(
                    s3_client, bucket, key,
                    content_type=request.content_type or "application/octet-stream",
                    ttl=ttl_sec,
                    use_external_endpoint=False,
                )

            headers = {}
            if request.content_type:
                headers["Content-Type"] = request.content_type

            method = "PUT"

        else:
            # Generate presigned GET URL
            url = generate_presigned_get(s3_client, bucket, key, ttl=ttl_sec, use_external_endpoint=False)
            headers = None
            method = "GET"

        expires_at = datetime.now(timezone.utc).replace(
            microsecond=0
        ) + __import__("datetime").timedelta(seconds=ttl_sec)

        logger.info(
            f"Generated presigned {request.operation.value} URL",
            extra={
                "project_id": project_id,
                "key": key,
                "ttl_sec": ttl_sec,
            },
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presigned URL: {e.response['Error']['Message']}",
        ) from e
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presigned URL",
        ) from e


@router.post(
    "/presign/batch",
    response_model=PresignBatchResponse,
    summary="Generate multiple presigned URLs",
    description="Generate presigned URLs for multiple files in a single request.",
)
async def generate_presigned_urls_batch(
    request: PresignBatchRequest,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> PresignBatchResponse:
    """Generate multiple presigned URLs."""

    results = []
    errors = []

    for idx, req in enumerate(request.requests):
        try:
            result = await generate_presigned_url(req, project_id, config)
            results.append(result)
        except HTTPException as e:
            errors.append({
                "index": idx,
                "key": req.key,
                "error": e.detail,
            })
        except Exception as e:
            errors.append({
                "index": idx,
                "key": req.key,
                "error": str(e),
            })

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
    config: S3Config = Depends(get_s3_config),
) -> FileMetadata:
    """Get file metadata."""

    # Validate and namespace the key
    key = validate_key(file_id, project_id)
    bucket_name = bucket or config.bucket

    try:
        s3_client = config.get_client()

        response = s3_client.head_object(Bucket=bucket_name, Key=key)

        return FileMetadata(
            key=key,
            bucket=bucket_name,
            size=response.get("ContentLength", 0),
            content_type=response.get("ContentType"),
            last_modified=response.get("LastModified", datetime.now(timezone.utc)).isoformat() + "Z",
            etag=response.get("ETag", "").strip('"'),
            metadata=response.get("Metadata", {}),
        )

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {key}",
            ) from e
        logger.error(f"S3 error getting file metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file metadata",
        ) from e


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
    config: S3Config = Depends(get_s3_config),
) -> None:
    """Delete a file."""

    key = validate_key(file_id, project_id)
    bucket_name = bucket or config.bucket

    try:
        s3_client = config.get_client()
        s3_client.delete_object(Bucket=bucket_name, Key=key)

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file",
        ) from e


@router.post(
    "/exists",
    summary="Check if files exist",
    description="Check if one or more files exist in storage.",
)
async def check_files_exist(
    keys: list[str],
    bucket: str | None = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> dict[str, bool]:
    """Check if files exist."""

    bucket_name = bucket or config.bucket
    results = {}

    try:
        s3_client = config.get_client()

        for key in keys:
            namespaced_key = validate_key(key, project_id)
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check files",
        ) from e
