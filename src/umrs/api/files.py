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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

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


_s3_config: Optional[S3Config] = None


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
    bucket: Optional[str] = Field(
        None, description="Bucket name (uses default if not specified)"
    )
    ttl_sec: Optional[int] = Field(
        None, ge=60, le=86400, description="TTL in seconds (60-86400)"
    )
    content_type: Optional[str] = Field(
        None, description="Content-Type for upload operations"
    )
    metadata: Optional[Dict[str, str]] = Field(
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
    headers: Optional[Dict[str, str]] = Field(
        None, description="Required headers for the request"
    )


class FileMetadata(BaseModel):
    """File metadata response."""

    key: str
    bucket: str
    size: int
    content_type: Optional[str] = None
    last_modified: Optional[str] = None
    etag: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class PresignBatchRequest(BaseModel):
    """Batch presigned URL request."""

    requests: list[PresignRequest] = Field(
        ..., min_length=1, max_length=100, description="Batch of presign requests"
    )


class PresignBatchResponse(BaseModel):
    """Batch presigned URL response."""

    results: list[PresignResponse]
    errors: list[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Dependencies
# =============================================================================


def get_project_id(
    authorization: Optional[str] = Header(None),
    x_project_id: Optional[str] = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.
    """
    if x_project_id:
        return x_project_id

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

    Ensures key is within project namespace.
    """
    # Normalize key
    key = key.strip("/")

    # Check if key is already namespaced
    if key.startswith(f"{project_id}/"):
        return key

    # Namespace the key
    return f"{project_id}/{key}"


# =============================================================================
# Endpoints
# =============================================================================


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
            params: Dict[str, Any] = {
                "Bucket": bucket,
                "Key": key,
            }

            if request.content_type:
                params["ContentType"] = request.content_type

            if request.metadata:
                params["Metadata"] = request.metadata

            url = s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params=params,
                ExpiresIn=ttl_sec,
            )

            headers = {}
            if request.content_type:
                headers["Content-Type"] = request.content_type

            method = "PUT"

        else:
            # Generate presigned GET URL
            url = s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                },
                ExpiresIn=ttl_sec,
            )
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
        )
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presigned URL",
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
    bucket: Optional[str] = None,
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
            )
        logger.error(f"S3 error getting file metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file metadata",
        )


@router.delete(
    "/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file",
    description="Delete a file by its ID (key).",
)
async def delete_file(
    file_id: str,
    bucket: Optional[str] = None,
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
            f"Deleted file",
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
        )


@router.post(
    "/exists",
    summary="Check if files exist",
    description="Check if one or more files exist in storage.",
)
async def check_files_exist(
    keys: list[str],
    bucket: Optional[str] = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> Dict[str, bool]:
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
        )
