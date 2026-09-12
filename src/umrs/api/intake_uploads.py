"""I3: Resumable/batch intake upload routes.

Chunked upload flow (intake-side counterpart of the presigned single-upload):

1. ``POST /umrs/uploads/manifests`` — create a manifest (target key + part count)
2. ``POST /umrs/uploads/manifests/{id}/parts/{n}/presign`` — presigned PUT for part N
3. ``POST /umrs/uploads/manifests/{id}/parts/{n}`` — record a confirmed part
4. ``POST /umrs/uploads/manifests/{id}/complete`` — verify all parts via HEAD,
   assemble server-side into the final object, write catalog ingestion, and
   mark the manifest assembled.

Resume semantics: manifests persist; re-PREAGING an uploaded part re-PUTs
only that part; ``complete`` is idempotent (returns the stored result when
the manifest is already assembled). Per-part provenance lives in the
manifest's ``parts_detail``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends
from fastapi import Path as FPath
from pydantic import BaseModel, Field

from umrs.data_catalog.intake_uploads import (
    IntakeUploadManifest,
    build_default_store,
    new_manifest,
)
from umrs.shared.storage.s3_presign import generate_presigned_put

from ..shared.storage.s3_config import S3Config, get_s3_config
from .files import get_project_id, validate_key
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/uploads", tags=["intake-uploads"])

_MAX_PARTS = 10000
_PART_KEY_TEMPLATE = "{target_key}.__parts/{manifest_id}/part-{part:05d}"

_store: Any = None


def _get_store() -> Any:
    """Lazily resolve the manifest store (test-patchable)."""
    global _store
    if _store is None:
        _store = build_default_store()
    return _store


def _part_key(target_key: str, manifest_id: str, part_no: int) -> str:
    return _PART_KEY_TEMPLATE.format(
        target_key=target_key, manifest_id=manifest_id, part=part_no
    )


class CreateManifestRequest(BaseModel):
    """Manifest creation body."""

    target_key: str = Field(
        ..., min_length=1, description="Project-scoped target object key"
    )
    expected_parts: int = Field(
        ..., ge=1, le=_MAX_PARTS, description="Total part count"
    )
    filename: str | None = Field(default=None, description="Original filename hint")
    project_id: str | None = Field(
        default=None, description="Project override (MVP mode)"
    )


class RecordPartRequest(BaseModel):
    """Part confirmation body."""

    etag: str = Field(default="", max_length=128)
    size_bytes: int = Field(default=0, ge=0)


def _require_manifest(manifest_id: str) -> IntakeUploadManifest:
    manifest = _get_store().get(manifest_id)
    if manifest is None:
        raise_typed(
            PlatformError.intake(
                "not_found",
                f"Manifest '{manifest_id}' not found"
            ),
        )
    return manifest


def _require_uploading(manifest: IntakeUploadManifest) -> None:
    if manifest.status == "assembled":
        raise_typed(
            PlatformError.intake(
                "conflict",
                "Manifest already assembled — complete is idempotent; use GET"
            ),
        )
    if manifest.status == "failed":
        raise_typed(
            PlatformError.intake(
                "conflict",
                f"Manifest failed: {manifest.error_note or 'unknown error'}"
            ),
        )


@router.post("/manifests", response_model=dict[str, Any])
def create_manifest(
    body: CreateManifestRequest,
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Create a chunked-upload manifest (I3)."""
    scoped_project = body.project_id or project_id
    target_key = validate_key(body.target_key, scoped_project)
    config = get_s3_config()
    manifest = new_manifest(
        project_id=scoped_project,
        target_key=target_key,
        expected_parts=body.expected_parts,
        owner=scoped_project,
    )
    manifest.target_bucket = config.bucket
    _get_store().create(manifest)
    logger.info(
        "Upload manifest created: id=%s project=%s parts=%d target=%s",
        manifest.manifest_id,
        scoped_project,
        body.expected_parts,
        target_key,
    )
    return {"status": "ok", "manifest": manifest.to_dict()}


@router.get("/manifests/{manifest_id}", response_model=dict[str, Any])
def get_manifest(manifest_id: str = FPath(...)) -> dict[str, Any]:
    """Manifest status with missing-part derivation (resume aid)."""
    manifest = _require_manifest(manifest_id)
    return {"status": "ok", "manifest": manifest.to_dict()}


@router.post(
    "/manifests/{manifest_id}/parts/{part_no}/presign", response_model=dict[str, Any]
)
def presign_part(
    manifest_id: str = FPath(...),
    part_no: int = FPath(..., ge=1),
) -> dict[str, Any]:
    """Presigned PUT URL for one part (client uploads the chunk here)."""
    manifest = _require_manifest(manifest_id)
    _require_uploading(manifest)
    if part_no > manifest.expected_parts:
        raise_typed(
            PlatformError.intake(
                "bad_request",
                f"part {part_no} exceeds expected_parts={manifest.expected_parts}"
            ),
        )
    config = get_s3_config()
    part_key = _part_key(manifest.target_key, manifest.manifest_id, part_no)
    url = generate_presigned_put(
        config.get_client(),
        config.bucket,
        part_key,
        ttl=int(os.getenv("UMRS_PRESIGNED_URL_TTL_SEC", "3600")),
        use_external_endpoint=False,
    )
    return {
        "status": "ok",
        "manifest_id": manifest.manifest_id,
        "part": part_no,
        "part_key": part_key,
        "url": url,
        "expires_in": int(os.getenv("UMRS_PRESIGNED_URL_TTL_SEC", "3600")),
    }


@router.post("/manifests/{manifest_id}/parts/{part_no}", response_model=dict[str, Any])
def record_part(
    body: RecordPartRequest,
    manifest_id: str = FPath(...),
    part_no: int = FPath(..., ge=1),
) -> dict[str, Any]:
    """Record a confirmed part upload (post-PUT handshake + provenance)."""
    manifest = _require_manifest(manifest_id)
    _require_uploading(manifest)
    if part_no > manifest.expected_parts:
        raise_typed(
            PlatformError.intake(
                "bad_request",
                f"part {part_no} exceeds expected_parts={manifest.expected_parts}"
            ),
        )
    if part_no not in manifest.received_parts:
        manifest.received_parts.append(part_no)
        manifest.received_parts.sort()
    manifest.parts_detail[str(part_no)] = {
        "etag": body.etag,
        "size_bytes": body.size_bytes,
    }
    _get_store().save(manifest)
    return {
        "status": "ok",
        "manifest_id": manifest.manifest_id,
        "received_parts": manifest.received_parts,
        "missing_parts": manifest.missing_parts(),
    }


def _head_part(client: Any, bucket: str, part_key: str) -> dict[str, Any] | None:
    """HEAD a part; None when absent."""
    try:
        return client.head_object(Bucket=bucket, Key=part_key)
    except ClientError:
        return None


def _assemble_parts(
    client: Any,
    bucket: str,
    manifest: IntakeUploadManifest,
    part_keys: list[str],
    tmp_path: str,
) -> None:
    """Stream each part into one temp file on disk."""
    with tempfile.TemporaryFile(dir=tmp_path) as tmp:
        for part_key in part_keys:
            response = client.get_object(Bucket=bucket, Key=part_key)
            body = response["Body"]
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.flush()
        tmp.seek(0)
        client.put_object(
            Bucket=bucket,
            Key=manifest.target_key,
            Body=tmp,
            Metadata={"upload_manifest_id": manifest.manifest_id},
        )


def _part_keys(manifest: IntakeUploadManifest) -> list[str]:
    """Ordered part keys for a manifest."""
    return [
        _part_key(manifest.target_key, manifest.manifest_id, n)
        for n in range(1, manifest.expected_parts + 1)
    ]


def _probe_parts(
    client: Any, bucket: str, manifest: IntakeUploadManifest
) -> tuple[dict[int, dict[str, Any]], int]:
    """HEAD every part. Returns (present-map, total-bytes)."""
    present: dict[int, dict[str, Any]] = {}
    size_total = 0
    for index, part_key in enumerate(_part_keys(manifest), start=1):
        head = _head_part(client, bucket, part_key)
        if head is not None:
            present[index] = head
            size_total += int(head.get("ContentLength") or 0)
    return present, size_total


def _raise_if_missing(manifest: IntakeUploadManifest, present: dict[int, Any]) -> None:
    """409 with the missing part list when the set is incomplete."""
    missing = [n for n in range(1, manifest.expected_parts + 1) if n not in present]
    if missing:
        raise_typed(
            PlatformError.intake(
                "conflict",
                { "error": "incomplete upload", "missing_parts": missing, "manifest_id": manifest.manifest_id, }
            ),
        )


def _merge_part_provenance(
    manifest: IntakeUploadManifest, present: dict[int, dict[str, Any]]
) -> None:
    """Refresh per-part provenance + received list from verified HEADs."""
    for index, head in present.items():
        manifest.parts_detail[str(index)] = {
            "etag": str(head.get("ETag") or "").strip('"'),
            "size_bytes": int(head.get("ContentLength") or 0),
        }
        if index not in manifest.received_parts:
            manifest.received_parts.append(index)
    manifest.received_parts.sort()


def _complete_manifest(
    manifest: IntakeUploadManifest, config: S3Config
) -> dict[str, Any]:
    """Verify + assemble + ingest. Returns the completion payload."""
    client = config.get_client()
    bucket = config.bucket

    present, size_total = _probe_parts(client, bucket, manifest)
    _raise_if_missing(manifest, present)

    _assemble_parts(
        client, bucket, manifest, _part_keys(manifest), tempfile.gettempdir()
    )
    final_head = client.head_object(Bucket=bucket, Key=manifest.target_key) or {}
    final_etag = str(final_head.get("ETag") or "").strip('"')

    _merge_part_provenance(manifest, present)
    manifest.status = "assembled"
    manifest.final_etag = final_etag
    manifest.size_bytes = size_total
    manifest.assembled_at = time.time()
    _get_store().save(manifest)

    from umrs.data_catalog.triggers import trigger_post_upload_ingestion

    job_id = trigger_post_upload_ingestion(
        client,
        bucket,
        s3_key=manifest.target_key,
        object_etag=final_etag,
        project_id=manifest.project_id,
        owner=manifest.owner or manifest.project_id,
        filename=Path(manifest.target_key).name,
    )
    return {
        "status": "assembled",
        "manifest": manifest.to_dict(),
        "final_s3_key": manifest.target_key,
        "final_etag": final_etag,
        "size_bytes": size_total,
        "job_id": job_id,
    }


@router.post("/manifests/{manifest_id}/complete", response_model=dict[str, Any])
def complete_manifest(manifest_id: str = FPath(...)) -> dict[str, Any]:
    """Verify, assemble, ingest, and close the manifest (idempotent)."""
    manifest = _require_manifest(manifest_id)
    if manifest.status == "assembled":
        return {
            "status": "assembled",
            "manifest": manifest.to_dict(),
            "idempotent": True,
        }
    _require_uploading(manifest)
    return _complete_manifest(manifest, get_s3_config())
