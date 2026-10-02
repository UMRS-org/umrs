"""Self-service personal buckets (ADR-079).

``POST /umrs/buckets`` creates ``<caller>.<project>.<alias>`` and its
``-output`` pair on the server's own object store and records the caller as
owner; ``GET /umrs/buckets`` lists the caller's buckets (personal and the
provisioned ones of their user config). The caller then moves data with the
files API (``bucket=``) and names the bucket in a workflow's ``bucket``
param; the HPC lane stages it because the ownership record matches the run
owner (ADR-074 D-5). No credential or presigned URL is ever returned.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from umrs.shared.storage.s3_config import get_s3_config
from umrs.shared.storage.user_buckets import (
    UserBucket,
    UserBucketError,
    bucket_name,
    get_user_bucket_store,
)

from ._verified_headers import CallerIdentity, get_verified_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/buckets", tags=["buckets"])


class CreateBucketRequest(BaseModel):
    project: str = Field(..., description="Your project / experiment name, e.g. opossum")
    alias: str = Field(..., description="Short bucket alias, e.g. videos")


class BucketInfo(BaseModel):
    bucket: str
    output_bucket: str
    owner: str
    project: str = ""
    alias: str = ""
    kind: str = "personal"  # personal (ADR-079) | provisioned (user config)


class BucketListResponse(BaseModel):
    buckets: list[BucketInfo] = Field(default_factory=list)


def _ensure_bucket(s3: Any, name: str) -> None:
    """Create *name* unless it exists; an existing bucket is fine only when
    the ownership checks below already allowed the caller to claim it."""
    try:
        s3.head_bucket(Bucket=name)
        return
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
    try:
        s3.create_bucket(Bucket=name)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code != "BucketAlreadyOwnedByYou":
            raise


def _provisioned_owner(name: str) -> str | None:
    from umrs.server import step_staging

    registry = step_staging._bucket_registry
    if registry is None:
        return None
    return step_staging._registered_owner(registry, name)


@router.post("", response_model=BucketInfo, status_code=201)
async def create_bucket(
    body: CreateBucketRequest,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> BucketInfo:
    """Create (or re-confirm) the caller's ``<user>.<project>.<alias>``."""
    try:
        name = bucket_name(caller.user_id, body.project, body.alias)
    except UserBucketError as exc:
        raise_typed(PlatformError.files("bucket_name_invalid", str(exc)), cause=exc)
    store = get_user_bucket_store()
    entry = UserBucket(name, caller.user_id, body.project.strip(), body.alias.strip())
    existing = await asyncio.to_thread(store.get, name)
    if existing is not None and existing.owner_id != caller.user_id:
        raise_typed(PlatformError.files("bucket_taken", f"Bucket {name!r} belongs to another user"))
    provisioned = _provisioned_owner(name)
    if provisioned is not None and provisioned != caller.user_id:
        raise_typed(PlatformError.files("bucket_taken", f"Bucket {name!r} belongs to another user"))
    s3 = get_s3_config().get_client()
    try:
        await asyncio.to_thread(_ensure_bucket, s3, name)
        await asyncio.to_thread(_ensure_bucket, s3, entry.output_bucket)
    except ClientError as exc:
        logger.error("creating personal bucket %s failed: %s", name, exc)
        raise_typed(PlatformError.files("bucket_create_failed", f"Could not create {name!r}"), cause=exc)
    if existing is None:
        try:
            await asyncio.to_thread(store.add, entry)
        except UserBucketError as exc:
            raise_typed(PlatformError.files("bucket_registry_unavailable", str(exc)), cause=exc)
    return BucketInfo(**entry.to_dict(), kind="personal")


@router.get("", response_model=BucketListResponse)
async def list_buckets(
    caller: CallerIdentity = Depends(get_verified_caller),
) -> BucketListResponse:
    """The caller's personal buckets plus provisioned ones from their config."""
    personal = await asyncio.to_thread(get_user_bucket_store().list_for, caller.user_id)
    rows = [BucketInfo(**b.to_dict(), kind="personal") for b in personal]
    seen = {b.bucket for b in personal}
    from umrs.server import step_staging

    registry = step_staging._bucket_registry
    if registry is not None:
        for path, entry in list(registry.items()):
            if getattr(entry, "s3_username", None) != caller.user_id or path in seen:
                continue
            if path.endswith("-output"):
                continue
            rows.append(
                BucketInfo(
                    bucket=path,
                    output_bucket=f"{path}-output",
                    owner=caller.user_id,
                    alias=str(getattr(entry, "bucket_alias", "") or ""),
                    kind="provisioned",
                )
            )
    return BucketListResponse(buckets=sorted(rows, key=lambda r: r.bucket))
