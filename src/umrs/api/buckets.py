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
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from umrs.shared.storage.s3_config import get_s3_config
from umrs.shared.storage.user_buckets import (
    UserBucket,
    UserBucketError,
    bucket_name,
    get_user_bucket_store,
)

from ._access import checked_access, resource_caller
from ._verified_headers import CallerIdentity, get_verified_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/buckets", tags=["buckets"])


class CreateBucketRequest(BaseModel):
    project: str = Field(..., description="Your project / experiment name, e.g. opossum")
    alias: str = Field(..., description="Short bucket alias, e.g. videos")


class BucketAccessUpdate(BaseModel):
    """ADR-080 object-access list update (canonical scope grammar)."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = owner only"
        ),
    )


class BucketInfo(BaseModel):
    bucket: str
    output_bucket: str
    owner: str
    project: str = ""
    alias: str = ""
    access: list[str] = Field(default_factory=list)
    kind: str = "personal"  # personal (ADR-079) | provisioned (user config) | shared (ADR-080)


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
    """Create (or re-confirm) the caller's ``<user>.<project>.<alias>``.
    # auth: verified
    """
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


def _provisioned_rows(user_id: str, seen: set[str]) -> list[BucketInfo]:
    """Provisioned buckets of the caller's user config (site-provided)."""
    from umrs.server import step_staging

    registry = step_staging._bucket_registry
    if registry is None:
        return []
    rows: list[BucketInfo] = []
    for path, entry in list(registry.items()):
        if getattr(entry, "s3_username", None) != user_id or path in seen:
            continue
        if path.endswith("-output"):
            continue
        rows.append(
            BucketInfo(
                bucket=path,
                output_bucket=f"{path}-output",
                owner=user_id,
                alias=str(getattr(entry, "bucket_alias", "") or ""),
                kind="provisioned",
            )
        )
    return rows


@router.get("", response_model=BucketListResponse)
async def list_buckets(
    caller: CallerIdentity = Depends(get_verified_caller),
) -> BucketListResponse:
    """The caller's personal buckets plus provisioned ones from their config
    plus buckets whose ADR-080 access list admits them (``kind="shared"``).
    # auth: verified
    """
    store = get_user_bucket_store()
    personal = await asyncio.to_thread(store.list_for, caller.user_id)
    rows = [BucketInfo(**b.to_dict(), kind="personal") for b in personal]
    seen = {b.bucket for b in personal}
    rows.extend(_provisioned_rows(caller.user_id, seen))
    shared = await asyncio.to_thread(store.list_shared_with, caller.user_id)
    rows.extend(
        BucketInfo(**entry.to_dict(), kind="shared")
        for entry in shared
        if entry.bucket not in seen
    )
    return BucketListResponse(buckets=sorted(rows, key=lambda r: r.bucket))


@router.get("/{bucket}/access")
async def get_bucket_access(
    bucket: str,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """The bucket's ADR-080 object-access list (owner, admin, or a caller
    the list admits).
    # auth: verified
    """
    store = get_user_bucket_store()
    entry = await asyncio.to_thread(store.get, bucket)
    if entry is None:
        raise_typed(PlatformError.files("not_found", f"Bucket {bucket!r} not found"))
    allowed = (
        entry.owner_id == caller.user_id
        or caller.is_admin
        or await asyncio.to_thread(store.can_access, bucket, caller.user_id)
    )
    if not allowed:
        raise_typed(PlatformError.files("not_found", f"Bucket {bucket!r} not found"))
    return {"bucket": entry.bucket, "owner": entry.owner_id, "access": list(entry.access)}


async def _refuse_access_change(
    store: Any, bucket: str, caller: CallerIdentity
) -> None:
    """Refuse a non-owner, non-admin access change. A caller the list does not
    admit gets files.not_found, as for an unknown bucket (GET .../access
    hides both too); a 403 told it the guessable name exists (H-1,
    2026-10-18). A caller the list admits already sees the bucket: 403."""
    if not await asyncio.to_thread(store.can_access, bucket, caller.user_id):
        raise_typed(PlatformError.files("not_found", f"Bucket {bucket!r} not found"))
    raise_typed(
        PlatformError.files(
            "bucket_denied", f"Only the owner of {bucket!r} may change its access"
        )
    )


@router.put("/{bucket}/access", response_model=BucketInfo)
async def set_bucket_access(
    bucket: str,
    body: BucketAccessUpdate,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> BucketInfo:
    """Replace the bucket's ADR-080 object-access list (owner or admin;
    audited). Governs OBJECT reads/writes; the bucket lifecycle stays
    owner-only; ``system`` entries are admin-only.
    # auth: verified
    """
    store = get_user_bucket_store()
    entry = await asyncio.to_thread(store.get, bucket)
    if entry is None:
        raise_typed(PlatformError.files("not_found", f"Bucket {bucket!r} not found"))
    if entry.owner_id != caller.user_id and not caller.is_admin:
        await _refuse_access_change(store, bucket, caller)
    guard = resource_caller(
        request,
        user_id=caller.user_id,
        project_id=caller.project_id or "",
        is_admin=caller.is_admin,
    )
    access = checked_access(
        body.access, guard, error=PlatformError.files, name="bucket_denied"
    )
    try:
        await asyncio.to_thread(store.set_access, bucket, access)
    except UserBucketError as exc:
        raise_typed(PlatformError.files("bucket_registry_unavailable", str(exc)), cause=exc)
    from .audit_events import emit_audit

    # Audited under the actor's verified project, never the bucket's project
    # segment: that segment is a free experiment label (ADR-079 D-2), so a
    # caller could otherwise file audit rows under any project (H-1, 2026-10-18).
    emit_audit(
        "bucket.access_changed",
        project_id=caller.project_id or "",
        details={
            "bucket": entry.bucket,
            "label": entry.project,
            "access": access,
            "by": caller.user_id,
        },
    )
    updated = await asyncio.to_thread(store.get, entry.bucket)
    assert updated is not None  # set_access would have failed otherwise
    return BucketInfo(**updated.to_dict(), kind="personal")
