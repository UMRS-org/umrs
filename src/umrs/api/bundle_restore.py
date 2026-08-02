"""K4: synchronous .kb bundle restore route.

`POST /umrs/knowledge/restore` — restores an exported knowledge bundle
from a server-side directory after manifest integrity verification
(when `UMRS_SECRET_KEY` configures HMAC). All restores are idempotent
(re-runnable for DR repair); rows are individually fault-isolated.

Synchronous by design, mirroring the export route's in-process model;
a durable-job variant can layer on later without changing the engine.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-restore"])


class RestoreRequest(BaseModel):
    """Bundle restore request."""

    bundle_dir: str = Field(
        ...,
        min_length=1,
        description="Server-side path to the bundle directory (contains manifest.json)",
    )
    project_id: str | None = Field(
        default=None,
        description="Retarget all restored rows to this project (defaults to the bundle's own project)",
    )
    verify: bool = Field(
        default=True,
        description="Verify manifest hash + HMAC before restoring (needs UMRS_SECRET_KEY)",
    )


@router.post("/restore", response_model=dict[str, Any])
async def restore_bundle(request: Request, body: RestoreRequest) -> dict[str, Any]:
    """K4: restore a .kb bundle with manifest verification."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    from umrs.knowledge.bundle_import import BundleFormatError, KnowledgeImportEngine
    from umrs.knowledge.store import KnowledgeStore

    engine = KnowledgeImportEngine(
        store=KnowledgeStore(pool),
        secret_key=os.getenv("UMRS_SECRET_KEY", ""),
    )
    try:
        result = await engine.import_project(
            body.bundle_dir,
            target_project_id=body.project_id,
            verify=body.verify,
        )
    except BundleFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "restore": result.to_dict()}
