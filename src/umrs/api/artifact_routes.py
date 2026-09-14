"""Artifact Registry HTTP API.

FastAPI router exposing artifact CRUD and lineage operations.

Routes:
    GET  /umrs/artifacts              (list with filters)
    POST /umrs/artifacts              (register artifact)
    GET  /umrs/artifacts/{id}         (get by id)
    GET  /umrs/artifacts/{id}/lineage (get lineage edges)
    PATCH /umrs/artifacts/{id}/metadata (update metadata)
    DELETE /umrs/artifacts/{id}       (delete artifact)
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Query, status

from ..artifact_registry.models import Artifact, ArtifactQuery
from ..artifact_registry.store import (
    ArtifactNotFoundError,
    ArtifactStore,
)
from .endpoints import _caller_is_admin
from .files import get_project_id
from .jobs import get_optional_job_store
from .typed_errors import PlatformError, raise_typed


def _scope_checked_project(
    caller_project: str,
    is_admin: bool,
    requested: str | None,
) -> str | None:
    """Project filter bound to the caller unless admin.

    Caller cannot enumerate another project's artifacts: requested filter is
    only honored when it matches the caller's own project (or admin).
    """
    if is_admin:
        return requested
    return caller_project


def _require_artifact_scope(artifact, caller_project: str, is_admin: bool) -> None:
    """404 (hide existence) on cross-project/non-owner artifact access."""
    if is_admin:
        return
    if artifact is not None and artifact.project_id == caller_project:
        return
    raise_typed(
        PlatformError.artifacts("not_found", "Artifact not found"),
    )


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/artifacts", tags=["artifact_registry"])


def _artifact_response(artifact: Artifact) -> dict:
    """Serialize an Artifact to dict for JSON response."""
    return artifact.to_dict()


@router.get("")
async def list_artifacts(
    artifact_type: str | None = Query(None, description="Filter by artifact type"),
    owner_id: str | None = Query(None, description="Filter by owner"),
    project_id: str | None = Query(None, description="Filter by project (admin only)"),
    execution_id: str | None = Query(None, description="Filter by execution"),
    name_prefix: str | None = Query(None, description="Filter by name prefix"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset"),
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> list[dict]:
    """List artifacts with optional filters."""
    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    query = ArtifactQuery(
        artifact_type=artifact_type,
        owner_id=owner_id,
        project_id=_scope_checked_project(caller_project, is_admin, project_id),
        execution_id=execution_id,
        name_prefix=name_prefix,
        limit=limit,
        offset=offset,
    )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        results = store.list(query)
        return [_artifact_response(r) for r in results]
    finally:
        store.close()


@router.post("")
async def register_artifact(
    artifact: Artifact,
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> dict:
    """Register a new artifact in the registry."""
    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    # Non-admin callers can only register into their own project.
    _require_artifact_scope(artifact, caller_project, is_admin)

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        created = store.create(artifact)
        return _artifact_response(created)
    finally:
        store.close()


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> dict:
    """Get an artifact by id."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise_typed(
            PlatformError.artifacts(
                "bad_request", f"Invalid artifact id: {artifact_id}"
            ),
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        artifact = store.get(aid)
        if artifact is None:
            raise_typed(
                PlatformError.artifacts(
                    "not_found", f"Artifact {artifact_id} not found"
                ),
            )
        _require_artifact_scope(artifact, caller_project, is_admin)
        return _artifact_response(artifact)
    finally:
        store.close()


@router.get("/{artifact_id}/lineage")
async def get_lineage(
    artifact_id: str,
    depth: int = Query(3, ge=1, le=10, description="Max depth"),
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> list[dict]:
    """Get lineage edges for an artifact."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise_typed(
            PlatformError.artifacts(
                "bad_request", f"Invalid artifact id: {artifact_id}"
            ),
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        artifact = store.get(aid)
        if artifact is None:
            raise_typed(
                PlatformError.artifacts(
                    "not_found", f"Artifact {artifact_id} not found"
                ),
            )
        _require_artifact_scope(artifact, caller_project, is_admin)
        edges = store.lineage(aid, depth=depth)
        return [
            {
                "artifact_id": str(e.artifact_id),
                "parent_artifact_id": str(e.parent_artifact_id),
                "relation": e.relation,
            }
            for e in edges
        ]
    finally:
        store.close()


@router.patch("/{artifact_id}/metadata")
async def update_metadata(
    artifact_id: str,
    patch: dict,
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> dict:
    """Update an artifact's metadata (merge, not replace)."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise_typed(
            PlatformError.artifacts(
                "bad_request", f"Invalid artifact id: {artifact_id}"
            ),
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        try:
            artifact = store.get(aid)
            _require_artifact_scope(artifact, caller_project, is_admin)
            store.update_metadata(aid, patch)
            return {"status": "ok"}
        except ArtifactNotFoundError as exc:
            raise_typed(PlatformError.artifacts("not_found", str(exc)), cause=exc)
    finally:
        store.close()


@router.delete("/{artifact_id}", status_code=status.HTTP_200_OK)
async def delete_artifact(
    artifact_id: str,
    caller_project: str = Depends(get_project_id),
    is_admin: bool = Depends(_caller_is_admin),
) -> dict:
    """Delete an artifact and its lineage."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise_typed(
            PlatformError.artifacts(
                "bad_request", f"Invalid artifact id: {artifact_id}"
            ),
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise_typed(
            PlatformError.artifacts("unavailable", "Artifact store not available"),
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory, conn_release=job_store.db_pool.putconn)
        artifact = store.get(aid)
        _require_artifact_scope(artifact, caller_project, is_admin)
        deleted = store.delete(aid)
        return {"deleted": deleted}
    finally:
        store.close()
