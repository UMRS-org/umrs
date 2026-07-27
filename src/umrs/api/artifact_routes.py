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

from fastapi import APIRouter, HTTPException, Query, status

from ..artifact_registry.models import Artifact, ArtifactQuery
from ..artifact_registry.store import (
    ArtifactNotFoundError,
    ArtifactStore,
)
from .jobs import get_optional_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/artifacts", tags=["artifact_registry"])


def _artifact_response(artifact: Artifact) -> dict:
    """Serialize an Artifact to dict for JSON response."""
    return artifact.to_dict()


@router.get("")
async def list_artifacts(
    artifact_type: str | None = Query(None, description="Filter by artifact type"),
    owner_id: str | None = Query(None, description="Filter by owner"),
    project_id: str | None = Query(None, description="Filter by project"),
    execution_id: str | None = Query(None, description="Filter by execution"),
    name_prefix: str | None = Query(None, description="Filter by name prefix"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset"),
) -> list[dict]:
    """List artifacts with optional filters."""
    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    query = ArtifactQuery(
        artifact_type=artifact_type,
        owner_id=owner_id,
        project_id=project_id,
        execution_id=execution_id,
        name_prefix=name_prefix,
        limit=limit,
        offset=offset,
    )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
        results = store.list(query)
        return [_artifact_response(r) for r in results]
    finally:
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())


@router.post("")
async def register_artifact(artifact: Artifact) -> dict:
    """Register a new artifact in the registry."""
    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
        created = store.create(artifact)
        return _artifact_response(created)
    finally:
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str) -> dict:
    """Get an artifact by id."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact id: {artifact_id}",
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
        artifact = store.get(aid)
        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact {artifact_id} not found",
            )
        return _artifact_response(artifact)
    finally:
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())


@router.get("/{artifact_id}/lineage")
async def get_lineage(
    artifact_id: str,
    depth: int = Query(3, ge=1, le=10, description="Max depth"),
) -> list[dict]:
    """Get lineage edges for an artifact."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact id: {artifact_id}",
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
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
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())


@router.patch("/{artifact_id}/metadata")
async def update_metadata(
    artifact_id: str,
    patch: dict,
) -> dict:
    """Update an artifact's metadata (merge, not replace)."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact id: {artifact_id}",
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
        try:
            store.update_metadata(aid, patch)
            return {"status": "ok"}
        except ArtifactNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
    finally:
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())


@router.delete("/{artifact_id}", status_code=status.HTTP_200_OK)
async def delete_artifact(artifact_id: str) -> dict:
    """Delete an artifact and its lineage."""
    try:
        aid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact id: {artifact_id}",
        )

    job_store = get_optional_job_store()
    if not job_store or not job_store.db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact store not available",
        )

    conn_factory = job_store.db_pool.getconn
    try:
        store = ArtifactStore(conn_factory)
        deleted = store.delete(aid)
        return {"deleted": deleted}
    finally:
        if job_store.db_pool:
            job_store.db_pool.putconn(conn_factory())
