"""
UMRS RAG Proxy Router.

Routes /umrs/rag/* to the RAG service inside the active mini-RiVault
container (port 8003), resolving the endpoint from the local
EndpointRegistry (no HTTP hop).

RAG indexing and retrieval run INSIDE mini-rivault containers,
never in Gateway or standalone UMRS services.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    get_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/rag", tags=["rag-proxy"])


def _trace_headers() -> dict[str, str]:
    """Build headers that propagate correlation ID to downstream services."""
    try:
        from umrs.server.correlation import get_request_id

        rid = get_request_id()
        if rid:
            return {"X-Request-ID": rid}
    except Exception:
        pass
    return {}


# RAG service port inside mini-rivault container
_RAG_PORT = int(os.environ.get("MINI_RIVAULT_RAG_PORT", "8003"))

# Direct RAG service URL for standalone deployments (bypasses endpoint discovery)
_RAG_DIRECT_URL = os.environ.get("UMRS_RAG_DIRECT_URL", "").strip()

# Timeouts
_CONNECT_TIMEOUT = float(os.environ.get("RAG_PROXY_CONNECT_TIMEOUT", "10"))
_READ_TIMEOUT = float(os.environ.get("RAG_PROXY_READ_TIMEOUT", "3600"))


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class RAGQueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., description="Query text")
    project_id: str = Field(..., description="Project ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_sources: bool = Field(True, description="Include source documents")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")


class RAGSource(BaseModel):
    """A source document from RAG retrieval."""

    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[RAGSource] = Field(default_factory=list)
    usage: dict[str, int] | None = None


class DocumentInfo(BaseModel):
    """Document information."""

    document_id: str
    filename: str
    project_id: str
    chunks: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Document upload response."""

    document_id: str
    filename: str
    status: str
    chunks: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_caller_identity(request: Request) -> tuple[str, str]:
    """Extract user_id and project_id from NGINX-set headers."""
    user_id = request.headers.get("x-user-id", "")
    verified_project_id = request.headers.get("x-verified-project-id", "").strip()
    project_id = verified_project_id or request.headers.get("x-project-id", "").strip()
    if (
        not verified_project_id
        and os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Verified-Project-ID header (Gateway must set this)",
        )
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Project-ID header (auth sidecar must set this)",
        )
    return user_id, project_id


def _discover_rag_endpoint(
    registry: EndpointRegistry,
    project_id: str,
) -> str:
    """
    Discover a ready mini-RiVault endpoint for RAG.

    RAG is model-agnostic; we find any usable endpoint for the project.
    Returns the RAG service base URL.

    If UMRS_RAG_DIRECT_URL is set, uses that directly (standalone mode)
    without requiring an active GPU endpoint.
    """
    # Standalone mode: RAG service runs natively on the front-end server
    if _RAG_DIRECT_URL:
        return _RAG_DIRECT_URL.rstrip("/")

    endpoints = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )

    if not endpoints:
        # Try system-tier endpoints
        endpoints = registry.list_all(
            project_id="system",
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )

    if not endpoints:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active endpoint available. Please provision an endpoint first.",
        )

    ep = endpoints[0]
    return f"http://{ep.host}:{_RAG_PORT}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request_body: RAGQueryRequest,
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """
    Query documents using RAG.

    Discovers active mini-rivault endpoint, forwards to RAG service.
    """
    _user_id, project_id = _get_caller_identity(request)

    if request_body.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: project_id mismatch",
        )

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/query"

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=5.0
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                rag_url, json=request_body.model_dump(), headers=_trace_headers()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"RAG service error: {response.text}",
                )

            data = response.json()
            return RAGQueryResponse(
                answer=data.get("answer", ""),
                sources=[
                    RAGSource(
                        document_id=src.get("document_id", ""),
                        content=src.get("content", ""),
                        score=src.get("score", 0.0),
                        metadata=src.get("metadata", {}),
                    )
                    for src in data.get("sources", [])
                ],
                usage=data.get("usage"),
            )

    except httpx.RequestError as e:
        logger.exception("RAG query request failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    registry: EndpointRegistry = Depends(get_registry),
):
    """Upload a document for RAG indexing."""
    _user_id, project_id = _get_caller_identity(request)

    # Governance quota check
    try:
        from .governance import check_quota

        check_quota(project_id, "rag_documents")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        pass  # governance not active

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/upload"

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=5.0
    )

    try:
        content = await file.read()

        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {
                "file": (
                    file.filename,
                    content,
                    file.content_type or "application/octet-stream",
                )
            }
            data = {"project_id": project_id}

            response = await client.post(
                rag_url, files=files, data=data, headers=_trace_headers()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Upload failed: {response.text}",
                )

            result = response.json()
            from .audit_events import emit_audit

            emit_audit(
                "rag.upload",
                user_id=_user_id,
                project_id=project_id,
                details={
                    "filename": file.filename,
                    "document_id": result.get("document_id", ""),
                    "chunks": result.get("chunks", 0),
                },
            )

            # Governance tracking
            from .governance import record_resource_created

            record_resource_created(
                "rag_documents",
                result.get("document_id", file.filename or "unknown"),
                _user_id,
                project_id=project_id,
            )

            return UploadResponse(
                document_id=result.get("document_id", ""),
                filename=file.filename,
                status="success",
                chunks=result.get("chunks", 0),
            )

    except httpx.RequestError as e:
        logger.exception("Upload request failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """List all documents in a project."""
    _user_id, project_id = _get_caller_identity(request)

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/documents"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                rag_url, params={"project_id": project_id}, headers=_trace_headers()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to list documents: {response.text}",
                )

            documents = response.json().get("documents", [])
            return [
                DocumentInfo(
                    document_id=doc.get("document_id", ""),
                    filename=doc.get("filename", ""),
                    project_id=doc.get("project_id", project_id),
                    chunks=doc.get("chunks", 0),
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]

    except httpx.RequestError as e:
        logger.exception("List documents failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """Delete a document from the project."""
    _user_id, project_id = _get_caller_identity(request)

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/documents/{document_id}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                rag_url, params={"project_id": project_id}, headers=_trace_headers()
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete: {response.text}",
                )

            # Governance tracking (best-effort)
            from .governance import record_resource_deleted

            record_resource_deleted("rag_documents", document_id, project_id)

            return {"status": "deleted", "document_id": document_id}

    except httpx.RequestError as e:
        logger.exception("Delete document failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e


@router.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document(
    document_id: str,
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """Get document metadata."""
    _user_id, project_id = _get_caller_identity(request)

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/documents/{document_id}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                rag_url, params={"project_id": project_id}, headers=_trace_headers()
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get document: {response.text}",
                )

            doc = response.json()
            return DocumentInfo(
                document_id=doc.get("document_id", ""),
                filename=doc.get("filename", ""),
                project_id=doc.get("project_id", project_id),
                chunks=doc.get("chunks", 0),
                metadata=doc.get("metadata", {}),
            )

    except httpx.RequestError as e:
        logger.exception("Get document failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e


@router.get("/stats")
async def rag_stats(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """Get RAG corpus statistics for the current project."""
    _user_id, project_id = _get_caller_identity(request)

    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/stats"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                rag_url, params={"project_id": project_id}, headers=_trace_headers()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get RAG stats: {response.text}",
                )

            return response.json()

    except httpx.RequestError as e:
        logger.exception("RAG stats request failed: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e
