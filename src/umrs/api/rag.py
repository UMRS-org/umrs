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

from umrs.knowledge.access_scope import (
    CallerScope,
    project_readable,
    project_writable,
)
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
    knowledge_nodes_minted: int | None = None


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


def _require_rag_scope(
    request: Request,
    target_project: str | None = None,
    *,
    write: bool = False,
) -> tuple[str, str]:
    """Identity + three-tier scope predicate for rag proxy routes (ADR-027 P0).

    Returns (user_id, verified_project_id). target_project defaults to the
    verified project. Reads use project_readable; writes use
    project_writable. Effects vs the legacy equality check: admin bypass,
    'system' corpus readable by any verified caller (writable by admin
    only), and 'user:<uid>' namespaces enforced owner-only.
    """
    user_id, project_id = _get_caller_identity(request)
    admin = request.headers.get("x-verified-is-admin", "").strip().lower() in (
        "true",
        "1",
    )
    caller = CallerScope(user_id=user_id, project_id=project_id, is_admin=admin)
    target = project_id if not target_project else target_project
    allowed = (
        project_writable(target, caller) if write else project_readable(target, caller)
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: no {'write' if write else 'read'} scope for '{target}'",
        )
    return user_id, project_id


# ADR-027 P1a: cap knowledge minting per upload (guard against giant docs).
_KNOWLEDGE_NODE_CAP = 100


async def _upload_success(
    request: Request,
    *,
    result: dict[str, Any],
    filename: str,
    mime_type: str,
    user_id: str,
    project_id: str,
) -> UploadResponse:
    """Governance record + knowledge minting + response for a good upload."""
    from .audit_events import emit_audit
    from .governance import record_resource_created

    emit_audit(
        "rag.upload",
        user_id=user_id,
        project_id=project_id,
        details={
            "filename": filename,
            "document_id": result.get("document_id", ""),
            "chunks": result.get("chunks", 0),
        },
    )
    record_resource_created(
        "rag_documents",
        result.get("document_id", filename or "unknown"),
        user_id,
        project_id=project_id,
    )

    # ADR-027 P1a: mint canonical knowledge nodes (never fails upload).
    minted = await _mint_knowledge_nodes_best_effort(
        request,
        user_id=user_id,
        project_id=project_id,
        document_id=result.get("document_id", ""),
        filename=filename,
        mime_type=mime_type,
        chunk_details=result.get("chunk_details") or [],
    )
    return UploadResponse(
        document_id=result.get("document_id", ""),
        filename=filename,
        status="success",
        chunks=result.get("chunks", 0),
        knowledge_nodes_minted=minted,
    )


async def _mint_knowledge_nodes(
    request: Request,
    *,
    user_id: str,
    project_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
    chunk_details: list[dict[str, Any]],
) -> int:
    """Mint canonical document + chunk nodes for a rag upload.

    ADR-027 P1a: uploaded documents become visible to the canonical
    knowledge graph / research loops, instead of living only in the rag
    corpus (closes the G4 feedback gap for the upload path).
    """
    import hashlib

    from umrs.knowledge.graph_registry import default_graph_id
    from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore

    from ._asyncpg_pool import knowledge_asyncpg_pool

    pool = await knowledge_asyncpg_pool(request)
    store = KnowledgeStore(pool)
    gid = default_graph_id(project_id)

    doc_node_id = f"ragdoc-{document_id}"
    content_hash = hashlib.sha256(
        f"{document_id}:{filename}".encode("utf-8")
    ).hexdigest()[:16]
    artifact_id = f"artifact-rag-{document_id}"
    await store.insert_artifact(
        artifact_id,
        content_hash,
        "document",
        project_id,
        source_ref=f"rag://{document_id}",
        mime_type=mime_type,
        metadata={
            "origin": "rag.upload",
            "owner_id": user_id,
            "filename": filename,
            "document_id": document_id,
        },
    )
    await store.upsert_node(
        KnowledgeNode(
            node_id=doc_node_id,
            project_id=project_id,
            label=filename,
            node_type="file",
            content={
                "kind": "document",
                "document_id": document_id,
                "filename": filename,
                "origin": "rag.upload",
                "owner_id": user_id,
                "artifact_id": artifact_id,
            },
            content_hash=content_hash,
            graph_id=gid,
        )
    )
    minted = 0
    for i, detail in enumerate(chunk_details[:_KNOWLEDGE_NODE_CAP]):
        chunk_id = f"{doc_node_id}-chunk{i}"
        await store.upsert_node(
            KnowledgeNode(
                node_id=chunk_id,
                project_id=project_id,
                label=f"chunk {i} of {filename}",
                node_type="entity",
                content={
                    "kind": "chunk",
                    "document_id": document_id,
                    "chunk_index": i,
                    "s3_object": detail.get("s3_object"),
                    "chunk_start": detail.get("chunk_start"),
                    "chunk_end": detail.get("chunk_end"),
                    "chunk_md5": detail.get("chunk_md5"),
                    "excerpt": str(detail.get("excerpt") or "")[:300],
                    "origin": "rag.upload",
                },
                content_hash=str(detail.get("chunk_md5") or ""),
                graph_id=gid,
            )
        )
        await store.upsert_edge(
            KnowledgeEdge(
                edge_id=f"{chunk_id}-part-of",
                project_id=project_id,
                source_id=chunk_id,
                target_id=doc_node_id,
                relation_type="part_of",
                content={"kind": "part_of", "origin": "rag.upload"},
                graph_id=gid,
            )
        )
        minted += 1
    return minted


async def _mint_knowledge_nodes_best_effort(
    request: Request,
    **kwargs: Any,
) -> int:
    """Knowledge minting must never fail an otherwise-successful upload."""
    try:
        return await _mint_knowledge_nodes(request, **kwargs)
    except Exception as exc:  # noqa: BLE001 — additive enrichment, fail-soft
        logger.warning(
            "knowledge node minting skipped for %s: %s",
            kwargs.get("document_id"),
            exc,
        )
        return 0


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
    _user_id, project_id = _require_rag_scope(
        request, request_body.project_id, write=False
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
    _user_id, project_id = _require_rag_scope(request, write=True)

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
            return await _upload_success(
                request,
                result=result,
                filename=file.filename or "",
                mime_type=file.content_type or "",
                user_id=_user_id,
                project_id=project_id,
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
    _user_id, project_id = _require_rag_scope(request, write=False)

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
    _user_id, project_id = _require_rag_scope(request, write=True)

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
    _user_id, project_id = _require_rag_scope(request, write=False)

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
    _user_id, project_id = _require_rag_scope(request, write=False)

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
