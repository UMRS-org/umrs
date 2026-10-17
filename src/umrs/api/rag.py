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
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
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
from umrs.rag.corpus_access import get_corpus_access_store, shared_corpus_readable

from ._access import checked_access, resource_caller
from .rag_retrieve import LEGACY_TOP_K
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

_ADAPTIVE_K_CAP = 12  # the retrieve verb's k_max bound

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
    top_k: int = Field(
        LEGACY_TOP_K,
        ge=1,
        le=20,
        description=(
            "Most passages: k in threshold mode. In adaptive mode the cut's "
            "k_max (capped at 12) when sent with a value other than the legacy "
            f"default {LEGACY_TOP_K}, which every pre-ADR-093 client serializes; "
            "unset or the legacy default means the site's UMRS_RAG_FINAL_K_MAX"
        ),
    )
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Legacy absolute similarity cut (top_k at or above it). Unset "
            "(ADR-093 D-3): the adaptive cut and calibrated sufficiency"
        ),
    )
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
    # ADR-093 D-4: citations, abstained, uncited, sufficiency, dropped passages.
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Extract user_id and project_id from NGINX-set headers (canonical, R1)."""
    from ._identity import verified_identity

    return verified_identity(request, require_project=True)


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

    ADR-080: a read the tier denies may still pass through the target
    corpus's access list (``rag_corpus_access`` — user/group/project/system
    scopes, group membership resolved against the 0097 authority). Writes
    never consult the list.
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
    if not allowed and not write:
        allowed = shared_corpus_readable(target, user_id)
    if not allowed:
        raise_typed(
            PlatformError.rag(
                "forbidden",
                f"Access denied: no {'write' if write else 'read'} scope for '{target}'",
            )
        )
    return user_id, project_id


# ADR-027 P1a: cap knowledge minting per upload (guard against giant docs).
_KNOWLEDGE_NODE_CAP = 100


class CorpusAccessUpdate(BaseModel):
    """ADR-080 corpus read-sharing update (canonical scope grammar)."""

    project_id: str = Field(..., min_length=1)
    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = owning project tier only"
        ),
    )


@router.get("/corpus-access")
def get_corpus_access(request: Request, project_id: str) -> dict[str, Any]:
    """The corpus's ADR-080 read-sharing list (any caller who may read it).
    # auth: verified
    """
    _require_rag_scope(request, project_id, write=False)
    return {
        "project_id": project_id,
        "access": get_corpus_access_store().get_access(project_id),
    }


@router.put("/corpus-access")
def put_corpus_access(request: Request, body: CorpusAccessUpdate) -> dict[str, Any]:
    """Replace the corpus's ADR-080 read-sharing list (audited).

    Corpus members (project_writable) or admins set it; ``system`` entries
    are admin-only. Governs READS (query/documents/search/stats) — ingest,
    upload and delete stay with the owning project.
    # auth: verified
    """
    user_id, _ = _require_rag_scope(request, body.project_id, write=True)
    admin = request.headers.get("x-verified-is-admin", "").strip().lower() in (
        "true",
        "1",
    )
    caller = resource_caller(
        request,
        user_id=user_id,
        project_id=body.project_id,
        is_admin=admin,
    )
    access = checked_access(body.access, caller, error=PlatformError.rag)
    get_corpus_access_store().set_access(body.project_id, access, set_by=user_id)
    from .audit_events import emit_audit

    emit_audit(
        "rag.corpus_access_changed",
        project_id=body.project_id,
        details={"project_id": body.project_id, "access": access, "by": user_id},
    )
    return {"project_id": body.project_id, "access": access}


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
    content_hash = hashlib.sha256(f"{document_id}:{filename}".encode()).hexdigest()[:16]
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
        raise_typed(
            PlatformError.rag(
                "unavailable",
                "No active endpoint available. Please provision an endpoint first.",
            )
        )

    ep = endpoints[0]
    return f"http://{ep.host}:{_RAG_PORT}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# ADR-027 D3/P2: answer from the knowledge spine. ``spine`` (default since the
# maintainer declared the legacy per-experience corpora disposable,
# 2026-10-09): the spine only, with an honest "nothing answers this";
# ``auto``: the spine when it has passages, else the endpoint corpus;
# ``endpoint``: the old proxy alone.
_QUERY_FACADE_ENV = "UMRS_RAG_QUERY_FACADE"


def _query_facade() -> str:
    value = os.environ.get(_QUERY_FACADE_ENV, "spine").strip().lower()
    return value if value in ("auto", "spine", "endpoint") else "spine"


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request_body: RAGQueryRequest,
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """Answer a question: the knowledge spine first, per UMRS_RAG_QUERY_FACADE.

    Uploads feed only the spine (WP-15), so the endpoint corpus alone could
    never answer from them.
    """
    _user_id, project_id = _require_rag_scope(
        request, request_body.project_id, write=False
    )
    facade = _query_facade()
    if facade != "endpoint":
        spine = await _query_via_spine(
            request, request_body, request_body.project_id or project_id
        )
        if spine.passages or facade == "spine":
            return _spine_response(spine, request_body)
        logger.info(
            "rag query: spine has no passages (%s); endpoint corpus next", spine.reason
        )
    return await _query_via_endpoint(request_body, registry, project_id)


def _spine_k_max(body: RAGQueryRequest) -> int | None:
    """The k the core is handed; ``None`` is the site default it applies.

    Threshold mode: ``top_k`` as sent. Adaptive mode: ``top_k`` (12 at most)
    only when it is not the legacy default — R4's ``RAGQuery`` and the eval
    harness always serialize ``top_k=5``, which overrode
    ``UMRS_RAG_FINAL_K_MAX`` on every call (review of ADR-093, 2026-10-10).
    """
    sent = "top_k" in body.model_fields_set
    if body.threshold is not None:
        return body.top_k if sent else None
    if not sent or body.top_k == LEGACY_TOP_K:
        return None
    return min(body.top_k, _ADAPTIVE_K_CAP)


def _answer_in_scope(
    request: Request, pool: Any, body: RAGQueryRequest, project: str, admin: bool
) -> Any:
    """The spine's answer over the project minus the graphs the caller may
    not read — the retrieve verb's whole-project scope (KG-05)."""
    from .rag_retrieve import hidden_graphs
    from .rag_spine import answer_from_spine

    return answer_from_spine(
        pool,
        body.query,
        project,
        k_max=_spine_k_max(body),
        threshold=body.threshold,
        include_ambient=admin,
        exclude_graph_ids=hidden_graphs(request, pool, project),
    )


async def _query_via_spine(
    request: Request, body: RAGQueryRequest, target_project: str
) -> Any:
    """Run the synchronous spine pipeline (DB, S3, one chat call) off the loop."""
    import asyncio

    from .rag_spine import SpineAnswer

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return SpineAnswer(reason="knowledge database unavailable")
    admin = request.headers.get("x-verified-is-admin", "").strip().lower() in (
        "true",
        "1",
    )
    try:
        return await asyncio.to_thread(
            _answer_in_scope, request, pool, body, target_project, admin
        )
    except Exception as exc:  # noqa: BLE001 — auto mode falls back to the endpoint corpus
        logger.warning("rag query: spine path failed: %s", exc)
        return SpineAnswer(reason=f"spine query failed: {type(exc).__name__}")


def _spine_source(passage: Any, model: str) -> RAGSource:
    scores = dict(passage.scores)
    return RAGSource(
        document_id=passage.source_id,
        content=passage.text,
        score=float(scores.get("dense", scores.get("fused", 0.0))),
        metadata={
            "text_source": passage.text_source,
            "source_name": passage.source_name,
            "artifact_id": passage.artifact_id,
            "section": passage.section,
            "span": list(passage.span) if passage.span else None,
            "rank": passage.rank,
            "scores": scores,
            "lanes": list(passage.lanes),
            "corpus_venue": "spine",
            "model": model,
        },
    )


def _spine_response(spine: Any, body: RAGQueryRequest) -> RAGQueryResponse:
    """The spine's answer, or why there is none; sources name their venue."""
    if spine.answer:
        answer = spine.answer
    elif spine.passages:
        answer = f"Could not generate an answer: {spine.reason}"
    else:
        answer = f"No knowledge in this project answers this ({spine.reason})."
    sources = [_spine_source(passage, spine.model) for passage in spine.passages]
    return RAGQueryResponse(
        answer=answer,
        sources=sources if body.include_sources else [],
        metadata=spine.metadata() if hasattr(spine, "metadata") else {},
    )


async def _query_via_endpoint(
    request_body: RAGQueryRequest, registry: EndpointRegistry, project_id: str
) -> RAGQueryResponse:
    """The per-experience corpus behind a mini-RiVault endpoint (pre-P2 path)."""
    rag_base = _discover_rag_endpoint(registry, project_id)
    rag_url = f"{rag_base}/rag/query"

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=5.0
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # An unset threshold is omitted: the endpoint corpus keeps its 0.7.
            response = await client.post(
                rag_url,
                json=request_body.model_dump(exclude_none=True),
                headers=_trace_headers(),
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
        raise_typed(
            PlatformError.rag("unavailable", "RAG service unavailable"), cause=e
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    registry: EndpointRegistry = Depends(get_registry),
):
    """Upload a document for RAG indexing.

    WP-15 dual-corpus dwell (owner decision 2026-09-11): NEW uploads write
    to the CENTRAL durable ingest plane at ``/umrs/knowledge/upload``;
    the endpoint-corpus proxy stays available only behind the explicit
    ``UMRS_RAG_UPLOAD_FACADE=endpoint`` opt-in (endpoint corpus is read-only
    for standing data). There is no silent dual-write.
    """
    _user_id, project_id = _require_rag_scope(request, write=True)

    if not file.filename:
        raise_typed(PlatformError.rag("bad_request", "Filename required"))

    # Dual-corpus dwell: the central ingest plane runs its OWN
    # governance/mutation gates (single audit record).
    import os as _os

    facade = _os.environ.get("UMRS_RAG_UPLOAD_FACADE", "central").lower()
    if facade != "endpoint":
        return await _upload_via_central(request, file, project_id)
    return await _upload_via_endpoint(request, file, registry, _user_id, project_id)


async def _upload_via_central(
    request: Request, file: UploadFile, project_id: str
) -> UploadResponse:
    """Dual-corpus dwell default: central ingest owns uploads (own gates)."""
    from .knowledge_upload import knowledge_upload

    result = await knowledge_upload(
        request=request, file=file, graph_id=None, project_id=project_id
    )
    return UploadResponse(
        document_id=result.get("job_id", ""),
        filename=file.filename or "",
        status="queued-central",
        chunks=0,
        knowledge_nodes_minted=0,
    )


async def _upload_via_endpoint(
    request: Request,
    file: UploadFile,
    registry: EndpointRegistry,
    user_id: str,
    project_id: str,
) -> UploadResponse:
    """Explicit opt-in legacy endpoint proxy (pre-dwell behaviour intact)."""
    try:
        from .governance import guard_ingest

        guard_ingest(
            project_id,
            "rag.upload",
            user_id=user_id,
            details={"filename": getattr(file, "filename", "")},
        )
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        logger.debug("governance not active for rag upload: %s", exc)

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
                user_id=user_id,
                project_id=project_id,
            )

    except httpx.RequestError as e:
        logger.exception("Upload request failed: %s", e)
        raise_typed(
            PlatformError.rag("unavailable", "RAG service unavailable"), cause=e
        )


def _documents_from(payload: Any) -> list[dict[str, Any]]:
    """The rag service answers GET /rag/documents with a bare list
    (``response_model=list[GatewayDocumentInfo]``); an older envelope
    ``{"documents": [...]}`` is still accepted. Anything else lists nothing
    (ops-watch #470, 2026-10-16: the proxy read ``.get`` from the list)."""
    if isinstance(payload, dict):
        payload = payload.get("documents", [])
    if not isinstance(payload, list):
        return []
    return [doc for doc in payload if isinstance(doc, dict)]


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
    project_id: str | None = Query(
        None, description="Corpus to list (default: the caller's)"
    ),
):
    """List all documents in a project.

    ``project_id`` was not a parameter, so the r4 client's project (``r4 rag
    docs <project>``, ``/rag docs <project>``) was dropped and the caller's own
    corpus listed under it (TUI audit 2026-10-17). It now names the corpus,
    checked like ``/query``: read tier, then the ADR-080 access list.
    """
    _user_id, caller_project = _require_rag_scope(request, project_id, write=False)
    # The caller's endpoint serves the call; the parameter names the corpus.
    rag_base = _discover_rag_endpoint(registry, caller_project)
    project_id = project_id or caller_project
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

            documents = _documents_from(response.json())
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
        raise_typed(
            PlatformError.rag("unavailable", "RAG service unavailable"), cause=e
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, request: Request) -> dict[str, Any]:
    """Delete a document from the project (ADR-093 MD-16, decided 2026-10-11).

    The id resolves to the knowledge artifact it names — the artifact id,
    the ``artifact-rag-<id>`` rag.upload minted, or the central upload's
    ingest job — and the artifact delete runs on it (one transaction; see
    ``knowledge_artifact_delete``). An id that resolves to no artifact of the
    project is 404, and nothing is recorded: never a "deleted" for a document
    that stays retrievable. Agents never delete.
    """
    from ._knowledge_mutation_guard import enforce_human_delete_authorization
    from .knowledge_artifact_delete import delete_rag_document

    user_id, project_id = _require_rag_scope(request, write=True)
    enforce_human_delete_authorization(request, operation="rag.document_delete")
    return await delete_rag_document(request, document_id, project_id, user_id)


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
                raise_typed(PlatformError.rag("not_found", "Document not found"))

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
        raise_typed(
            PlatformError.rag("unavailable", "RAG service unavailable"), cause=e
        )


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
        raise_typed(
            PlatformError.rag("unavailable", "RAG service unavailable"), cause=e
        )


# ADR-093 D-2: the retrieve verb and its fetch handle, mounted under this
# router's prefix (/umrs/rag/retrieve, /umrs/rag/passages/{node_id}).
from .rag_retrieve import register as _register_retrieve

_register_retrieve(router)
