"""
UMRS Context Storage API (Phase 3).

Provides durable server-side conversation, message, summary, and
memory persistence. The local r4 session store acts as a *cache*; this
module holds the authoritative conversation state.

Endpoints
---------
- POST /umrs/context/conversations      Create a conversation
- GET  /umrs/context/conversations      List user conversations
- GET  /umrs/context/conversations/{id} Get conversation with messages
- POST /umrs/context/conversations/{id}/messages  Append messages
- POST /umrs/context/pack               Assemble context window (summary + recent + memory)
- POST /umrs/context/memory             Store memory item
- POST /umrs/context/memory/search      Semantic memory search

Storage
-------
Backed by Postgres tables from ``0010_context_storage.sql``.
Falls back to in-memory buffer if ``UMRS_DB_URL`` is unset (dev mode).
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from .typed_errors import PlatformError, raise_typed


def _pg_kwargs() -> dict:
    """R-14 session hygiene kwargs for every per-call context connection."""
    from umrs.shared.pg_settings import pg_connect_kwargs

    return pg_connect_kwargs("context")


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/context", tags=["context-storage"])


# =============================================================================
# Models
# =============================================================================


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""

    user_id: str
    project_id: str | None = None
    title: str = ""
    model: str = ""
    system_prompt: str = ""


class ConversationResponse(BaseModel):
    """Conversation metadata."""

    conversation_id: str
    user_id: str
    project_id: str | None = None
    title: str
    model: str
    system_prompt: str
    rev: int
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    """A single message to append."""

    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str
    token_count: int = 0
    model: str | None = None
    tool_calls: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """A stored message."""

    message_id: str
    conversation_id: str
    role: str
    content: str
    token_count: int
    seq: int
    model: str | None = None
    tool_calls: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ContextPackRequest(BaseModel):
    """Request to assemble context for an LLM call."""

    conversation_id: str
    max_tokens: int = 4096
    include_summary: bool = True
    include_memory: bool = True
    include_rag: bool = False
    memory_query: str | None = None
    memory_limit: int = 5
    rag_query: str | None = None
    rag_namespace_id: str | None = None
    rag_limit: int = 5


class ContextPackResponse(BaseModel):
    """Assembled context payload."""

    conversation_id: str
    messages: list[MessageResponse]
    summary: str | None = None
    memory_items: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_context: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = 0
    token_breakdown: dict[str, int] = Field(default_factory=dict)


class MemoryItemCreate(BaseModel):
    """Request to store a memory item."""

    user_id: str
    project_id: str | None = None
    content: str
    source: str = "conversation"
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItemResponse(BaseModel):
    """A stored memory item."""

    memory_id: str
    user_id: str
    project_id: str | None = None
    content: str
    source: str
    importance: float
    access_count: int
    created_at: str


class MemorySearchRequest(BaseModel):
    """Request to search memory items."""

    user_id: str
    project_id: str | None = None
    query: str
    limit: int = 10


class MemorySearchResponse(BaseModel):
    """Memory search results."""

    items: list[MemoryItemResponse]
    count: int


# =============================================================================
# Backend (DB with in-memory fallback for dev)
# =============================================================================


class _ContextStore:
    """Context storage backend with optional Postgres persistence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._db_url = os.getenv("UMRS_DB_URL", "").strip() or None
        self._psycopg2 = None
        if self._db_url:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "UMRS_DB_URL set but psycopg2 unavailable; "
                    "context storage will use in-memory store only"
                )

        # In-memory fallback structures
        self._conversations: dict[str, dict] = {}
        self._messages: dict[str, list[dict]] = defaultdict(list)
        self._summaries: dict[str, list[dict]] = defaultdict(list)
        self._memory_items: list[dict] = []

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    # --- Conversations ---

    def create_conversation(self, req: ConversationCreate) -> ConversationResponse:
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if self.db_enabled:
            try:
                return self._db_create_conversation(conv_id, req, now)
            except Exception as exc:
                logger.warning("DB create_conversation failed: %s", exc)

        # In-memory fallback
        record = {
            "conversation_id": conv_id,
            "user_id": req.user_id,
            "project_id": req.project_id,
            "title": req.title,
            "model": req.model,
            "system_prompt": req.system_prompt,
            "rev": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conversations[conv_id] = record
        return ConversationResponse(**record)

    def get_conversation(self, conversation_id: str) -> ConversationResponse | None:
        if self.db_enabled:
            try:
                return self._db_get_conversation(conversation_id)
            except Exception as exc:
                logger.warning("DB get_conversation failed: %s", exc)

        with self._lock:
            record = self._conversations.get(conversation_id)
        if record is None:
            return None
        return ConversationResponse(**record)

    def list_conversations(
        self, user_id: str, limit: int = 50
    ) -> list[ConversationResponse]:
        if self.db_enabled:
            try:
                return self._db_list_conversations(user_id, limit)
            except Exception as exc:
                logger.warning("DB list_conversations failed: %s", exc)

        with self._lock:
            items = [v for v in self._conversations.values() if v["user_id"] == user_id]
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return [ConversationResponse(**c) for c in items[:limit]]

    # --- Messages ---

    def append_messages(
        self, conversation_id: str, messages: list[MessageCreate]
    ) -> list[MessageResponse]:
        if self.db_enabled:
            try:
                return self._db_append_messages(conversation_id, messages)
            except Exception as exc:
                logger.warning("DB append_messages failed: %s", exc)

        now = datetime.now(timezone.utc).isoformat()
        results: list[MessageResponse] = []
        with self._lock:
            existing = self._messages[conversation_id]
            seq_start = len(existing)
            for i, msg in enumerate(messages):
                record = {
                    "message_id": str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "token_count": msg.token_count,
                    "seq": seq_start + i,
                    "model": msg.model,
                    "tool_calls": msg.tool_calls,
                    "metadata": msg.metadata,
                    "created_at": now,
                }
                existing.append(record)
                results.append(MessageResponse(**record))
            # Bump rev
            conv = self._conversations.get(conversation_id)
            if conv:
                conv["rev"] += 1
                conv["updated_at"] = now
        return results

    def get_messages(
        self, conversation_id: str, limit: int = 100, offset: int = 0
    ) -> list[MessageResponse]:
        if self.db_enabled:
            try:
                return self._db_get_messages(conversation_id, limit, offset)
            except Exception as exc:
                logger.warning("DB get_messages failed: %s", exc)

        with self._lock:
            msgs = self._messages.get(conversation_id, [])
        sliced = msgs[offset : offset + limit]
        return [MessageResponse(**m) for m in sliced]

    # --- Context Pack ---

    def pack_context(self, req: ContextPackRequest) -> ContextPackResponse:
        """Assemble context window: summary + recent messages + memory + RAG."""
        # Validate conversation exists
        conv = self.get_conversation(req.conversation_id)
        if conv is None:
            raise_typed(PlatformError.contextapi("not_found", "Conversation not found"))

        messages = self.get_messages(req.conversation_id, limit=req.max_tokens // 4)
        summary, summary_tokens = self._pack_summary(req)
        memory_items, memory_tokens = self._pack_memory(req, conv)
        retrieval_context, rag_tokens = self._pack_rag(req, conv)

        message_tokens = sum(m.token_count for m in messages)
        total_tokens = message_tokens + summary_tokens + memory_tokens + rag_tokens

        return ContextPackResponse(
            conversation_id=req.conversation_id,
            messages=messages,
            summary=summary,
            memory_items=memory_items,
            retrieval_context=retrieval_context,
            total_tokens=total_tokens,
            token_breakdown={
                "messages": message_tokens,
                "summary": summary_tokens,
                "memory": memory_tokens,
                "retrieval": rag_tokens,
            },
        )

    def _pack_summary(self, req: ContextPackRequest) -> tuple[str | None, int]:
        """Latest conversation summary + its token count (when requested)."""
        if not req.include_summary:
            return None, 0
        summaries = self._get_summaries(req.conversation_id)
        if not summaries:
            return None, 0
        return summaries[-1].get("content"), summaries[-1].get("token_count", 0)

    def _pack_memory(
        self, req: ContextPackRequest, conv: Any
    ) -> tuple[list[dict[str, Any]], int]:
        """Memory items for the pack (~4 chars per token estimate)."""
        if not (req.include_memory and req.memory_query):
            return [], 0
        mem_results = self.search_memory(
            MemorySearchRequest(
                user_id=conv.user_id,
                project_id=conv.project_id,
                query=req.memory_query,
                limit=req.memory_limit,
            )
        )
        items = [
            {"content": m.content, "importance": m.importance}
            for m in mem_results.items
        ]
        return items, sum(len(m["content"]) // 4 for m in items)

    def _pack_rag(
        self, req: ContextPackRequest, conv: Any
    ) -> tuple[list[dict[str, Any]], int]:
        """RAG retrieval: keyword search over memory items with a different query."""
        if not (req.include_rag and req.rag_query):
            return [], 0
        rag_results = self.search_memory(
            MemorySearchRequest(
                user_id=conv.user_id,
                project_id=req.rag_namespace_id or conv.project_id,
                query=req.rag_query,
                limit=req.rag_limit,
            )
        )
        items = [
            {"content": m.content, "source": m.source, "importance": m.importance}
            for m in rag_results.items
        ]
        return items, sum(len(r["content"]) // 4 for r in items)

    def _get_summaries(self, conversation_id: str) -> list[dict]:
        if self.db_enabled:
            try:
                return self._db_get_summaries(conversation_id)
            except Exception as exc:
                logger.warning("DB get_summaries failed: %s", exc)
        with self._lock:
            return list(self._summaries.get(conversation_id, []))

    # --- Memory ---

    def store_memory(self, req: MemoryItemCreate) -> MemoryItemResponse:
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if self.db_enabled:
            try:
                return self._db_store_memory(memory_id, req, now)
            except Exception as exc:
                logger.warning("DB store_memory failed: %s", exc)

        record = {
            "memory_id": memory_id,
            "user_id": req.user_id,
            "project_id": req.project_id,
            "content": req.content,
            "source": req.source,
            "importance": req.importance,
            "access_count": 0,
            "created_at": now,
        }
        with self._lock:
            self._memory_items.append(record)
        return MemoryItemResponse(**record)

    def search_memory(self, req: MemorySearchRequest) -> MemorySearchResponse:
        """Search memory items. Uses keyword matching in dev mode."""
        if self.db_enabled:
            try:
                return self._db_search_memory(req)
            except Exception as exc:
                logger.warning("DB search_memory failed: %s", exc)

        # In-memory keyword fallback
        query_lower = req.query.lower()
        with self._lock:
            candidates = [
                m
                for m in self._memory_items
                if m["user_id"] == req.user_id
                and (req.project_id is None or m.get("project_id") == req.project_id)
                and query_lower in m["content"].lower()
            ]
        candidates.sort(key=lambda x: x.get("importance", 0), reverse=True)
        items = [MemoryItemResponse(**m) for m in candidates[: req.limit]]
        return MemorySearchResponse(items=items, count=len(items))

    # ==========================================================================
    # Postgres implementations
    # ==========================================================================

    def _db_create_conversation(
        self, conv_id: str, req: ConversationCreate, now: str
    ) -> ConversationResponse:
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations
                        (conversation_id, user_id, project_id, title, model, system_prompt)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING conversation_id, user_id, project_id, title, model,
                              system_prompt, rev, created_at, updated_at
                    """,
                    (
                        conv_id,
                        req.user_id,
                        req.project_id,
                        req.title,
                        req.model,
                        req.system_prompt,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return ConversationResponse(
            conversation_id=str(row[0]),
            user_id=row[1],
            project_id=row[2],
            title=row[3],
            model=row[4],
            system_prompt=row[5],
            rev=row[6],
            created_at=row[7].isoformat()
            if hasattr(row[7], "isoformat")
            else str(row[7]),
            updated_at=row[8].isoformat()
            if hasattr(row[8], "isoformat")
            else str(row[8]),
        )

    def _db_get_conversation(self, conversation_id: str) -> ConversationResponse | None:
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT conversation_id, user_id, project_id, title, model,
                           system_prompt, rev, created_at, updated_at
                    FROM conversations WHERE conversation_id = %s
                    """,
                    (conversation_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return ConversationResponse(
            conversation_id=str(row[0]),
            user_id=row[1],
            project_id=row[2],
            title=row[3],
            model=row[4],
            system_prompt=row[5],
            rev=row[6],
            created_at=row[7].isoformat()
            if hasattr(row[7], "isoformat")
            else str(row[7]),
            updated_at=row[8].isoformat()
            if hasattr(row[8], "isoformat")
            else str(row[8]),
        )

    def _db_list_conversations(
        self, user_id: str, limit: int
    ) -> list[ConversationResponse]:
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT conversation_id, user_id, project_id, title, model,
                           system_prompt, rev, created_at, updated_at
                    FROM conversations WHERE user_id = %s
                    ORDER BY updated_at DESC LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            ConversationResponse(
                conversation_id=str(r[0]),
                user_id=r[1],
                project_id=r[2],
                title=r[3],
                model=r[4],
                system_prompt=r[5],
                rev=r[6],
                created_at=r[7].isoformat()
                if hasattr(r[7], "isoformat")
                else str(r[7]),
                updated_at=r[8].isoformat()
                if hasattr(r[8], "isoformat")
                else str(r[8]),
            )
            for r in rows
        ]

    def _db_append_messages(
        self, conversation_id: str, messages: list[MessageCreate]
    ) -> list[MessageResponse]:
        assert self._psycopg2 and self._db_url
        import json as _json

        results: list[MessageResponse] = []
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                # Get next seq number
                cur.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE conversation_id = %s",
                    (conversation_id,),
                )
                next_seq = cur.fetchone()[0]

                for i, msg in enumerate(messages):
                    msg_id = str(uuid.uuid4())
                    seq = next_seq + i
                    cur.execute(
                        """
                        INSERT INTO messages
                            (message_id, conversation_id, role, content, token_count,
                             seq, model, tool_calls, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        RETURNING message_id, conversation_id, role, content, token_count,
                                  seq, model, tool_calls, metadata, created_at
                        """,
                        (
                            msg_id,
                            conversation_id,
                            msg.role,
                            msg.content,
                            msg.token_count,
                            seq,
                            msg.model,
                            _json.dumps(msg.tool_calls) if msg.tool_calls else None,
                            _json.dumps(msg.metadata),
                        ),
                    )
                    row = cur.fetchone()
                    results.append(
                        MessageResponse(
                            message_id=str(row[0]),
                            conversation_id=str(row[1]),
                            role=row[2],
                            content=row[3],
                            token_count=row[4],
                            seq=row[5],
                            model=row[6],
                            tool_calls=row[7],
                            metadata=row[8] if row[8] else {},
                            created_at=row[9].isoformat()
                            if hasattr(row[9], "isoformat")
                            else str(row[9]),
                        )
                    )

                # Bump conversation rev
                cur.execute(
                    """
                    UPDATE conversations SET rev = rev + 1, updated_at = NOW()
                    WHERE conversation_id = %s
                    """,
                    (conversation_id,),
                )
            conn.commit()
        return results

    def _db_get_messages(
        self, conversation_id: str, limit: int, offset: int
    ) -> list[MessageResponse]:
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT message_id, conversation_id, role, content, token_count,
                           seq, model, tool_calls, metadata, created_at
                    FROM messages WHERE conversation_id = %s
                    ORDER BY seq LIMIT %s OFFSET %s
                    """,
                    (conversation_id, limit, offset),
                )
                rows = cur.fetchall()
        return [
            MessageResponse(
                message_id=str(r[0]),
                conversation_id=str(r[1]),
                role=r[2],
                content=r[3],
                token_count=r[4],
                seq=r[5],
                model=r[6],
                tool_calls=r[7],
                metadata=r[8] if r[8] else {},
                created_at=r[9].isoformat()
                if hasattr(r[9], "isoformat")
                else str(r[9]),
            )
            for r in rows
        ]

    def _db_get_summaries(self, conversation_id: str) -> list[dict]:
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary_id, content, token_count, messages_start, messages_end, created_at
                    FROM conversation_summaries WHERE conversation_id = %s
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (conversation_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "summary_id": str(r[0]),
                "content": r[1],
                "token_count": r[2],
                "messages_start": r[3],
                "messages_end": r[4],
                "created_at": r[5].isoformat()
                if hasattr(r[5], "isoformat")
                else str(r[5]),
            }
            for r in rows
        ]

    def _db_store_memory(
        self, memory_id: str, req: MemoryItemCreate, now: str
    ) -> MemoryItemResponse:
        assert self._psycopg2 and self._db_url
        import json as _json

        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_items
                        (memory_id, user_id, project_id, content, source, importance, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING memory_id, user_id, project_id, content, source,
                              importance, access_count, created_at
                    """,
                    (
                        memory_id,
                        req.user_id,
                        req.project_id,
                        req.content,
                        req.source,
                        req.importance,
                        _json.dumps(req.metadata),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return MemoryItemResponse(
            memory_id=str(row[0]),
            user_id=row[1],
            project_id=row[2],
            content=row[3],
            source=row[4],
            importance=row[5],
            access_count=row[6],
            created_at=row[7].isoformat()
            if hasattr(row[7], "isoformat")
            else str(row[7]),
        )

    def _db_search_memory(self, req: MemorySearchRequest) -> MemorySearchResponse:
        """Search memory using keyword matching (vector search requires embeddings)."""
        assert self._psycopg2 and self._db_url
        with self._psycopg2.connect(self._db_url, **_pg_kwargs()) as conn:
            with conn.cursor() as cur:
                # Use trigram/keyword matching; full vector search requires
                # embedding generation which is an async pipeline concern.
                cur.execute(
                    """
                    SELECT memory_id, user_id, project_id, content, source,
                           importance, access_count, created_at
                    FROM memory_items
                    WHERE user_id = %s
                      AND (%s IS NULL OR project_id = %s)
                      AND content ILIKE %s
                    ORDER BY importance DESC
                    LIMIT %s
                    """,
                    (
                        req.user_id,
                        req.project_id,
                        req.project_id,
                        f"%{req.query}%",
                        req.limit,
                    ),
                )
                rows = cur.fetchall()
        items = [
            MemoryItemResponse(
                memory_id=str(r[0]),
                user_id=r[1],
                project_id=r[2],
                content=r[3],
                source=r[4],
                importance=r[5],
                access_count=r[6],
                created_at=r[7].isoformat()
                if hasattr(r[7], "isoformat")
                else str(r[7]),
            )
            for r in rows
        ]
        return MemorySearchResponse(items=items, count=len(items))


# Global store instance
_context_store: _ContextStore | None = None


def get_context_store() -> _ContextStore:
    global _context_store
    if _context_store is None:
        _context_store = _ContextStore()
    return _context_store


# =============================================================================
# API Routes
# =============================================================================


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(req: ConversationCreate) -> ConversationResponse:
    """Create a new conversation."""
    return get_context_store().create_conversation(req)


@router.get("/conversations")
def list_conversations(
    user_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
) -> list[ConversationResponse]:
    """List conversations for a user."""
    return get_context_store().list_conversations(user_id, limit)


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> ConversationResponse:
    """Get a single conversation."""
    conv = get_context_store().get_conversation(conversation_id)
    if conv is None:
        raise_typed(PlatformError.contextapi("not_found", "Conversation not found"))
    return conv


@router.post(
    "/conversations/{conversation_id}/messages", status_code=status.HTTP_201_CREATED
)
def append_messages(
    conversation_id: str, messages: list[MessageCreate]
) -> list[MessageResponse]:
    """Append messages to a conversation."""
    conv = get_context_store().get_conversation(conversation_id)
    if conv is None:
        raise_typed(PlatformError.contextapi("not_found", "Conversation not found"))
    return get_context_store().append_messages(conversation_id, messages)


@router.post("/pack")
def pack_context(req: ContextPackRequest) -> ContextPackResponse:
    """Assemble context window for an LLM call.

    Combines conversation summary, recent messages, memory items,
    and optional RAG retrieval into a single payload with token breakdown.
    """
    return get_context_store().pack_context(req)


@router.post("/memory", status_code=status.HTTP_201_CREATED)
def store_memory(req: MemoryItemCreate) -> MemoryItemResponse:
    """Store a memory item."""
    return get_context_store().store_memory(req)


@router.post("/memory/search")
def search_memory(req: MemorySearchRequest) -> MemorySearchResponse:
    """Search memory items."""
    return get_context_store().search_memory(req)
