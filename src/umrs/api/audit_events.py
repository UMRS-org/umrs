"""Structured audit event emitter for UMRS API operations.

Emits JSON-structured log lines for security-relevant operations:
- Endpoint creation/destruction
- MCP server registration/deregistration
- RAG document uploads/deletions
- Privilege escalation attempts / auth failures
- Model staging / download requests

Events are written to the standard Python logger in structured JSON format
so they can be ingested by any log aggregator (journald, fluentd, ELK, etc).

Schema (v1):
    event_id     — unique UUID4 for this event
    ts           — unix timestamp (float, seconds)
    event        — dot-separated event name (e.g. "endpoint.created")
    actor        — user who triggered the action
    project_id   — project context
    request_id   — correlation ID from gateway/client (for distributed tracing)
    outcome      — "success" | "denied" | "error"
    scope        — "system" | "project" | "user"
    details      — arbitrary event-specific payload

Usage:
    from .audit_events import emit_audit

    emit_audit("endpoint.created", actor="alice", project_id="proj-1",
               details={"model": "qwen-7b", "tier": "user"})
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("umrs.audit")

# Audit log level: uses INFO by default. Set UMRS_AUDIT_LEVEL=DEBUG to lower.
_AUDIT_LEVEL = getattr(
    logging, os.getenv("UMRS_AUDIT_LEVEL", "INFO").upper(), logging.INFO
)

# Schema version for forward compatibility.
AUDIT_SCHEMA_VERSION = "1"

# In-memory ring buffer of recent audit events for trace inspection.
# Size configurable via UMRS_AUDIT_BUFFER_SIZE (default 1000).
_AUDIT_BUFFER_SIZE = int(os.getenv("UMRS_AUDIT_BUFFER_SIZE", "1000"))
_audit_buffer: list[dict[str, Any]] = []
_audit_buffer_lock = __import__("threading").Lock()

# Optional persistent store — set via set_audit_store() during sidecar init.
_audit_store: Any = None


def set_audit_store(store) -> None:
    """Register a persistent AuditStore for write-through."""
    global _audit_store
    _audit_store = store


def _get_correlation_id() -> str:
    """Retrieve correlation ID from request context (best-effort)."""
    try:
        from umrs.server.correlation import get_request_id

        return get_request_id()
    except Exception:
        return ""


def emit_audit(
    event: str,
    *,
    actor: str = "",
    user_id: str = "",
    project_id: str = "",
    details: dict[str, Any] | None = None,
    outcome: str = "success",
    scope: str = "",
    request_id: str = "",
) -> None:
    """Emit a structured audit event.

    Parameters
    ----------
    event : str
        Dot-separated event name (e.g. "endpoint.created", "mcp.registered").
    actor : str
        The authenticated user performing the action. Falls back to user_id.
    user_id : str
        Deprecated alias for actor (kept for backward compatibility).
    project_id : str
        The project context (if applicable).
    details : dict
        Arbitrary event-specific payload.
    outcome : str
        "success", "denied", "error".
    scope : str
        "system", "project", or "user". Inferred from context if empty.
    request_id : str
        Explicit correlation ID override. Auto-detected from middleware if empty.
    """
    effective_actor = actor or user_id
    effective_request_id = request_id or _get_correlation_id()

    record = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "ts": time.time(),
        "event_id": str(uuid.uuid4()),
        "event": event,
        "actor": effective_actor,
        "project_id": project_id,
        "request_id": effective_request_id,
        "outcome": outcome,
        "scope": scope,
        "details": details or {},
    }

    logger.log(_AUDIT_LEVEL, json.dumps(record, default=str))

    # Store in ring buffer for fast recent queries
    with _audit_buffer_lock:
        _audit_buffer.append(record)
        if len(_audit_buffer) > _AUDIT_BUFFER_SIZE:
            del _audit_buffer[: len(_audit_buffer) - _AUDIT_BUFFER_SIZE]

    # Write through to persistent store if available
    if _audit_store is not None:
        try:
            _audit_store.write(record)
        except Exception:
            logger.debug("Audit persistent write failed", exc_info=True)


def query_audit_events(
    *,
    request_id: str = "",
    event: str = "",
    actor: str = "",
    since: float = 0.0,
    until: float = 0.0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query audit events, preferring persistent store if available.

    Returns matching audit events, most recent first.
    Falls back to in-memory ring buffer when no persistent store is configured.
    """
    # Try persistent store first
    if _audit_store is not None:
        try:
            return _audit_store.query(
                request_id=request_id,
                event=event,
                actor=actor,
                since=since,
                until=until,
                limit=limit,
            )
        except Exception:
            logger.debug("Persistent audit query failed, falling back to buffer")

    # Fall back to in-memory buffer
    with _audit_buffer_lock:
        results = list(_audit_buffer)

    if request_id:
        results = [r for r in results if r.get("request_id") == request_id]
    if event:
        results = [r for r in results if event in r.get("event", "")]
    if actor:
        results = [r for r in results if r.get("actor") == actor]
    if since > 0:
        results = [r for r in results if r.get("ts", 0) >= since]
    if until > 0:
        results = [r for r in results if r.get("ts", 0) <= until]

    results.reverse()
    return results[:limit]
