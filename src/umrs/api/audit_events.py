"""Structured audit event emitter for UMRS API operations.

Emits JSON-structured log lines for security-relevant operations:
- Endpoint creation/destruction
- MCP server registration/deregistration
- RAG document uploads/deletions
- Privilege escalation attempts / auth failures
- Model staging / download requests

Events are written to the standard Python logger in structured JSON format
so they can be ingested by any log aggregator (journald, fluentd, ELK, etc).

Usage:
    from .audit_events import emit_audit

    emit_audit("endpoint.created", user_id="alice", project_id="proj-1",
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
_AUDIT_LEVEL = getattr(logging, os.getenv("UMRS_AUDIT_LEVEL", "INFO").upper(), logging.INFO)


def emit_audit(
    event: str,
    *,
    user_id: str = "",
    project_id: str = "",
    details: dict[str, Any] | None = None,
    outcome: str = "success",
) -> None:
    """Emit a structured audit event.

    Parameters
    ----------
    event : str
        Dot-separated event name (e.g. "endpoint.created", "mcp.registered").
    user_id : str
        The authenticated user performing the action.
    project_id : str
        The project context (if applicable).
    details : dict
        Arbitrary event-specific payload.
    outcome : str
        "success", "denied", "error".
    """
    record = {
        "ts": time.time(),
        "event_id": str(uuid.uuid4()),
        "event": event,
        "user_id": user_id,
        "project_id": project_id,
        "outcome": outcome,
        "details": details or {},
    }

    logger.log(_AUDIT_LEVEL, json.dumps(record, default=str))
