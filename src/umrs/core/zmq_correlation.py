"""ZMQ correlation ID generation and extraction utilities.

Provides a consistent way to generate and extract correlation IDs in ZMQ
control-plane messages. This bridges the gap between HTTP (which uses
X-Request-ID via CorrelationIDMiddleware) and ZMQ (which previously had
no correlation mechanism).

The correlation ID is stored as a JSON key `"correlation_id"` inside the
control message payload. Old servers ignore the unknown key; new servers
extract it and log it with every command.

Usage:
    from umrs.core.zmq_correlation import (
        ZmqCorrelation,
    )

    client: ZmqCorrelation.create_payload({"cmd": "CHECK", "data": {...}})
    # Returns: {"cmd": "CHECK", "data": {...}, "correlation_id": "a1b2c3d4e5f6"}

    server: cid = ZmqCorrelation.extract_correlation_id(payload)
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Maximum length for a stored correlation ID (hex chars from uuid4).
_MAX_CID_LENGTH = 64


def generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns a UUID4 hex string truncated to 16 characters (64-bit entropy).
    This is compact enough to fit in ZMQ frame overhead while still being
    sufficiently random to avoid collisions across concurrent requests.
    """
    return uuid.uuid4().hex[:16]


def create_correlation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add a correlation ID to a payload dict.

    If the payload already has a correlation_id, it is preserved.
    Otherwise, a new one is generated.

    Args:
        payload: The original payload dict.

    Returns:
        A new dict with the correlation_id key added.
    """
    result = dict(payload)  # shallow copy
    if "correlation_id" not in result:
        result["correlation_id"] = generate_correlation_id()
    return result


def dumps_with_correlation_id(payload: dict[str, Any]) -> str:
    """Serialize a dict to JSON with a correlation_id added.

    This is the recommended helper for client-side ZMQ JSON payloads.
    It wraps create_correlation_payload and json.dumps in one call.

    Args:
        payload: The original payload dict.

    Returns:
        A JSON string with correlation_id included.
    """
    return json.dumps(create_correlation_payload(payload), ensure_ascii=False)


def extract_correlation_id(payload: dict[str, Any]) -> str | None:
    """Extract the correlation ID from a payload dict.

    Args:
        payload: The payload dict (parsed from JSON).

    Returns:
        The correlation_id string, or None if not present.
    """
    cid = payload.get("correlation_id")
    if cid and isinstance(cid, str) and len(cid) > _MAX_CID_LENGTH:
        # Truncate oversize values to prevent abuse.
        return cid[:_MAX_CID_LENGTH]
    return cid if cid and isinstance(cid, str) else None


def log_correlation_id(correlation_id: str, log_fn, *args: Any, **kwargs: Any) -> None:
    """Log a message that includes the correlation ID.

    This is a convenience wrapper to ensure the request_id is always
    set when logging from ZMQ handlers.

    Args:
        correlation_id: The correlation ID string.
        log_fn: The logging function (e.g., logger.info).
        *args, **kwargs: Passed to the log function.
    """
    log_fn(f"[cid={correlation_id}] " + " ".join(str(a) for a in args), **kwargs)
