"""KnowledgeEvent — pub/sub event with persistence hooks.

CRUD for knowledge ingestion lifecycle events (parse, extract, embed, index).
No umrs.api or umrs.server imports — strict boundary.

Usage:
    # Register consumers
    from umrs.knowledge.events import KnowledgeEvent

    async def my_consumer(event: KnowledgeEvent) -> None:
        ...

    KnowledgeEvent.register_consumer(my_consumer)

    # Emit
    event = KnowledgeEvent(
        event_type="indexed",
        job_id="job-123",
        attempt_id="attempt-456",
        artifact_id="art-789",
        source_ref="s3://bucket/file.pdf",
        details={"document_count": 5},
    )
    await event.emit()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Sentinel for "no timestamp" so callers can override if needed
_UNSET: datetime = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Allowed event_type values — keep in sync with SSE/audit consumers
_VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "parsed",
    "extracted",
    "embedded",
    "indexed",
    "archived",
    "failed",
    "resumed",
})

# Consumer callback type
_ConsumerCallback = Callable[[KnowledgeEvent], Coroutine[Any, Any, None]]


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class _DuplicateConsumerError(Exception):
    """Raised when trying to register the same consumer twice."""


class _UnknownConsumerError(Exception):
    """Raised when trying to unregister a consumer that was never registered."""


@dataclass(frozen=True, slots=True)
class KnowledgeEvent:
    """Knowledge ingestion lifecycle event.

    Immutable snapshot of a single knowledge event. Consumers receive the
    same instance — do not mutate it.

    Fields:
        event_type: One of parsed/extracted/embedded/indexed/archived/failed/resumed
        job_id: UMRS durable job id that produced this event
        attempt_id: Specific attempt within the job
        artifact_id: Knowledge artifact id (content-hash or UUID)
        source_ref: Original source (s3://, file://, git://, http://)
        details: Arbitrary key-value metadata (no secrets)
        timestamp: Event wall-clock UTC
        event_id: Unique event identifier (auto-generated)
    """

    event_type: str
    job_id: str
    attempt_id: str
    artifact_id: str
    source_ref: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        """Validate event_type against allowlist."""
        if self.event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {self.event_type!r}. "
                f"Allowed: {sorted(_VALID_EVENT_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for DB storage (knowledge_events table).

        Keys match the migration 0056 column names.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "artifact_id": self.artifact_id,
            "source_ref": self.source_ref,
            "details": self.details,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEvent:
        """Reconstruct from a dict (e.g. read from DB)."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            job_id=data["job_id"],
            attempt_id=data["attempt_id"],
            artifact_id=data["artifact_id"],
            source_ref=data["source_ref"],
            details=data.get("details", {}),
            timestamp=ts or _utcnow(),
        )

    def to_json(self) -> dict[str, Any]:
        """Flat dict safe for SSE payloads (no secrets).

        Timestamps are ISO-8601 strings. 'details' keys are sanitized:
        string/int/float/bool values pass through; containers are dropped
        to keep SSE payloads small and predictable.
        """
        sanitized: dict[str, Any] = {}
        for k, v in self.details.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            # Drop lists/dicts/None — keep payload flat
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "artifact_id": self.artifact_id,
            "source_ref": self.source_ref,
            "details": sanitized,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> KnowledgeEvent:
        """Reconstruct from an SSE-compatible JSON dict."""
        ts_str = data.get("timestamp")
        ts = datetime.fromisoformat(ts_str) if ts_str else _utcnow()
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            job_id=data["job_id"],
            attempt_id=data["attempt_id"],
            artifact_id=data["artifact_id"],
            source_ref=data["source_ref"],
            details=data.get("details", {}),
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Pub/sub registry (static)
    # ------------------------------------------------------------------

    _consumers: list[_ConsumerCallback] = field(default=list, repr=False, init=False, hash=False)

    @classmethod
    def register_consumer(cls, callback: _ConsumerCallback) -> None:
        """Register a consumer (async callable receiving KnowledgeEvent).

        Raises _DuplicateConsumerError if already registered.
        """
        if callback in cls._consumers:
            raise _DuplicateConsumerError(
                f"Consumer {callback.__qualname__} already registered"
            )
        cls._consumers.append(callback)
        logger.debug("Registered consumer: %s", callback.__qualname__)

    @classmethod
    def unregister_consumer(cls, callback: _ConsumerCallback) -> None:
        """Unregister a previously registered consumer.

        Raises _UnknownConsumerError if never registered.
        """
        if callback not in cls._consumers:
            raise _UnknownConsumerError(
                f"Consumer {callback.__qualname__} not registered"
            )
        cls._consumers.remove(callback)
        logger.debug("Unregistered consumer: %s", callback.__qualname__)

    @classmethod
    def clear_consumers(cls) -> int:
        """Remove all consumers. Returns count removed."""
        n = len(cls._consumers)
        cls._consumers.clear()
        logger.debug("Cleared %d consumers", n)
        return n

    @classmethod
    def consumer_count(cls) -> int:
        """Return number of registered consumers."""
        return len(cls._consumers)

    # ------------------------------------------------------------------
    # Broadcast + audit
    # ------------------------------------------------------------------

    async def emit(self) -> None:
        """Iterate registered consumers, call each in order, log to audit.

        Consumers run sequentially. A consumer failure is logged and
        does NOT stop subsequent consumers.

        Audit log entry: [timestamp] event_type=<type> job_id=<id>
        """
        ts = self.timestamp
        logger.info(
            "KnowledgeEvent emit: type=%s job_id=%s artifact_id=%s consumers=%d",
            self.event_type,
            self.job_id,
            self.artifact_id,
            len(self._consumers),
        )

        failures = 0
        for cb in self._consumers:
            try:
                await cb(self)
            except Exception:
                failures += 1
                logger.exception(
                    "KnowledgeEvent consumer %s failed for %s",
                    cb.__qualname__,
                    self.event_id,
                )

        if failures:
            logger.warning(
                "KnowledgeEvent emit completed with %d consumer failure(s)",
                failures,
            )
