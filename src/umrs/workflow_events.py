"""Workflow events — event-driven automation for the scientific workflow.

M3 Phase 1: Persistent Scientific Model.

Instead of a procedural workflow, this module introduces events that
subsystems can react to. This makes autonomous research much easier
because the system can react to changes in knowledge state automatically.

Events:
  ArtifactUploaded     → triggers ingestion
  KnowledgeExtracted   → triggers embedding
  EmbeddingCompleted   → triggers graph update
  GraphUpdated         → triggers verification
  VerificationCompleted → triggers planning
  PlannerStarted       → triggers experiment
  ExperimentFinished   → triggers finding registration
  FindingRegistered    → triggers knowledge update
  KnowledgeUpdated     → triggers planner re-evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

__all__ = [
    "WorkflowEvent",
    "WorkflowEventData",
    "EventHandler",
    "WorkflowEventDispatcher",
]


class WorkflowEvent(str, Enum):
    """Types of workflow events."""

    ARTIFACT_UPLOADED = "artifact_uploaded"
    KNOWLEDGE_EXTRACTED = "knowledge_extracted"
    EMBEDDING_COMPLETED = "embedding_completed"
    GRAPH_UPDATED = "graph_updated"
    VERIFICATION_COMPLETED = "verification_completed"
    PLANNER_STARTED = "planner_started"
    EXPERIMENT_FINISHED = "experiment_finished"
    FINDING_REGISTERED = "finding_registered"
    KNOWLEDGE_UPDATED = "knowledge_updated"


@dataclass
class WorkflowEventData:
    """Data associated with a workflow event.

    Attributes:
        event_id: Unique identifier for this event
        event_type: The type of event
        project_id: The project this event belongs to
        object_id: The object that triggered this event (if applicable)
        timestamp: When the event was created
        data: Additional event-specific data
    """

    event_id: str = ""
    event_type: WorkflowEvent = WorkflowEvent.ARTIFACT_UPLOADED
    project_id: str = ""
    object_id: str = ""
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = f"evt-{uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "project_id": self.project_id,
            "object_id": self.object_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


EventHandler = Callable[[WorkflowEventData], None]


class WorkflowEventDispatcher:
    """Dispatcher for workflow events.

    Subsystems register handlers for specific event types. When an
    event is dispatched, all registered handlers for that event type
    are called.

    Usage:
        dispatcher = WorkflowEventDispatcher()

        @dispatcher.on(WorkflowEvent.ARTIFACT_UPLOADED)
        def handle_upload(event):
            print(f"Artifact uploaded: {event.object_id}")

        dispatcher.dispatch(WorkflowEventData(
            event_type=WorkflowEvent.ARTIFACT_UPLOADED,
            object_id="paper-001",
        ))
    """

    def __init__(self) -> None:
        self._handlers: dict[WorkflowEvent, list[EventHandler]] = {}
        self._event_log: list[WorkflowEventData] = []

    def on(
        self,
        event_type: WorkflowEvent,
    ) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register a handler for an event type.

        Usage:
            @dispatcher.on(WorkflowEvent.ARTIFACT_UPLOADED)
            def handler(event):
                ...
        """

        def decorator(handler: EventHandler) -> EventHandler:
            self.register(event_type, handler)
            return handler

        return decorator

    def register(
        self,
        event_type: WorkflowEvent,
        handler: EventHandler,
    ) -> None:
        """Register a handler for an event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def unregister(
        self,
        event_type: WorkflowEvent,
        handler: EventHandler,
    ) -> None:
        """Unregister a handler for an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def dispatch(self, event: WorkflowEventData) -> None:
        """Dispatch an event to all registered handlers.

        If a handler raises an exception, the exception is logged but
        does not prevent other handlers from being called.
        """
        self._event_log.append(event)
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass  # Log in production

    def get_event_log(
        self,
        *,
        event_type: WorkflowEvent | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[WorkflowEventData]:
        """Get the event log with optional filters.

        Args:
            event_type: Filter by event type
            project_id: Filter by project ID
            limit: Maximum number of events to return

        Returns:
            List of matching events, newest first
        """
        results = []
        for event in reversed(self._event_log):
            if event_type and event.event_type != event_type:
                continue
            if project_id and event.project_id != project_id:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results

    def clear(self) -> None:
        """Clear all handlers and event log."""
        self._handlers.clear()
        self._event_log.clear()
