"""Tests for the workflow events module (M3 Phase 1)."""

from __future__ import annotations

from umrs.workflow_events import (
    WorkflowEvent,
    WorkflowEventData,
    WorkflowEventDispatcher,
)


def test_workflow_event_enum():
    """Test WorkflowEvent enum values."""
    assert WorkflowEvent.ARTIFACT_UPLOADED == "artifact_uploaded"
    assert WorkflowEvent.KNOWLEDGE_EXTRACTED == "knowledge_extracted"
    assert WorkflowEvent.EMBEDDING_COMPLETED == "embedding_completed"
    assert WorkflowEvent.GRAPH_UPDATED == "graph_updated"
    assert WorkflowEvent.VERIFICATION_COMPLETED == "verification_completed"
    assert WorkflowEvent.PLANNER_STARTED == "planner_started"
    assert WorkflowEvent.EXPERIMENT_FINISHED == "experiment_finished"
    assert WorkflowEvent.FINDING_REGISTERED == "finding_registered"
    assert WorkflowEvent.KNOWLEDGE_UPDATED == "knowledge_updated"


def test_event_data_defaults():
    """Test WorkflowEventData defaults."""
    event = WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
        object_id="obj-001",
    )

    assert event.event_id != ""  # Auto-generated
    assert event.event_type == WorkflowEvent.ARTIFACT_UPLOADED
    assert event.project_id == "proj-001"
    assert event.object_id == "obj-001"
    assert event.timestamp != ""  # Auto-generated
    assert event.data == {}


def test_event_data_to_dict():
    """Test WorkflowEventData to_dict."""
    event = WorkflowEventData(
        event_type=WorkflowEvent.GRAPH_UPDATED,
        project_id="proj-001",
        object_id="obj-001",
        data={"key": "value"},
    )

    data = event.to_dict()

    assert data["event_type"] == "graph_updated"
    assert data["project_id"] == "proj-001"
    assert data["object_id"] == "obj-001"
    assert data["data"] == {"key": "value"}


def test_dispatcher_init():
    """Test dispatcher initialization."""
    dispatcher = WorkflowEventDispatcher()
    assert dispatcher is not None


def test_dispatcher_register_and_dispatch():
    """Test registering a handler and dispatching an event."""
    dispatcher = WorkflowEventDispatcher()

    received_events = []

    def handler(event: WorkflowEventData) -> None:
        received_events.append(event)

    dispatcher.register(WorkflowEvent.ARTIFACT_UPLOADED, handler)

    event = WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
        object_id="obj-001",
    )
    dispatcher.dispatch(event)

    assert len(received_events) == 1
    assert received_events[0].object_id == "obj-001"


def test_dispatcher_decorator():
    """Test the decorator-based handler registration."""
    dispatcher = WorkflowEventDispatcher()

    received = []

    @dispatcher.on(WorkflowEvent.KNOWLEDGE_EXTRACTED)
    def handler(event: WorkflowEventData) -> None:
        received.append(event)

    event = WorkflowEventData(
        event_type=WorkflowEvent.KNOWLEDGE_EXTRACTED,
        project_id="proj-001",
        object_id="obj-001",
    )
    dispatcher.dispatch(event)

    assert len(received) == 1


def test_dispatcher_multiple_handlers():
    """Test that multiple handlers for the same event are all called."""
    dispatcher = WorkflowEventDispatcher()

    calls = []

    @dispatcher.on(WorkflowEvent.GRAPH_UPDATED)
    def handler1(event: WorkflowEventData) -> None:
        calls.append("handler1")

    @dispatcher.on(WorkflowEvent.GRAPH_UPDATED)
    def handler2(event: WorkflowEventData) -> None:
        calls.append("handler2")

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.GRAPH_UPDATED,
        project_id="proj-001",
    ))

    assert calls == ["handler1", "handler2"]


def test_dispatcher_unregister():
    """Test unregistering a handler."""
    dispatcher = WorkflowEventDispatcher()

    received = []

    def handler(event: WorkflowEventData) -> None:
        received.append(event)

    dispatcher.register(WorkflowEvent.ARTIFACT_UPLOADED, handler)
    dispatcher.unregister(WorkflowEvent.ARTIFACT_UPLOADED, handler)

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
    ))

    assert len(received) == 0


def test_dispatcher_event_log():
    """Test that dispatched events are logged."""
    dispatcher = WorkflowEventDispatcher()

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
        object_id="obj-001",
    ))
    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.GRAPH_UPDATED,
        project_id="proj-001",
        object_id="obj-002",
    ))

    log = dispatcher.get_event_log()
    assert len(log) == 2


def test_dispatcher_event_log_filter():
    """Test filtering the event log."""
    dispatcher = WorkflowEventDispatcher()

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
    ))
    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.GRAPH_UPDATED,
        project_id="proj-001",
    ))
    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-002",
    ))

    # Filter by event type
    log = dispatcher.get_event_log(event_type=WorkflowEvent.ARTIFACT_UPLOADED)
    assert len(log) == 2

    # Filter by project
    log = dispatcher.get_event_log(project_id="proj-001")
    assert len(log) == 2


def test_dispatcher_clear():
    """Test clearing handlers and event log."""
    dispatcher = WorkflowEventDispatcher()

    @dispatcher.on(WorkflowEvent.ARTIFACT_UPLOADED)
    def handler(event: WorkflowEventData) -> None:
        pass

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
    ))

    dispatcher.clear()

    log = dispatcher.get_event_log()
    assert len(log) == 0


def test_dispatcher_handler_exception_does_not_crash():
    """Test that a handler exception does not crash the dispatcher."""
    dispatcher = WorkflowEventDispatcher()

    received = []

    @dispatcher.on(WorkflowEvent.ARTIFACT_UPLOADED)
    def good_handler(event: WorkflowEventData) -> None:
        received.append(event)

    @dispatcher.on(WorkflowEvent.ARTIFACT_UPLOADED)
    def bad_handler(event: WorkflowEventData) -> None:
        raise RuntimeError("Handler error")

    dispatcher.dispatch(WorkflowEventData(
        event_type=WorkflowEvent.ARTIFACT_UPLOADED,
        project_id="proj-001",
    ))

    # The good handler should still have been called
    assert len(received) == 1
