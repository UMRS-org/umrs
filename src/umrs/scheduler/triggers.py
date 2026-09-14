"""UMRS webhook→scheduler wiring — inbound trigger surface.

Allows webhook events to trigger scheduled actions, and allows
the scheduler to subscribe to webhook events as triggers.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WebhookTrigger:
    """A trigger that fires a scheduled job when a webhook event matches."""

    trigger_id: str = ""
    name: str = ""
    event_type: str = ""  # e.g. "job.completed", "pipeline.completed"
    event_filter: dict[str, Any] = field(default_factory=dict)  # Match payload fields
    schedule_id: str = ""  # Target schedule to trigger
    job_type: str = ""  # Job type to submit
    job_config: dict[str, Any] = field(default_factory=dict)
    owner_id: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.trigger_id and self.name:
            self.trigger_id = hashlib.sha256(
                f"{self.owner_id}:{self.name}".encode()
            ).hexdigest()[:16]


class TriggerMatcher:
    """Evaluates whether a webhook event matches a trigger's filter."""

    @staticmethod
    def matches(
        trigger: WebhookTrigger, event_type: str, payload: dict[str, Any]
    ) -> bool:
        """Check if an event matches a trigger's criteria."""
        if trigger.event_type and trigger.event_type != event_type:
            return False

        # Check filter: all specified fields must match
        for key, expected in trigger.event_filter.items():
            actual = payload.get(key)
            if actual != expected:
                return False

        return True


@dataclass
class TriggerResult:
    """Result of evaluating a trigger."""

    trigger: WebhookTrigger
    matched: bool
    job_submitted: bool = False
    error: str | None = None


class WebhookSchedulerBridge:
    """Wires webhook events to the scheduler as inbound triggers.

    Usage::

        bridge = WebhookSchedulerBridge()

        # Register a trigger
        trigger = WebhookTrigger(
            name="sync-on-upload",
            event_type="upload.completed",
            event_filter={"project_id": "proj-123"},
            job_type="connector_sync",
            job_config={"connector": "filesystem", "path": "/data"},
            owner_id="alice",
        )
        bridge.register(trigger)

        # When a webhook event arrives:
        results = bridge.evaluate(event_type, payload)

    The bridge does NOT submit jobs directly. It returns
    ``TriggerResult`` objects that the caller (UMRS event loop)
    uses to submit durable jobs through the normal job system.
    """

    def __init__(self) -> None:
        self._triggers: dict[str, WebhookTrigger] = {}
        self._on_trigger: Callable[[WebhookTrigger, dict[str, Any]], None] | None = None

    def register(self, trigger: WebhookTrigger) -> None:
        """Register a webhook trigger."""
        self._triggers[trigger.trigger_id] = trigger
        logger.info(
            "Registered webhook trigger: %s (event=%s)",
            trigger.name,
            trigger.event_type,
        )

    def unregister(self, trigger_id: str) -> bool:
        """Unregister a webhook trigger."""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False

    def set_callback(
        self,
        callback: Callable[[WebhookTrigger, dict[str, Any]], None],
    ) -> None:
        """Set a callback invoked when a trigger fires.

        The callback receives the matched trigger and the event payload.
        Typically this submits a durable job.
        """
        self._on_trigger = callback

    def evaluate(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[TriggerResult]:
        """Evaluate all triggers against an event.

        Returns a list of results for triggers that matched.
        Fires the callback for each match if one is set.
        """
        results: list[TriggerResult] = []

        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue

            if not TriggerMatcher.matches(trigger, event_type, payload):
                continue

            result = TriggerResult(trigger=trigger, matched=True)

            if self._on_trigger is not None:
                try:
                    self._on_trigger(trigger, payload)
                    result.job_submitted = True
                except Exception as exc:
                    result.error = str(exc)
                    logger.warning(
                        "Trigger callback failed for %s: %s",
                        trigger.name,
                        exc,
                    )

            results.append(result)

        return results

    def list_triggers(self, owner_id: str | None = None) -> list[WebhookTrigger]:
        """List all triggers, optionally filtered by owner."""
        triggers = list(self._triggers.values())
        if owner_id:
            triggers = [t for t in triggers if t.owner_id == owner_id]
        return triggers

    def get_trigger(self, trigger_id: str) -> WebhookTrigger | None:
        return self._triggers.get(trigger_id)

    def enable(self, trigger_id: str) -> bool:
        t = self._triggers.get(trigger_id)
        if t:
            t.enabled = True
            return True
        return False

    def disable(self, trigger_id: str) -> bool:
        t = self._triggers.get(trigger_id)
        if t:
            t.enabled = False
            return True
        return False
