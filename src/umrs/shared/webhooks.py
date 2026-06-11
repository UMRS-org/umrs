"""
B4 — Webhook Notifications

Event-driven webhook system for notifying external services about UMRS events
(job completion, pipeline errors, model registration, etc.).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Supported webhook event types."""

    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    MODEL_REGISTERED = "model.registered"
    MODEL_DEPRECATED = "model.deprecated"
    UPLOAD_COMPLETED = "upload.completed"


class DeliveryStatus(str, Enum):
    """Delivery attempt status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class WebhookEvent:
    """Typed event envelope."""

    event_id: str
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "umrs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class WebhookSubscription:
    """A webhook endpoint subscription."""

    subscription_id: str
    url: str
    event_types: set[EventType] = field(default_factory=lambda: set(EventType))
    secret: str = ""
    enabled: bool = True
    description: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, event_type: EventType) -> bool:
        return self.enabled and event_type in self.event_types


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt."""

    event_id: str
    subscription_id: str
    url: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    status_code: int | None = None
    attempt_number: int = 1
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class DeliveryRecord:
    """Full delivery record for an event-subscription pair."""

    event_id: str
    subscription_id: str
    attempts: list[DeliveryAttempt] = field(default_factory=list)
    final_status: DeliveryStatus = DeliveryStatus.PENDING

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def _compute_signature(payload: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class WebhookDelivery:
    """Handles HTTP delivery of webhook events with retry and dead-letter queue."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.dead_letter: deque[DeliveryRecord] = deque(maxlen=1000)
        self._delivery_log: list[DeliveryRecord] = []
        self._lock = threading.Lock()

    def deliver(
        self,
        event: WebhookEvent,
        subscription: WebhookSubscription,
    ) -> DeliveryRecord:
        """Deliver an event to a subscription endpoint with retries."""
        record = DeliveryRecord(
            event_id=event.event_id,
            subscription_id=subscription.subscription_id,
        )

        payload_json = event.to_json()

        for attempt_num in range(1, self.max_retries + 1):
            attempt = DeliveryAttempt(
                event_id=event.event_id,
                subscription_id=subscription.subscription_id,
                url=subscription.url,
                attempt_number=attempt_num,
            )

            start = time.monotonic()
            try:
                headers = {
                    "Content-Type": "application/json",
                    "X-UMRS-Event": event.event_type.value,
                    "X-UMRS-Event-ID": event.event_id,
                    "X-UMRS-Timestamp": str(event.timestamp),
                }

                if subscription.secret:
                    sig = _compute_signature(payload_json, subscription.secret)
                    headers["X-UMRS-Signature"] = f"sha256={sig}"

                req = urllib.request.Request(
                    subscription.url,
                    data=payload_json.encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    attempt.status_code = resp.status
                    if 200 <= resp.status < 300:
                        attempt.status = DeliveryStatus.DELIVERED
                    else:
                        attempt.status = DeliveryStatus.FAILED
                        attempt.error = f"HTTP {resp.status}"

            except Exception as exc:
                attempt.status = DeliveryStatus.FAILED
                attempt.error = str(exc)

            attempt.duration_ms = (time.monotonic() - start) * 1000
            attempt.timestamp = time.time()
            record.attempts.append(attempt)

            if attempt.status == DeliveryStatus.DELIVERED:
                record.final_status = DeliveryStatus.DELIVERED
                break

            if attempt_num < self.max_retries:
                time.sleep(self.retry_delay_seconds * attempt_num)

        if record.final_status != DeliveryStatus.DELIVERED:
            record.final_status = DeliveryStatus.DEAD_LETTER
            with self._lock:
                self.dead_letter.append(record)

        with self._lock:
            self._delivery_log.append(record)

        return record

    @property
    def delivery_log(self) -> list[DeliveryRecord]:
        with self._lock:
            return list(self._delivery_log)


class WebhookManager:
    """Subscription registry and event dispatcher.

    Usage::

        manager = WebhookManager()
        manager.subscribe(
            subscription_id="sub-001",
            url="https://example.com/webhook",
            event_types={EventType.JOB_COMPLETED, EventType.JOB_FAILED},
            secret="my-secret",
        )

        # When an event occurs:
        event = WebhookEvent(
            event_id="evt-001",
            event_type=EventType.JOB_COMPLETED,
            payload={"job_id": "j-123", "status": "completed"},
        )
        records = manager.dispatch(event)
    """

    def __init__(
        self,
        delivery: WebhookDelivery | None = None,
        async_dispatch: bool = False,
    ) -> None:
        self._subscriptions: dict[str, WebhookSubscription] = {}
        self._delivery = delivery or WebhookDelivery()
        self._async_dispatch = async_dispatch
        self._lock = threading.Lock()
        self._event_log: deque[WebhookEvent] = deque(maxlen=10000)
        self._filters: list[Callable[[WebhookEvent], bool]] = []

    def subscribe(
        self,
        subscription_id: str,
        url: str,
        event_types: set[EventType] | None = None,
        secret: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> WebhookSubscription:
        sub = WebhookSubscription(
            subscription_id=subscription_id,
            url=url,
            event_types=event_types or set(EventType),
            secret=secret,
            description=description,
            metadata=metadata or {},
        )
        with self._lock:
            self._subscriptions[subscription_id] = sub
        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def list_subscriptions(self) -> list[WebhookSubscription]:
        with self._lock:
            return list(self._subscriptions.values())

    def get_subscription(self, subscription_id: str) -> WebhookSubscription | None:
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def add_filter(self, fn: Callable[[WebhookEvent], bool]) -> None:
        self._filters.append(fn)

    def dispatch(self, event: WebhookEvent) -> list[DeliveryRecord]:
        """Dispatch an event to all matching subscriptions."""
        self._event_log.append(event)

        # Apply filters
        for f in self._filters:
            if not f(event):
                return []

        with self._lock:
            matching = [
                s for s in self._subscriptions.values() if s.matches(event.event_type)
            ]

        if not matching:
            return []

        records: list[DeliveryRecord] = []

        if self._async_dispatch:
            threads = []
            for sub in matching:
                t = threading.Thread(
                    target=self._deliver_and_collect,
                    args=(event, sub, records),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=60.0)
        else:
            for sub in matching:
                record = self._delivery.deliver(event, sub)
                records.append(record)

        return records

    def _deliver_and_collect(
        self,
        event: WebhookEvent,
        sub: WebhookSubscription,
        records: list[DeliveryRecord],
    ) -> None:
        record = self._delivery.deliver(event, sub)
        records.append(record)

    @property
    def event_log(self) -> list[WebhookEvent]:
        return list(self._event_log)

    @property
    def dead_letter_queue(self) -> deque[DeliveryRecord]:
        return self._delivery.dead_letter
