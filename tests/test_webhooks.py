"""Tests for B4 — Webhook Notifications."""

import json
from unittest.mock import MagicMock, patch

from umrs.shared.webhooks import (
    DeliveryStatus,
    EventType,
    WebhookDelivery,
    WebhookEvent,
    WebhookManager,
    WebhookSubscription,
    _compute_signature,
)


class TestWebhookEvent:
    def test_creation(self):
        e = WebhookEvent(
            event_id="evt-001",
            event_type=EventType.JOB_COMPLETED,
            payload={"job_id": "j1"},
        )
        assert e.event_type == EventType.JOB_COMPLETED
        assert e.event_id == "evt-001"

    def test_to_dict(self):
        e = WebhookEvent(
            event_id="evt-002",
            event_type=EventType.PIPELINE_STARTED,
            payload={"pipeline": "p1"},
        )
        d = e.to_dict()
        assert d["event_type"] == "pipeline.started"
        assert d["event_id"] == "evt-002"
        assert "timestamp" in d

    def test_to_json(self):
        e = WebhookEvent(
            event_id="evt-003",
            event_type=EventType.MODEL_REGISTERED,
            payload={"model": "m1"},
        )
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["event_type"] == "model.registered"


class TestWebhookSubscription:
    def test_default_values(self):
        s = WebhookSubscription(
            subscription_id="s1",
            url="https://example.com/hook",
        )
        assert s.enabled is True
        assert s.secret == ""
        assert len(s.event_types) > 0  # defaults to all event types

    def test_matches(self):
        s = WebhookSubscription(
            subscription_id="s1",
            url="https://example.com/hook",
            event_types={EventType.JOB_COMPLETED, EventType.JOB_FAILED},
        )
        assert s.matches(EventType.JOB_COMPLETED)
        assert not s.matches(EventType.PIPELINE_STARTED)


class TestComputeSignature:
    def test_deterministic(self):
        payload = '{"test": true}'
        secret = "my-secret"
        sig1 = _compute_signature(payload, secret)
        sig2 = _compute_signature(payload, secret)
        assert sig1 == sig2
        assert len(sig1) > 0

    def test_different_secrets(self):
        payload = '{"test": true}'
        sig1 = _compute_signature(payload, "secret1")
        sig2 = _compute_signature(payload, "secret2")
        assert sig1 != sig2


class TestWebhookDelivery:
    def test_deliver_success(self):
        delivery = WebhookDelivery(max_retries=1, timeout_seconds=5)
        event = WebhookEvent(
            event_id="evt-100", event_type=EventType.JOB_COMPLETED, payload={"id": "1"}
        )
        sub = WebhookSubscription(subscription_id="s1", url="https://httpbin.org/post")

        with patch("umrs.shared.webhooks.urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            record = delivery.deliver(event, sub)
            assert record.final_status == DeliveryStatus.DELIVERED
            assert record.attempt_count == 1


class TestWebhookManager:
    def test_subscribe(self):
        mgr = WebhookManager()
        sub = mgr.subscribe(
            subscription_id="sub-001",
            url="https://example.com/hook",
            event_types={EventType.JOB_COMPLETED},
            secret="secret",
        )
        assert sub.subscription_id == "sub-001"
        assert sub.url == "https://example.com/hook"

    def test_unsubscribe(self):
        mgr = WebhookManager()
        mgr.subscribe(subscription_id="sub-001", url="https://example.com/hook")
        assert mgr.unsubscribe("sub-001")
        assert not mgr.unsubscribe("nonexistent")

    def test_list_subscriptions(self):
        mgr = WebhookManager()
        mgr.subscribe(subscription_id="s1", url="https://a.com/hook")
        mgr.subscribe(subscription_id="s2", url="https://b.com/hook")
        subs = mgr.list_subscriptions()
        assert len(subs) == 2

    def test_dispatch_fires_matching(self):
        mgr = WebhookManager()
        with patch.object(mgr, "_delivery") as mock_delivery:
            mock_record = MagicMock()
            mock_record.final_status = DeliveryStatus.DELIVERED
            mock_delivery.deliver.return_value = mock_record

            mgr.subscribe(
                subscription_id="s1",
                url="https://example.com/hook",
                event_types={EventType.JOB_COMPLETED},
            )
            event = WebhookEvent(
                event_id="evt-500",
                event_type=EventType.JOB_COMPLETED,
                payload={"job_id": "j1"},
            )
            mgr.dispatch(event)
            assert mock_delivery.deliver.called

    def test_dispatch_skips_non_matching(self):
        mgr = WebhookManager()
        with patch.object(mgr, "_delivery") as mock_delivery:
            mgr.subscribe(
                subscription_id="s1",
                url="https://example.com/hook",
                event_types={EventType.JOB_COMPLETED},
            )
            event = WebhookEvent(
                event_id="evt-501",
                event_type=EventType.PIPELINE_STARTED,
                payload={"pipeline": "p1"},
            )
            mgr.dispatch(event)
            assert not mock_delivery.deliver.called

    def test_event_log(self):
        mgr = WebhookManager()
        with patch.object(mgr, "_delivery") as mock_delivery:
            mock_record = MagicMock()
            mock_record.final_status = DeliveryStatus.DELIVERED
            mock_delivery.deliver.return_value = mock_record

            mgr.subscribe(subscription_id="s1", url="https://example.com/hook")
            event = WebhookEvent(
                event_id="evt-600",
                event_type=EventType.JOB_COMPLETED,
                payload={"id": "1"},
            )
            mgr.dispatch(event)
            assert len(mgr._event_log) == 1
