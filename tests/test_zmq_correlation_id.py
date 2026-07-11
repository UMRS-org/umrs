"""Tests for ZMQ correlation ID propagation (G-001).

These tests verify that:
1. Client-side JSON payloads include a correlation_id
2. Server-side _parse_json_payload extracts and logs correlation_id
3. Old clients (without correlation_id) still work (backward compat)
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from umrs.core.zmq_correlation import (
    dumps_with_correlation_id,
    extract_correlation_id,
    generate_correlation_id,
)


class TestGenerateCorrelationId:
    """Test correlation ID generation."""

    def test_returns_16_char_hex(self):
        cid = generate_correlation_id()
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_returns_unique_ids(self):
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100  # all unique in 100 attempts


class TestCreateCorrelationPayload:
    """Test payload wrapping."""

    def test_adds_correlation_id(self):
        payload = {"cmd": "CHECK", "data": "foo"}
        result = dumps_with_correlation_id(payload)
        data = json.loads(result)
        assert "correlation_id" in data
        assert data["cmd"] == "CHECK"
        assert data["data"] == "foo"

    def test_preserves_existing_correlation_id(self):
        cid = "custom1234567890"  # 16 chars
        payload = {"correlation_id": cid, "cmd": "CHECK"}
        result = dumps_with_correlation_id(payload)
        data = json.loads(result)
        assert data["correlation_id"] == cid  # unchanged

    def test_does_not_mutate_original(self):
        payload = {"cmd": "CHECK"}
        dumps_with_correlation_id(payload)
        assert "correlation_id" not in payload  # original unchanged


class TestExtractCorrelationId:
    """Test correlation ID extraction from payloads."""

    def test_extracts_present_id(self):
        cid = extract_correlation_id({"correlation_id": "abc123def456", "other": "val"})
        assert cid == "abc123def456"

    def test_returns_none_when_missing(self):
        assert extract_correlation_id({"other": "val"}) is None

    def test_truncates_oversize(self):
        cid = extract_correlation_id({"correlation_id": "x" * 200})
        assert len(cid) == 64  # _MAX_CID_LENGTH

    def test_non_string_ignored(self):
        assert extract_correlation_id({"correlation_id": 123}) is None


class TestServerSideExtraction:
    """Test _parse_json_payload extracts correlation_id."""

    def test_parse_extracts_and_logs_correlation_id(self, caplog):
        """_parse_json_payload should extract and log correlation_id from JSON."""
        from umrs.core.MessageQueue import MessageQueue

        # Build a real instance (doesn't need ZMQ sockets for this test)
        mq = MessageQueue.__new__(MessageQueue)
        # Minimal init to set _CONTROL_PARSERS (class attr)
        mq._CONTROL_PARSERS = MessageQueue._CONTROL_PARSERS

        # Build a JSON payload with correlation_id
        payload_str = dumps_with_correlation_id({"workflow_name": "test"})

        with caplog.at_level(logging.INFO):
            cmd, result = mq._parse_json_payload(
                ["RUN_WORKFLOW", "client-1", payload_str], "sender-1"
            )

        assert cmd == "RUN_WORKFLOW"
        assert "correlation_id=" in caplog.text

    def test_parse_handles_no_correlation_id(self, caplog):
        """Old clients without correlation_id should work fine."""
        from umrs.core.MessageQueue import MessageQueue

        mq = MessageQueue.__new__(MessageQueue)
        mq._CONTROL_PARSERS = MessageQueue._CONTROL_PARSERS

        payload_str = json.dumps({"workflow_name": "test"})

        with caplog.at_level(logging.INFO):
            cmd, result = mq._parse_json_payload(
                ["RUN_WORKFLOW", "client-1", payload_str], "sender-1"
            )

        assert cmd == "RUN_WORKFLOW"
        # No correlation_id logged (backward compat)

    def test_parse_handles_invalid_json(self, caplog):
        """Garbage payload should not crash the parser."""
        from umrs.core.MessageQueue import MessageQueue

        mq = MessageQueue.__new__(MessageQueue)
        mq._CONTROL_PARSERS = MessageQueue._CONTROL_PARSERS

        with caplog.at_level(logging.INFO):
            cmd, result = mq._parse_json_payload(
                ["CHECK", "client-1", "not-json"], "sender-1"
            )

        assert cmd == "CHECK"  # Still returns cmd


class TestBackwardCompatibility:
    """Old clients (no correlation_id) must still work."""

    def test_simple_payload_still_works(self):
        """A simple payload without correlation_id parses correctly."""
        from umrs.core.MessageQueue import MessageQueue

        mq = MessageQueue.__new__(MessageQueue)
        mq._CONTROL_PARSERS = MessageQueue._CONTROL_PARSERS

        payload_str = json.dumps({"cmd": "CHECK"})
        cmd, result = mq._parse_json_payload(
            ["CHECK", "client-1", payload_str], "sender-1"
        )

        assert cmd == "CHECK"
        client_id, payload_str_result, zmq_sender_id = result
        assert client_id == "client-1"
        assert payload_str_result == payload_str

    def test_empty_payload_still_works(self):
        """Empty payload string still works."""
        from umrs.core.MessageQueue import MessageQueue

        mq = MessageQueue.__new__(MessageQueue)
        mq._CONTROL_PARSERS = MessageQueue._CONTROL_PARSERS

        cmd, result = mq._parse_json_payload(["CHECK", "client-1"], "sender-1")
        assert cmd == "CHECK"
