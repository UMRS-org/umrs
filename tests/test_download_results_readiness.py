"""Tests for download_results readiness helpers and CLI flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from umrs.client.download_results import (  # noqa: E402
    classify_wait_reply,
    format_wait_progress,
    incomplete_error_payload,
    parse_process_objects_readiness,
)


def _status_payload(entries: list[dict], *, message: str | None = None) -> dict:
    payload: dict = {
        "status": "ok",
        "client": "alice",
        "summary": {"summary": "ok", "data": entries},
    }
    if message is not None:
        payload["message"] = message
    return payload


def _entry(status: str, bucket: str, *, thread_id: str = "t1") -> dict:
    return {
        "thread_id": thread_id,
        "status": status,
        "updated_at": "2026-07-27T00:00:00Z",
        "work": [
            {
                "user": "alice",
                "bucket": bucket,
                "work_type": "PROCESS_OBJECTS",
                "nb_objects": 1,
            }
        ],
    }


class TestParseProcessObjectsReadiness:
    def test_running_for_output_bucket_is_incomplete(self):
        payload = _status_payload(
            [_entry("Running", "alice.xp.images")]
        )
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is False
        assert result.status_known is True
        assert result.incomplete_statuses == ("Running",)
        assert "incomplete" in result.message
        assert "Running" in result.message

    def test_done_for_matching_bucket_is_ready(self):
        payload = _status_payload([_entry("Done", "alice.xp.images")])
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is True
        assert result.status_known is True
        assert result.incomplete_statuses == ()

    def test_idle_is_terminal(self):
        payload = _status_payload([_entry("Idle", "alice.xp.images")])
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is True

    def test_direct_bucket_name_match(self):
        payload = _status_payload(
            [_entry("Processing", "alice.xp.images-output")]
        )
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is False
        assert "Processing" in result.incomplete_statuses

    def test_no_status_entries_is_not_ready(self):
        payload = {
            "status": "error",
            "message": "No status entries",
            "client": "alice",
        }
        result = parse_process_objects_readiness(payload, "alice.xp.images-output")
        assert result.ready is False
        assert result.status_known is False
        assert "unknown" in result.message.lower() or "No status entries" in result.message

    def test_no_status_for_key_is_not_ready(self):
        payload = {
            "status": "error",
            "message": "No status for 'PROCESS_OBJECTS'",
            "client": "alice",
        }
        result = parse_process_objects_readiness(payload, "bucket-out")
        assert result.ready is False
        assert result.status_known is False

    def test_none_payload_is_not_ready(self):
        result = parse_process_objects_readiness(None, "bucket-out")
        assert result.ready is False
        assert result.status_known is False

    def test_unrelated_running_work_does_not_block(self):
        payload = _status_payload(
            [_entry("Running", "other.user.bucket", thread_id="other")]
        )
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is True
        assert result.status_known is True

    def test_mixed_done_and_running_for_same_bucket(self):
        payload = _status_payload(
            [
                _entry("Done", "alice.xp.images", thread_id="old"),
                _entry("Running", "alice.xp.images", thread_id="new"),
            ]
        )
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is False
        assert result.incomplete_statuses == ("Running",)

    def test_dict_shaped_data_is_accepted(self):
        payload = {
            "status": "ok",
            "summary": {
                "data": {
                    "t1": _entry("Running", "alice.xp.images"),
                }
            },
        }
        result = parse_process_objects_readiness(
            payload, "alice.xp.images-output"
        )
        assert result.ready is False


class TestFormatWaitProgress:
    def test_includes_status_and_elapsed(self):
        readiness = parse_process_objects_readiness(
            _status_payload([_entry("Running", "b")]),
            "b-output",
        )
        text = format_wait_progress(readiness, elapsed_sec=125)
        assert "Running" in text
        assert "2m 5s" in text


class TestControlFlowHelpers:
    def test_incomplete_error_payload_no_wait(self):
        readiness = parse_process_objects_readiness(
            _status_payload([_entry("Running", "b")]),
            "b-output",
        )
        payload = incomplete_error_payload(
            readiness,
            full_bucket_name="b-output",
            subdir="",
            modality="image",
            output_format="json",
        )
        assert payload["status"] == "error"
        assert payload["task_status"] == ["Running"]
        assert "incomplete" in payload["message"]

    def test_incomplete_error_payload_timeout(self):
        readiness = parse_process_objects_readiness(None, "b-output")
        payload = incomplete_error_payload(
            readiness,
            full_bucket_name="b-output",
            subdir="x",
            modality="image",
            output_format="text",
            elapsed_sec=10,
            timed_out=True,
            max_wait_sec=10,
        )
        assert "Timed out after 10s" in payload["message"]
        assert payload["elapsed_sec"] == 10
        assert payload["task_status"] == "unknown"

    def test_classify_wait_reply(self):
        assert classify_wait_reply("TIMEOUT") == "timeout"
        assert classify_wait_reply("UPDATE") == "update"
        assert classify_wait_reply("NO_SUCH_BUCKET") == "hard_error"
        assert classify_wait_reply("SERVER_OVERLOADED") == "hard_error"
        assert classify_wait_reply("WTF") == "unexpected"


class TestDownloadResultsCliFlags:
    def test_wait_flags_default_and_no_wait(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="action")
        dr_p = sub.add_parser("download_results")
        dr_p.add_argument("bucket_alias")
        dr_p.add_argument("local_path")
        wait_group = dr_p.add_mutually_exclusive_group()
        wait_group.add_argument(
            "--wait", dest="wait", action="store_true", default=True
        )
        wait_group.add_argument(
            "--no-wait", dest="wait", action="store_false"
        )
        dr_p.add_argument("--timeout", type=int, default=3600)

        args = parser.parse_args(
            ["download_results", "alias", "/tmp/out"]
        )
        assert args.wait is True
        assert args.timeout == 3600

        args = parser.parse_args(
            ["download_results", "alias", "/tmp/out", "--no-wait"]
        )
        assert args.wait is False

        args = parser.parse_args(
            ["download_results", "alias", "/tmp/out", "--timeout", "30"]
        )
        assert args.timeout == 30
        assert args.wait is True

        src = Path(__file__).resolve().parents[1] / "src" / "client_umrs.py"
        text = src.read_text(encoding="utf-8")
        assert "--no-wait" in text
        assert 'dest="wait"' in text
        assert "--timeout" in text
        assert "incomplete_error_payload" in text
        assert "classify_wait_reply" in text
