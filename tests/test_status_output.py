from __future__ import annotations

import argparse
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import client_umrs
from client_umrs import _fmt_cell, _resolve_status_target, action_list_buckets, format_status_text
from umrs.client.control_commands import configure_status_parser
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.server.provisioning import reconcile_status_on_startup
from umrs.shared.pretty_print import pretty_print_thread_status


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_pretty_print_thread_status_shows_snapshot_and_last_update():
    message = pretty_print_thread_status(
        {
            "startup": {
                "thread_id": "snapshot",
                "status": "Done",
                "updated_at": "2026-04-19T02:11:24Z",
                "work": [
                    {
                        "work_type": "process_objects",
                        "user": "alice",
                        "nb_objects": 2,
                        "modality_name": "text",
                        "bucket": "user1-textfiles",
                        "data_base": "pgvector_alice",
                        "table_name": "text_embeddings",
                    }
                ],
            }
        },
        running_only=False,
        print_output=False,
    )

    assert "snapshot" in message
    assert "Last Update" in message
    assert "2026-04-19" in message  # date part only, time converted to local


def test_format_status_text_shows_generated_and_last_update_timestamps():
    rendered = format_status_text(
        {
            "status": "ok",
            "client": "alice",
            "summary": {
                "generated_at": "2026-04-19T02:12:00Z",
                "last_updated_at": "2026-04-19T02:11:24Z",
                "summary": "Server status:\nall good",
            },
        },
        "process_objects",
    )

    assert "2026-04-19" in rendered  # timestamps present


def test_format_status_text_explains_missing_reload_status():
    rendered = format_status_text(
        {
            "status": "error",
            "message": "No status for 'reload'",
        },
        "reload",
    )

    assert "Reload is synchronous" in rendered
    assert "status all" in rendered
    assert "bucket list" in rendered


def test_fmt_cell_pads_before_applying_style(monkeypatch):
    monkeypatch.setattr(client_umrs, "_use_color", lambda: True)

    rendered = _fmt_cell("Alias", 20, style=client_umrs._bold)

    assert _strip_ansi(rendered) == f"{'Alias':20s}"


def test_action_list_buckets_keeps_header_columns_aligned_with_color(monkeypatch):
    monkeypatch.setattr(client_umrs, "_use_color", lambda: True)
    bucket_cfg = {
        "test1-text": SimpleNamespace(
            username="test1",
            xp_name="default",
            modality_name="text",
            task="embedding",
            path_to_bucket="test1.default.test1-text",
        )
    }

    rendered = action_list_buckets(bucket_cfg, "test1", "text")
    lines = [_strip_ansi(line) for line in rendered.splitlines()]

    assert lines[1] == "  " + "  ".join(
        [
            f"{'Experience':10s}",
            f"{'Alias':10s}",
            f"{'Modality':8s}",
            f"{'Task':9s}",
            "Path",
        ]
    )
    assert lines[3] == "  " + "  ".join(
        [
            f"{'default':10s}",
            f"{'test1-text':10s}",
            f"{'text':8s}",
            f"{'embedding':9s}",
            "test1.default.test1-text",
        ]
    )


def test_resolve_status_target_supports_all_flag_alias():
    parser = argparse.ArgumentParser()
    configure_status_parser(parser)

    args = parser.parse_args(["--all"])

    assert _resolve_status_target(args) == "all"


def test_resolve_status_target_rejects_all_flag_with_specific_key():
    parser = argparse.ArgumentParser()
    configure_status_parser(parser)

    args = parser.parse_args(["--all", "check"])

    try:
        _resolve_status_target(args)
    except ValueError as exc:
        assert "either '--all' or a specific status key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting status selectors")


def test_reconcile_status_on_startup_records_snapshot_status():
    active_threads = MagicMock()
    active_threads._json_safe_thread_status.side_effect = lambda data: [
        {
            "thread_id": data["startup"]["thread_id"],
            "status": data["startup"]["status"],
            "updated_at": data["startup"]["updated_at"],
            "work": data["startup"]["work"],
        }
    ]
    buckets_cfg = {
        "user1-textfiles": SimpleNamespace(
            s3_username="alice",
            db_name="pgvector_alice",
            table_name="text_embeddings",
            modality_name="text",
            num_objects=3,
        )
    }

    reconcile_status_on_startup(active_threads, buckets_cfg)

    assert active_threads.record_status.call_count == 2
    first_call = active_threads.record_status.call_args_list[0]
    assert first_call.args[0] == "alice"
    assert first_call.args[1] == TaskCmd.PROCESS_OBJECTS  # code passes uppercase
    payload = first_call.args[2]
    assert payload["data"][0]["thread_id"] == "snapshot"
    assert payload["last_updated_at"] == payload["data"][0]["updated_at"]
