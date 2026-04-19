from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from client_umrs import format_status_text
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.server.provisioning import reconcile_status_on_startup
from umrs.shared.pretty_print import pretty_print_thread_status


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
    assert "2026-04-19T02:11:24Z" in message


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

    assert "generated 2026-04-19T02:12:00Z; last update 2026-04-19T02:11:24Z" in rendered


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
    assert first_call.args[1] == TaskCmd.PROCESS_OBJECTS.lower()
    payload = first_call.args[2]
    assert payload["data"][0]["thread_id"] == "snapshot"
    assert payload["last_updated_at"] == payload["data"][0]["updated_at"]