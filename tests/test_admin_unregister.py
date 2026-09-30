"""Pins for _maybe_unregister_from_server_config (2026-09-30).

Found during the batch-benchmark cleanup: `admin user delete --purge`
removed the user files but left the conf_umrs.json entries behind —
the daemon would have re-provisioned 128 benchmark users on its next
reload because the unregister verb was never implemented (only the
register half existed).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from umrs.cli.admin.main import _maybe_unregister_from_server_config


class _Store:
    def __init__(self, dir_path) -> None:
        self.dir = dir_path


def test_unregister_removes_relative_entry_and_keeps_others(tmp_path):
    conf = tmp_path / "conf_umrs.json"
    conf.write_text(
        json.dumps(
            {
                "users": [
                    "./user_files/real_user.json",
                    "./user_files/user1_1.json",
                    "/other/abs/path/user1_1.json",
                    "./user_files/another.json",
                ]
            }
        )
    )
    store_dir = tmp_path / "user_files"
    store_dir.mkdir()
    args = SimpleNamespace(server_config=str(conf))

    _maybe_unregister_from_server_config(args, _Store(store_dir), "user1_1")

    users = json.loads(conf.read_text())["users"]
    # The relative form pointing at the deleted file is dropped; the absolute
    # entry points at a DIFFERENT file and must be preserved.
    assert users == [
        "./user_files/real_user.json",
        "/other/abs/path/user1_1.json",
        "./user_files/another.json",
    ]


def test_unregister_noop_when_user_absent(tmp_path):
    conf = tmp_path / "conf_umrs.json"
    conf.write_text(json.dumps({"users": ["./user_files/real_user.json"]}))
    args = SimpleNamespace(server_config=str(conf))

    _maybe_unregister_from_server_config(args, _Store(tmp_path / "user_files"), "ghost")

    assert json.loads(conf.read_text())["users"] == ["./user_files/real_user.json"]


def test_unregister_noop_without_config_flag(tmp_path):
    args = SimpleNamespace(server_config=None)
    # Must not raise even though no conf path exists.
    _maybe_unregister_from_server_config(args, _Store(tmp_path), "user1_1")
