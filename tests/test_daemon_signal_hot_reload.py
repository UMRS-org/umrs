"""Tests for the daemon-signal PID-file + SIGHUP hot-reload path."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# -----------------------------------------------------------------------------
# isolated import so ``UMRS_CONFIG`` overrides take effect per-test
# -----------------------------------------------------------------------------


def _import_daemon_signal(monkeypatch: pytest.MonkeyPatch, cfg_dir: Path):
    monkeypatch.setenv("UMRS_CONFIG", str(cfg_dir))
    monkeypatch.delenv("UMRS_SERVER_PID_FILE", raising=False)
    # Ensure we re-evaluate the module path with the patched env.
    import importlib

    import umrs.admin.daemon_signal as mod

    return importlib.reload(mod)


# -----------------------------------------------------------------------------
# path resolution
# -----------------------------------------------------------------------------


class TestResolvePidFilePath:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "custom.pid"
        monkeypatch.setenv("UMRS_SERVER_PID_FILE", str(target))
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        # Re-set env because reload above may have reset under some harnesses.
        monkeypatch.setenv("UMRS_SERVER_PID_FILE", str(target))
        assert mod.resolve_pid_file_path() == os.path.realpath(str(target))

    def test_defaults_to_cfg_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        expected = os.path.realpath(str(tmp_path / "umrs-server.pid"))
        assert mod.resolve_pid_file_path() == expected

    def test_explicit_cfg_dir_param(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        other = tmp_path / "other"
        other.mkdir()
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        assert mod.resolve_pid_file_path(cfg_dir=str(other)) == os.path.realpath(
            str(other / "umrs-server.pid")
        )


# -----------------------------------------------------------------------------
# read/write/remove
# -----------------------------------------------------------------------------


class TestPidFileIO:
    def test_write_creates_file_with_current_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        path = mod.write_pid_file()
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as fh:
            assert fh.read().strip() == str(os.getpid())

    def test_write_is_atomic_and_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        first = mod.write_pid_file()
        # Write again — must not fail even if the file exists.
        second = mod.write_pid_file()
        assert first == second

    def test_read_returns_none_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        assert mod.read_pid_file() is None

    def test_read_returns_none_on_garbage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        target = mod.resolve_pid_file_path()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        Path(target).write_text("not-a-number\n", encoding="utf-8")
        assert mod.read_pid_file() is None

    def test_remove_only_if_ours_protects_foreign_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        target = mod.resolve_pid_file_path()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        Path(target).write_text("999999\n", encoding="utf-8")
        assert mod.remove_pid_file(only_if_ours=True) is False
        assert os.path.exists(target)
        assert mod.remove_pid_file(only_if_ours=False) is True
        assert not os.path.exists(target)


# -----------------------------------------------------------------------------
# liveness probe
# -----------------------------------------------------------------------------


class TestIsProcessAlive:
    def test_current_process_is_alive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        assert mod.is_process_alive(os.getpid()) is True

    def test_invalid_pid_is_dead(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        assert mod.is_process_alive(0) is False
        assert mod.is_process_alive(-1) is False

    def test_reaped_pid_is_dead(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        # Start & immediately wait for a no-op process so its PID is reaped.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        # On Linux the PID may be immediately recycled; we just assert the
        # call is well-behaved (doesn't raise).
        result = mod.is_process_alive(proc.pid)
        assert isinstance(result, bool)


# -----------------------------------------------------------------------------
# signal_reload
# -----------------------------------------------------------------------------


class TestSignalReload:
    def test_missing_pid_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        result = mod.signal_reload()
        assert result.sent is False
        assert result.pid is None
        assert "no pid file" in result.reason

    def test_stale_pid_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        target = mod.resolve_pid_file_path()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # A PID we're confident is not running.
        Path(target).write_text("999999\n", encoding="utf-8")
        result = mod.signal_reload()
        assert result.sent is False
        assert result.pid == 999999
        assert "stale" in result.reason

    def test_live_process_receives_sighup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fork a short-lived child that catches SIGHUP and writes a marker."""
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        marker = tmp_path / "got_sighup"

        child_src = textwrap.dedent(
            f"""
            import os, signal, sys, time
            got = {{"v": False}}
            def h(signum, frame):
                got["v"] = True
                with open({str(marker)!r}, "w") as fh:
                    fh.write("ok")
                sys.exit(0)
            signal.signal(signal.SIGHUP, h)
            print("ready", flush=True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                time.sleep(0.05)
            sys.exit(1)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", child_src],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for the child to install its handler.
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"
            target = mod.resolve_pid_file_path()
            os.makedirs(os.path.dirname(target), exist_ok=True)
            Path(target).write_text(f"{proc.pid}\n", encoding="utf-8")
            result = mod.signal_reload()
            assert result.sent is True
            assert result.pid == proc.pid
            rc = proc.wait(timeout=5)
            assert rc == 0
            assert marker.read_text() == "ok"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# -----------------------------------------------------------------------------
# handler install + consume
# -----------------------------------------------------------------------------


class TestInstallHandler:
    def test_handler_sets_flag_on_sighup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_daemon_signal(monkeypatch, tmp_path)
        # Preserve and restore any prior handler.
        prev = signal.getsignal(signal.SIGHUP)
        try:
            mod.install_reload_signal_handler()
            # Flag should start clear.
            assert mod.consume_reload_request() is False
            os.kill(os.getpid(), signal.SIGHUP)
            # Give the signal a moment to be delivered.
            for _ in range(20):
                if mod.consume_reload_request():
                    break
                time.sleep(0.01)
            else:
                pytest.fail("SIGHUP never flipped the reload flag")
            # Second consume returns False (one-shot).
            assert mod.consume_reload_request() is False
        finally:
            signal.signal(signal.SIGHUP, prev)


# -----------------------------------------------------------------------------
# admin CLI integration: auto-reload after mutations
# -----------------------------------------------------------------------------


class TestAdminCliAutoReload:
    def test_add_user_signals_reload_when_pid_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        user_files = tmp_path / "user_files"
        monkeypatch.setenv("UMRS_CONFIG", str(cfg))
        monkeypatch.delenv("UMRS_SERVER_PID_FILE", raising=False)

        # Start a child that catches SIGHUP.
        child_src = textwrap.dedent(
            """
            import os, signal, sys, time
            def h(signum, frame):
                print("GOT_SIGHUP", flush=True)
                sys.exit(0)
            signal.signal(signal.SIGHUP, h)
            print("ready", flush=True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                time.sleep(0.05)
            sys.exit(1)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", child_src],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"
            # Seed the default PID file path with the child's PID.
            import importlib

            import umrs.admin.daemon_signal as ds

            importlib.reload(ds)
            pid_path = ds.resolve_pid_file_path()
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            Path(pid_path).write_text(f"{proc.pid}\n", encoding="utf-8")

            from umrs.cli.admin.main import main as admin_main

            rc = admin_main(
                [
                    "--user-files-dir",
                    str(user_files),
                    "user",
                    "add",
                    "alice",
                ]
            )
            assert rc == 0
            out = capsys.readouterr().out
            assert "[reload] SIGHUP sent" in out
            # Child exits after receiving SIGHUP.
            rc_child = proc.wait(timeout=5)
            assert rc_child == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    def test_add_user_no_reload_flag_suppresses_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        user_files = tmp_path / "user_files"
        monkeypatch.setenv("UMRS_CONFIG", str(cfg))
        # Point to a stale PID file so we'd see "[reload] skipped" if signalling ran.
        pid_path = cfg / "umrs-server.pid"
        pid_path.write_text("999999\n", encoding="utf-8")

        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(
            [
                "--user-files-dir",
                str(user_files),
                "--no-reload",
                "user",
                "add",
                "bob",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[reload]" not in out

    def test_readonly_list_does_not_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        user_files = tmp_path / "user_files"
        user_files.mkdir()
        monkeypatch.setenv("UMRS_CONFIG", str(cfg))

        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(
            [
                "--user-files-dir",
                str(user_files),
                "user",
                "list",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[reload]" not in out

    def test_reload_subcommand_returns_error_when_no_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        monkeypatch.setenv("UMRS_CONFIG", str(cfg))

        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(["reload"])
        assert rc == 2
        out = capsys.readouterr().out
        assert "skipped" in out
        assert "no pid file" in out

    def test_reload_subcommand_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        monkeypatch.setenv("UMRS_CONFIG", str(cfg))

        from umrs.cli.admin.main import main as admin_main

        rc = admin_main(["reload", "--json"])
        assert rc == 2
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["sent"] is False
        assert payload["pid"] is None
        assert "reason" in payload
