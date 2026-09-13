"""6b-3 pins: venue safety + cancel-as-interrupt (fail-without-change)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from umrs.jobs.download_safety import (
    DownloadVenueError,
    assert_download_safe,
    broadcast_cancel,
    free_bytes_for,
    min_free_bytes,
    mount_type_for,
    register_cancel_listener,
)


@pytest.fixture(autouse=True)
def _isolate_cancel_bus():
    """The cancel bus is a process-global listener list; backend constructors
    self-register listeners that outlive their test. Snapshot/restore so a
    broadcast inside one test never notifies another test's listeners."""
    from umrs.jobs import download_safety as ds

    snapshot = list(ds._listeners)
    ds._listeners.clear()
    try:
        yield
    finally:
        ds._listeners.clear()
        ds._listeners.extend(snapshot)


class TestMountParsing:
    def test_overlay_refused_before_first_byte(self, tmp_path):
        with (
            patch("umrs.jobs.download_safety._mount_entries", return_value=[("/", "overlay")]),
            pytest.raises(DownloadVenueError, match="union filesystem"),
        ):
            assert_download_safe(str(tmp_path))

    def test_mounted_shared_fs_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_DOWNLOAD_MIN_FREE_BYTES", "1")
        with patch(
            "umrs.jobs.download_safety._mount_entries",
            return_value=[
                ("/", "ext4"),
                (str(tmp_path), "lustre"),
            ],
        ):
            out = assert_download_safe(str(tmp_path / "some" / "dir"))
        assert out["mount_type"] == "lustre"
        assert out["free_bytes"] > 0

    def test_free_space_floor_enforced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_DOWNLOAD_MIN_FREE_BYTES", str(10**20))
        with (
            patch("umrs.jobs.download_safety._mount_entries", return_value=[("/", "ext4")]),
            pytest.raises(DownloadVenueError, match="bytes free"),
        ):
            assert_download_safe(str(tmp_path))

    def test_floor_env_zero_disables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_DOWNLOAD_MIN_FREE_BYTES", "0")
        with patch("umrs.jobs.download_safety._mount_entries", return_value=[("/", "ext4")]):
            out = assert_download_safe(str(tmp_path))
        assert out["min_free_bytes"] == 0

    def test_mount_type_longest_prefix(self):
        with patch(
            "umrs.jobs.download_safety._mount_entries",
            return_value=[
                ("/", "ext4"),
                ("/home", "lustre"),
                ("/home/mnagaso/models", "fuse-overlayfs"),
            ],
        ):
            assert mount_type_for("/home/mnagaso/models/x") == "fuse-overlayfs"


class TestFreeBytes:
    def test_staeded(self, tmp_path):
        assert free_bytes_for(str(tmp_path)) > 0

    def test_missing_path_uses_parent(self, tmp_path):
        assert free_bytes_for(str(tmp_path / "deep" / "nest")) > 0


class TestCancelBus:
    def test_broadcast_notifies_all_and_isolated_failures(self):
        got = []
        good = lambda j: got.append(j)

        def boom(_):
            raise RuntimeError("local listener died")

        register_cancel_listener(good)
        register_cancel_listener(boom)
        notified = broadcast_cancel("job-x")
        assert got == ["job-x"]
        assert notified == 1  # isolated per-listener
        # registry resets across suite runs: pop for isolation
        from umrs.jobs import download_safety as ds

        ds._listeners.remove(good)
        ds._listeners.remove(boom)


class TestCancelInterruptsDownload:
    def _backend(self):
        from umrs.jobs.model_download import ModelDownloadBackend

        backend = ModelDownloadBackend(
            model_store=MagicMock(),
            path_validator=MagicMock(is_allowed=lambda p: True),
        )
        return backend

    def test_cancel_marks_attempt_and_cancels_handle(self):
        backend = self._backend()

        async def napping():
            await asyncio.sleep(60)

        # get_event_loop() has no current-loop guarantee on the main thread
        # once any earlier test closed it — own the loop explicitly.
        loop = asyncio.new_event_loop()
        try:
            handle = loop.create_task(napping())
            backend._handles["att-9"] = handle
            assert backend.cancel_attempt("att-9") is True
            assert "att-9" in backend._cancelled_attempts
            assert backend.cancel_attempt("missing") is False
        finally:
            handle.cancel()
            loop.close()

    def test_callback_guard_raises_mid_loop(self):
        from umrs.jobs.download_safety import DownloadCancelledError

        backend = self._backend()
        backend._cancelled_attempts.add("att-1")
        with pytest.raises(DownloadCancelledError):
            backend._raise_if_cancelled("att-1")
        backend._raise_if_cancelled("att-2")  # no-op, never a false cancel

    def test_durable_cancel_listener_hits_own_jobs_only(self):
        backend = self._backend()
        ctx_mine = MagicMock(job_id="job-mine", attempt_id="att-1")
        ctx_other = MagicMock(job_id="job-other", attempt_id="att-2")
        backend.download_registry._tasks = {
            ctx_mine.attempt_id: ctx_mine,
            ctx_other.attempt_id: ctx_other,
        }
        backend._on_durable_cancel("job-mine")
        assert "att-1" in backend._cancelled_attempts
        assert "att-2" not in backend._cancelled_attempts
