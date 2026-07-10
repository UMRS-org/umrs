"""Regression tests for ghost endpoints caused by stale job info files.

A Slurm job writes an info file (``rivault-job-<id>.info``) to the shared
filesystem at startup. That file is only deleted by ``cleanup()``. When a job
dies by walltime, preemption, node failure, or an external ``scancel`` — i.e.
without going through UMRS ``cleanup()`` — the file is orphaned and still says
``RIVAULT_STATUS=running``.

``get_job_status`` previously trusted the info file verbatim, so a dead job kept
reporting ``RUNNING``/``READY``. ``_filter_live_ready_endpoints`` therefore never
pruned it and ``r4 endpoint list`` showed a ghost ``ready`` endpoint long after
``squeue`` was empty.

The fix cross-checks READY/RUNNING info-file states against Slurm and reports the
terminal state (removing the stale file) when Slurm *positively* reports the job
FAILED/STOPPED. A transient ``UNKNOWN`` miss must NOT delete a healthy endpoint.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend
from umrs.orchestrator.slurm_backend import SlurmBackend

_STALE_RUNNING_INFO = (
    "RIVAULT_STATUS=running\n"
    "RIVAULT_NODE=ng-dgx-m0-09.cloud.r-ccs.riken.jp\n"
    "RIVAULT_LITELLM_PORT=12221\n"
    "RIVAULT_VLLM_PORT=8000\n"
    "RIVAULT_MODEL_PATH=/models/Qwen3.6-35B-A3B-NVFP4\n"
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRemoteSlurmGhostPruning:
    def _backend(self) -> RemoteSlurmBackend:
        return RemoteSlurmBackend.__new__(RemoteSlurmBackend)

    def test_dead_job_with_stale_running_info_reports_terminal(self):
        backend = self._backend()
        backend._read_remote_job_info = AsyncMock(return_value=_STALE_RUNNING_INFO)
        # squeue miss -> sacct positively FAILED.
        backend._query_slurm_job_status = AsyncMock(return_value=(JobStatus.FAILED, None))
        backend._remove_stale_info_file = AsyncMock()

        info = _run(backend.get_job_status("207461"))

        assert info.status == JobStatus.FAILED
        assert info.endpoint is None
        backend._remove_stale_info_file.assert_awaited_once_with("207461")

    def test_unknown_slurm_state_preserves_running_endpoint(self):
        # Transient squeue/sacct miss must NOT delete a live endpoint.
        backend = self._backend()
        backend._read_remote_job_info = AsyncMock(return_value=_STALE_RUNNING_INFO)
        backend._query_slurm_job_status = AsyncMock(return_value=(JobStatus.UNKNOWN, None))
        backend._remove_stale_info_file = AsyncMock()
        backend._endpoint_via_tunnel = AsyncMock(side_effect=lambda ep: ep)

        info = _run(backend.get_job_status("207461"))

        assert info.status == JobStatus.RUNNING
        assert info.endpoint is not None
        backend._remove_stale_info_file.assert_not_awaited()

    def test_running_job_confirmed_by_slurm_is_preserved(self):
        backend = self._backend()
        backend._read_remote_job_info = AsyncMock(return_value=_STALE_RUNNING_INFO)
        # squeue RUNNING maps to STARTING (non-terminal) -> keep info status.
        backend._query_slurm_job_status = AsyncMock(return_value=(JobStatus.STARTING, None))
        backend._remove_stale_info_file = AsyncMock()
        backend._endpoint_via_tunnel = AsyncMock(side_effect=lambda ep: ep)

        info = _run(backend.get_job_status("207461"))

        assert info.status == JobStatus.RUNNING
        assert info.endpoint is not None
        backend._remove_stale_info_file.assert_not_awaited()


class TestLocalSlurmGhostPruning:
    def _backend(self, info_dir) -> SlurmBackend:
        backend = SlurmBackend.__new__(SlurmBackend)
        backend.config = SimpleNamespace(slurm_job_info_dir=str(info_dir))
        return backend

    def test_dead_job_with_stale_running_info_reports_terminal(self, tmp_path):
        backend = self._backend(tmp_path)
        info_path = tmp_path / "rivault-job-207461.info"
        info_path.write_text(_STALE_RUNNING_INFO)
        backend._query_slurm_job_status = AsyncMock(return_value=JobStatus.FAILED)

        info = _run(backend.get_job_status("207461"))

        assert info.status == JobStatus.FAILED
        assert info.endpoint is None
        assert not info_path.exists()  # stale file removed

    def test_unknown_slurm_state_preserves_running_endpoint(self, tmp_path):
        backend = self._backend(tmp_path)
        info_path = tmp_path / "rivault-job-207461.info"
        info_path.write_text(_STALE_RUNNING_INFO)
        backend._query_slurm_job_status = AsyncMock(return_value=JobStatus.UNKNOWN)

        info = _run(backend.get_job_status("207461"))

        assert info.status == JobStatus.RUNNING
        assert info.endpoint is not None
        assert info_path.exists()  # preserved on inconclusive verification
