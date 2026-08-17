"""CompanionWorkerSupervisor tests (wave T3b m4, venue spawn acceptance)."""

from __future__ import annotations

import os
import signal
import time
from contextlib import suppress
from pathlib import Path

import pytest
from umrs.jobs.companion_worker import (
    WORKER_MAX_ATTEMPTS,
    CompanionWorkerSpec,
)
from umrs.jobs.companion_worker_supervisor import (
    AtCapacity,
    CompanionWorkerSupervisor,
    SpawnConfig,
    WorkerSpawnError,
)


class SpyManager:
    """Minimal Duck-typed manager (same API picks used by CompanionWorkerManager)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.states: dict[str, str] = {}

    def get(self, registry_key: str):
        row = self.rows.get(registry_key)
        return dict(row) if row else None

    def register_active(self, spec: CompanionWorkerSpec, *, job_id: str):
        row = {
            "registry_key": spec.registry_key(),
            "owner_user_id": spec.owner_user_id,
            "project_id": spec.project_id,
            "job_id": job_id,
            "session_token_fingerprint": spec.session_token_fingerprint,
            "state": "active",
            "attempt_index": 0,
            "attempts_max": WORKER_MAX_ATTEMPTS,
        }
        self.rows[spec.registry_key()] = row
        return dict(row)

    def heartbeat(self, registry_key: str, *, attempt_index: int, attempts_max: int = 6):
        row = self.rows.get(registry_key)
        if row is None:
            return None
        row["attempt_index"] = attempt_index
        row["attempts_max"] = attempts_max
        if attempt_index >= attempts_max - 1:
            row["state"] = "draining"
        return dict(row)

    def evict(self, registry_key: str) -> None:
        if registry_key in self.rows:
            self.rows[registry_key]["state"] = "evicted"


def _spec(owner: str = "alice", project: str = "default") -> CompanionWorkerSpec:
    return CompanionWorkerSpec(owner_user_id=owner, project_id=project)


def _config(tmp_path: Path, **overrides) -> SpawnConfig:
    opts: dict = {
        "workers_root": tmp_path,
        "attempts_max": 3,
        "beat_interval_sec": 0.2,
        "term_grace_sec": 1.0,
        "respawn_jitter_sec": 0.01,
    }
    opts.update(overrides)
    return SpawnConfig(**opts)


def _wait_for(fn, timeout: float = 5.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        result = fn()
        if result:
            return
        time.sleep(0.03)
    raise AssertionError("wait_for timed out")


def _modify_exit(proc, *, kill=False):
    with suppress(ProcessLookupError):
        os.kill(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)


class TestSupervisorSpawn:
    def test_spawn_registers_and_consumes_beats(self, tmp_path):
        manager = SpyManager()
        supervisor = CompanionWorkerSupervisor(manager, config=_config(tmp_path))
        try:
            spawn = supervisor.spawn(_spec(), job_id="job-w1")
            _wait_for(lambda: spawn.beats_seen >= 1 or spawn.process.poll() is not None)
            assert spawn.process.poll() is None
            assert spawn.socket_path.exists()
            assert oct(spawn.workdir.stat().st_mode & 0o777) == "0o700"
            row = manager.get("alice@default")
            assert row is not None and row["state"] == "active"
        finally:
            supervisor.stop_all()

    def test_capacity_gate_blocks_overrun(self, tmp_path):
        manager = SpyManager()
        c = _config(tmp_path, max_workers=1)
        supervisor = CompanionWorkerSupervisor(manager, config=c)
        try:
            supervisor.spawn(_spec("alice"), job_id="job-w1")
            with pytest.raises(AtCapacity):
                supervisor.spawn(_spec("bob"), job_id="job-w2")
        finally:
            supervisor.stop_all()


class TestRestart:
    def test_kill9_respawns_with_incremented_attempt(self, tmp_path):
        manager = SpyManager()
        supervisor = CompanionWorkerSupervisor(manager, config=_config(tmp_path))
        try:
            spawn = supervisor.spawn(_spec(), job_id="job-w1")
            _modify_exit(spawn.process, kill=True)
            _wait_for(lambda: spawn.process.poll() is not None)
            supervisor.supervise_tick()
            _wait_for(lambda: manager.get("alice@default")["attempt_index"] == 1)
            current = manager.get("alice@default")
            assert current["state"] == "active"
            live = supervisor._workers["alice@default"]
            assert live.process.poll() is None
            assert live.command_attempt == 1
        finally:
            supervisor.stop_all()

    def test_restart_budget_exhaustion_blocks_spawn(self, tmp_path):
        manager = SpyManager()
        supervisor = CompanionWorkerSupervisor(manager, config=_config(tmp_path, attempts_max=2))
        try:
            spawn = supervisor.spawn(_spec(), job_id="job-w1")
            _modify_exit(spawn.process, kill=True)
            _wait_for(lambda: spawn.process.poll() is not None)
            supervisor.supervise_tick()
            second = supervisor._workers["alice@default"]
            _wait_for(lambda: second.command_attempt >= 1)
            # Heartbeat drives the drain transition once restart budget is at
            # the last attempt (attempt_index = attempts_max - 1 = 1).
            _wait_for(
                lambda: (
                    second.beats_seen >= 1 or manager.get("alice@default")["state"] == "draining"
                )
            )
            _modify_exit(second.process, kill=True)
            _wait_for(lambda: second.process.poll() is not None)
            supervisor.supervise_tick()
            exhausted = manager.get("alice@default")
            assert exhausted["state"] == "draining"
            with pytest.raises(WorkerSpawnError, match="draining"):
                supervisor.spawn(_spec(), job_id="job-w3")
        finally:
            supervisor.stop_all()


class TestWorkerEnvContract:
    def test_worker_env_carries_required_keys_and_no_auth_material(self, tmp_path):
        sent_env: dict[str, str] = {}

        manager = SpyManager()
        supervisor = CompanionWorkerSupervisor(manager, config=_config(tmp_path))
        try:
            spawn = supervisor.spawn(_spec(), job_id="job-w1")
            sent_env.update(spawn.env_snapshot)
        finally:
            supervisor.stop_all()
        assert sent_env.get("R4_COMPANION_WORKER") == "1"
        assert sent_env.get("R4_COMPANION_REGISTRY_KEY") == "alice@default"
        assert "SECRET" not in "".join(sent_env)
        assert "TOKEN" not in "".join(sent_env)
