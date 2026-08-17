"""CompanionWorkerSupervisor — spawn + supervise per-(user, project) workers (T3b m4).

Venue per the design doc and m4 review guidance:
  * per-(user, project) long-lived worker, spawned behind the 0070 registry.
  * Control plane = per-worker Unix socket (UID-scoped dir, 0700) — same host
    as the supervisor (data0 single node reality); ZMQ reserved for
    cross-host boundaries.
  * Heartbeats are PUSHED by the worker over its socket; the supervisor
    consumes them and feeds CompanionWorkerManager.heartbeat() — one
    heartbeat mechanism at two grains (worker liveness here, turn attempts
    via CompanionTurnAttemptRegistry above).
  * Isolation model for T3b: logical (registry-key), NOT OS-setuid; socket
    permissions (0700 dir, worker-owned socket) are the boundary. OS-level
    setuid workers are a T4 hardening option, not m4's scope.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .companion_worker import (
    CompanionWorkerManager,
    CompanionWorkerSpec,
    WorkerNotFound,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 8
BEAT_EXPECT_MIN = 1  # beats expected before a worker counts as alive
SCANTIFY_TERM_TIME = 0.05  # supervisor tick between liveness checks
_SUN_PATH_LIMIT = 103  # macOS/Linux Unix-domain socket path max


class AtCapacity(RuntimeError):
    """Active workers already meet the cap; the surface translates to 429."""


class WorkerSpawnError(RuntimeError):
    """The process failed to launch or died during handshake."""


@dataclass
class SpawnConfig:
    """Spawn-time knobs; the socket directory root follows the data0 dev rule of
    a supervisor-owned staging area over /tmp when unset.
    """

    workers_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("UMRS_COMPANION_WORKERS_DIR", "/run/umrs/companion-workers")
        )
    )
    max_workers: int = field(
        default_factory=lambda: int(
            os.environ.get("UMRS_COMPANION_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))
        )
    )
    attempts_max: int = 6
    beat_interval_sec: float = 5.0
    term_grace_sec: float = 2.0
    respawn_jitter_sec: float = 1.0
    # Test seams — override command template and workers_root; never parsed here.
    command_template: tuple[str, ...] = (sys.executable, "-m", "umrs.jobs.worker_entry")


@dataclass
class _WorkerInstance:
    registry_key: str
    process: subprocess.Popen
    workdir: Path
    socket_path: Path
    command_attempt: int = 0
    env_snapshot: dict[str, str] = field(default_factory=dict)
    _beat_thread: Optional[threading.Thread] = None
    _beat_stop: threading.Event = field(default_factory=threading.Event)
    _server_sock: Optional[socket.socket] = None
    beats_seen: int = 0


class CompanionWorkerSupervisor:
    """Spawn + supervise per-(user, project) companion workers.

    The supervisor's lifecycle semantics:
      spawn → manager.register_active() (job_id FK) → worker connects socket,
      supervisor consumes beats, each beat → manager.heartbeat(attempt_index).
      Process exit → respawn under same manager (attempt_index += jitter
      backoff); after attempts_max-1 attempts the row drains per manager
      policy and the supervisor refuses further respawns.
    """

    def __init__(
        self,
        manager: CompanionWorkerManager,
        *,
        config: Optional[SpawnConfig] = None,
    ) -> None:
        self._manager = manager
        self._config = config or SpawnConfig()
        self._workers: dict[str, _WorkerInstance] = {}
        # Supervisor-owned tick thread isn't started by the base class — callers
        # drive supervise_tick() in tests/scripts; the runtime polls it.

    # -- spawning ------------------------------------------------------------

    def spawn(self, spec: CompanionWorkerSpec, *, job_id: str) -> _WorkerInstance:
        """spawn a worker; register active row; consume first beats."""
        key = spec.registry_key()
        manager_row = self._manager.get(key)
        if manager_row and manager_row.get("state") == "draining":
            raise WorkerSpawnError(f"worker {key} is draining (restart budget spent)")
        active_now = sum(1 for w in self._workers.values() if w.process.poll() is None)
        if active_now >= self._config.max_workers:
            raise AtCapacity(f"worker cap {self._config.max_workers} reached")

        row = manager_row or self._manager.register_active(spec, job_id=job_id)
        attempt_index = int(row.get("attempt_index") or 0)

        workdir = self._config.workers_root / secrets.token_hex(4)
        workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(workdir, 0o700)
        socket_path = self._socket_path(workdir)

        env = os.environ.copy()
        env.update(
            {
                "R4_COMPANION_WORKER": "1",
                "R4_COMPANION_REGISTRY_KEY": key,
                "R4_COMPANION_PROJECT_ID": spec.project_id,
                "R4_COMPANION_OWNER_USER_ID": spec.owner_user_id,
                "R4_COMPANION_SOCKET_PATH": str(socket_path),
                "R4_COMPANION_BEAT_SEC": str(self._config.beat_interval_sec),
                "R4_COMPANION_JOB_ID": job_id,
            }
        )
        try:
            process = subprocess.Popen(
                list(self._config.command_template),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:  # pragma: no cover — defensive seam
            shutil.rmtree(workdir, ignore_errors=True)
            raise WorkerSpawnError(str(exc)) from exc

        instance = _WorkerInstance(
            registry_key=key,
            process=process,
            workdir=workdir,
            socket_path=socket_path,
            command_attempt=attempt_index,
            env_snapshot={
                k: v for k, v in env.items() if k.startswith("R4_COMPANION_")
            },
        )
        self._workers[key] = instance
        self._start_beat_listener(instance)
        return instance

    # -- heartbeat plumbing (push model, one mechanism) -----------------------

    def _start_beat_listener(self, instance: _WorkerInstance) -> None:
        def _run() -> None:
            while not instance._beat_stop.is_set():
                try:
                    self._consume_one_beat(instance)
                except (OSError, TimeoutError):
                    time.sleep(SCANTIFY_TERM_TIME)

        thread = threading.Thread(
            target=_run,
            name=f"companion-beat-{instance.registry_key[:24]}",
            daemon=True,
        )
        instance._beat_thread = thread
        thread.start()

    def _consume_one_beat(self, instance: _WorkerInstance) -> None:
        server = self._ensure_server(instance)
        if server is None:
            time.sleep(SCANTIFY_TERM_TIME)
            return
        try:
            client, _ = server.accept()
        except (OSError, TimeoutError):
            return
        with client:
            data = client.recv(4096).decode("utf-8", errors="replace")
        if not data.strip():
            return
        # One beat consumed; manager heartbeat push with the per-instance
        # attempt index (restart count).
        instance.beats_seen += 1
        self._manager.heartbeat(
            instance.registry_key,
            attempt_index=instance.command_attempt,
            attempts_max=self._config.attempts_max,
        )

    def _socket_path(self, workdir: Path) -> Path:
        """Unix sockets cap at _SUN_PATH_LIMIT; long roots fall back to /tmp.

        macOS pytest tmpdirs routinely exceed the limit (the first wave of
        venue probes hang-to-timeout because of this silent bind failure),
        so clamp explicitly and keep the fallback path readable.
        """
        candidate = workdir / "worker.sock"
        if len(str(candidate)) <= _SUN_PATH_LIMIT:
            return candidate
        return Path(f"/tmp/cw-{secrets.token_hex(6)}.sock")

    def _ensure_server(self, instance: _WorkerInstance) -> Optional[socket.socket]:
        if instance._server_sock is not None:
            return instance._server_sock
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(instance.socket_path))
            # Socket is the per-worker control plane — 0700 dir + factory mode,
            # so only the supervisor's uid can reach it (documented boundary).
            os.chmod(instance.socket_path, stat.S_IRUSR | stat.S_IWUSR)
            server.listen(2)
            server.settimeout(0.2)
        except OSError as exc:
            logger.error(
                "companion supervisor: bind(%s) failed: %s",
                instance.socket_path,
                exc,
            )
            server.close()
            return None
        instance._server_sock = server
        return server

    # -- supervision / restart ------------------------------------------------

    def supervise_tick(self) -> None:
        """One pass: check for dead workers, respawn where budget remains."""
        for key, instance in list(self._workers.items()):
            if instance.process.poll() is None:
                continue
            self._shutdown_instance(
                instance, reason=f"exit {instance.process.returncode}"
            )
            row = self._fetch_row_or_none(key)
            if row is None or str(row.get("state") or "") != "active":
                continue
            spec = self._spec_from_row(row)
            try:
                time.sleep(
                    self._config.respawn_jitter_sec * (1 + instance.command_attempt)
                )
                next_instance = self.spawn(spec, job_id=str(row.get("job_id") or ""))
                next_instance.command_attempt = instance.command_attempt + 1
            except (AtCapacity, WorkerSpawnError) as exc:
                logger.warning("companion worker respawn blocked: %s", exc)

    def _spec_from_row(self, row: dict) -> CompanionWorkerSpec:
        return CompanionWorkerSpec(
            owner_user_id=str(row.get("owner_user_id") or ""),
            project_id=str(row.get("project_id") or ""),
            session_token_fingerprint=str(row.get("session_token_fingerprint") or ""),
        )

    def _fetch_row_or_none(self, registry_key: str) -> Optional[dict]:
        try:
            return self._manager.get(registry_key)
        except WorkerNotFound:
            return None

    def _shutdown_instance(self, instance: _WorkerInstance, *, reason: str) -> None:
        instance._beat_stop.set()
        if instance.process.poll() is None:
            try:
                os.killpg(instance.process.pid, signal.SIGTERM)
                instance.process.wait(timeout=self._config.term_grace_sec)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                with suppress(ProcessLookupError):
                    os.killpg(instance.process.pid, signal.SIGKILL)
        with suppress(OSError):
            if instance._server_sock is not None:
                instance._server_sock.close()
        with suppress(OSError):
            instance.socket_path.unlink(missing_ok=True)
        shutil.rmtree(instance.workdir, ignore_errors=True)
        logger.info(
            "companion worker shutdown: %s (%s)",
            instance.registry_key,
            reason,
        )

    # -- lifecycle / capacity reporting --------------------------------------

    def stop_all(self) -> None:
        for instance in list(self._workers.values()):
            self._shutdown_instance(instance, reason="supervisor stop_all")
        self._workers.clear()

    def stop_worker(self, registry_key: str) -> None:
        instance = self._workers.pop(registry_key, None)
        if instance is not None:
            self._shutdown_instance(instance, reason="supervisor stop_worker")
        self._manager.evict(registry_key)

    @property
    def active_count(self) -> int:
        return sum(1 for w in self._workers.values() if w.process.poll() is None)

    def at_capacity(self) -> bool:
        return self.active_count >= self._config.max_workers
