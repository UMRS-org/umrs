"""UMRS client connector — wraps ``client_umrs.py`` subprocess calls."""

from __future__ import annotations

import json
import logging
import subprocess
import time

from umrs.shared.config import get_resolved_umrs_subdir, get_users_default_cfg_subdir

logger = logging.getLogger(__name__)


def _default_user_config_path() -> str:
    return get_resolved_umrs_subdir(get_users_default_cfg_subdir(), "user_rivault.json")


class UMRSClientConnector:
    """Drive the UMRS CLI from a pipeline.

    This connector invokes ``client_umrs.py`` as a subprocess, which is the
    standard way to interact with the UMRS ZMQ server from demo containers.

    Args:
        client_bin: Path to ``client_umrs.py``
            (default: ``/opt/umrs/client_umrs.py``).
        user_config: Path to the user JSON config file.
        server_ip: UMRS ZMQ server host (passed via ``--server-ip``
            if provided).
        container: If set, commands are prefixed with
            ``podman exec <container> ...`` so the connector can be used
            from the host.
    """

    def __init__(self, client_bin: str = "/opt/umrs/client_umrs.py",
                 user_config: str | None = None,
                 server_ip: str | None = None,
                 container: str | None = None):
        self.client_bin = client_bin
        self.user_config = user_config or _default_user_config_path()
        self.server_ip = server_ip
        self.container = container

    # ── public API ───────────────────────────────────────────────────────

    def upload_file(self, local_path: str, bucket_alias: str) -> str:
        """Upload a single file to the UMRS-managed S3 bucket."""
        self._run_client("upload_file", local_path, bucket_alias)
        return local_path

    def upload_dir(self, local_dir: str, bucket_alias: str) -> str:
        """Upload a directory of files."""
        self._run_client("upload_dir", local_dir, bucket_alias)
        return local_dir

    def trigger_process(self) -> None:
        """Trigger UMRS to process newly uploaded objects (not always needed;
        the server auto-detects new objects in many configurations)."""
        # The server detects new objects automatically on upload.
        # This is a no-op placeholder for pipelines that need explicit trigger.
        logger.info("Process trigger: UMRS auto-processes on upload")

    def get_status(self) -> dict:
        """Query the status of all tasks (``status all --output json``)."""
        out = self._run_client("--output", "json", "status", "all",
                               capture=True)
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {}

    def wait_for_embeddings(self, timeout: float = 600.0,
                            poll_interval: float = 10.0) -> bool:
        """Poll ``status all`` until all tasks finish.

        Returns True if all succeeded, False if any errored.
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                status = self.get_status()
            except Exception as exc:
                logger.debug("Status poll error: %s", exc)
                time.sleep(poll_interval)
                continue

            running, errors = self._parse_task_states(status)
            if running == 0 and errors == 0:
                logger.info("All embedding tasks completed")
                return True
            if running == 0 and errors > 0:
                logger.warning("%d task(s) ended in error", errors)
                return False

            elapsed = time.monotonic() - start
            logger.info("Tasks running: %d, errors: %d (%.0fs elapsed)",
                        running, errors, elapsed)
            time.sleep(poll_interval)

        logger.error("Embedding wait timed out after %.0fs", timeout)
        return False

    def export_embeddings(self, bucket_alias: str, local_path: str,
                          fmt: str = "csv",
                          profile: str = "minimal",
                          db_host: str | None = None,
                          db_user: str | None = None) -> str:
        """Export embeddings to a local file."""
        args = ["--no-check-server", "export_embeddings",
                bucket_alias, local_path, "--format", fmt, "--profile", profile]
        if db_host:
            args.extend(["--db-host", db_host])
        if db_user:
            args.extend(["--db-user", db_user])
        self._run_client(*args)
        return local_path

    # ── internal ─────────────────────────────────────────────────────────

    def _build_cmd(self, *args: str) -> list[str]:
        cmd = []
        if self.container:
            cmd = ["podman", "exec", self.container]
        cmd.extend(["python3", self.client_bin, "-c", self.user_config])
        if self.server_ip:
            cmd.extend(["--server-ip", self.server_ip])
        cmd.extend(args)
        return cmd

    def _run_client(self, *args: str, capture: bool = False) -> str:
        cmd = self._build_cmd(*args)
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("UMRS client error:\n%s", result.stderr)
            result.check_returncode()
        if capture:
            return result.stdout
        return ""

    @staticmethod
    def _parse_task_states(status: dict) -> tuple:
        """Return (running_count, error_count) from a status JSON payload."""
        running = 0
        errors = 0

        def _iter_threads(payload):
            summaries = payload.get("summaries")
            if isinstance(summaries, dict):
                for entry in summaries.values():
                    yield from entry.get("data", [])
            summary = payload.get("summary")
            if isinstance(summary, dict):
                data = summary.get("data")
                if isinstance(data, list):
                    yield from data

        for t in _iter_threads(status):
            s = str(t.get("status", "")).lower()
            if s == "running":
                running += 1
            elif s == "error":
                errors += 1
        return running, errors
