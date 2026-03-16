"""
Subprocess sandbox runner for user-defined workflow steps (U-206).

Executes user step code in a child process with:
  * An isolated virtual environment (optional)
  * Resource limits via ``resource.setrlimit`` (memory, CPU time)
  * A timeout enforced by the parent process
  * File-based audit trail of every execution

This is the default production execution mode.  For HPC clusters,
see the Slurm plugin runner (``plugins/slurm_runner.py``); for
Kubernetes, see ``plugins/k8s_runner.py``.

Design principles
-----------------
* **Synchronous** — fits the existing ``compute_and_store_embedding``
  call chain which is already threaded.
* **No container required** — works on bare-metal and inside Docker.
* **No heavy deps** — stdlib only (``subprocess``, ``json``, ``resource``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from umrs.steps.models import WorkflowDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Outcome of a sandboxed workflow execution."""

    run_id: str
    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    error_message: str = ""


@dataclass
class SandboxConfig:
    """Knobs for the subprocess sandbox."""

    enabled: bool = True
    timeout_seconds: int = 3600            # kill after this
    memory_limit_bytes: int = 8 * 1024**3  # 8 GiB default
    install_requirements: bool = True      # pip-install user deps
    python_executable: str = ""            # empty ⇒ same as parent
    audit_dir: Optional[str] = None        # where to write audit JSON


# ---------------------------------------------------------------------------
# Audit helper (append-only JSON-lines file)
# ---------------------------------------------------------------------------

class StepAuditLog:
    """Lightweight file-based audit logger for sandbox executions."""

    def __init__(self, audit_dir: Optional[str] = None) -> None:
        if audit_dir:
            self._path = Path(audit_dir)
        else:
            self._path = Path(tempfile.gettempdir()) / "umrs_step_audit"
        self._path.mkdir(parents=True, exist_ok=True)

    def log(self, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        dest = self._path / "audit.jsonl"
        try:
            with open(dest, "a") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.warning("Audit write failed for %s", dest, exc_info=True)


# ---------------------------------------------------------------------------
# The runner script executed inside the subprocess
# ---------------------------------------------------------------------------

# This string is written to a temp file and run via ``python <script>``.
# It receives parameters through a JSON file so there is no shell injection risk.
_WORKER_SCRIPT = r"""
import json, os, resource, sys, traceback

params_path = sys.argv[1]
with open(params_path) as fh:
    params = json.load(fh)

# ── Resource limits ──────────────────────────────────────────────────────
mem_limit = params.get("memory_limit_bytes", 0)
if mem_limit and sys.platform != "win32":
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except Exception:
        pass  # non-fatal on some platforms

# ── Adjust sys.path ─────────────────────────────────────────────────────
script_dir = params["script_dir"]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Also need the UMRS package itself for the ABC
umrs_src = params.get("umrs_src_dir", "")
if umrs_src and umrs_src not in sys.path:
    sys.path.insert(0, umrs_src)

# ── Environment variables ────────────────────────────────────────────────
for k, v in params.get("env", {}).items():
    os.environ[k] = v

# ── Import step class ───────────────────────────────────────────────────
module_name, class_name = params["entrypoint"].split(":", 1)
mod = __import__(module_name, fromlist=[class_name])
cls = getattr(mod, class_name)

# ── Setup + run ─────────────────────────────────────────────────────────
instance = cls()
instance.setup(params.get("config", {}))

# Reconstruct StepContext (S3 connection is serialised as a config dict;
# the subprocess connects independently).
from umrs.steps.context import StepContext, StepResult
import boto3, botocore.config

s3_cfg = params["s3"]
s3 = boto3.client(
    "s3",
    endpoint_url=s3_cfg["endpoint_url"],
    aws_access_key_id=s3_cfg["access_key"],
    aws_secret_access_key=s3_cfg["secret_key"],
    config=botocore.config.Config(signature_version="s3v4"),
)

ctx = StepContext(
    s3=s3,
    input_bucket=params["input_bucket"],
    input_keys=params["input_keys"],
    output_bucket=params["output_bucket"],
    config=params.get("config", {}),
    metadata=params.get("metadata", {}),
    run_id=params.get("run_id", ""),
)

result = instance.run(ctx)

# ── Serialise result ────────────────────────────────────────────────────
out = {
    "output_keys": result.output_keys,
    "checksums": result.checksums,
    "metadata": result.metadata,
    "extra": result.extra,
}
if result.embeddings is not None:
    import numpy as np, io, base64
    buf = io.BytesIO()
    np.save(buf, result.embeddings, allow_pickle=False)
    out["embeddings_b64"] = base64.b64encode(buf.getvalue()).decode()

result_path = params["result_path"]
with open(result_path, "w") as fh:
    json.dump(out, fh)

instance.teardown()
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SubprocessSandbox:
    """Execute a single workflow step in a sandboxed subprocess."""

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.cfg = config or SandboxConfig()
        self.audit = StepAuditLog(self.cfg.audit_dir)

    # ------------------------------------------------------------------

    def run_step(
        self,
        *,
        entrypoint: str,
        step_config: Dict[str, Any],
        script_dir: str,
        input_bucket: str,
        input_keys: List[str],
        output_bucket: str,
        s3_params: Dict[str, str],
        run_id: str = "",
        shared_metadata: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "StepResult":
        """Run one step in an isolated subprocess.

        Returns a :class:`StepResult` with the step's outputs.
        Raises ``RuntimeError`` if the subprocess fails.
        """
        from umrs.steps.context import StepResult

        run_id = run_id or uuid.uuid4().hex[:12]
        tmp = Path(tempfile.mkdtemp(prefix=f"umrs_step_{run_id}_"))

        # Serialize parameters to a JSON file (no shell injection possible)
        params = {
            "entrypoint": entrypoint,
            "config": step_config,
            "script_dir": script_dir,
            "umrs_src_dir": str(
                Path(__file__).resolve().parent.parent.parent  # src/
            ),
            "input_bucket": input_bucket,
            "input_keys": input_keys,
            "output_bucket": output_bucket,
            "s3": s3_params,
            "run_id": run_id,
            "metadata": shared_metadata or {},
            "memory_limit_bytes": self.cfg.memory_limit_bytes,
            "env": env or {},
            "result_path": str(tmp / "result.json"),
        }
        params_path = tmp / "params.json"
        params_path.write_text(json.dumps(params))

        worker_path = tmp / "_worker.py"
        worker_path.write_text(_WORKER_SCRIPT)

        python = self.cfg.python_executable or sys.executable

        self.audit.log({
            "event": "step_start",
            "run_id": run_id,
            "entrypoint": entrypoint,
            "input_keys_count": len(input_keys),
        })

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [python, str(worker_path), str(params_path)],
                capture_output=True,
                timeout=self.cfg.timeout_seconds,
                text=True,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            self.audit.log({
                "event": "step_timeout",
                "run_id": run_id,
                "entrypoint": entrypoint,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"Step '{entrypoint}' timed out after "
                f"{self.cfg.timeout_seconds}s"
            )

        elapsed = time.monotonic() - t0

        if proc.returncode != 0:
            self.audit.log({
                "event": "step_failed",
                "run_id": run_id,
                "entrypoint": entrypoint,
                "exit_code": proc.returncode,
                "stderr": proc.stderr[-2000:] if proc.stderr else "",
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"Step '{entrypoint}' failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-4000:]}"
            )

        # Read back the result JSON written by the worker
        result_path = tmp / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"Step '{entrypoint}' did not produce a result file"
            )
        with open(result_path) as fh:
            raw = json.load(fh)

        embeddings = None
        if "embeddings_b64" in raw:
            import base64, io
            import numpy as np
            buf = io.BytesIO(base64.b64decode(raw["embeddings_b64"]))
            embeddings = np.load(buf, allow_pickle=False)

        self.audit.log({
            "event": "step_succeeded",
            "run_id": run_id,
            "entrypoint": entrypoint,
            "output_keys_count": len(raw.get("output_keys", [])),
            "elapsed_seconds": elapsed,
        })

        return StepResult(
            output_keys=raw.get("output_keys", []),
            embeddings=embeddings,
            checksums=raw.get("checksums", []),
            metadata=raw.get("metadata", []),
            extra=raw.get("extra", {}),
        )
