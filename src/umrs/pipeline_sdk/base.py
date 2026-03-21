"""
UMRS Pipeline SDK — Abstract base class for user-defined AI pipelines.

A Pipeline defines a standard lifecycle that all UMRS pipelines follow:

    configure  →  ingest  →  compute  →  collect  →  verify  →  present

Users subclass `Pipeline` and implement the hooks they need.  The `run()`
method orchestrates the full lifecycle with logging and error handling.

Two compute patterns are supported through the same interface:
  - **Embedding pipelines** (Slurm / vLLM / local):  upload data → UMRS server
    computes embeddings → results land in pgvector.
  - **Task-queue pipelines** (BentoML): upload data → submit task → poll
    status → retrieve results.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

class ComputeBackend(Enum):
    """How the pipeline dispatches computation."""
    UMRS_EMBEDDING = "umrs_embedding"   # Slurm / vLLM / local via UMRS server
    BENTOML_TASK = "bentoml_task"       # BentoML async task queue
    LOCAL = "local"                     # In-process (for dev / small datasets)


@dataclass
class PipelineConfig:
    """Configuration produced by ``Pipeline.configure()``.

    Fields that don't apply to a given pipeline may be left as ``None``.
    """
    name: str
    compute_backend: ComputeBackend

    # --- UMRS embedding backend ---
    user_config_path: Optional[str] = None      # path to user_*.json
    umrs_server_ip: Optional[str] = None        # ZMQ control plane host
    umrs_client_bin: Optional[str] = None       # path to client_umrs.py

    # --- BentoML task-queue backend ---
    bentoml_url: Optional[str] = None           # e.g. http://localhost:3000
    bentoml_endpoint: Optional[str] = None      # e.g. "infer"

    # --- S3 ---
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket: Optional[str] = None

    # --- Database ---
    db_host: Optional[str] = None
    db_port: int = 5432
    db_user: Optional[str] = None
    db_password: Optional[str] = None

    # --- RAG (optional) ---
    rag_url: Optional[str] = None

    # --- Compute limits ---
    timeout_seconds: float = 600.0
    poll_interval: float = 5.0

    # --- Free-form extra settings ---
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataItem:
    """One piece of input data that has been ingested.

    Attributes:
        key: Unique identifier (e.g. S3 key, file path).
        source: Where the data came from (local path, URL, ...).
        metadata: Free-form metadata (mime type, size, ...).
    """
    key: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskHandle:
    """A reference to a submitted computation.

    Attributes:
        task_id: Identifier returned by the compute backend.
        endpoint: Which endpoint/command was used.
        data_item: The input item this task processes.
        metadata: Extra info (slurm job id, bentoml endpoint, ...).
    """
    task_id: str
    endpoint: str
    data_item: Optional[DataItem] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """One output from the pipeline.

    Attributes:
        task_handle: The handle that produced this result.
        status: "success", "failure", "skipped".
        data: The actual output payload (model-specific dict).
        error: Error message if status == "failure".
    """
    task_handle: TaskHandle
    status: str = "success"
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ── Phase tracking ───────────────────────────────────────────────────────────

class Phase(Enum):
    CONFIGURE = "configure"
    INGEST = "ingest"
    COMPUTE = "compute"
    COLLECT = "collect"
    VERIFY = "verify"
    PRESENT = "present"


# ── Abstract base class ─────────────────────────────────────────────────────

class Pipeline(abc.ABC):
    """Abstract base class for UMRS pipelines.

    Subclass this and override the hooks you need.  At minimum you must
    implement ``configure``, ``ingest``, ``compute``, and ``collect``.
    The ``verify`` and ``present`` hooks have default no-op implementations.

    Example::

        class MyPipeline(Pipeline):
            def configure(self) -> PipelineConfig:
                return PipelineConfig(name="my-pipe", ...)

            def ingest(self, config):
                ...
                return [DataItem(key="file.txt", source="/data/file.txt")]

            def compute(self, config, items):
                ...
                return [TaskHandle(task_id="123", endpoint="embed")]

            def collect(self, config, handles):
                ...
                return [Result(task_handle=h, data={...})]

        MyPipeline().run()
    """

    def __init__(self):
        self._config: Optional[PipelineConfig] = None
        self._log = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ── Lifecycle hooks (override these) ─────────────────────────────────

    @abc.abstractmethod
    def configure(self) -> PipelineConfig:
        """Return a ``PipelineConfig`` that describes this pipeline.

        Called first.  Set up connections, read config files, and return
        the config object that will be passed to subsequent phases.
        """
        ...

    @abc.abstractmethod
    def ingest(self, config: PipelineConfig) -> List[DataItem]:
        """Upload / register input data and return ``DataItem`` handles.

        For embedding pipelines this typically uploads files to S3.
        For task-queue pipelines this may just list existing S3 keys.
        """
        ...

    @abc.abstractmethod
    def compute(self, config: PipelineConfig,
                items: List[DataItem]) -> List[TaskHandle]:
        """Submit computation and return ``TaskHandle`` references.

        - **Embedding backend**: trigger UMRS ``process_objects`` or
          ``recompute``.
        - **BentoML backend**: submit tasks via the task-queue client.
        """
        ...

    @abc.abstractmethod
    def collect(self, config: PipelineConfig,
                handles: List[TaskHandle]) -> List[Result]:
        """Wait for computation to finish and return ``Result`` objects.

        Implementations typically poll for status until all handles
        resolve to success or failure.
        """
        ...

    def verify(self, config: PipelineConfig,
               results: List[Result]) -> bool:
        """Optional quality check on results.  Return True if OK.

        Default implementation returns True if no results have
        status == "failure".
        """
        failures = [r for r in results if r.status == "failure"]
        if failures:
            self._log.warning(
                "%d / %d results failed", len(failures), len(results)
            )
            return False
        return True

    def present(self, config: PipelineConfig,
                results: List[Result]) -> None:
        """Optional visualization / export step.

        Default implementation prints a summary table.
        """
        from .helpers import format_results_table
        print(format_results_table(results))

    # ── Orchestrator ─────────────────────────────────────────────────────

    def run(self) -> List[Result]:
        """Execute the full pipeline lifecycle.

        Returns the list of ``Result`` objects, even if some failed.
        """
        start = time.monotonic()

        self._log_phase(Phase.CONFIGURE)
        config = self.configure()
        self._config = config

        self._log_phase(Phase.INGEST)
        items = self.ingest(config)
        self._log.info("Ingested %d item(s)", len(items))

        self._log_phase(Phase.COMPUTE)
        handles = self.compute(config, items)
        self._log.info("Submitted %d task(s)", len(handles))

        self._log_phase(Phase.COLLECT)
        results = self.collect(config, handles)
        successes = sum(1 for r in results if r.status == "success")
        self._log.info(
            "Collected %d result(s): %d success, %d other",
            len(results), successes, len(results) - successes,
        )

        self._log_phase(Phase.VERIFY)
        ok = self.verify(config, results)
        if ok:
            self._log.info("Verification passed")
        else:
            self._log.warning("Verification FAILED")

        self._log_phase(Phase.PRESENT)
        self.present(config, results)

        elapsed = time.monotonic() - start
        self._log.info("Pipeline %r finished in %.1fs", config.name, elapsed)
        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _log_phase(self, phase: Phase):
        self._log.info("─── %s ───", phase.value.upper())
