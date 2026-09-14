"""Lightweight in-process workflow runner.

Executes a workflow definition (parsed ``workflow.json``) using the
classes referenced by each step's ``entrypoint``.  Designed to support
the built-in templates (``VisionStep``, ``LLMStep``, ``OCRStep``)
without requiring a Slurm/K8s backend.

This runner is **process-local**.  Heavy / multi-node executions still
need to go through the durable Job/Attempt machinery.  The runner is
the path used when ``ResourceSpec.execution_mode`` is missing or
explicitly ``in_process``.

Each step receives a :class:`StepContext` whose S3 client is an
in-memory shim so the templates can write their JSON output keys
without configuring a real S3 endpoint.  The shim records every
uploaded key in the record's ``artifacts`` list and stashes the bytes
in ``record.results['blobs']`` for the API's ``/results`` endpoint to
serve back inline.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from umrs.metrics import METRICS
from umrs.pipeline_sdk.dag import DAGScheduler, DAGValidationError
from umrs.pipeline_sdk.step import (
    Step,
    StepContext,
    StepPolicy,
    StepResult,
    call_with_deadline,
)
from umrs.pipeline_sdk.workflow import StepSpec
from umrs.server.step_executor import (
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_PAYLOAD_S3_METADATA_KEY,
    StepExecutionError,
    StepExecutor,
    build_step_executor,
)
from umrs.server.step_router import (
    AffinityHints,
    RoutingDecision,
    StepRouter,
    StepRoutingError,
    build_default_router,
    get_feedback_store,
)
from umrs.server.workflow_execution_registry import ExecutionRecord

logger = logging.getLogger(__name__)


_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_MAX_PARALLEL_STEPS = 8
_IMPORT_LOCK = threading.Lock()


def _max_parallel_steps() -> int:
    """Return the configured cap on concurrent step executions per level."""
    raw = os.getenv("UMRS_RUNNER_MAX_PARALLEL_STEPS")
    if not raw:
        return _DEFAULT_MAX_PARALLEL_STEPS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid UMRS_RUNNER_MAX_PARALLEL_STEPS=%r; using default %d",
            raw,
            _DEFAULT_MAX_PARALLEL_STEPS,
        )
        return _DEFAULT_MAX_PARALLEL_STEPS
    return max(1, value)


# ----------------------------------------------------------------------
# In-memory S3 shim
# ----------------------------------------------------------------------


class _InMemoryS3Client:
    """Minimal stand-in that supports ``put_object`` / ``get_object``.

    Steps in this runner only need to upload small JSON / metadata
    objects; large artefacts should be written to a real S3 endpoint
    via the durable-job backend.  We deliberately keep this shim tiny.

    Thread-safety: under parallel level execution multiple steps may
    write blobs concurrently. ``put_object`` therefore guards the
    ``record.results['blobs']`` dict mutation with the runner-supplied
    lock. ``add_artifact`` is already lock-protected on the record.
    """

    def __init__(
        self,
        record: ExecutionRecord,
        *,
        state_lock: threading.Lock | None = None,
    ) -> None:
        self._record = record
        self._lock = state_lock or threading.Lock()
        record.results.setdefault("blobs", {})

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, **_: Any
    ) -> dict[str, Any]:
        full_key = f"{Bucket}/{Key}"
        with self._lock:
            self._record.results["blobs"][full_key] = Body
        self._record.add_artifact(full_key)
        return {"ETag": '"in-memory"', "VersionId": "0"}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        full_key = f"{Bucket}/{Key}"
        with self._lock:
            body = self._record.results.get("blobs", {}).get(full_key)
        if body is None:
            raise KeyError(full_key)

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(body)}

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        full_key = f"{Bucket}/{Key}"
        with self._lock:
            body = self._record.results.get("blobs", {}).get(full_key)
        if body is None:
            raise KeyError(full_key)
        return {"ContentLength": len(body)}


# ----------------------------------------------------------------------
# Step loading
# ----------------------------------------------------------------------


def _is_stale_module(module: Any, source_root: str) -> bool:
    """Whether *module* was loaded from under the upload tree."""
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str) and os.path.realpath(module_file).startswith(
        source_root
    ):
        return True
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return False
    try:
        return any(
            os.path.realpath(str(path_entry)).startswith(source_root)
            for path_entry in module_path
        )
    except TypeError:
        return False


def _purge_workflow_modules(source_path: str) -> None:
    """Drop cached Python modules loaded from *source_path*.

    Uploaded workflows can be re-uploaded with the same module names
    (for example ``opossum_step``).  Without clearing ``sys.modules``,
    the long-lived UMRS server process keeps executing stale code until
    restart.  Remove any module whose ``__file__`` or ``__path__`` lives
    under the uploaded workflow tree so the next import sees fresh code.
    """
    source_root = os.path.realpath(source_path)
    stale_names = [
        name
        for name, module in list(sys.modules.items())
        if _is_stale_module(module, source_root)
    ]
    for name in stale_names:
        sys.modules.pop(name, None)

    if stale_names:
        logger.info(
            "Purged %d cached workflow module(s) from %s",
            len(stale_names),
            source_root,
        )

    importlib.invalidate_caches()


def _load_step_class(entrypoint: str) -> type[Step]:
    if not _ENTRYPOINT_RE.match(entrypoint):
        raise ValueError(f"Invalid step entrypoint: {entrypoint!r}")
    module_name, class_name = entrypoint.split(":", 1)
    # Serialise concurrent imports: when independent steps in the same
    # DAG wave run on threads, two parallel ``importlib.import_module``
    # calls into the same package tree can deadlock Python's import
    # system. The lock is process-wide and only held for the brief
    # import + getattr; it does not affect step ``run()`` parallelism.
    with _IMPORT_LOCK:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"{class_name} not found in {module_name}")
    if not isinstance(cls, type) or not issubclass(cls, Step):
        raise TypeError(f"{entrypoint} does not point to a Step subclass")
    return cls


# ----------------------------------------------------------------------
# DAG ordering
# ----------------------------------------------------------------------


def _topo_sort(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topologically sort steps by ``depends_on``.

    Steps without explicit deps run in declaration order after all of
    their preceding peers — preserving backwards compatibility with
    workflow.json files that rely on declaration order.

    Kept for backwards compatibility with any external caller; the
    runner itself now uses :class:`umrs.pipeline_sdk.dag.DAGScheduler`
    so that independent steps in the same wave can run in parallel.
    """
    by_name: dict[str, dict[str, Any]] = {s["name"]: s for s in steps if s.get("name")}
    visited: dict[str, str] = {}  # name -> "temp" | "perm"
    ordered: list[dict[str, Any]] = []

    def _visit(node_name: str) -> None:
        state = visited.get(node_name)
        if state == "perm":
            return
        if state == "temp":
            raise ValueError(f"Cycle detected at step {node_name!r}")
        visited[node_name] = "temp"
        node = by_name.get(node_name)
        if node is not None:
            for dep in node.get("depends_on") or []:
                if dep in by_name:
                    _visit(dep)
            ordered.append(node)
        visited[node_name] = "perm"

    for s in steps:
        name = s.get("name")
        if name:
            _visit(name)
    return ordered


class _DAGAdapter:
    """Tiny duck-typed adapter exposing ``name`` and ``depends_on`` for the
    ``DAGScheduler``, which expects ``StepSpec``-shaped objects.

    We can't construct full ``StepSpec`` instances here because per-step
    validation (e.g. unknown ``model_type``) would abort the whole
    workflow before the runner gets a chance to record per-step failures
    in the same shape as today.
    """

    __slots__ = ("_dict", "depends_on", "name")

    def __init__(self, raw: dict[str, Any], synthetic_name: str) -> None:
        self.name: str = raw.get("name") or synthetic_name
        deps = raw.get("depends_on")
        # Preserve None vs [] distinction for DAGScheduler's "linear
        # fallback when no step declares depends_on" rule.
        self.depends_on: list[str] | None = (
            list(deps) if isinstance(deps, list) else None
        )
        self._dict = raw


def _build_levels(
    steps_spec: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return (levels, all_named_specs) computed from the workflow spec.

    Each ``levels[i]`` is the list of step dicts that can run concurrently
    in wave *i*. Steps without a ``name`` get a synthetic ``step_<idx>``
    name so the DAG layer can address them, matching the legacy runner
    behaviour where unnamed steps got the same fallback label.
    """
    adapters: list[_DAGAdapter] = []
    canonical_specs: list[dict[str, Any]] = []
    for idx, raw in enumerate(steps_spec):
        synthetic = f"step_{idx}"
        adapter = _DAGAdapter(raw, synthetic)
        # Stamp the resolved name back so downstream code can rely on it
        # without re-deriving the synthetic fallback.
        spec_with_name = dict(raw)
        spec_with_name["name"] = adapter.name
        adapters.append(adapter)
        canonical_specs.append(spec_with_name)

    scheduler = DAGScheduler(adapters)
    level_indices = scheduler.execution_levels()
    levels: list[list[dict[str, Any]]] = [
        [canonical_specs[idx] for idx in level] for level in level_indices
    ]
    return levels, canonical_specs


# ----------------------------------------------------------------------
# Retry / deadline policy (W3 — ADR-021)
# ----------------------------------------------------------------------


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed from a ``time.monotonic()`` start point."""
    return (time.monotonic() - started) * 1000.0


def _sleep_between_attempts(seconds: float, record: ExecutionRecord) -> bool:
    """Sleep between retry attempts, watching the record's cancel flag.

    Returns ``True`` when cancellation was requested mid-backoff.
    """
    deadline = time.monotonic() + seconds
    while True:
        if record.cancel_requested:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.5, remaining))


def _run_executor_attempt(
    executor: StepExecutor,
    *,
    entrypoint: str,
    config: dict[str, Any],
    ctx: StepContext,
    step_name: str,
    timeout_seconds: float | None,
) -> StepResult:
    """One executor attempt wrapped in the step's soft deadline.

    A deadline breach is normalised into a run-phase
    :class:`StepExecutionError` so the retry loop treats it like any
    other step failure (retriable, then terminal).
    """
    try:
        return call_with_deadline(
            lambda: executor.execute(entrypoint=entrypoint, config=config, ctx=ctx),
            timeout_seconds=timeout_seconds,
            label=f"step '{step_name}'",
        )
    except TimeoutError as exc:
        raise StepExecutionError("run", str(exc)) from exc


def _record_final_attempt_failure(
    exc: StepExecutionError,
    *,
    decision: RoutingDecision,
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
    run_started: float,
    step_name: str,
) -> None:
    """Annotate the routing log + feed the failure EWMA for a terminal failure.

    ``setup()`` failures occur before the step touched an endpoint, so
    they carry no routing outcome and are skipped (pre-W3 semantics).
    Failures say nothing useful about an endpoint's healthy
    throughput, so only the failure-rate EWMA is fed — the latency
    EWMA is intentionally untouched.
    """
    if exc.phase == "setup":
        return
    with state_lock:
        entry = routing_log.get(step_name)
        if entry is not None:
            entry["exec_time_ms"] = _elapsed_ms(run_started)
            entry["outcome"] = "failed"
    if decision.endpoint_id:
        get_feedback_store().record_failure(decision.endpoint_id)


def _execute_with_retry(
    *,
    executor: StepExecutor,
    spec: dict[str, Any],
    config: dict[str, Any],
    ctx: StepContext,
    record: ExecutionRecord,
    step_name: str,
    decision: RoutingDecision,
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
) -> tuple[StepResult | None, str | None, int, float]:
    """Drive one step through its retry + soft-deadline policy.

    Returns ``(result, error, attempts, elapsed_ms)`` — exactly one of
    ``result``/``error`` is non-None.

    Policy fields are read from the raw step spec under the same names
    :class:`umrs.pipeline_sdk.workflow.StepSpec` declares
    (``max_retries`` / ``retry_delay`` / ``timeout_seconds``), so the
    server runner and the ``WorkflowPipeline``/``dag`` executor honour
    one policy vocabulary (ADR-021, W3).

    Intermediate attempt failures do NOT touch the routing log or the
    failure EWMA — only the *final* failure feeds the flaky-endpoint
    signal, matching pre-W3 single-attempt semantics.
    """
    policy = StepPolicy.from_mapping(spec)
    max_attempts = policy.max_retries + 1
    retry_delay = policy.retry_delay
    timeout_seconds = policy.timeout_seconds
    last_msg = ""
    run_started = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            record.set_step(
                step_name,
                status="running",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "last_error": last_msg,
                },
            )
        try:
            result = _run_executor_attempt(
                executor,
                entrypoint=spec["entrypoint"],
                config=config,
                ctx=ctx,
                step_name=step_name,
                timeout_seconds=timeout_seconds,
            )
            return result, None, attempt, _elapsed_ms(run_started)
        except StepExecutionError as exc:
            last_msg = f"{exc.phase}() raised: {exc.message}"
            logger.exception(
                "Step %s crashed during %s() (attempt %d/%d)",
                step_name,
                exc.phase,
                attempt,
                max_attempts,
            )
            if attempt < max_attempts:
                logger.warning(
                    "Step %s attempt %d/%d failed: %s. Retrying in %.1fs...",
                    step_name,
                    attempt,
                    max_attempts,
                    last_msg,
                    retry_delay,
                )
                if not _sleep_between_attempts(retry_delay, record):
                    continue
                last_msg += "; cancel requested during retry backoff"
            if attempt > 1:
                last_msg = f"{last_msg} (after {attempt} attempts)"
            _record_final_attempt_failure(
                exc,
                decision=decision,
                routing_log=routing_log,
                state_lock=state_lock,
                run_started=run_started,
                step_name=step_name,
            )
            return None, last_msg, attempt, _elapsed_ms(run_started)

    raise RuntimeError("unreachable: retry loop always returns")


# ----------------------------------------------------------------------
# Per-step phase helpers (complexity decomposition of _execute_step)
# ----------------------------------------------------------------------


def _validate_step_entrypoint(spec: dict[str, Any]) -> str | None:
    """Fail-fast entrypoint load check. Returns error message or None."""
    try:
        _load_step_class(spec["entrypoint"])
    except Exception as exc:  # noqa: BLE001
        return f"Failed to load step: {exc}"
    return None


def _assemble_step_config(
    spec: dict[str, Any], record_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge step config with workflow params (params are defaults)."""
    config = dict(spec.get("config") or {})
    config.setdefault("step_name", spec["name"])
    config.setdefault("_workflow_params", dict(record_params))
    for key, value in (record_params or {}).items():
        config.setdefault(key, value)
    return config


def _resolve_step_decision(
    *,
    spec: dict[str, Any],
    config: dict[str, Any],
    step_name: str,
    wave_index: int,
    router: StepRouter,
    project_id: str | None,
    user_id: str | None,
    affinity_state: dict[str, str],
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
) -> tuple[RoutingDecision, str | None]:
    """Resolve the step's routing decision. Never raises.

    Returns ``(decision, error_msg)`` — ``StepRoutingError`` becomes a
    routing-error log entry plus error message; unexpected router
    exceptions degrade to the env fallback decision.
    """
    try:
        spec_with_config = dict(spec)
        spec_with_config["config"] = config
        step_spec_obj = StepSpec.from_dict(spec_with_config)
        # Build affinity hints from the endpoints the parent steps landed
        # on. Read under the shared lock so we observe a consistent view
        # of completed parents (parents are by definition in earlier
        # waves, so this is mostly a memory-visibility safeguard).
        parent_endpoint_ids: dict[str, str] = {}
        with state_lock:
            for parent in spec.get("depends_on") or []:
                ep_id = affinity_state.get(parent)
                if ep_id:
                    parent_endpoint_ids[parent] = ep_id
        affinity_hints = AffinityHints.from_step(
            step_spec_obj, parent_endpoint_ids=parent_endpoint_ids
        )
        decision = router.resolve(
            step_spec_obj,
            project_id=project_id,
            user_id=user_id,
            affinity_hints=affinity_hints,
        )
        return decision, None
    except StepRoutingError as exc:
        msg = str(exc)
        with state_lock:
            routing_log[step_name] = {
                "resolved_via": "routing_error",
                "error": msg,
                "wave_index": wave_index,
            }
        return RoutingDecision(resolved_via="routing_error"), msg
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Step %s router raised non-StepRoutingError: %s; using env fallback",
            step_name,
            exc,
        )
        return RoutingDecision(resolved_via="env_fallback_after_error"), None


def _apply_routing_overrides(
    spec: dict[str, Any], decision_dict: dict[str, Any]
) -> tuple[Any, Any]:
    """Slice 10/12: resolve per-step image + GPU overrides.

    Records the *resolved* values into ``decision_dict`` (routing-log
    entry) — the hint when present, else the deployment-wide default
    (``UMRS_STEP_IMAGE`` / ``UMRS_STEP_GPU``) — and returns
    ``(image_hint, gpu_hint)`` for metadata construction.
    """
    routing_hints = (
        spec.get("routing_hints") if isinstance(spec.get("routing_hints"), dict) else {}
    )
    image_hint = routing_hints.get("image")
    if image_hint is not None:
        decision_dict["image"] = image_hint
    else:
        decision_dict["image"] = os.getenv("UMRS_STEP_IMAGE", "")
    gpu_hint = routing_hints.get("gpu")
    if gpu_hint is not None:
        decision_dict["gpu"] = gpu_hint
    else:
        decision_dict["gpu"] = os.environ.get("UMRS_STEP_GPU", "0").strip() == "1"
    return image_hint, gpu_hint


def _collect_upstream_keys(
    spec: dict[str, Any], parent_results_snapshot: dict[str, StepResult]
) -> list[str]:
    """Fan-in: flatten output keys of all declared parents."""
    upstream_keys: list[str] = []
    for dep in spec.get("depends_on") or []:
        dep_result = parent_results_snapshot.get(dep)
        if dep_result is not None:
            upstream_keys.extend(dep_result.output_keys)
    return upstream_keys


def _payload_s3_hint(metadata: dict[str, Any]) -> tuple[bool, str, str]:
    """Slice 11: payload S3 hint. Returns ``(enabled, bucket, prefix)``.

    When ``UMRS_STEP_PAYLOAD_S3=1`` the runner passes a structured
    config dict to the executor so it can decide whether to upload
    the pickle payload to S3 (for large payloads) or stick with the
    local bind-mount (for small ones). The executor owns the
    threshold logic; the runner is just a hint transport.
    """
    if os.environ.get("UMRS_STEP_PAYLOAD_S3", "").strip() != "1":
        return False, "", ""
    payload_bucket = os.environ.get("UMRS_S3_BUCKET", "").strip()
    if not payload_bucket:
        return False, "", ""
    payload_prefix = os.environ.get("UMRS_S3_PREFIX", "umrs").strip()
    metadata[EXECUTOR_PAYLOAD_S3_METADATA_KEY] = {
        "bucket": payload_bucket,
        "prefix": payload_prefix,
    }
    return True, payload_bucket, payload_prefix


def _annotate_payload_mode(
    routing_log: dict[str, dict[str, Any]],
    step_name: str,
    state_lock: threading.Lock,
    enabled: bool,
    bucket: str,
    prefix: str,
) -> None:
    """Record payload transport intent in the routing log."""
    with state_lock:
        entry = routing_log.get(step_name)
        if entry is None:
            return
        if enabled:
            entry["payload_mode"] = "s3"
            entry["payload_s3_bucket"] = bucket
            entry["payload_s3_prefix"] = prefix
        else:
            entry["payload_mode"] = "local"


def _record_step_success(
    *,
    record: ExecutionRecord,
    step_name: str,
    result: StepResult,
    decision: RoutingDecision,
    exec_time_ms: float,
    attempt: int,
    parent_results: dict[str, StepResult],
    routing_log: dict[str, dict[str, Any]],
    affinity_state: dict[str, str],
    state_lock: threading.Lock,
) -> None:
    """Success bookkeeping: parent results, affinity, routing-log
    outcome, latency EWMA feedback, and the step record update."""
    with state_lock:
        parent_results[step_name] = result
        # Record the endpoint this step landed on so dependents can
        # prefer the same endpoint via affinity. Empty endpoint_id
        # (e.g. mock / env-fallback paths) is intentionally not stored
        # so it does not skew downstream selection.
        if decision.endpoint_id:
            affinity_state[step_name] = decision.endpoint_id
        # Outcome telemetry: feed real exec time back into the routing
        # log. Future slices can aggregate this per-endpoint to refine
        # the latency_score that the router consumes.
        entry = routing_log.get(step_name)
        if entry is not None:
            entry["exec_time_ms"] = exec_time_ms
            entry["outcome"] = "succeeded"

    # Feed the realised exec time into the process-wide feedback store
    # so the router's next decision for this endpoint is biased by
    # observed performance. Done OUTSIDE ``state_lock`` to keep the
    # critical section minimal — the store has its own lock.
    if decision.endpoint_id:
        get_feedback_store().record(decision.endpoint_id, exec_time_ms)

    extra = _serialise_step_extra(result)
    if attempt > 1:
        # Surface retry history on the successful record so `r4 workflow
        # status`/detail consumers can see how many attempts the step took.
        extra = dict(extra)
        extra["attempts"] = attempt
    record.set_step(
        step_name,
        status="succeeded",
        finished_at=time.time(),
        progress=1.0,
        extra=extra,
    )


# ----------------------------------------------------------------------
# Manual approval gates (W5)
# ----------------------------------------------------------------------

_GATE_DEFAULT_TIMEOUT_SECONDS = 86400.0


def _gate_timeout_seconds(spec: dict[str, Any]) -> float:
    """Approval deadline for a gate step (default 24h, env-overridable)."""
    raw = spec.get("gate_timeout_seconds")
    if raw is None:
        return float(
            os.environ.get(
                "UMRS_GATE_TIMEOUT_SECONDS", str(_GATE_DEFAULT_TIMEOUT_SECONDS)
            )
        )
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Step %s: invalid gate_timeout_seconds=%r; using default",
            spec.get("name"),
            raw,
        )
        return _GATE_DEFAULT_TIMEOUT_SECONDS


def _await_gate_decision(
    record: ExecutionRecord, event: threading.Event, timeout_seconds: float
) -> str:
    """Block on an approval event with cancel + deadline watch.

    Returns ``"approved"`` / ``"timeout"`` / ``"cancelled"``.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        if record.cancel_requested:
            return "cancelled"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        if event.wait(timeout=min(0.5, remaining)):
            return "approved"


def _execute_manual_gate(
    spec: dict[str, Any],
    *,
    wave_index: int,
    record: ExecutionRecord,
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
) -> tuple[str, StepResult | None, str | None]:
    """W5: manual approval gate (schema step ``gate: true``).

    No executor is invoked; the step parks the worker thread on a
    per-gate ``threading.Event`` until an approval decision arrives via
    ``POST /umrs/workflows/executions/{id}/steps/{step}/approve``, the
    approval deadline expires (timeout escalation → step failure), or
    the execution is cancelled. Decisions are recorded through
    ``record.resolve_gate`` so the audit trail lands in
    ``results.gate_audit`` (durable via the record's persist hook).
    Gate outcomes are intentionally NOT retried by the W3 retry loop.
    """
    step_name = spec["name"]
    timeout = _gate_timeout_seconds(spec)
    event = record.prepare_gate(step_name)
    record.set_step(step_name, status="waiting_approval")
    with state_lock:
        routing_log[step_name] = {
            "resolved_via": "manual_gate",
            "wave_index": wave_index,
            "gate_timeout_seconds": timeout,
        }

    wait_started = time.monotonic()
    outcome = _await_gate_decision(record, event, timeout)
    waited_seconds = time.monotonic() - wait_started
    if outcome != "approved":
        reason = (
            "cancelled while waiting for approval"
            if outcome == "cancelled"
            else f"approval deadline of {timeout:.0f}s expired"
        )
        msg = f"manual gate '{step_name}': {reason}"
        record.set_step(step_name, status="failed", error=msg, finished_at=time.time())
        return step_name, None, msg

    decision = dict(record.gate_decisions.get(step_name) or {})
    if decision.get("approved"):
        result = StepResult(
            output_keys=[],
            extra={"gate": decision, "waited_seconds": waited_seconds},
        )
        record.set_step(
            step_name,
            status="succeeded",
            finished_at=time.time(),
            progress=1.0,
            extra={"gate": decision, "waited_seconds": waited_seconds},
        )
        return step_name, result, None

    approver = decision.get("approver") or "unknown approver"
    comment = str(decision.get("comment") or "").strip()
    suffix = f": {comment}" if comment else ""
    msg = f"manual gate '{step_name}' rejected by {approver}{suffix}"
    record.set_step(step_name, status="failed", error=msg, finished_at=time.time())
    return step_name, None, msg


# ----------------------------------------------------------------------
# Per-step execution
# ----------------------------------------------------------------------


def _execute_step(
    spec: dict[str, Any],
    *,
    wave_index: int,
    record: ExecutionRecord,
    router: StepRouter,
    project_id: str | None,
    user_id: str | None,
    parent_results_snapshot: dict[str, StepResult],
    parent_results: dict[str, StepResult],
    routing_log: dict[str, dict[str, Any]],
    affinity_state: dict[str, str],
    s3_client: _InMemoryS3Client,
    bucket: str,
    state_lock: threading.Lock,
    executor: StepExecutor,
) -> tuple[str, StepResult | None, str | None]:
    """Execute a single step. Returns ``(step_name, result_or_None, error_or_None)``.

    All shared-state writes (``parent_results``, ``routing_log``) happen
    under ``state_lock``. ``record.set_step`` / ``record.update`` are
    already thread-safe via the record's own lock.

    The ``executor`` argument abstracts the ``setup → run → teardown``
    lifecycle (Slice 8). Default is in-process; setting
    ``UMRS_STEP_EXECUTOR=subprocess`` swaps in process isolation
    without changing any other behaviour.
    """
    step_name = spec["name"]
    record.set_step(step_name, status="running", started_at=time.time())

    # ---- W5: manual approval gate — no executor, wait for decision -------
    if spec.get("gate") or spec.get("manual_gate"):
        return _execute_manual_gate(
            spec,
            wave_index=wave_index,
            record=record,
            routing_log=routing_log,
            state_lock=state_lock,
        )

    # ---- Validate entrypoint early (fail-fast for bad workflows) ---------
    err = _validate_step_entrypoint(spec)
    if err is not None:
        record.set_step(step_name, status="failed", error=err, finished_at=time.time())
        return step_name, None, err

    config = _assemble_step_config(spec, record.params)

    # ---- Routing ----------------------------------------------------------
    decision, routing_err = _resolve_step_decision(
        spec=spec,
        config=config,
        step_name=step_name,
        wave_index=wave_index,
        router=router,
        project_id=project_id,
        user_id=user_id,
        affinity_state=affinity_state,
        routing_log=routing_log,
        state_lock=state_lock,
    )
    if routing_err is not None:
        record.set_step(
            step_name, status="failed", error=routing_err, finished_at=time.time()
        )
        return step_name, None, f"routing failed: {routing_err}"

    _apply_routing_decision_to_config(config, decision)
    decision_dict = decision.to_dict()
    decision_dict["wave_index"] = wave_index

    # ---- Slice 10/12: per-step image + GPU overrides ----------------------
    image_hint, gpu_hint = _apply_routing_overrides(spec, decision_dict)

    with state_lock:
        routing_log[step_name] = decision_dict

    # ---- Build context ----------------------------------------------------
    upstream_keys = _collect_upstream_keys(spec, parent_results_snapshot)
    metadata: dict[str, Any] = {}
    if image_hint is not None:
        metadata[EXECUTOR_IMAGE_METADATA_KEY] = image_hint
    if gpu_hint is not None:
        metadata[EXECUTOR_GPU_METADATA_KEY] = gpu_hint
    payload_s3_enabled, payload_bucket, payload_s3_prefix = _payload_s3_hint(metadata)
    _annotate_payload_mode(
        routing_log,
        step_name,
        state_lock,
        payload_s3_enabled,
        payload_bucket,
        payload_s3_prefix,
    )

    ctx = StepContext(
        s3=s3_client,
        input_bucket=bucket,
        input_keys=upstream_keys,
        output_bucket=bucket + "-output",
        # Keep source input_bucket for trigger objects; parent artifacts
        # live in the shared output bucket (common {input}-output pattern).
        parent_output_bucket=(bucket + "-output")
        if upstream_keys or (spec.get("depends_on") or [])
        else "",
        config=config,
        metadata=metadata,
        run_id=record.execution_id,
        parent_results=dict(parent_results_snapshot),
    )

    # ---- Execute (Slice 8: pluggable executor) ---------------------------
    # The executor drives setup → run → teardown. It translates any
    # phase-specific failure into ``StepExecutionError`` so we have
    # uniform exception handling regardless of inproc / subprocess /
    # future RPC backends. W3: per-step retry + soft deadline applied
    # here via ``_execute_with_retry`` (ADR-021 policy vocabulary).
    result, error, attempt, exec_time_ms = _execute_with_retry(
        executor=executor,
        spec=spec,
        config=config,
        ctx=ctx,
        record=record,
        step_name=step_name,
        decision=decision,
        routing_log=routing_log,
        state_lock=state_lock,
    )
    if error is not None:
        record.set_step(
            step_name, status="failed", error=error, finished_at=time.time()
        )
        return step_name, None, error
    assert result is not None  # error/result are mutex in _execute_with_retry

    _record_step_success(
        record=record,
        step_name=step_name,
        result=result,
        decision=decision,
        exec_time_ms=exec_time_ms,
        attempt=attempt,
        parent_results=parent_results,
        routing_log=routing_log,
        affinity_state=affinity_state,
        state_lock=state_lock,
    )
    return step_name, result, None


# ----------------------------------------------------------------------
# Runner phase helpers (complexity decomposition of run_workflow_record)
# ----------------------------------------------------------------------


def _emit_workflow_metric(
    record: ExecutionRecord, executor: StepExecutor | None, outcome: str
) -> None:
    """X1: workflow-run counter + end-to-end duration. Never raises."""
    try:
        name = type(executor).__name__ if executor is not None else "none"
        started = record.started_at or record.submitted_at
        METRICS.workflow_run(name, outcome, max(0.0, time.time() - started))
    except Exception:
        logger.debug("workflow metrics emission failed", exc_info=True)


def _prepare_source_path(source_path: str | None) -> None:
    """Insert the workflow source dir into ``sys.path`` + purge stale
    cached modules so ``_load_step_class`` sees fresh code."""
    if not source_path or not os.path.isdir(source_path):
        return
    abs_source = os.path.abspath(source_path)
    if abs_source not in sys.path:
        sys.path.insert(0, abs_source)
    _purge_workflow_modules(abs_source)


def _find_s3cfg() -> str | None:
    """Locate the S3 config file across the known paths."""
    candidates = [
        os.path.expanduser("~/.config/umrs/s3cfg"),
        "/etc/umrs/s3cfg",
        os.path.expanduser("~/.config/umrs/s3cfg_mnagaso"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _boto3_client_from_s3cfg(s3cfg_path: str, bucket: str) -> Any:
    """Build a boto3 S3 client from an s3cfg file."""
    import configparser

    import boto3

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(s3cfg_path)
    s3_host = config.get("default", "host_base", fallback="localhost")
    use_https = config.getboolean("default", "use_https", fallback=False)
    access_key = config.get("default", "access_key", fallback="")
    secret_key = config.get("default", "secret_key", fallback="")
    verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=verify_ssl,
    )
    logger.info("Using real S3 client (endpoint=%s, bucket=%s)", endpoint_url, bucket)
    return client


def _resolve_s3_client(
    record: ExecutionRecord, state_lock: threading.Lock
) -> tuple[Any, str]:
    """Pick the S3 client for this run; returns ``(client, bucket)``.

    A real bucket in workflow params triggers the boto3 path; any
    config problem degrades to the in-memory shim with the synthetic
    ``wf-runner`` bucket. ``bucket`` coming in is already the
    *physical* S3 bucket name — the client resolves the user-facing
    alias to ``{user}.{project}.{alias}`` before sending RUN_WORKFLOW.
    The step context derives the output bucket as ``<bucket>-output``,
    so the output bucket must be provisioned alongside the input bucket.
    """
    bucket = record.params.get("bucket", "wf-runner")
    if bucket == "wf-runner":
        return _InMemoryS3Client(record, state_lock=state_lock), bucket
    try:
        s3cfg_path = _find_s3cfg()
        if s3cfg_path is None:
            logger.warning("S3 config not found. Falling back to in-memory S3.")
        else:
            return _boto3_client_from_s3cfg(s3cfg_path, bucket), bucket
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to create real S3 client: %s. Falling back to in-memory.", exc
        )
    return _InMemoryS3Client(record, state_lock=state_lock), "wf-runner"


_EXECUTION_MODE_EXECUTOR = {
    "in_process": "inproc",
    "subprocess": "subprocess",
    "slurm": "slurm",
}


def _executor_name_from_execution_mode(mode: Any) -> str | None:
    """Map a workflow-level ``execution_mode`` to the step-executor name.

    The workflow's declared mode is authoritative for its own steps: it must
    not inherit the provisioning-level ``UMRS_STEP_EXECUTOR`` that
    ``SiteConfig.inject_env()`` plants in the long-lived server process
    (semantic collision — user runs failed at setup when the site kind is
    ``slurm`` but the (in-process) workflow lacked an image).  Unknown or
    absent modes keep the legacy env-based default.
    """
    if not isinstance(mode, str):
        return None
    return _EXECUTION_MODE_EXECUTOR.get(mode.strip().lower())


def _resolve_runner_defaults(
    router: StepRouter | None,
    executor: StepExecutor | None,
    name: str | None = None,
) -> tuple[StepRouter, StepExecutor]:
    """Fill router/executor with defaults when not injected (tests inject).

    ``name`` is the workflow-authoritative executor name (from
    ``execution_mode``); ``None`` preserves the env-derived selection.
    """
    if router is None:
        router = build_default_router()
    if executor is None:
        executor = build_step_executor(name=name)
    return router, executor


def _mark_wave_skipped(
    level: list[dict[str, Any]],
    *,
    wave_index: int,
    record: ExecutionRecord,
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
    completed: int,
    total: int,
) -> int:
    """Record a post-failure wave as skipped; advance progress."""
    for spec in level:
        _mark_skipped(
            spec,
            wave_index=wave_index,
            record=record,
            routing_log=routing_log,
            state_lock=state_lock,
        )
        completed += 1
        record.update(progress=completed / total)
    return completed


def _run_wave(
    *,
    wave_index: int,
    level: list[dict[str, Any]],
    record: ExecutionRecord,
    router: StepRouter,
    project_id: str | None,
    user_id: str | None,
    executor: StepExecutor,
    parent_results: dict[str, StepResult],
    routing_log: dict[str, dict[str, Any]],
    affinity_state: dict[str, str],
    s3_client: Any,
    bucket: str,
    state_lock: threading.Lock,
    max_workers_cap: int,
) -> list[str]:
    """Execute one DAG wave (directly when linear, pooled when parallel).

    Returns collected ``step: error`` strings. Sibling-tolerant: a step
    exception does not cancel in-flight siblings; the level barrier is
    awaited and the caller marks the workflow failed.
    """
    # Snapshot parent_results once per wave so siblings see a stable
    # view (they're independent by definition; this is mostly for
    # safety against mid-wave mutation).
    with state_lock:
        snapshot = dict(parent_results)
    wave_started = time.monotonic()
    wave_errors: list[str] = []

    if len(level) == 1:
        name, _result, err = _execute_step(
            level[0],
            wave_index=wave_index,
            record=record,
            router=router,
            project_id=project_id,
            user_id=user_id,
            parent_results_snapshot=snapshot,
            parent_results=parent_results,
            routing_log=routing_log,
            affinity_state=affinity_state,
            s3_client=s3_client,
            bucket=bucket,
            state_lock=state_lock,
            executor=executor,
        )
        if err is not None:
            wave_errors.append(f"{name}: {err}")
    else:
        _run_wave_parallel(
            level,
            snapshot=snapshot,
            wave_errors=wave_errors,
            max_workers_cap=max_workers_cap,
            wave_index=wave_index,
            record=record,
            router=router,
            project_id=project_id,
            user_id=user_id,
            executor=executor,
            parent_results=parent_results,
            routing_log=routing_log,
            affinity_state=affinity_state,
            s3_client=s3_client,
            bucket=bucket,
            state_lock=state_lock,
        )

    wave_duration_ms = (time.monotonic() - wave_started) * 1000.0
    logger.info(
        "workflow=%s wave=%d size=%d duration_ms=%.1f errors=%d",
        record.workflow_name,
        wave_index,
        len(level),
        wave_duration_ms,
        len(wave_errors),
    )
    return wave_errors


def _run_wave_parallel(
    level: list[dict[str, Any]],
    *,
    snapshot: dict[str, StepResult],
    wave_errors: list[str],
    max_workers_cap: int,
    wave_index: int,
    record: ExecutionRecord,
    router: StepRouter,
    project_id: str | None,
    user_id: str | None,
    executor: StepExecutor,
    parent_results: dict[str, StepResult],
    routing_log: dict[str, dict[str, Any]],
    affinity_state: dict[str, str],
    s3_client: Any,
    bucket: str,
    state_lock: threading.Lock,
) -> None:
    """Fan a parallel wave over the thread pool and collect errors."""
    workers = min(len(level), max_workers_cap)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=f"wf-{record.execution_id}-w{wave_index}",
    ) as pool:
        futures = [
            pool.submit(
                _execute_step,
                spec,
                wave_index=wave_index,
                record=record,
                router=router,
                project_id=project_id,
                user_id=user_id,
                parent_results_snapshot=snapshot,
                parent_results=parent_results,
                routing_log=routing_log,
                affinity_state=affinity_state,
                s3_client=s3_client,
                bucket=bucket,
                state_lock=state_lock,
                executor=executor,
            )
            for spec in level
        ]
        # Wait for *all* futures (sibling completion). We do not
        # cancel in-flight steps even if one raises; this keeps
        # any partial outputs consistent with what the step
        # already wrote to the record.
        for future in futures:
            try:
                name, _result, err = future.result()
            except Exception as exc:
                # _execute_step is supposed to capture all
                # exceptions internally; this branch is purely
                # defensive.
                logger.exception("Unexpected exception escaping _execute_step")
                wave_errors.append(f"<unknown>: {exc}")
                continue
            if err is not None:
                wave_errors.append(f"{name}: {err}")


def _finalize_run(
    record: ExecutionRecord,
    *,
    first_error: str | None,
    parent_results: dict[str, StepResult],
    levels: list[list[dict[str, Any]]],
    executor: StepExecutor,
    state_lock: threading.Lock,
) -> None:
    """Persist step outputs and set the terminal record status."""
    if record.cancel_requested and first_error is not None:
        # Cancel delivered during a failing wave (e.g. W5 gate cancel):
        # the cancel wins over the step failure for the terminal status.
        record.update(status="cancelled", error=first_error, finished_at=time.time())
        _emit_workflow_metric(record, executor, "cancelled")
        _try_shutdown_executor(executor)
        return
    with state_lock:
        record.results["steps"] = {
            name: _serialise_step_extra(res) for name, res in parent_results.items()
        }
    if first_error is not None:
        record.update(
            status="failed",
            error=first_error,
            finished_at=time.time(),
        )
        _emit_workflow_metric(record, executor, "failed")
        _try_shutdown_executor(executor)
        return

    with state_lock:
        last_level = levels[-1] if levels else []
        if last_level:
            # Pick a deterministic "final": last step in the last wave by
            # declaration order. With multiple terminal steps this matches
            # the legacy behaviour (which used the topo-sort tail).
            last_name = last_level[-1]["name"]
            last_result = parent_results.get(last_name)
            if last_result is not None:
                record.results["final"] = _serialise_step_extra(last_result)

    record.update(
        status="succeeded",
        progress=1.0,
        finished_at=time.time(),
    )
    _emit_workflow_metric(record, executor, "succeeded")
    _try_shutdown_executor(executor)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def run_workflow_record(
    record: ExecutionRecord,
    *,
    source_path: str | None,
    workflow_json: dict[str, Any],
    router: StepRouter | None = None,
    project_id: str | None = None,
    user_id: str | None = None,
    executor: StepExecutor | None = None,
) -> None:
    """Execute the workflow described by ``workflow_json`` in-process.

    Updates ``record`` as steps progress. Sets a terminal status
    (``succeeded`` / ``failed`` / ``cancelled``) before returning.

    Steps that share a wave (no ``depends_on`` ordering between them)
    run concurrently on a thread pool capped at
    ``UMRS_RUNNER_MAX_PARALLEL_STEPS`` (default 8). Failures are
    sibling-tolerant: a step exception does **not** cancel in-flight
    siblings; the level barrier is awaited, the workflow is marked
    ``failed``, and dependent steps in later waves are recorded as
    ``skipped`` rather than executed.

    ``router`` is the step-level resolver. When ``None`` a default router
    is constructed; that default has no registry hooks and falls back to
    env vars, which preserves today's behaviour for tests.

    ``executor`` is the per-step execution backend (Slice 8). When
    ``None`` it is built from ``UMRS_STEP_EXECUTOR`` (deployment override),
    else the workflow manifest ``execution_mode``, else ``inproc``. Tests
    inject a custom executor to assert isolation semantics without
    spawning real subprocesses.
    """
    record.update(status="running", started_at=time.time())
    _prepare_source_path(source_path)

    steps_spec: list[dict[str, Any]] = list(workflow_json.get("steps", []))
    if not steps_spec:
        record.update(
            status="succeeded",
            finished_at=time.time(),
            progress=1.0,
        )
        _emit_workflow_metric(record, executor, "succeeded")
        return

    worker = _executor_name_from_execution_mode(workflow_json.get("execution_mode"))
    router, executor = _resolve_runner_defaults(router, executor, name=worker)

    try:
        levels, canonical_specs = _build_levels(steps_spec)
    except (DAGValidationError, ValueError) as exc:
        record.update(
            status="failed",
            error=str(exc),
            finished_at=time.time(),
        )
        _emit_workflow_metric(record, executor, "failed")
        return

    state_lock = threading.Lock()
    s3_client, bucket = _resolve_s3_client(record, state_lock)

    parent_results: dict[str, StepResult] = {}
    routing_log: dict[str, dict[str, Any]] = {}
    # Per-execution map: step_name -> endpoint_id of the endpoint the
    # step ran on. Populated under ``state_lock`` after each step
    # succeeds; consumed by dependents to bias routing toward the same
    # endpoint (best-effort tiebreaker, not a constraint).
    affinity_state: dict[str, str] = {}
    record.results.setdefault("_routing", routing_log)

    total = sum(len(level) for level in levels)
    completed = 0
    max_workers_cap = _max_parallel_steps()
    first_error: str | None = None

    for wave_index, level in enumerate(levels):
        if record.cancel_requested:
            record.update(status="cancelled", finished_at=time.time())
            _emit_workflow_metric(record, executor, "cancelled")
            _try_shutdown_executor(executor)
            return
        if first_error is not None:
            # A previous wave produced a failure: mark dependents skipped
            # without launching them.
            completed = _mark_wave_skipped(
                level,
                wave_index=wave_index,
                record=record,
                routing_log=routing_log,
                state_lock=state_lock,
                completed=completed,
                total=total,
            )
            continue

        wave_errors = _run_wave(
            wave_index=wave_index,
            level=level,
            record=record,
            router=router,
            project_id=project_id,
            user_id=user_id,
            executor=executor,
            parent_results=parent_results,
            routing_log=routing_log,
            affinity_state=affinity_state,
            s3_client=s3_client,
            bucket=bucket,
            state_lock=state_lock,
            max_workers_cap=max_workers_cap,
        )
        completed += len(level)
        record.update(progress=completed / total)

        if wave_errors and first_error is None:
            first_error = wave_errors[0]

    _finalize_run(
        record,
        first_error=first_error,
        parent_results=parent_results,
        levels=levels,
        executor=executor,
        state_lock=state_lock,
    )


def _try_shutdown_executor(executor: StepExecutor) -> None:
    """Call ``shutdown()`` if the executor supports it (pool executors)."""
    shutdown = getattr(executor, "shutdown", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("executor shutdown failed: %s", exc)


def _mark_skipped(
    spec: dict[str, Any],
    *,
    wave_index: int,
    record: ExecutionRecord,
    routing_log: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
) -> None:
    """Mark a step as ``skipped`` because an earlier wave failed."""
    step_name = spec["name"]
    now = time.time()
    record.set_step(
        step_name,
        status="skipped",
        started_at=now,
        finished_at=now,
        error="skipped due to earlier failure",
    )
    with state_lock:
        routing_log[step_name] = {
            "resolved_via": "skipped",
            "wave_index": wave_index,
        }


def _serialise_step_extra(result: StepResult) -> dict[str, Any]:
    """Return a JSON-safe view of ``result.extra`` plus output keys."""
    safe_extra: dict[str, Any] = {}
    for key, value in (result.extra or {}).items():
        try:
            import json as _json

            _json.dumps(value, default=str)
            safe_extra[key] = value
        except Exception:  # noqa: BLE001
            safe_extra[key] = repr(value)
    return {"output_keys": list(result.output_keys or []), "extra": safe_extra}


def get_blob(record: ExecutionRecord, key: str) -> bytes | None:
    """Return raw bytes for an artifact key, or ``None`` if missing."""
    return record.results.get("blobs", {}).get(key)


def list_artifacts(record: ExecutionRecord) -> Iterable[tuple[str, int]]:
    """Yield ``(key, size_bytes)`` pairs for every uploaded artifact."""
    blobs: dict[str, bytes] = record.results.get("blobs", {}) or {}
    for key, body in blobs.items():
        yield key, len(body)


def _apply_routing_decision_to_config(
    config: dict[str, Any], decision: RoutingDecision
) -> None:
    """Merge a routing decision into the per-step config dict.

    Existing values in ``config`` win — workflow authors can still pin a
    specific ``api_base`` / ``model`` per step. The decision is stored
    under ``_routing`` (underscore prefix = internal) so newer templates
    can introspect *how* they were resolved.
    """
    config["_routing"] = decision.to_dict()
    if decision.api_base:
        config.setdefault("api_base", decision.api_base)
    if decision.api_key:
        config.setdefault("api_key", decision.api_key)
    if decision.model:
        config.setdefault("model", decision.model)
    if decision.headers:
        existing_headers = dict(config.get("headers") or {})
        for k, v in decision.headers.items():
            existing_headers.setdefault(k, v)
        config["headers"] = existing_headers
