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
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from umrs.pipeline_sdk.dag import DAGScheduler, DAGValidationError
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
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

    __slots__ = ("name", "depends_on", "_dict")

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

    # ---- Validate entrypoint early (fail-fast for bad workflows) ---------
    # The executor will re-resolve the class itself; this just gives the
    # user a clean error before we touch the router on a hopeless step.
    try:
        _load_step_class(spec["entrypoint"])
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load step: {exc}"
        record.set_step(step_name, status="failed", error=msg, finished_at=time.time())
        return step_name, None, msg

    config = dict(spec.get("config") or {})
    config.setdefault("step_name", step_name)
    config.setdefault("_workflow_params", dict(record.params))
    for key, value in (record.params or {}).items():
        config.setdefault(key, value)

    # ---- Routing ----------------------------------------------------------
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
    except StepRoutingError as exc:
        msg = str(exc)
        record.set_step(step_name, status="failed", error=msg, finished_at=time.time())
        # Record a routing entry so callers can see why this step failed.
        with state_lock:
            routing_log[step_name] = {
                "resolved_via": "routing_error",
                "error": msg,
                "wave_index": wave_index,
            }
        return step_name, None, f"routing failed: {msg}"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Step %s router raised non-StepRoutingError: %s; using env fallback",
            step_name,
            exc,
        )
        decision = RoutingDecision(resolved_via="env_fallback_after_error")

    _apply_routing_decision_to_config(config, decision)
    decision_dict = decision.to_dict()
    decision_dict["wave_index"] = wave_index

    # ---- Slice 10: per-step image override -------------------------------
    # Lift ``routing_hints.image`` into the step context's metadata so
    # :class:`PodmanExecutor` can pick it up. The executor performs
    # the actual validation (empty/non-string → ``setup``-phase
    # error) so the runner stays a thin transport. Recording the hint
    # in the routing log here makes it easy to debug "which image did
    # this step run in" without correlating against workflow.json.
    image_hint = (
        spec.get("routing_hints", {}).get("image")
        if isinstance(spec.get("routing_hints"), dict)
        else None
    )
    if image_hint is not None:
        decision_dict["image"] = image_hint
    else:
        # Surface the deployment-wide default in the routing log too,
        # so every entry shows the image the step is expected to run
        # in (empty string when no PodmanExecutor is configured).
        decision_dict["image"] = os.getenv("UMRS_STEP_IMAGE", "")

    # ---- Slice 12: per-step GPU override ---------------------------------
    # Same pattern as image: lift ``routing_hints.gpu`` (bool) into
    # the step context's metadata so PodmanExecutor can decide per-step
    # whether to attach a GPU device. The routing log always records
    # the *resolved* value — i.e. the hint when present, else the
    # deployment-wide default (``UMRS_STEP_GPU``).
    routing_hints = (
        spec.get("routing_hints") if isinstance(spec.get("routing_hints"), dict) else {}
    )
    gpu_hint = routing_hints.get("gpu")
    if gpu_hint is not None:
        decision_dict["gpu"] = gpu_hint
    else:
        decision_dict["gpu"] = os.environ.get("UMRS_STEP_GPU", "0").strip() == "1"

    with state_lock:
        routing_log[step_name] = decision_dict

    # ---- Build context ----------------------------------------------------
    upstream_keys: list[str] = []
    for dep in spec.get("depends_on") or []:
        dep_result = parent_results_snapshot.get(dep)
        if dep_result is not None:
            upstream_keys.extend(dep_result.output_keys)

    metadata: dict[str, Any] = {}
    if image_hint is not None:
        metadata[EXECUTOR_IMAGE_METADATA_KEY] = image_hint
    if gpu_hint is not None:
        metadata[EXECUTOR_GPU_METADATA_KEY] = gpu_hint

    # ---- Slice 11: payload S3 hint -----------------------------------
    # When ``UMRS_STEP_PAYLOAD_S3=1`` the runner passes a structured
    # config dict to the executor so it can decide whether to upload
    # the pickle payload to S3 (for large payloads) or stick with the
    # local bind-mount (for small ones). The executor owns the
    # threshold logic; the runner is just a hint transport.
    payload_s3_enabled = False
    if os.environ.get("UMRS_STEP_PAYLOAD_S3", "").strip() == "1":
        payload_bucket = os.environ.get("UMRS_S3_BUCKET", "").strip()
        if payload_bucket:
            payload_s3_prefix = os.environ.get("UMRS_S3_PREFIX", "umrs").strip()
            metadata[EXECUTOR_PAYLOAD_S3_METADATA_KEY] = {
                "bucket": payload_bucket,
                "prefix": payload_s3_prefix,
            }
            payload_s3_enabled = True

    # Record payload transport intent in routing log.
    with state_lock:
        entry = routing_log.get(step_name)
        if entry is not None:
            if payload_s3_enabled:
                entry["payload_mode"] = "s3"
                entry["payload_s3_bucket"] = payload_bucket  # type: ignore[possibly-undefined]
                entry["payload_s3_prefix"] = payload_s3_prefix  # type: ignore[possibly-undefined]
            else:
                entry["payload_mode"] = "local"

    ctx = StepContext(
        s3=s3_client,
        input_bucket=bucket,
        input_keys=upstream_keys,
        output_bucket=bucket,
        config=config,
        metadata=metadata,
        run_id=record.execution_id,
        parent_results=dict(parent_results_snapshot),
    )

    # ---- Execute (Slice 8: pluggable executor) ---------------------------
    # The executor drives setup → run → teardown. It translates any
    # phase-specific failure into ``StepExecutionError`` so we have
    # uniform exception handling regardless of inproc / subprocess /
    # future RPC backends.
    run_started = time.monotonic()
    try:
        result = executor.execute(
            entrypoint=spec["entrypoint"],
            config=config,
            ctx=ctx,
        )
    except StepExecutionError as exc:
        logger.exception("Step %s crashed during %s()", step_name, exc.phase)
        msg = f"{exc.phase}() raised: {exc.message}"
        # setup() failures don't carry a routing outcome — they fail
        # before the step ever touched the endpoint. Only annotate the
        # routing log and feed the failure store on run/worker errors.
        if exc.phase != "setup":
            with state_lock:
                entry = routing_log.get(step_name)
                if entry is not None:
                    entry["exec_time_ms"] = (time.monotonic() - run_started) * 1000.0
                    entry["outcome"] = "failed"
            # Slice 7: feed the failure into the per-endpoint failure-rate
            # EWMA so the router will start avoiding flaky endpoints. The
            # latency EWMA is intentionally NOT touched — failures say
            # nothing useful about an endpoint's healthy throughput.
            if decision.endpoint_id:
                get_feedback_store().record_failure(decision.endpoint_id)
        record.set_step(step_name, status="failed", error=msg, finished_at=time.time())
        return step_name, None, msg
    exec_time_ms = (time.monotonic() - run_started) * 1000.0

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
    record.set_step(
        step_name,
        status="succeeded",
        finished_at=time.time(),
        progress=1.0,
        extra=_serialise_step_extra(result),
    )
    return step_name, result, None


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
    ``None`` it is built from ``UMRS_STEP_EXECUTOR`` (default
    ``inproc``). Tests inject a custom executor to assert isolation
    semantics without spawning real subprocesses.
    """
    record.update(status="running", started_at=time.time())

    steps_spec: list[dict[str, Any]] = list(workflow_json.get("steps", []))
    if not steps_spec:
        record.update(
            status="succeeded",
            finished_at=time.time(),
            progress=1.0,
        )
        return

    if router is None:
        router = build_default_router()
    if executor is None:
        executor = build_step_executor()

    try:
        levels, canonical_specs = _build_levels(steps_spec)
    except (DAGValidationError, ValueError) as exc:
        record.update(
            status="failed",
            error=str(exc),
            finished_at=time.time(),
        )
        return

    state_lock = threading.Lock()
    s3_client = _InMemoryS3Client(record, state_lock=state_lock)
    bucket = "wf-runner"
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
            _try_shutdown_executor(executor)
            return
        if first_error is not None:
            # A previous wave produced a failure: mark dependents skipped
            # without launching them.
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
            continue

        # Snapshot parent_results once per wave so siblings see a stable
        # view (they're independent by definition; this is mostly for
        # safety against mid-wave mutation).
        with state_lock:
            snapshot = dict(parent_results)

        wave_started = time.monotonic()
        wave_errors: list[str] = []

        if len(level) == 1:
            spec = level[0]
            name, _result, err = _execute_step(
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
            if err is not None:
                wave_errors.append(f"{name}: {err}")
            completed += 1
        else:
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
                    except Exception as exc:  # noqa: BLE001
                        # _execute_step is supposed to capture all
                        # exceptions internally; this branch is purely
                        # defensive.
                        logger.exception("Unexpected exception escaping _execute_step")
                        wave_errors.append(f"<unknown>: {exc}")
                        completed += 1
                        continue
                    if err is not None:
                        wave_errors.append(f"{name}: {err}")
                    completed += 1

        wave_duration_ms = (time.monotonic() - wave_started) * 1000.0
        logger.info(
            "workflow=%s wave=%d size=%d duration_ms=%.1f errors=%d",
            record.workflow_name,
            wave_index,
            len(level),
            wave_duration_ms,
            len(wave_errors),
        )
        record.update(progress=completed / total)

        if wave_errors and first_error is None:
            first_error = wave_errors[0]

    if first_error is not None:
        record.update(
            status="failed",
            error=first_error,
            finished_at=time.time(),
        )
        # Still persist whatever step results we have for debugging.
        with state_lock:
            record.results["steps"] = {
                name: _serialise_step_extra(res) for name, res in parent_results.items()
            }
        _try_shutdown_executor(executor)
        return

    # ---- Success: persist final outputs ----------------------------------
    with state_lock:
        record.results["steps"] = {
            name: _serialise_step_extra(res) for name, res in parent_results.items()
        }
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
    _try_shutdown_executor(executor)


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
