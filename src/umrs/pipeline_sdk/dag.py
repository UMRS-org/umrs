"""
DAG Scheduler for workflow steps.

Supports branching and parallel execution of steps based on
``depends_on`` declarations in ``StepSpec``.  Steps without explicit
dependencies default to linear chaining (previous step).

Usage::

    from umrs.pipeline_sdk.dag import DAGScheduler

    scheduler = DAGScheduler(workflow.steps)
    for level in scheduler.execution_levels():
        # All steps in `level` can run in parallel
        for spec in level:
            ...
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from umrs.pipeline_sdk.step import (
    Step,
    StepContext,
    StepResult,
    call_with_deadline,
)

if TYPE_CHECKING:
    from umrs.jobs.execution_trace_store import ExecutionTraceStore

logger = logging.getLogger(__name__)


class DAGValidationError(Exception):
    """Raised when the step DAG is invalid."""


@dataclass
class _StepNode:
    """Internal node representation for DAG scheduling."""

    name: str
    spec_index: int
    parents: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)


class DAGScheduler:
    """Topological scheduler for workflow steps.

    Resolves ``depends_on`` declarations into execution levels where
    all steps within a level are independent and can run concurrently.

    When no step has ``depends_on`` set, the scheduler falls back to
    simple linear ordering (each step depends on the previous one).

    Args:
        step_specs: The list of ``StepSpec`` from a ``WorkflowDefinition``.
    """

    def __init__(self, step_specs: list) -> None:
        self._specs = step_specs
        self._nodes: dict[str, _StepNode] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct the directed graph from StepSpec declarations."""
        # Create nodes
        for idx, spec in enumerate(self._specs):
            self._nodes[spec.name] = _StepNode(
                name=spec.name,
                spec_index=idx,
            )

        # Determine if any step has explicit depends_on
        has_explicit_deps = any(spec.depends_on is not None for spec in self._specs)

        if not has_explicit_deps:
            # Linear chain: each step depends on the previous one
            for i in range(1, len(self._specs)):
                prev = self._specs[i - 1].name
                curr = self._specs[i].name
                self._nodes[curr].parents.add(prev)
                self._nodes[prev].children.add(curr)
        else:
            # Explicit DAG mode
            for spec in self._specs:
                node = self._nodes[spec.name]
                if spec.depends_on is not None:
                    for dep_name in spec.depends_on:
                        if dep_name not in self._nodes:
                            raise DAGValidationError(
                                f"Step '{spec.name}' depends on unknown "
                                f"step '{dep_name}'"
                            )
                        node.parents.add(dep_name)
                        self._nodes[dep_name].children.add(spec.name)
                # depends_on=[] means root step (no parents)
                # depends_on=None treated as "unset" — in explicit mode,
                # steps without depends_on are also treated as roots

    def _kahn_drain(self, queue: deque, in_degree: dict[str, int]) -> int:
        """Drain the zero-in-degree frontier; return visited node count."""
        visited_count = 0
        while queue:
            name = queue.popleft()
            visited_count += 1
            for child in self._nodes[name].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return visited_count

    def _detect_cycle_members(self) -> list[str]:
        """Kahn traversal; return step names still carrying in-degree (cycle)."""
        in_degree: dict[str, int] = {
            name: len(node.parents) for name, node in self._nodes.items()
        }
        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        visited_count = self._kahn_drain(queue, in_degree)

        if visited_count == len(self._nodes):
            return []
        return [name for name, deg in in_degree.items() if deg > 0]

    def validate(self) -> None:
        """Check for cycles and unreachable steps.

        Raises:
            DAGValidationError: If the graph contains cycles or
                unreachable steps.
        """
        cycle_members = self._detect_cycle_members()
        if cycle_members:
            raise DAGValidationError(f"Cycle detected involving steps: {cycle_members}")

    def execution_levels(self) -> list[list[int]]:
        """Return step indices grouped by execution level.

        Level 0 contains steps with no dependencies (roots).
        Level N contains steps whose parents are all in levels < N.
        Steps within the same level can run in parallel.

        Returns:
            List of lists, where each inner list contains spec indices
            for steps that can execute concurrently.
        """
        self.validate()

        in_degree: dict[str, int] = {
            name: len(node.parents) for name, node in self._nodes.items()
        }
        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        levels: list[list[int]] = []

        while queue:
            level_names = list(queue)
            queue.clear()
            level_indices = [self._nodes[name].spec_index for name in level_names]
            levels.append(sorted(level_indices))

            for name in level_names:
                for child in self._nodes[name].children:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return levels

    def topological_order(self) -> list[int]:
        """Return step indices in a valid topological order.

        This is a flattened version of :meth:`execution_levels`.
        """
        return [idx for level in self.execution_levels() for idx in level]

    def parent_names(self, step_name: str) -> set[str]:
        """Return the set of parent step names for *step_name*."""
        return set(self._nodes[step_name].parents)

    def is_linear(self) -> bool:
        """Return True if the DAG is a simple linear chain.

        This is used to decide whether the legacy linear executor
        can be used (avoiding thread pool overhead).
        """
        return all(len(level) == 1 for level in self.execution_levels())


def execute_dag(
    scheduler: DAGScheduler,
    step_instances: list[Step],
    step_specs: list,
    *,
    s3: Any,
    input_bucket: str,
    input_keys: list[str],
    output_bucket: str,
    run_id: str,
    shared_metadata: dict[str, Any],
    trace_store: ExecutionTraceStore | None = None,
    job_id: str = "",
    execution_id: str = "",
) -> dict[str, StepResult]:
    """Execute steps according to DAG ordering.

    For linear DAGs, steps run sequentially. For DAGs with parallelism,
    steps within the same level run concurrently using a thread pool.

    Args:
        scheduler: Pre-validated DAGScheduler.
        step_instances: Loaded Step instances (same order as step_specs).
        step_specs: StepSpec list from workflow definition.
        s3: S3 client.
        input_bucket: Source bucket.
        input_keys: Initial S3 keys.
        output_bucket: Final output bucket.
        run_id: Unique run identifier.
        shared_metadata: Mutable metadata dict shared across steps.

    Returns:
        Mapping of step_name → StepResult for all executed steps.
    """
    levels = scheduler.execution_levels()
    results: dict[str, StepResult] = {}

    # Track current keys per step for fan-in/fan-out
    step_output_keys: dict[str, list[str]] = {}
    step_output_bucket: dict[str, str] = {}

    def _run_step(spec_idx: int) -> None:
        spec = step_specs[spec_idx]
        step = step_instances[spec_idx]
        parents = scheduler.parent_names(spec.name)
        current_keys, current_bucket = _step_inputs(
            parents=parents,
            input_bucket=input_bucket,
            input_keys=input_keys,
            step_output_keys=step_output_keys,
            step_output_bucket=step_output_bucket,
        )

        # Route all workflow step outputs to the dedicated output bucket.
        # Writing intermediate artifacts back into the watched input bucket
        # causes the workflow to recursively retrigger on its own outputs.
        step_out_bucket = output_bucket

        # Collect parent results for fan-in
        parent_results = {
            pname: results[pname] for pname in parents if pname in results
        }

        logger.info(
            "  step %d [%s]: %d keys → bucket=%s (parents=%s)",
            spec_idx + 1,
            spec.name,
            len(current_keys),
            step_out_bucket,
            list(parents) if parents else "root",
        )

        ctx = StepContext(
            s3=s3,
            input_bucket=current_bucket,
            input_keys=current_keys,
            output_bucket=step_out_bucket,
            config=spec.config,
            metadata=shared_metadata,
            run_id=run_id,
            parent_results=parent_results,
        )

        trace_id = _start_step_trace(
            trace_store,
            job_id=job_id or run_id,
            execution_id=execution_id or run_id,
            spec_name=spec.name,
            parents=parents,
            shared_metadata=shared_metadata,
        )
        result = _run_step_with_retry(
            step,
            ctx,
            spec,
            trace_store=trace_store,
            trace_id=trace_id,
            shared_metadata=shared_metadata,
        )

        results[spec.name] = result
        if result.output_keys:
            step_output_keys[spec.name] = result.output_keys
            step_output_bucket[spec.name] = step_out_bucket

    for level in levels:
        _dispatch_level(level, _run_step)

    return results


def _step_inputs(
    *,
    parents: set[str],
    input_bucket: str,
    input_keys: list[str],
    step_output_keys: dict[str, list[str]],
    step_output_bucket: dict[str, str],
) -> tuple[list[str], str]:
    """Fan-in: merge parent outputs into the step's (keys, bucket)."""
    if not parents:
        # Determine input keys: merge from all parents
        return list(input_keys), input_bucket
    merged_keys: list[str] = []
    merged_bucket = input_bucket
    for parent_name in parents:
        merged_keys.extend(step_output_keys.get(parent_name, input_keys))
        merged_bucket = step_output_bucket.get(parent_name, input_bucket)
    if not merged_keys:
        merged_keys = list(input_keys)
    return merged_keys, merged_bucket


def _start_step_trace(
    trace_store: ExecutionTraceStore | None,
    *,
    job_id: str,
    execution_id: str,
    spec_name: str,
    parents: set[str],
    shared_metadata: dict[str, Any],
) -> str:
    """Create + start a step trace when tracing is configured; else ''."""
    if not trace_store:
        return ""
    trace_id = str(uuid.uuid4())
    # Look up parent trace IDs for DAG lineage
    parent_trace_id = None
    if parents:
        first_parent = next(iter(parents))
        parent_trace_id = shared_metadata.get(f"_trace_id_{first_parent}")
    trace_store.create_trace(
        trace_id=trace_id,
        job_id=job_id,
        execution_id=execution_id,
        step_name=spec_name,
        step_type="pipeline_step",
        parent_trace_id=parent_trace_id,
    )
    trace_store.start_trace(trace_id)
    return trace_id


def _run_step_with_retry(
    step: Step,
    ctx: StepContext,
    spec: Any,
    *,
    trace_store: ExecutionTraceStore | None,
    trace_id: str,
    shared_metadata: dict[str, Any],
) -> StepResult:
    """Execute one step under its retry (+ W3 soft-deadline) policy.

    Trace state mirrors the attempt outcome: failed after the final
    attempt, completed with outputs on success.
    """
    last_exc: Exception | None = None
    for attempt in range(spec.max_retries + 1):
        try:
            result = call_with_deadline(
                lambda: step.run(ctx),
                timeout_seconds=spec.timeout_seconds,
                label=f"step '{spec.name}'",
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < spec.max_retries:
                import time

                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s. Retrying in %.1fs...",
                    spec.name,
                    attempt + 1,
                    spec.max_retries + 1,
                    exc,
                    spec.retry_delay,
                )
                time.sleep(spec.retry_delay)
            else:
                if trace_store and trace_id:
                    trace_store.fail_trace(trace_id, str(last_exc))
                raise last_exc from None

    if trace_store and trace_id:
        trace_store.complete_trace(trace_id, outputs_ref=result.to_dict())
        shared_metadata[f"_trace_id_{spec.name}"] = trace_id
    return result


def _dispatch_level(level: list[int], run_step: Any) -> None:
    """Run a wave: directly when single, via thread pool when parallel."""
    if len(level) == 1:
        # Single step — run directly (no thread overhead)
        run_step(level[0])
        return
    # Parallel execution within this level
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        futures = {pool.submit(run_step, idx): idx for idx in level}
        for future in as_completed(futures):
            future.result()  # re-raises exceptions
