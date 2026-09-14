"""Affinity-aware routing tests for ``run_workflow_record``.

Covers the runner's per-execution side of the affinity feature:
* The endpoint_id chosen for a parent step propagates to its children
  via ``AffinityHints``, so a dependent step running in a later wave
  prefers the same endpoint when the capability set has multiple
  satisfying candidates.
* Sibling branches that don't depend on each other do not share affinity
  state — each branch is judged on its own parent chain.
* ``routing_hints={"affinity": "off"}`` on a step disables the
  tiebreaker for that step only.

These tests exercise the runner end-to-end with a fake StepRouter whose
registry exposes two same-tier capability candidates, so the only thing
that can flip selection is affinity.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_router import StepRouter, reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0")
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


@pytest.fixture
def probe_module():
    mod_name = "_runner_affinity_probe"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    yield mod
    sys.modules.pop(mod_name, None)


def _entry(class_name: str) -> str:
    return f"_runner_affinity_probe:{class_name}"


def _two_endpoint_router() -> StepRouter:
    """Router with two same-tier vision endpoints (ep-1, ep-2).

    Without affinity, ``vlm-a`` (declaration order) wins, so the parent
    naturally lands on ``ep-1``. The test that wants the parent on
    ``ep-2`` flips the candidate order via a routing_hint or a different
    fixture; for this base helper we only need that BOTH endpoints
    satisfy the capability.
    """

    def find_models(caps: list[str], mt: str | None) -> list[dict[str, Any]]:
        return [
            {"name": "vlm-a", "scope": "system", "model_type": "vision"},
            {"name": "vlm-b", "scope": "system", "model_type": "vision"},
        ]

    def find_endpoint(model_name: str, *_a: Any, **_k: Any) -> dict[str, Any] | None:
        if model_name == "vlm-a":
            return {"litellm_url": "http://a", "job_id": "ep-1"}
        if model_name == "vlm-b":
            return {"litellm_url": "http://b", "job_id": "ep-2"}
        return None

    return StepRouter(find_models=find_models, find_endpoint=find_endpoint)


class _RecordingStep(Step):
    """Step that captures the routing decision the runner injected."""

    name = "rec"

    def setup(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(
            extra={
                "endpoint_id": (ctx.config.get("_routing") or {}).get("endpoint_id"),
                "affinity_used": (ctx.config.get("_routing") or {}).get("affinity_used"),
                "affinity_matched_parents": (ctx.config.get("_routing") or {}).get(
                    "affinity_matched_parents"
                ),
            }
        )


# ---------------------------------------------------------------------------
# 1. Parent endpoint propagates to dependent step.
# ---------------------------------------------------------------------------


def test_runner_propagates_parent_endpoint_to_dependent(probe_module, tmp_path, monkeypatch):
    """Parent lands on ep-1; child has two viable candidates and must
    pick ep-1 via affinity."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    probe_module._RecordingStep = _RecordingStep  # type: ignore[attr-defined]
    router = _two_endpoint_router()

    registry = get_execution_registry()
    record = registry.create(workflow_name="aff", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "aff",
            "steps": [
                {
                    "name": "parent",
                    "entrypoint": _entry("_RecordingStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
                {
                    "name": "child",
                    "entrypoint": _entry("_RecordingStep"),
                    "depends_on": ["parent"],
                    "requires": ["vision"],
                    "model_type": "vision",
                },
            ],
        },
        router=router,
    )

    assert record.status == "succeeded", record.error
    routing = record.results["_routing"]
    # Parent had no affinity input; declaration order picks ep-1.
    assert routing["parent"]["endpoint_id"] == "ep-1"
    assert routing["parent"]["affinity_used"] is False
    # Child saw parent on ep-1 -> affinity preserves ep-1 (a strong
    # affinity tie just means "same as parent"; declaration order would
    # also pick ep-1, so we additionally check the affinity flags).
    assert routing["child"]["endpoint_id"] == "ep-1"
    assert routing["child"]["affinity_used"] is True
    assert routing["child"]["affinity_matched_parents"] == ["parent"]


# ---------------------------------------------------------------------------
# 2. Unrelated branches don't share affinity.
# ---------------------------------------------------------------------------


def test_runner_skips_affinity_for_unrelated_branch(probe_module, tmp_path, monkeypatch):
    """A step that does not declare a parent must not pick up affinity
    from an unrelated step that ran earlier in another wave."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    probe_module._RecordingStep = _RecordingStep  # type: ignore[attr-defined]
    router = _two_endpoint_router()

    registry = get_execution_registry()
    record = registry.create(workflow_name="aff2", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "aff2",
            "steps": [
                {
                    "name": "left",
                    "entrypoint": _entry("_RecordingStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
                # ``right`` runs in the same wave as ``left`` (no
                # depends_on), so even though ``left`` will land on
                # ep-1, ``right`` has no parent and therefore no hints.
                {
                    "name": "right",
                    "entrypoint": _entry("_RecordingStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
            ],
        },
        router=router,
    )

    assert record.status == "succeeded", record.error
    routing = record.results["_routing"]
    assert routing["left"]["affinity_used"] is False
    assert routing["right"]["affinity_used"] is False
    assert routing["right"]["affinity_matched_parents"] == []


# ---------------------------------------------------------------------------
# 3. ``routing_hints.affinity = "off"`` disables affinity per step.
# ---------------------------------------------------------------------------


def test_routing_hints_off_disables_affinity_in_runner(probe_module, tmp_path, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    # Slice 6: this test verifies declaration-order tiebreak when
    # affinity is off. Disable the feedback signal so the parent's
    # realised exec_time_ms doesn't bias child selection — feedback
    # has its own dedicated tests.
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")

    probe_module._RecordingStep = _RecordingStep  # type: ignore[attr-defined]
    router = _two_endpoint_router()

    registry = get_execution_registry()
    record = registry.create(workflow_name="aff3", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "aff3",
            "steps": [
                {
                    "name": "parent",
                    "entrypoint": _entry("_RecordingStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
                {
                    "name": "child",
                    "entrypoint": _entry("_RecordingStep"),
                    "depends_on": ["parent"],
                    "requires": ["vision"],
                    "model_type": "vision",
                    "routing_hints": {"affinity": "off"},
                },
            ],
        },
        router=router,
    )

    assert record.status == "succeeded", record.error
    routing = record.results["_routing"]
    # Parent landed on ep-1; affinity is OFF for child -> child is
    # selected purely on declaration order. Both endpoints are viable,
    # so ep-1 still wins, but ``affinity_used`` MUST be False.
    assert routing["child"]["endpoint_id"] == "ep-1"
    assert routing["child"]["affinity_used"] is False
    assert routing["child"]["affinity_matched_parents"] == []


# ---------------------------------------------------------------------------
# 4. Outcome telemetry: exec_time_ms and outcome are recorded.
# ---------------------------------------------------------------------------


def test_runner_records_exec_time_outcome_in_routing_log(probe_module, tmp_path, monkeypatch):
    """The runner must annotate each routing entry with the realised
    exec_time_ms and an outcome string after the step finishes."""
    import time as _time

    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    class _SlowStep(Step):
        name = "slow"

        def run(self, ctx: StepContext) -> StepResult:
            _time.sleep(0.02)  # ensure measurable >0
            return StepResult(extra={"ok": True})

    probe_module._SlowStep = _SlowStep  # type: ignore[attr-defined]
    router = _two_endpoint_router()

    registry = get_execution_registry()
    record = registry.create(workflow_name="aff4", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "aff4",
            "steps": [
                {
                    "name": "slow",
                    "entrypoint": _entry("_SlowStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
            ],
        },
        router=router,
    )
    assert record.status == "succeeded", record.error
    entry = record.results["_routing"]["slow"]
    assert entry["outcome"] == "succeeded"
    # Sleep was 20ms; allow a generous lower bound.
    assert entry["exec_time_ms"] >= 10.0
    # Telemetry sanity: load/latency/queue scores all present (zeros
    # because endpoint metadata is empty in this fixture).
    assert "load_score" in entry
    assert "latency_score" in entry
    assert "queue_score" in entry


# ---------------------------------------------------------------------------
# 5. Feedback store integration (Slice 6).
# ---------------------------------------------------------------------------


def test_runner_records_outcome_into_feedback_store(probe_module, tmp_path, monkeypatch):
    """After a successful step, the chosen endpoint's EWMA must be set."""
    import time as _time

    from umrs.server.step_router import get_feedback_store

    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    class _SlowStep(Step):
        name = "slow"

        def run(self, ctx: StepContext) -> StepResult:
            _time.sleep(0.02)
            return StepResult(extra={"ok": True})

    probe_module._SlowStep = _SlowStep  # type: ignore[attr-defined]
    router = _two_endpoint_router()

    registry = get_execution_registry()
    record = registry.create(workflow_name="aff5", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "aff5",
            "steps": [
                {
                    "name": "slow",
                    "entrypoint": _entry("_SlowStep"),
                    "requires": ["vision"],
                    "model_type": "vision",
                },
            ],
        },
        router=router,
    )
    assert record.status == "succeeded", record.error
    # The runner records realised exec time into the feedback store
    # keyed by the chosen endpoint id.
    chosen_ep = record.results["_routing"]["slow"]["endpoint_id"]
    norm = get_feedback_store().get_norm(chosen_ep)
    assert norm > 0.0
