"""Provenance diff engine (MEM-B7).

Compares two provenance records and their execution traces to produce
a structured diff showing what changed between two workflow runs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldDiff:
    """A single changed field."""
    field: str
    old: Any
    new: Any


@dataclass
class StepDiff:
    """Diff for a single execution step."""
    step_name: str
    status: str  # "added", "removed", "changed", "unchanged"
    field_diffs: list[FieldDiff] = field(default_factory=list)


@dataclass
class ProvenanceDiff:
    """Structured diff between two provenance records."""
    job_id_a: str
    job_id_b: str
    job_level_diffs: list[FieldDiff] = field(default_factory=list)
    step_diffs: list[StepDiff] = field(default_factory=list)
    artifact_diffs: list[FieldDiff] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id_a": self.job_id_a,
            "job_id_b": self.job_id_b,
            "job_level_diffs": [
                {"field": d.field, "old": d.old, "new": d.new}
                for d in self.job_level_diffs
            ],
            "step_diffs": [
                {
                    "step_name": sd.step_name,
                    "status": sd.status,
                    "field_diffs": [
                        {"field": d.field, "old": d.old, "new": d.new}
                        for d in sd.field_diffs
                    ],
                }
                for sd in self.step_diffs
            ],
            "artifact_diffs": [
                {"field": d.field, "old": d.old, "new": d.new}
                for d in self.artifact_diffs
            ],
            "summary": self.summary,
        }


# Fields to compare at the job/provenance level
_JOB_COMPARE_FIELDS = [
    "job_type", "status", "execution_authority", "restart_policy",
    "total_attempts",
]

_MODEL_COMPARE_FIELDS = ["name", "path", "version", "source"]


def compute_provenance_diff(
    prov_a: dict[str, Any],
    prov_b: dict[str, Any],
    traces_a: list[dict[str, Any]] | None = None,
    traces_b: list[dict[str, Any]] | None = None,
) -> ProvenanceDiff:
    """Compare two provenance records and optional execution traces.

    Args:
        prov_a: Provenance dict for job A (baseline).
        prov_b: Provenance dict for job B (comparison).
        traces_a: Execution traces for job A (optional).
        traces_b: Execution traces for job B (optional).

    Returns:
        ProvenanceDiff with structured comparison.
    """
    diff = ProvenanceDiff(
        job_id_a=prov_a.get("job_id", ""),
        job_id_b=prov_b.get("job_id", ""),
    )
    diff.job_level_diffs.extend(_job_level_diffs(prov_a, prov_b))
    diff.artifact_diffs.extend(_artifact_diffs(prov_a, prov_b))
    diff.step_diffs.extend(_step_diffs(traces_a, traces_b))
    diff.summary = _summary_counts(diff)
    return diff

def _job_level_diffs(prov_a: dict[str, Any], prov_b: dict[str, Any]) -> list[FieldDiff]:
    diffs = _changed_fields(prov_a, prov_b, _JOB_COMPARE_FIELDS)
    model_a = prov_a.get("model") or {}
    model_b = prov_b.get("model") or {}
    diffs.extend(_changed_fields(model_a, model_b, _MODEL_COMPARE_FIELDS, prefix="model."))
    diffs.extend(_changed_fields(prov_a, prov_b, ["total_duration_seconds"]))
    return diffs


def _changed_fields(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: list[str],
    *,
    prefix: str = "",
) -> list[FieldDiff]:
    return [
        FieldDiff(field=f"{prefix}{field}", old=left.get(field), new=right.get(field))
        for field in fields
        if left.get(field) != right.get(field)
    ]


def _artifact_diffs(prov_a: dict[str, Any], prov_b: dict[str, Any]) -> list[FieldDiff]:
    arts_a = _artifact_map(prov_a)
    arts_b = _artifact_map(prov_b)
    diffs: list[FieldDiff] = []
    for uri in sorted(set(arts_a) | set(arts_b)):
        artifact_diff = _artifact_diff_for_uri(uri, arts_a, arts_b)
        if artifact_diff is not None:
            diffs.append(artifact_diff)
    return diffs


def _artifact_map(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {artifact.get("uri", ""): artifact for artifact in provenance.get("artifacts", [])}


def _artifact_diff_for_uri(
    uri: str,
    arts_a: dict[str, dict[str, Any]],
    arts_b: dict[str, dict[str, Any]],
) -> FieldDiff | None:
    if uri in arts_a and uri not in arts_b:
        return FieldDiff(field=uri, old="present", new="absent")
    if uri not in arts_a and uri in arts_b:
        return FieldDiff(field=uri, old="absent", new="present")
    a_hash = arts_a[uri].get("sha256")
    b_hash = arts_b[uri].get("sha256")
    if a_hash and b_hash and a_hash != b_hash:
        return FieldDiff(field=f"{uri}:sha256", old=a_hash, new=b_hash)
    return None


def _step_diffs(
    traces_a: list[dict[str, Any]] | None,
    traces_b: list[dict[str, Any]] | None,
) -> list[StepDiff]:
    if traces_a is None or traces_b is None:
        return []
    steps_a = {trace.get("step_name", ""): trace for trace in traces_a}
    steps_b = {trace.get("step_name", ""): trace for trace in traces_b}
    return [_step_diff(step, steps_a, steps_b) for step in sorted(set(steps_a) | set(steps_b))]


def _step_diff(
    step: str,
    steps_a: dict[str, dict[str, Any]],
    steps_b: dict[str, dict[str, Any]],
) -> StepDiff:
    if step in steps_a and step not in steps_b:
        return StepDiff(step_name=step, status="removed")
    if step not in steps_a and step in steps_b:
        return StepDiff(step_name=step, status="added")
    field_diffs = _compare_trace_fields(steps_a[step], steps_b[step])
    status = "changed" if field_diffs else "unchanged"
    return StepDiff(step_name=step, status=status, field_diffs=field_diffs)


def _summary_counts(diff: ProvenanceDiff) -> dict[str, int]:
    status_counts = Counter(step_diff.status for step_diff in diff.step_diffs)
    return {
        "job_field_changes": len(diff.job_level_diffs),
        "artifact_changes": len(diff.artifact_diffs),
        "steps_changed": status_counts["changed"],
        "steps_added": status_counts["added"],
        "steps_removed": status_counts["removed"],
        "steps_unchanged": status_counts["unchanged"],
        "total_steps": len(diff.step_diffs),
    }


_TRACE_COMPARE_FIELDS = [
    "status", "step_type", "inputs_hash", "outputs_hash",
]


def _compare_trace_fields(
    ta: dict[str, Any], tb: dict[str, Any],
) -> list[FieldDiff]:
    """Compare two trace dicts and return field-level diffs."""
    diffs: list[FieldDiff] = []

    for f in _TRACE_COMPARE_FIELDS:
        old_val = ta.get(f)
        new_val = tb.get(f)
        if old_val != new_val:
            diffs.append(FieldDiff(field=f, old=old_val, new=new_val))

    # Compare model refs
    ma = ta.get("model_ref") or {}
    mb = tb.get("model_ref") or {}
    for f in _MODEL_COMPARE_FIELDS:
        old_val = ma.get(f)
        new_val = mb.get(f)
        if old_val != new_val:
            diffs.append(FieldDiff(field=f"model_ref.{f}", old=old_val, new=new_val))

    return diffs
