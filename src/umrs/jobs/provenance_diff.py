"""Provenance diff engine (MEM-B7).

Compares two provenance records and their execution traces to produce
a structured diff showing what changed between two workflow runs.
"""

from __future__ import annotations

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

    # Compare job-level fields
    for f in _JOB_COMPARE_FIELDS:
        old_val = prov_a.get(f)
        new_val = prov_b.get(f)
        if old_val != new_val:
            diff.job_level_diffs.append(FieldDiff(field=f, old=old_val, new=new_val))

    # Compare model refs
    model_a = prov_a.get("model") or {}
    model_b = prov_b.get("model") or {}
    for f in _MODEL_COMPARE_FIELDS:
        old_val = model_a.get(f)
        new_val = model_b.get(f)
        if old_val != new_val:
            diff.job_level_diffs.append(
                FieldDiff(field=f"model.{f}", old=old_val, new=new_val)
            )

    # Compare duration
    dur_a = prov_a.get("total_duration_seconds")
    dur_b = prov_b.get("total_duration_seconds")
    if dur_a != dur_b:
        diff.job_level_diffs.append(
            FieldDiff(field="total_duration_seconds", old=dur_a, new=dur_b)
        )

    # Compare artifacts
    arts_a = {a.get("uri", ""): a for a in prov_a.get("artifacts", [])}
    arts_b = {a.get("uri", ""): a for a in prov_b.get("artifacts", [])}

    for uri in sorted(set(arts_a) | set(arts_b)):
        if uri in arts_a and uri not in arts_b:
            diff.artifact_diffs.append(
                FieldDiff(field=uri, old="present", new="absent")
            )
        elif uri not in arts_a and uri in arts_b:
            diff.artifact_diffs.append(
                FieldDiff(field=uri, old="absent", new="present")
            )
        else:
            a_hash = arts_a[uri].get("sha256")
            b_hash = arts_b[uri].get("sha256")
            if a_hash and b_hash and a_hash != b_hash:
                diff.artifact_diffs.append(
                    FieldDiff(field=f"{uri}:sha256", old=a_hash, new=b_hash)
                )

    # Compare execution traces (step-level)
    if traces_a is not None and traces_b is not None:
        steps_a = {t.get("step_name", ""): t for t in traces_a}
        steps_b = {t.get("step_name", ""): t for t in traces_b}
        all_steps = sorted(set(steps_a) | set(steps_b))

        for step in all_steps:
            if step in steps_a and step not in steps_b:
                diff.step_diffs.append(StepDiff(step_name=step, status="removed"))
            elif step not in steps_a and step in steps_b:
                diff.step_diffs.append(StepDiff(step_name=step, status="added"))
            else:
                ta = steps_a[step]
                tb = steps_b[step]
                field_diffs = _compare_trace_fields(ta, tb)
                status = "changed" if field_diffs else "unchanged"
                diff.step_diffs.append(
                    StepDiff(step_name=step, status=status, field_diffs=field_diffs)
                )

    # Summary counts
    changed = sum(1 for sd in diff.step_diffs if sd.status == "changed")
    added = sum(1 for sd in diff.step_diffs if sd.status == "added")
    removed = sum(1 for sd in diff.step_diffs if sd.status == "removed")
    unchanged = sum(1 for sd in diff.step_diffs if sd.status == "unchanged")

    diff.summary = {
        "job_field_changes": len(diff.job_level_diffs),
        "artifact_changes": len(diff.artifact_diffs),
        "steps_changed": changed,
        "steps_added": added,
        "steps_removed": removed,
        "steps_unchanged": unchanged,
        "total_steps": len(diff.step_diffs),
    }

    return diff


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
