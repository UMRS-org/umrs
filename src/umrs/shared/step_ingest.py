"""Step-level ingest selection (ADR-105 D-10, slice A105-S7, 2026-11-24).

A durable workflow run auto-ingests its outputs into the project's knowledge
store (``spec["ingest_results"]``, ``jobs/adapters.enqueue_auto_ingest``).
Until this slice every stored blob of a run was ingested, and the per-step
``outputs[].ingest`` flag a workflow could declare was never read.

A step may now declare, in its ``workflow.json`` entry::

    "outputs": [{"name": "table", "path_or_ref": "results/", "ingest": true}]

The declared flags are the run's ingest allowlist:

* no step declares an ``ingest`` flag: every stored output is ingested, as
  before (no selection is recorded);
* any step declares one: only the keys of steps whose output says
  ``ingest: true`` are ingested; with ``path_or_ref`` only the step's keys
  equal to it or under it (``ref/…``). Every other stored output is kept,
  registered and listed on the job's receipt, but not ingested. A run
  whose flags admit nothing enqueues no ingest job.

The selection is the run's step output keys (``StepResult.output_keys``)
matched against those declarations; the runner records it as
``record.results["ingest_keys"]`` and ``jobs.workflow_run`` marks each
stored output that falls outside it ``"ingest": false``.

Run outputs are ``s3_uri`` references into the project's own store: they
need no egress, so the ADR-099 web policy and the site's URL allowlist
(``knowledge.ingest_web_policy``, ``knowledge.ingest_allowlist``) have
nothing to decide for them; they still bind URL and git ingest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

INGEST_KEYS_RESULT = "ingest_keys"


def _declared_outputs(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs = step.get("outputs")
    if not isinstance(outputs, list):
        return []
    return [o for o in outputs if isinstance(o, Mapping)]


def declares_ingest(steps: Iterable[Mapping[str, Any]]) -> bool:
    """Whether any step output carries an explicit ``ingest`` flag."""
    return any("ingest" in o for s in steps for o in _declared_outputs(s))


def _under(key: str, ref: str) -> bool:
    root = ref.rstrip("/")
    return key == ref or key == root or key.startswith(root + "/")


def _output_keys(output: Mapping[str, Any], keys: list[str]) -> set[str]:
    """The step keys one ``ingest: true`` output admits."""
    if output.get("ingest") is not True:
        return set()
    ref = output.get("path_or_ref")
    if not isinstance(ref, str) or not ref.strip():
        return set(keys)
    return {k for k in keys if _under(k, ref.strip())}


def _result_keys(result: Any) -> list[str]:
    keys = getattr(result, "output_keys", None) or []
    return [str(k) for k in keys]


def ingest_selection(
    steps: Iterable[Mapping[str, Any]], results: Mapping[str, Any]
) -> list[str] | None:
    """The output keys this run may ingest, or None (no flag declared: all).

    ``results`` maps a step name to its ``StepResult``.
    """
    steps = list(steps)
    if not declares_ingest(steps):
        return None
    selected: set[str] = set()
    for step in steps:
        keys = _result_keys(results.get(str(step.get("name", ""))))
        for output in _declared_outputs(step):
            selected |= _output_keys(output, keys)
    return sorted(selected)


def blob_ingest_flag(results: Mapping[str, Any] | None, key: str) -> dict[str, bool]:
    """``{"ingest": False}`` for a stored output outside the run's selection,
    else ``{}`` (the descriptor keeps its pre-selection shape)."""
    keys = (results or {}).get(INGEST_KEYS_RESULT)
    if not isinstance(keys, list) or key in keys:
        return {}
    return {"ingest": False}
