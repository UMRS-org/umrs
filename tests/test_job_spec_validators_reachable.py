"""U3 S1 dead code: every job-lane spec validator is reachable.

``_validate_job_spec`` runs only after ``_parse_job_type`` has accepted the
type, so a validator keyed by a type outside ``VALID_JOB_TYPES`` can never
run. Three such validators (knowledge_reindex, companion_worker,
research_loop) were removed; this keeps the table honest.
"""

from __future__ import annotations

from umrs.server.job_tracking_handler import _JOB_SPEC_VALIDATORS, VALID_JOB_TYPES


def test_every_validator_is_for_a_submittable_type():
    unreachable = {t.value for t in _JOB_SPEC_VALIDATORS} - VALID_JOB_TYPES
    assert not unreachable, f"validators that can never run: {sorted(unreachable)}"
