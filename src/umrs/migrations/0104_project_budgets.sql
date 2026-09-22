-- 0104: per-project loop budgets (gap row D-36, ADR-067).
--
-- Caps the autonomous loop may spend per project. Enforcement today is
-- wall-clock (measurable at every iteration boundary: the engine driver
-- stops the loop with state_reason budget_exhausted:wall_clock when the
-- elapsed time of a RUNNING loop exceeds max_wall_clock_hours).
-- max_gpu_hours is RECORDED now and enforced when job-cost accounting
-- lands (named leg B in ADR-067 — GPU usage is not yet attributable per
-- loop across shared clusters, so it is never silently approximated).

CREATE TABLE IF NOT EXISTS project_budgets (
    project_id          TEXT PRIMARY KEY,
    max_gpu_hours       REAL NOT NULL DEFAULT 0,
    max_wall_clock_hours REAL NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE project_budgets IS
    'Per-project autonomous-loop budgets (ADR-067): 0 = uncapped. Wall-clock enforced at iteration boundaries; GPU cap recorded, enforcement = leg B (job-cost accounting).';
