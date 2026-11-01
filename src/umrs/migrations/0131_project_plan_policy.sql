-- 0131: per-project plan -> commit policy (ADR-105 D-4, slice S3).
--
-- project_budgets (0104) gains the two settings the workflow commit route
-- reads (src/umrs/workflow_plans.py, read_project_plan_policy):
--
-- auto_commit_max_gpu_hours  NULL = no threshold: a commit flagged `auto`
--                            is refused (fail closed). Otherwise an `auto`
--                            commit passes only when the re-resolved plan's
--                            max_gpu_hours is <= this value.
-- require_plan               true = POST /umrs/workflows/{name}/trigger,
--                            run-definition triggers and the ZMQ
--                            RUN_WORKFLOW command are refused for this
--                            project; POST .../plan then .../commit is the
--                            only way to start a run.
--
-- A project without a project_budgets row has no threshold and
-- require_plan false. Idempotent (IF NOT EXISTS); no data rewrite.

ALTER TABLE project_budgets
    ADD COLUMN IF NOT EXISTS auto_commit_max_gpu_hours DOUBLE PRECISION NULL
        CHECK (auto_commit_max_gpu_hours IS NULL OR auto_commit_max_gpu_hours >= 0);

ALTER TABLE project_budgets
    ADD COLUMN IF NOT EXISTS require_plan BOOLEAN NOT NULL DEFAULT false;
