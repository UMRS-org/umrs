-- 0107: per-project data-class policy (ADR-062, gap row D-51).
--
-- Every project carries data_class ∈ {sensitive, non_sensitive}; the
-- default is the SAFE class (sensitive): inference on self-deployed
-- models only; public-kind sources refuse for that project's sessions
-- and jobs. Widening to non_sensitive is an explicit per-project act
-- attributable to its owner (the PATCH route is owner-or-admin).
--
-- Enforcement lands at the routing layer (the ADR's named follow-up);
-- this column is the policy object of record.

ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS data_class TEXT NOT NULL DEFAULT 'sensitive'
        CHECK (data_class IN ('sensitive', 'non_sensitive'));

COMMENT ON COLUMN projects.data_class IS
    'ADR-062 model locality: sensitive (default) = public-kind inference sources refused for this project''s sessions and jobs; widening is an explicit owner act.';
