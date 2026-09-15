-- Migration 0047: Knowledge namespace isolation (MEM-C4)
--
-- Adds structured namespace registry and visibility policies.
-- Extends the namespace column from 0035_rag_namespace.sql with
-- project-scoped and experiment-scoped partitions.

CREATE TABLE IF NOT EXISTS knowledge_namespaces (
    namespace_id   VARCHAR(64)  PRIMARY KEY,
    name           TEXT         NOT NULL,
    scope          TEXT         NOT NULL CHECK (scope IN ('default', 'project', 'experiment', 'user')),
    project_id     TEXT,
    experiment_id  TEXT,
    owner_id       TEXT,
    description    TEXT         DEFAULT '',
    is_active      BOOLEAN      DEFAULT TRUE,
    item_count     INTEGER      DEFAULT 0,
    created_at     TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kns_project
    ON knowledge_namespaces(project_id)
    WHERE project_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_kns_scope
    ON knowledge_namespaces(scope);

CREATE INDEX IF NOT EXISTS idx_kns_active
    ON knowledge_namespaces(is_active)
    WHERE is_active = TRUE;

-- Visibility policies for jobs/workflows
CREATE TABLE IF NOT EXISTS namespace_visibility_policies (
    policy_id          VARCHAR(64)  PRIMARY KEY,
    job_id             TEXT,
    workflow_name      TEXT,
    visible_namespaces TEXT         DEFAULT '',
    include_default    BOOLEAN      DEFAULT TRUE,
    include_project    BOOLEAN      DEFAULT TRUE,
    include_experiment BOOLEAN      DEFAULT FALSE,
    created_at         TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_nvp_job
    ON namespace_visibility_policies(job_id)
    WHERE job_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_nvp_workflow
    ON namespace_visibility_policies(workflow_name)
    WHERE workflow_name IS NOT NULL;
