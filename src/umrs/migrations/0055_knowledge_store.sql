-- Migration 0055: Knowledge store tables (TASK-002)
--
-- Creates the core knowledge persistence tables:
--   knowledge_artifacts     – content-addressed artifact store
--   knowledge_ir            – intermediate representations
--   knowledge_nodes         – graph nodes (extracted entities)
--   knowledge_edges         – graph edges (relationships)
--   verification_reports    – versioned verification reports
--   verification_findings   – findings with transition rules
--   readiness_approvals     – project readiness approval state
--   loop_iterations         – research loop iteration history
--   compiled_skills         – compiled skill definitions

-- =============================================================================
-- 1. Knowledge Artifacts (content-addressed storage metadata)
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_artifacts (
    artifact_id    TEXT             PRIMARY KEY,
    content_hash   TEXT             NOT NULL UNIQUE,
    kind           TEXT             NOT NULL CHECK (kind IN ('document', 'code', 'table', 'image', 'audio', 'video', 'embedding', 'other')),
    project_id     TEXT             NOT NULL,
    source_ref     TEXT             DEFAULT '',
    size_bytes     BIGINT           DEFAULT 0,
    mime_type      TEXT             DEFAULT '',
    status         TEXT             DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    metadata       JSONB            DEFAULT '{}',
    created_at     TIMESTAMPTZ      DEFAULT now(),
    updated_at     TIMESTAMPTZ      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artifact_project
    ON knowledge_artifacts(project_id, status);

CREATE INDEX IF NOT EXISTS idx_artifact_kind
    ON knowledge_artifacts(kind, project_id);

CREATE INDEX IF NOT EXISTS idx_artifact_hash
    ON knowledge_artifacts(content_hash) WHERE status = 'active';

COMMENT ON TABLE knowledge_artifacts IS 'Content-addressed artifact metadata. S3 storage keyed by content_hash.';

-- =============================================================================
-- 2. Knowledge Intermediate Representations (IRs)
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_ir (
    ir_id          TEXT             PRIMARY KEY,
    artifact_id    TEXT             NOT NULL REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE,
    version        INTEGER          DEFAULT 1,
    ir_type        TEXT             NOT NULL CHECK (ir_type IN ('code', 'document', 'schema', 'catalog', 'embedding')),
    content        JSONB            NOT NULL DEFAULT '{}',
    content_hash   TEXT             NOT NULL,
    size_bytes     BIGINT           DEFAULT 0,
    status         TEXT             DEFAULT 'current' CHECK (status IN ('current', 'archived', 'superseded')),
    created_at     TIMESTAMPTZ      DEFAULT now(),
    UNIQUE (artifact_id, version)
);

CREATE INDEX IF NOT EXISTS idx_ir_artifact
    ON knowledge_ir(artifact_id, status);

-- knowledge_ir carries no project_id column (see KnowledgeIRRow in
-- knowledge/store.py); index ir_type alone. The original (ir_type,
-- project_id) was authored against a column never created here.
CREATE INDEX IF NOT EXISTS idx_ir_type
    ON knowledge_ir(ir_type);

-- Unique guard for the ON CONFLICT (artifact_id, version) upsert path.
-- The CREATE TABLE above already declares it for fresh installations;
-- this index converges environments that created the table pre-0055.
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_ir_artifact_version_key
    ON knowledge_ir(artifact_id, version);

COMMENT ON TABLE knowledge_ir IS 'Versioned intermediate representations. ON CONFLICT DO NOTHING for idempotent inserts.';

-- =============================================================================
-- 3. Knowledge Graph Nodes
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    node_id        TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    label          TEXT             NOT NULL,
    node_type      TEXT             NOT NULL CHECK (node_type IN ('class', 'function', 'file', 'module', 'concept', 'entity', 'relationship')),
    content        JSONB            DEFAULT '{}',
    content_hash   TEXT             NOT NULL,
    source_ir      TEXT             REFERENCES knowledge_ir(ir_id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ      DEFAULT now(),
    updated_at     TIMESTAMPTZ      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_node_project
    ON knowledge_nodes(project_id, node_type);

CREATE INDEX IF NOT EXISTS idx_node_type
    ON knowledge_nodes(node_type);

CREATE INDEX IF NOT EXISTS idx_node_source_ir
    ON knowledge_nodes(source_ir) WHERE source_ir IS NOT NULL;

-- =============================================================================
-- 4. Knowledge Graph Edges
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_edges (
    edge_id        TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    source_id      TEXT             NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
    target_id      TEXT             NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
    relation_type  TEXT             NOT NULL,
    content        JSONB            DEFAULT '{}',
    created_at     TIMESTAMPTZ      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_edge_source
    ON knowledge_edges(source_id);

CREATE INDEX IF NOT EXISTS idx_edge_target
    ON knowledge_edges(target_id);

CREATE INDEX IF NOT EXISTS idx_edge_project
    ON knowledge_edges(project_id, relation_type);

-- =============================================================================
-- 5. Verification Reports (immutable, versioned)
-- =============================================================================
CREATE TABLE IF NOT EXISTS verification_reports (
    report_id      TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    version        INTEGER          NOT NULL,
    spec_name      TEXT             NOT NULL,
    confidence     NUMERIC(3, 2)    DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    total_checks   INTEGER          DEFAULT 0,
    passed_checks  INTEGER          DEFAULT 0,
    failed_checks  INTEGER          DEFAULT 0,
    skipped_checks INTEGER          DEFAULT 0,
    degraded       BOOLEAN          DEFAULT FALSE,
    degraded_reason TEXT,
    created_at     TIMESTAMPTZ      DEFAULT now(),
    UNIQUE (project_id, spec_name, version)
);

CREATE INDEX IF NOT EXISTS idx_report_project
    ON verification_reports(project_id, spec_name);

CREATE INDEX IF NOT EXISTS idx_report_confidence
    ON verification_reports(confidence) WHERE confidence < 0.8;

-- =============================================================================
-- 6. Verification Findings (transition rules)
-- =============================================================================
CREATE TABLE IF NOT EXISTS verification_findings (
    finding_key    TEXT             PRIMARY KEY,
    report_id      TEXT             NOT NULL REFERENCES verification_reports(report_id) ON DELETE CASCADE,
    artifact_id    TEXT             REFERENCES knowledge_artifacts(artifact_id),
    category       TEXT             NOT NULL,
    severity       TEXT             NOT NULL CHECK (severity IN ('blocking', 'warning', 'info')),
    status         TEXT             DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'resolved', 'waived')),
    title          TEXT             NOT NULL,
    description    TEXT             DEFAULT '',
    evidence       TEXT             DEFAULT '',
    waived_evidence TEXT,
    waiver_expires TIMESTAMPTZ,
    created_at     TIMESTAMPTZ      DEFAULT now(),
    updated_at     TIMESTAMPTZ      DEFAULT now(),
    UNIQUE (report_id, category, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_finding_status
    ON verification_findings(status);

CREATE INDEX IF NOT EXISTS idx_finding_severity
    ON verification_findings(severity) WHERE status = 'open';

CREATE INDEX IF NOT EXISTS idx_finding_project
    ON verification_findings(report_id, status);

-- =============================================================================
-- 7. Readiness Approvals
-- =============================================================================
CREATE TABLE IF NOT EXISTS readiness_approvals (
    approval_id    TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    threshold      NUMERIC(3, 2)    NOT NULL CHECK (threshold >= 0.0 AND threshold <= 1.0),
    approved_by    TEXT             NOT NULL,
    approved_at    TIMESTAMPTZ      DEFAULT now(),
    expires_at     TIMESTAMPTZ,
    reason         TEXT             DEFAULT '',
    audited        BOOLEAN          DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_approval_project
    ON readiness_approvals(project_id, approved_at);

CREATE INDEX IF NOT EXISTS idx_approval_active
    ON readiness_approvals(project_id, expires_at)
    WHERE expires_at IS NULL OR expires_at > now();

-- =============================================================================
-- 8. Loop Iterations
-- =============================================================================
CREATE TABLE IF NOT EXISTS loop_iterations (
    iteration_id   TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    loop_name      TEXT             NOT NULL,
    iteration_num  INTEGER          NOT NULL,
    plan_id        TEXT             DEFAULT '',
    workflow_run_ids TEXT[]         DEFAULT '{}',
    status         TEXT             DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'running', 'completed', 'failed', 'cancelled')),
    plan_snapshot  JSONB            DEFAULT '{}',
    emit_snapshot  JSONB            DEFAULT '{}',
    lint_result    JSONB            DEFAULT '{}',
    trigger_snapshot JSONB          DEFAULT '{}',
    collect_snapshot JSONB         DEFAULT '{}',
    ingest_result  JSONB            DEFAULT '{}',
    human_gate     BOOLEAN          DEFAULT FALSE,
    human_approved_by TEXT,
    created_at     TIMESTAMPTZ      DEFAULT now(),
    updated_at     TIMESTAMPTZ      DEFAULT now(),
    UNIQUE (project_id, loop_name, iteration_num)
);

CREATE INDEX IF NOT EXISTS idx_loop_project
    ON loop_iterations(project_id, loop_name);

CREATE INDEX IF NOT EXISTS idx_loop_status
    ON loop_iterations(status);

CREATE INDEX IF NOT EXISTS idx_loop_workflow
    ON loop_iterations(project_id, status)
    WHERE workflow_run_ids IS NOT NULL AND array_length(workflow_run_ids, 1) > 0;

-- =============================================================================
-- 9. Compiled Skills
-- =============================================================================
CREATE TABLE IF NOT EXISTS compiled_skills (
    skill_id       TEXT             PRIMARY KEY,
    project_id     TEXT             NOT NULL,
    skill_name     TEXT             NOT NULL,
    skill_version  TEXT             DEFAULT '1.0.0',
    compiled_from  TEXT[]           DEFAULT '{}',
    skill_content  JSONB            NOT NULL DEFAULT '{}',
    content_hash   TEXT             NOT NULL,
    status         TEXT             DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'archived')),
    created_at     TIMESTAMPTZ      DEFAULT now(),
    updated_at     TIMESTAMPTZ      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_skill_project
    ON compiled_skills(project_id, skill_name);

CREATE INDEX IF NOT EXISTS idx_skill_version
    ON compiled_skills(project_id, skill_name, skill_version);

-- =============================================================================
-- Migration complete
-- =============================================================================
