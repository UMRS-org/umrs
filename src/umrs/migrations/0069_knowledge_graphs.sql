-- Migration 0069: Knowledge graph registry + graph-aware storage + proposals
--
-- Phase 1 of the multi-graph capability (see docs/knowledge-graph-convergence-2026-08-14.md):
--   1. knowledge_graphs — first-class graph identity/registry
--      (identity, ownership, optional project association, lifecycle, provenance)
--   2. knowledge_nodes/knowledge_edges gain graph_id (scoped canonical storage)
--   3. knowledge_graph_proposals — staged graph-extension proposals with
--      approval/decision audit, consumed by the agent mutation guard
--
-- Backward compatibility:
--   * Existing rows have graph_id '' and are treated as members of the
--     project's DEFAULT graph (deterministic id from project_id) at read time.
--   * No backfill UPDATE is required; reads fall back for legacy rows.
--
-- No destructive deletion: graphs are archived (status), never dropped.

CREATE TABLE IF NOT EXISTS knowledge_graphs (
    graph_id         TEXT         PRIMARY KEY,
    name             TEXT         NOT NULL,
    description      TEXT         NOT NULL DEFAULT '',
    owner_id         TEXT         NOT NULL DEFAULT '',
    project_id       TEXT,
    kind             TEXT         NOT NULL DEFAULT 'project',
    status           TEXT         NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'archived')),
    ontology_id      TEXT,
    ontology_version TEXT,
    provenance       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kgraphs_project
    ON knowledge_graphs(project_id)
    WHERE project_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_kgraphs_status
    ON knowledge_graphs(status);

CREATE INDEX IF NOT EXISTS idx_kgraphs_name
    ON knowledge_graphs(lower(name));

COMMENT ON TABLE knowledge_graphs IS
    'First-class knowledge graph registry: identity/ownership/lifecycle/provenance. '
    'Project default graphs have deterministic ids (sha256 of project:<project_id>).';

-- ── Graph-aware node/edge storage ─────────────────────────────────────────

ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS graph_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_graph
    ON knowledge_nodes(graph_id);

ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS graph_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_knowledge_edges_graph
    ON knowledge_edges(graph_id);

-- ── Graph extension proposals (approval-gated mutations) ──────────────────

CREATE TABLE IF NOT EXISTS knowledge_graph_proposals (
    proposal_id         TEXT         PRIMARY KEY,
    target_graph_id     TEXT         NOT NULL REFERENCES knowledge_graphs(graph_id),
    reason              TEXT         NOT NULL DEFAULT '',
    missing_information JSONB        NOT NULL DEFAULT '[]'::jsonb,
    source_artifacts    JSONB        NOT NULL DEFAULT '[]'::jsonb,
    proposed_operations JSONB        NOT NULL DEFAULT '[]'::jsonb,
    created_by          TEXT         NOT NULL DEFAULT '',
    session_id          TEXT         NOT NULL DEFAULT '',
    status              TEXT         NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'declined', 'executed')),
    decided_by          TEXT         NOT NULL DEFAULT '',
    decided_at          TIMESTAMPTZ,
    execution           JSONB,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kgraph_proposals_target
    ON knowledge_graph_proposals(target_graph_id);

CREATE INDEX IF NOT EXISTS idx_kgraph_proposals_status
    ON knowledge_graph_proposals(status);

COMMENT ON TABLE knowledge_graph_proposals IS
    'Staged graph-extension proposals. Agent-originated graph mutations require an '
    'approved proposal (decision persisted here); execution provenance in execution JSONB.';
