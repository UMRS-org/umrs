-- 0112: skill review state — the D-27 skill-edit plane (merge-on-green).
--
-- Loop-proposed skill proposals (compile_knowledge_skills, V-09) land
-- disabled with a pending_review marker. This column is the review-plane
-- state: the proposal block (status + lineage) at compile time, merged with
-- the human decision block (decision, reviewer, decided_at, note, and the
-- approve's green test receipt) at review time. Approve = the merge: it is
-- the only transition that flips enabled to true, and the API gate refuses
-- it unless the caller carries a green test receipt (design suggestion 8:
-- loop-proposed skills merge only when tests pass).

ALTER TABLE skills
    ADD COLUMN IF NOT EXISTS review_state JSONB;

-- Backfill in-flight proposals so existing pending_review rows are reviewable
-- through the new plane (idempotent; decided rows keep their state).
UPDATE skills
    SET review_state = jsonb_build_object(
        'status', 'pending_review',
        'source_ir_ids', COALESCE(manifest->'source_ir_ids', '[]'::jsonb),
        'source_artifact_ids', COALESCE(manifest->'source_artifact_ids', '[]'::jsonb)
    )
    WHERE manifest->>'_compiled_skill' = 'true'
      AND enabled = FALSE
      AND review_state IS NULL;

COMMENT ON COLUMN skills.review_state IS
    'D-27 skill-edit plane: pending proposal block + the human review decision (approve requires a green test receipt — merge-on-green); approve is the only transition that enables a compiled proposal.';
