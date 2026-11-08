-- 0136: the signed intent of a run under the person's own HPC account
-- (ADR-112 wave B2, plan §5 B2.0 §10.4).
--
-- intent_b64 is the base64 of the exact payload bytes the person signed,
-- intent_signature its armored SSHSIG, intent_id sha256(payload) as hex. The
-- server never verifies the signature (the person's gate does, at each
-- step's submit); it keeps both so every step can hand them to the gate.
-- One run per intent: a unique partial index on intent_id. Nullable: every
-- other run carries none. record_json carries intent_id only (the run status
-- shows the id, never the payload or the signature).

ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS intent_b64 TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS intent_signature TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS intent_id CHAR(64);
CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_executions_intent_id
    ON workflow_executions (intent_id) WHERE intent_id IS NOT NULL;
