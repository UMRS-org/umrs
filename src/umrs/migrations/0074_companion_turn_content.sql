-- 0074: companion turn content plane (wave T3c).
--
-- The venue must persist what it executes: m5 stored only the message
-- hash, so attempts were un-executable after process loss, and the ADR-028
-- replay plane had no response body to stream. Adds the durable fields the
-- T3c roundtrip needs; all additive/idempotent.
ALTER TABLE companion_turns
    ADD COLUMN IF NOT EXISTS user_message TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS response_content TEXT,
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'ask';
