-- 0111: persisted per-project ask-policy default (D-33 tail, ADR-066).
--
-- Loops started without an explicit mode inherit the project's persisted
-- default (mode + trigger classes); an explicit request always wins.
-- Same owner-or-admin verb shape as data_class (the widening is an
-- explicit attributable act).

ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS ask_mode TEXT NOT NULL DEFAULT ''
        CONSTRAINT projects_ask_mode_check
        CHECK (ask_mode IN ('', 'auto', 'ask', 'profile')),
    ADD COLUMN IF NOT EXISTS pause_triggers JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN projects.ask_mode IS
    'D-33 persisted default: loops started without an explicit mode inherit it ('''' = request default wins); set via PATCH action=set-ask-policy (owner-or-admin).';
COMMENT ON COLUMN projects.pause_triggers IS
    'D-33 persisted default trigger classes for ask_mode=profile loops; the explicit request always overrides.';
