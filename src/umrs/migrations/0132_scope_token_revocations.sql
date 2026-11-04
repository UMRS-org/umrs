-- 0132: scope-token revocations (ADR-108 D-3, wave W2b).
--
-- A work level (a loop, a teamup run) acts for a user under a signed scope
-- token (umrs/scopes/tokens.py). Tokens are stateless; stopping a level
-- revokes its path and everything beneath it, so every front (they share
-- this database) refuses its token and its children's from then on.
--
-- A row revokes path_prefix for one user in one project: a token whose path
-- equals the prefix or starts with "<prefix>/" is refused. Rows are only
-- added; a token's own expiry ends it anyway (at most 72 h), so old rows can
-- be pruned by revoked_at without changing any decision.

CREATE TABLE IF NOT EXISTS scope_token_revocations (
    user_id     TEXT        NOT NULL,
    project_id  TEXT        NOT NULL,
    path_prefix TEXT        NOT NULL,
    revoked_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, project_id, path_prefix)
);

CREATE INDEX IF NOT EXISTS scope_token_revocations_revoked_at
    ON scope_token_revocations (revoked_at);
