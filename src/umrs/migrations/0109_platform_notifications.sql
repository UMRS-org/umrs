-- 0109: durable platform notifications (D-36 leg B: admin notification on
-- budget stop; the web dashboard's notifications surface becomes store-backed).
--
-- One row per notification; read_at marks consumption (never a DELETE —
-- the row is the audit of what was surfaced, when it was surfaced).

CREATE TABLE IF NOT EXISTS platform_notifications (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    user_id     TEXT NOT NULL DEFAULT '',
    project_id  TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    read_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_platform_notifications_unread
    ON platform_notifications (user_id, project_id)
    WHERE read_at IS NULL;

COMMENT ON TABLE platform_notifications IS
    'Durable platform notifications (D-36 leg B): budget-exhausted stops and kindred operator-visible events; read_at marks consumption, rows are never deleted.';
