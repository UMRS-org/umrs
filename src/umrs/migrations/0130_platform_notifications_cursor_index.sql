-- 0130: platform_notifications cursor index (ADR-105 D-5, pushed job events).
--
-- Every live job watcher polls GET /umrs/jobs/events, which reads
--   WHERE user_id = $1 AND project_id = $2 AND id > $3 AND kind LIKE 'job.%'
--   ORDER BY id ASC LIMIT n             (NotificationStore.list_after)
--   SELECT MAX(id) ... same scope        (NotificationStore.max_id, tail=true)
-- job.* rows are cursor-read and never stamped read_at, so 0109's partial
-- index (WHERE read_at IS NULL) cannot serve these reads. This index does:
-- an equality prefix on (user_id, project_id), then id for the range/order.
--
-- Idempotent (IF NOT EXISTS). Retention of job.* rows is open (no policy yet).

CREATE INDEX IF NOT EXISTS platform_notifications_user_project_id_idx
    ON platform_notifications (user_id, project_id, id);
