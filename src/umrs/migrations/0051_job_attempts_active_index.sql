-- 0051: Add index for active-attempt queries.
-- get_active_attempts() runs a full scan of job_attempts every poll cycle
-- (default 5 s) looking for status IN ('submitted', 'running'). This index
-- makes that query O(log N) instead of O(N).

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_job_attempts_active
    ON job_attempts (status, created_at)
    WHERE status IN ('submitted', 'running');

COMMENT ON INDEX idx_job_attempts_active IS
    'Partial index for active-attempt polling (get_active_attempts)';
