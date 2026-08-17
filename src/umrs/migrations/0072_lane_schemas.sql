-- 0072: S5 lane segregation scaffold — one Postgres schema per surface lane.
--
-- Wave: UMRS Stability & Governed Surface Alignment. The schemas are EMPTY
-- today; the S9 expand/contract migrations move lane-owned tables here
-- (ALTER TABLE ... SET SCHEMA, transaction-local per table). Creation is
-- additive and idempotent; nothing reads these until
-- UMRS_LANE_SCHEMAS_ENABLED=true (identity mapping to `public` before).
CREATE SCHEMA IF NOT EXISTS lane_stable;
CREATE SCHEMA IF NOT EXISTS lane_beta;
CREATE SCHEMA IF NOT EXISTS lane_internal;
