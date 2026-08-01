#!/usr/bin/env bash
# 11 — Backbone durability operator walkthrough. Requires a UMRS-shell host
# with psycopg2 available (e.g. umrs-front with the deployed checkout env).
set -euo pipefail

DB_URL="${DB_URL:?export DB_URL=postgresql://user:pass@host:5432/umrs_db}"

echo "=== 1. Migration status"
python -m umrs.migrations --db-url "$DB_URL" --status

echo "=== 2. Factory defaults (durable when DB env set)"
python - <<'PY'
from umrs.scientific_memory import build_default_store
from umrs.connectors.sync_state import build_default_store as connectors_store
print(type(build_default_store()).__name__)
print(type(connectors_store()).__name__)
PY

echo "=== 3. Backbone tables present"
psql "$DB_URL" -c "SELECT table_name FROM information_schema.tables
  WHERE table_schema='public' AND (table_name LIKE 'sat_%' OR table_name
  IN ('hypotheses','workflow_executions','connector_sync_state',
       'connector_seen_hashes','knowledge_artifacts'));"
