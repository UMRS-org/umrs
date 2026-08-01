# 32 — Backbone Durability (UMRS, operator demo)

Operator-facing walkthrough of the 2026-08 durability wave: migrations,
factory-resolved backends, and restart recovery proofs.

Run against a UMRS deployment with Postgres configured (e.g. umrs-front).

## 1. Migration status

```bash
python -m umrs.migrations --db-url postgresql://user:pass@host:5432/umrs_db --status
# Expected on a rehearsed host: Current version 0064 / Pending migrations: 0
```

## 2. Which stores are durable?

Every research satellite follows the same factory: `pg|memory|auto`.
With `DB_HOST`/`DATABASE_URL` set, `auto` → Postgres; unset → in-memory.

```bash
python - <<'PY'
from umrs.scientific_memory import build_default_store
from umrs.connectors.sync_state import build_default_store as connectors_store
from umrs.workflow.pg_registry import build_execution_registry
print(type(build_default_store()).__name__)
print(type(connectors_store()).__name__)
print(type(build_execution_registry()).__name__)
PY
```

On umrs-front this prints `PgScientificMemoryStore`, `PgSyncStateStore`,
`PgWorkflowExecutionRegistry` — all durable because the factories resolved
through the shared pool.

## 3. Tables now covering the backbone

```bash
psql ... -d umrs_db -c "SELECT table_name FROM information_schema.tables
  WHERE table_schema='public' AND (table_name LIKE 'sat_%' OR table_name
  IN ('hypotheses','workflow_executions','connector_sync_state',
       'connector_seen_hashes','knowledge_artifacts'));"
```

## 4. Restart recovery proof

1. Trigger any workflow: `r4 workflow trigger <name>` (noted execution id)
2. Restart the UMRS server hosting it (compose restart).
3. `r4 workflow status <execution_id>` — record is read-through from
   `workflow_executions` and continues reporting timestamps.

## 5. Connector dead letters

Rerun `r4 data intake` after a failed connector to see the capped
replayable DLQ preserved on restart (in `connector_sync_state.dlq`).
