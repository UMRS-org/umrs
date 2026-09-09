"""Retention sweep entrypoint (R3-2): `python -m umrs.security.retention`.

Run from cron/ops cadence against the shared pool. Env:
    UMRS_AUDIT_RETENTION_DAYS (default 90)   — user_source_audit + secret_audit TTL
    UMRS_GRANT_TOMBSTONE_RETENTION_DAYS (default 180) — source_grants tombstone TTL
    UMRS_AUDIT_EXPORT_DIR (optional)         — JSONL export before purge
No credential material in any row purged or exported.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _export_if_configured(store, export_dir: str, now_iso: str) -> str | None:
    if not export_dir:
        return None
    rows = store.export_audit(since_days=3650)
    path = Path(export_dir) / f"audit-export-{now_iso[:10]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return str(path)


def main() -> int:
    from umrs.grants.source_store import SourceGrantStore
    from umrs.security.sealed_sources import SealedSourceStore
    from umrs.shared.pg import get_default_sync_pool

    pool = get_default_sync_pool()
    if pool is None:
        print("retention: no shared pool resolvable from env", file=sys.stderr)
        return 2

    audit_days = int(os.environ.get("UMRS_AUDIT_RETENTION_DAYS", "90"))
    tombstone_days = int(os.environ.get("UMRS_GRANT_TOMBSTONE_RETENTION_DAYS", "180"))
    export_dir = os.environ.get("UMRS_AUDIT_EXPORT_DIR", "").strip()

    sources = SealedSourceStore(pool)
    grants = SourceGrantStore(pool)
    now_iso = datetime.now(timezone.utc).isoformat()

    export_path = _export_if_configured(sources, export_dir, now_iso)
    counts = sources.purge_audit_older_than(audit_days)
    tombstones = grants.purge_tombstoned_older_than(tombstone_days)

    result = {
        "audit_days": audit_days,
        "tombstone_days": tombstone_days,
        "user_source_audit_purged": counts["user_source_audit"],
        "secret_audit_purged": counts["secret_audit"],
        "source_grants_tombstones_purged": tombstones,
        "export_path": export_path,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
