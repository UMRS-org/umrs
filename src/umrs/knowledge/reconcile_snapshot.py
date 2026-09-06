"""Optional reconciliation-snapshot summary for the v1.2 export manifest.

ADR-032 precondition #5 asks the exporter to consume a checked snapshot
emitted by the reconciliation job instead of reading planes ad hoc. Full
snapshot-consumption is a schema round of its own; this module provides
the honest *attach* leg: when a sync DSN is reachable we run
``plane_reconciler.scan_project`` and embed a deterministic summary
(state counts + content digest) in the manifest's
``reconciliation_snapshot`` field. When no DSN is configured we return
``None`` and the manifest keeps the documented interim ``null``.

The manifest already declares ``provenance_canonical: false`` for the
ad-hoc read both paths share today.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class _OneConnPool:
    """Minimal getconn/putconn shim around one opened connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def getconn(self) -> Any:
        return self._conn

    def putconn(self, _conn: Any) -> None:
        return None


def summarize_reconcile_snapshot(project_id: str, *, db_url: str = "") -> dict[str, Any] | None:
    """Summarize the reconcile scan into a manifest payload, or None."""
    dsn = db_url or os.getenv("UMRS_DB_URL") or os.getenv("DATABASE_URL", "")
    if not dsn:
        return None
    try:
        import psycopg2  # noqa: PLC0415 - runtime dep, resolved server-side

        from .plane_reconciler import scan_project

        conn = psycopg2.connect(dsn)
        try:
            report = scan_project(_OneConnPool(conn), project_id)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - snapshot is best-effort metadata
        logger.warning("reconciliation snapshot unavailable for %s: %s", project_id, exc)
        return None

    digest = hashlib.sha256(
        json.dumps(
            sorted(
                (a.document_id, _signature_for(a))
                for a in report.assets
            ),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "states": report.counts,
        "assets_total": len(report.assets),
        "catalogue_available": report.catalogue_available,
        "digest": digest,
    }


def _signature_for(asset: Any) -> str:
    """Stable per-asset fingerprint for the snapshot digest."""
    return json.dumps(
        {
            "state": asset.state,
            "chunks": asset.chunk_count,
            "canonical": asset.has_canonical,
            "catalogue": asset.has_catalogue,
            "embeddings": asset.has_embeddings,
        },
        sort_keys=True,
    )


__all__ = ["summarize_reconcile_snapshot"]
