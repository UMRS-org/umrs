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
from dataclasses import dataclass
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


def scan_reconcile_report(project_id: str, *, db_url: str = "") -> Any | None:
    """Run the plane reconciler for one project; None when no DSN / on failure."""
    dsn = db_url or os.getenv("UMRS_DB_URL") or os.getenv("DATABASE_URL", "")
    if not dsn:
        return None
    try:
        import psycopg2

        from umrs.shared.pg_settings import pg_connect_kwargs

        from .plane_reconciler import scan_project

        conn = psycopg2.connect(dsn, **pg_connect_kwargs("reconcile-snapshot"))
        try:
            return scan_project(_OneConnPool(conn), project_id)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - snapshot is best-effort metadata
        logger.warning(
            "reconciliation snapshot unavailable for %s: %s", project_id, exc
        )
        return None


def summarize_report(report: Any) -> dict[str, Any]:
    """Deterministic manifest summary of one reconcile report."""
    digest = hashlib.sha256(
        json.dumps(
            sorted((a.document_id, _signature_for(a)) for a in report.assets),
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


def summarize_reconcile_snapshot(
    project_id: str, *, db_url: str = ""
) -> dict[str, Any] | None:
    """Summarize the reconcile scan into a manifest payload, or None."""
    report = scan_reconcile_report(project_id, db_url=db_url)
    return summarize_report(report) if report is not None else None


# ADR-032 precondition #5: export eligibility is snapshot-competent assets
# only — COMPLETE or EMBEDDING_PENDING, never PARTIAL or ORPHANED.
EXPORT_ELIGIBLE_STATES = ("COMPLETE", "EMBEDDING_PENDING")


@dataclass
class ExportSnapshot:
    """A checked snapshot the exporter consumes (precondition #5)."""

    summary: dict[str, Any]
    excluded_node_ids: frozenset[str] = frozenset()


def export_snapshot_from_report(report: Any) -> ExportSnapshot:
    """Derive the exporter's exclusions from one reconcile report.

    Canonical-plane rows of ineligible assets (their document + chunk nodes)
    and orphaned chunk nodes stay out of the bundle; the summary records each
    exclusion so the package states what it omitted and why.
    """
    excluded: set[str] = set(report.orphaned_chunks)
    excluded_assets: list[dict[str, Any]] = []
    for asset in sorted(report.assets, key=lambda a: a.document_id):
        if asset.state in EXPORT_ELIGIBLE_STATES:
            continue
        excluded.update(asset.doc_node_ids)
        excluded.update(asset.chunk_node_ids)
        excluded_assets.append(
            {
                "document_id": asset.document_id,
                "state": asset.state,
                "reasons": asset.reasons,
            }
        )
    summary = summarize_report(report)
    summary.update(
        {
            "consumed": True,
            "export_eligible_states": list(EXPORT_ELIGIBLE_STATES),
            "excluded_assets": excluded_assets,
            "orphaned_chunks_excluded": len(report.orphaned_chunks),
            "excluded_node_count": len(excluded),
        }
    )
    return ExportSnapshot(summary=summary, excluded_node_ids=frozenset(excluded))


def take_export_snapshot(project_id: str, *, db_url: str = "") -> ExportSnapshot | None:
    """The checked snapshot for an export, or None (no DSN / scan failure)."""
    report = scan_reconcile_report(project_id, db_url=db_url)
    return export_snapshot_from_report(report) if report is not None else None


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


__all__ = [
    "EXPORT_ELIGIBLE_STATES",
    "ExportSnapshot",
    "export_snapshot_from_report",
    "scan_reconcile_report",
    "summarize_reconcile_snapshot",
    "take_export_snapshot",
]
