"""Plane reconciliation/checker (ADR-031 §Operational enforcement, normative).

Owner: server-side durable reconciliation job (this module is its compute
core — attempt-scoped reads only, no writes). Classifies each asset/document
into exactly one consistency state per Invariant 4:

    COMPLETE           catalogue ✓ canonical ✓ embeddings ✓
    EMBEDDING_PENDING  catalogue ✓ canonical ✓ embeddings ✗ (non-error)
    PARTIAL            catalogue ✓ canonical ✗  OR  catalogue ✗ canonical ✓
                       OR  canonical doc with no chunk nodes
    ORPHANED           a row references an identity that no longer exists in
                       its parent plane (vector row with no node; chunk node
                       with no document node)

Planes per ADR-031: catalogue (catalog_objects / metadata), canonical
(knowledge_nodes doc+chunk rows), vector (knowledge_node_embeddings, 0075).
Legacy corpus tables are NOT a plane here — legacy-only corpora must be
canonicalized first (Invariant 2), the checker never auto-mints.

This wave ships classification + scan + report only (read-only). Mutation
(requeue/repair marking) is deliberately deferred — the checker's state is
derived and stamped nowhere yet; consistency_state is computed per scan, per
ADR-031 Invariant 4 ("derived, never stored per-plane").
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("umrs.knowledge.plane_reconciler")

# ── SQL (read-only) ─────────────────────────────────────────────────────

_DOCS_SQL = """
SELECT node_id, content->>'document_id'
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'document'
"""

_CHUNKS_SQL = """
SELECT node_id, content->>'document_id', content->>'s3_object'
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'chunk'
"""

_VECTORS_SQL = """
SELECT node_id, count(*)
  FROM knowledge_node_embeddings
 GROUP BY node_id
"""

_CATALOGUE_BY_FILENAME_SQL = """
SELECT s3_key
  FROM catalog_objects
 WHERE project_id = %s
"""

_CATALOGUE_TABLE_PROBE_SQL = "SELECT to_regclass('catalog_objects')"

# ── Model ───────────────────────────────────────────────────────────────

STATE_COMPLETE = "COMPLETE"
STATE_EMBEDDING_PENDING = "EMBEDDING_PENDING"
STATE_PARTIAL = "PARTIAL"
STATE_ORPHANED = "ORPHANED"

STATES = (STATE_COMPLETE, STATE_EMBEDDING_PENDING, STATE_PARTIAL, STATE_ORPHANED)


@dataclass
class Asset:
    document_id: str
    doc_node_ids: list[str] = field(default_factory=list)
    chunk_node_ids: list[str] = field(default_factory=list)
    s3_objects: set[str] = field(default_factory=set)
    vector_rows_by_chunk: dict[str, int] = field(default_factory=dict)
    catalogue_keys: set[str] = field(default_factory=set)  # basename-matched
    state: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def has_catalogue(self) -> bool:
        return bool(self.catalogue_keys)

    @property
    def has_canonical(self) -> bool:
        return bool(self.doc_node_ids)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_node_ids)

    @property
    def has_embeddings(self) -> bool:
        return any(rows > 0 for rows in self.vector_rows_by_chunk.values())


@dataclass
class ReconcileReport:
    project_id: str
    assets: list[Asset]
    orphaned_vectors: list[str]
    orphaned_chunks: list[str]
    catalogue_available: bool = True

    @property
    def counts(self) -> dict[str, int]:
        out = {state: 0 for state in STATES}
        for asset in self.assets:
            if asset.state in out:
                out[asset.state] += 1
        out[STATE_ORPHANED] += len(self.orphaned_vectors) + len(self.orphaned_chunks)
        return out


# ── Classification ──────────────────────────────────────────────────────


def _signature(asset: Asset) -> str:
    """Class-rule key: cat-can-chunks-vec booleans, zero-chunks beats all."""
    if asset.has_canonical and asset.chunk_count == 0:
        return "zero-chunks"
    flags = (
        "1" if asset.has_catalogue else "0",
        "1" if asset.has_canonical else "0",
        "1" if asset.has_embeddings else "0",
    )
    return "".join(flags)


_RULES: dict[str, tuple[str, list[str]]] = {
    "zero-chunks": (STATE_PARTIAL, ["canonical document has zero chunk nodes"]),
    "111": (STATE_COMPLETE, []),
    "110": (STATE_EMBEDDING_PENDING, ["no embedding rows for chunks (non-error)"]),
    "100": (STATE_PARTIAL, ["catalogue row without canonical document"]),
    "011": (STATE_PARTIAL, ["canonical document without catalogue row"]),
    "010": (STATE_PARTIAL, ["canonical document without catalogue row"]),
    "001": (STATE_PARTIAL, ["document with no plane rows (unreachable state)"]),
    "000": (STATE_PARTIAL, ["document with no plane rows (unreachable state)"]),
}


def classify(asset: Asset) -> Asset:
    """Assign per-asset consistency state per Invariant 4 rules."""
    state, reasons = _RULES[_signature(asset)]
    asset.state, asset.reasons = state, list(reasons)
    return asset


# ── Scan ────────────────────────────────────────────────────────────────


def _rows(pool: Any, sql: str, params: tuple) -> Iterator[tuple]:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            yield from cur.fetchall()
    finally:
        pool.putconn(conn)


def _collect_docs_and_chunks(
    pool: Any, project_id: str
) -> tuple[dict[str, Asset], list[str]]:
    assets: dict[str, Asset] = {}
    for node_id, doc_id in _rows(pool, _DOCS_SQL, (project_id,)):
        if not doc_id:
            continue
        asset = assets.setdefault(str(doc_id), Asset(document_id=str(doc_id)))
        asset.doc_node_ids.append(str(node_id))
    orphaned_chunks: list[str] = []
    for node_id, doc_id, s3_object in _rows(pool, _CHUNKS_SQL, (project_id,)):
        asset = assets.get(str(doc_id)) if doc_id else None
        if asset is None:
            orphaned_chunks.append(str(node_id))
            continue
        asset.chunk_node_ids.append(str(node_id))
        if s3_object:
            asset.s3_objects.add(str(s3_object))
    return assets, orphaned_chunks


def _merge_vectors(
    pool: Any, assets: dict[str, Asset]
) -> tuple[dict[str, int], list[str]]:
    vector_by_chunk = {
        str(node_id): int(rows or 0)
        for node_id, rows in _rows(pool, _VECTORS_SQL, ())
    }
    chunk_ids: set[str] = set()
    for asset in assets.values():
        chunk_ids.update(asset.chunk_node_ids)
        for chunk_id in asset.chunk_node_ids:
            asset.vector_rows_by_chunk[chunk_id] = vector_by_chunk.get(chunk_id, 0)
    orphaned = [nid for nid, rows in vector_by_chunk.items() if nid not in chunk_ids and rows > 0]
    return vector_by_chunk, orphaned


def _attach_catalogue(pool: Any, project_id: str, assets: dict[str, Asset]) -> bool:
    """Attach s3_keys to assets when the catalogue plane is readable here."""
    probe_rows = list(_rows(pool, _CATALOGUE_TABLE_PROBE_SQL, ()))
    if not probe_rows or not (probe_rows[0] and probe_rows[0][0]):
        logger.warning("catalog_objects not found in this pool — catalogue column marked unknown")
        return False
    basename_by_doc: dict[str, str] = {}
    for asset_id, asset in assets.items():
        for s3_object in asset.s3_objects:
            basename_by_doc.setdefault(s3_object.rsplit("/", 1)[-1], asset_id)
    for (s3_key,) in _rows(pool, _CATALOGUE_BY_FILENAME_SQL, (project_id,)):
        base = str(s3_key).rstrip("/").rsplit("/", 1)[-1]
        if base in basename_by_doc:
            assets[basename_by_doc[base]].catalogue_keys.add(str(s3_key))
    return True


def _finalize(assets: dict[str, Asset], catalogue_available: bool) -> None:
    for asset in assets.values():
        if not catalogue_available and asset.has_canonical:
            asset.state = (
                STATE_COMPLETE if asset.has_embeddings else STATE_EMBEDDING_PENDING
            )
            asset.reasons = ["catalogue plane not readable in this pool"]
            continue
        classify(asset)


def scan_project(pool: Any, project_id: str) -> ReconcileReport:
    """Read the three planes for one project + classify every asset.

    Fail-closed on pool absence: a checker without its primary store must
    not produce a misleading COMPLETE-everything snapshot.
    """
    if pool is None:
        raise RuntimeError("reconciliation requires the shared umrs_db pool")

    assets, orphaned_chunks = _collect_docs_and_chunks(pool, project_id)
    _vector_map, orphaned_vectors = _merge_vectors(pool, assets)
    catalogue_available = _attach_catalogue(pool, project_id, assets)
    _finalize(assets, catalogue_available)

    return ReconcileReport(
        project_id=project_id,
        assets=list(assets.values()),
        orphaned_vectors=orphaned_vectors,
        orphaned_chunks=orphaned_chunks,
        catalogue_available=catalogue_available,
    )


def main(argv: list[str] | None = None) -> int:
    """`python -m umrs.knowledge.plane_reconciler --dsn ... --project X`."""
    import argparse
    from psycopg2 import pool as pg_pool

    parser = argparse.ArgumentParser(
        description="ADR-031 plane reconciliation scan (read-only report).",
    )
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--show-samples", type=int, default=5)
    args = parser.parse_args(argv)

    from umrs.shared.pg_settings import pg_connect_kwargs

    pool = pg_pool.ThreadedConnectionPool(
        minconn=1, maxconn=2, dsn=args.dsn, **pg_connect_kwargs("plane-reconciler")
    )
    try:
        report = scan_project(pool, args.project)
    finally:
        pool.closeall()

    counts = report.counts
    print(f"project: {report.project_id}")
    print(f"  {STATE_COMPLETE:<20} {counts[STATE_COMPLETE]}")
    print(f"  {STATE_EMBEDDING_PENDING:<20} {counts[STATE_EMBEDDING_PENDING]}")
    print(f"  {STATE_PARTIAL:<20} {counts[STATE_PARTIAL]}")
    print(f"  {STATE_ORPHANED:<20} {counts[STATE_ORPHANED]}")
    for asset in report.assets[: args.show_samples]:
        note = f" ({asset.reasons[0]})" if asset.reasons else ""
        print(f"    · {asset.document_id}: {asset.state}{note}")
    if report.orphaned_vectors:
        print(f"    ⚠ {len(report.orphaned_vectors)} orphaned vector rows")
    if report.orphaned_chunks:
        print(f"    ⚠ {len(report.orphaned_chunks)} orphaned chunk nodes")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI path
    raise SystemExit(main())
