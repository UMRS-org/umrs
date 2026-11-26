"""Postgres side of the term registry (ADR-081 D-4/D-5/D-9, slice KN-S7).

- :func:`fetch_snapshot` reads the admitted vocabulary of one graph: the
  graph's ontology record when that record has an admitted set
  (``knowledge_ontology_terms``), else the bundled ``r4-core@2`` record.
  A graph without a row, without an ontology, or with an ontology that has
  no admitted set (a 0091 record, ``r4-default``) resolves to ``r4-core@2``
  in warn mode, whatever its ``ontology_mode`` says: enforce applies only to
  the ontology the graph was bound to. Returns None when ``r4-core@2`` is
  not seeded (a database before the seed migration): the caller then writes
  untyped rows, as before W2a.
- :func:`record_gaps` counts vocabulary misses in ``knowledge_vocab_gaps``.
- :func:`record_batch` appends a ``knowledge_write_batches`` row and bumps
  ``knowledge_graphs.graph_version``.
- :func:`bundled_snapshot` builds a snapshot from package data alone (no
  database), for raw-SQL minting helpers and tests.

Lookups are scoped: a graph's ontology record is the caller project's live
record first, then the global one (ADR-116 D-5); another project's record
is never read.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .term_types import (
    CORE_REF,
    ORIGIN_ACTORS,
    Gap,
    RegistrySnapshot,
    TermSpec,
    Violation,
    bundled_record_id,
)

__all__ = [
    "SnapshotCache",
    "bundled_snapshot",
    "fetch_snapshot",
    "record_batch",
    "record_gaps",
]

_GRAPH_SQL = (
    "SELECT ontology_id, ontology_version, ontology_mode "
    "FROM knowledge_graphs WHERE graph_id = $1"
)
# The project's live record first, then the global one; only a record with
# an admitted set counts.
_RECORD_SQL = (
    "SELECT o.record_id FROM knowledge_ontologies o "
    "WHERE o.ontology_id = $1 AND o.version = $2 AND o.retired_at IS NULL "
    "AND (o.project_id = $3 OR o.project_id IS NULL) "
    "AND EXISTS (SELECT 1 FROM knowledge_ontology_terms ot WHERE ot.record_id = o.record_id) "
    "ORDER BY o.project_id NULLS LAST LIMIT 1"
)
_SEEDED_SQL = "SELECT EXISTS (SELECT 1 FROM knowledge_ontology_terms WHERE record_id = $1)"
_TERM_COLUMNS = (
    "t.term_id, t.iri, t.kind, t.label, t.parents, t.bucket, t.domain_classes, "
    "t.range_classes, t.datatype, t.unit, t.transitive, t.functional, t.inverse_of, "
    "t.aliases, t.synonyms, t.definition, t.required_on, t.origin, t.verified, "
    "t.source, t.status, t.replaced_by, t.mappings"
)
_TERMS_SQL = (
    f"SELECT {_TERM_COLUMNS} FROM knowledge_ontology_terms ot "
    "JOIN knowledge_terms t ON t.term_id = ot.term_id "
    "WHERE ot.record_id = $1 ORDER BY t.term_id"
)
_CLOSURE_SQL = (
    "SELECT c.ancestor, c.descendant, c.depth FROM knowledge_term_closure c "
    "JOIN knowledge_ontology_terms ot ON ot.term_id = c.descendant AND ot.record_id = $1 "
    "ORDER BY c.ancestor, c.descendant"
)
_GAP_SQL = (
    "INSERT INTO knowledge_vocab_gaps (graph_id, kind, value, writer, n, sample_id) "
    "VALUES ($1, $2, $3, $4, $5, $6) "
    "ON CONFLICT (graph_id, kind, value) DO UPDATE "
    "SET n = knowledge_vocab_gaps.n + EXCLUDED.n, last_seen = now(), "
    "writer = EXCLUDED.writer, sample_id = EXCLUDED.sample_id"
)
_BATCH_SQL = (
    "INSERT INTO knowledge_write_batches (batch_id, graph_id, project_id, writer, "
    "origin_actor, ontology_ref, mode, counts, violations) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)"
)
_BUMP_SQL = (
    "UPDATE knowledge_graphs SET graph_version = graph_version + 1 "
    "WHERE graph_id = $1"
)


def _json_value(raw: Any, default: Any) -> Any:
    """A jsonb column read with or without a codec."""
    if raw is None:
        return default
    if isinstance(raw, (str, bytes)):
        return json.loads(raw)
    return raw


def _term_from_row(row: Mapping[str, Any]) -> TermSpec:
    return TermSpec(
        term_id=row["term_id"],
        iri=row["iri"],
        kind=row["kind"],
        label=row["label"],
        parents=tuple(row["parents"] or ()),
        bucket=row["bucket"],
        domain_classes=tuple(row["domain_classes"] or ()),
        range_classes=tuple(row["range_classes"] or ()),
        datatype=row["datatype"],
        unit=row["unit"],
        transitive=bool(row["transitive"]),
        functional=bool(row["functional"]),
        inverse_of=row["inverse_of"],
        aliases=tuple(row["aliases"] or ()),
        synonyms=tuple(_json_value(row["synonyms"], [])),
        definition=row["definition"] or "",
        required_on=tuple(row["required_on"] or ()),
        origin=row["origin"],
        verified=bool(row["verified"]),
        source=row["source"] or "",
        status=row["status"],
        replaced_by=row["replaced_by"],
        mappings=_json_value(row["mappings"], {}),
    )


async def _bound_record(
    conn: Any, graph: Mapping[str, Any] | None, project_id: str
) -> tuple[str, str] | None:
    """(record_id, ref) of the graph's own ontology, when it has an admitted set."""
    if graph is None or not graph["ontology_id"]:
        return None
    version = graph["ontology_version"] or ""
    record_id = await conn.fetchval(
        _RECORD_SQL, graph["ontology_id"], version, project_id
    )
    if record_id is None:
        return None
    return str(record_id), f"{graph['ontology_id']}@{version}"


async def fetch_snapshot(
    conn: Any, *, project_id: str, graph_id: str
) -> RegistrySnapshot | None:
    """The admitted vocabulary of ``graph_id`` (module docstring)."""
    graph = await conn.fetchrow(_GRAPH_SQL, graph_id)
    bound = await _bound_record(conn, graph, project_id)
    if bound is None:
        record_id, ref, mode = bundled_record_id(CORE_REF), CORE_REF, "warn"
        if not await conn.fetchval(_SEEDED_SQL, record_id):
            return None
    else:
        record_id, ref = bound
        mode = str(graph["ontology_mode"] or "warn") if graph is not None else "warn"
    terms = tuple(_term_from_row(r) for r in await conn.fetch(_TERMS_SQL, record_id))
    closure = tuple(
        (r["ancestor"], r["descendant"], int(r["depth"]))
        for r in await conn.fetch(_CLOSURE_SQL, record_id)
    )
    return RegistrySnapshot(
        ontology_ref=ref, mode=mode, terms=terms, closure=closure, record_id=record_id
    )


async def record_gaps(
    conn: Any, *, graph_id: str, writer: str, gaps: Iterable[Gap]
) -> int:
    """Count each (kind, value) miss once per call with its multiplicity.
    Returns the number of distinct gaps written."""
    counted = Counter((g.kind, g.value) for g in gaps if g.value)
    samples = {(g.kind, g.value): g.sample_id for g in gaps}
    for (kind, value), n in sorted(counted.items()):
        await conn.execute(
            _GAP_SQL, graph_id, kind, value, writer, n, samples.get((kind, value), "")
        )
    return len(counted)


async def record_batch(
    conn: Any,
    *,
    batch_id: str,
    graph_id: str,
    project_id: str,
    writer: str,
    origin_actor: str,
    ontology_ref: str,
    mode: str,
    counts: Mapping[str, int],
    violations: Iterable[Violation],
) -> None:
    """Append the write-batch audit row and bump the graph's version.

    ``origin_actor`` outside ``ORIGIN_ACTORS`` raises ValueError before any
    write (the table CHECK would refuse it too).
    """
    if origin_actor not in ORIGIN_ACTORS:
        raise ValueError(f"unknown origin_actor {origin_actor!r}")
    await conn.execute(
        _BATCH_SQL,
        batch_id,
        graph_id,
        project_id,
        writer,
        origin_actor,
        ontology_ref,
        mode,
        json.dumps(dict(counts), sort_keys=True),
        json.dumps([v.as_json() for v in violations]),
    )
    await conn.execute(_BUMP_SQL, graph_id)


def bundled_snapshot(ref: str = CORE_REF, *, mode: str = "warn") -> RegistrySnapshot:
    """A snapshot of the bundled ontology ``ref`` from package data alone."""
    from .ontologies import bundled_by_ref
    from .term_closure import admitted_set, build_closure

    docs = bundled_by_ref()
    admitted = admitted_set(ref, docs)
    every = {t.term_id: t for doc in docs.values() for t in doc.terms}
    terms = tuple(every[term_id] for term_id in sorted(admitted))
    closure = build_closure(every.values())
    rows = tuple(
        (ancestor, descendant, depth)
        for (ancestor, descendant), depth in sorted(closure.items())
        if descendant in admitted
    )
    return RegistrySnapshot(
        ontology_ref=ref,
        mode=mode,
        terms=terms,
        closure=rows,
        record_id=bundled_record_id(ref),
    )


class SnapshotCache:
    """Per-store memo of graph snapshots (``ttl_s`` seconds, bounded).

    Terms are immutable and an admitted set changes only on registration
    (W2b), so a short TTL bounds how long a mode flip or a new binding takes
    to reach writers. A None snapshot (registry not seeded) is cached too.
    """

    def __init__(self, ttl_s: float = 30.0, max_entries: int = 256) -> None:
        self._ttl_s = ttl_s
        self._max = max_entries
        self._entries: dict[tuple[str, str], tuple[float, RegistrySnapshot | None]] = {}

    def get(self, key: tuple[str, str]) -> tuple[bool, RegistrySnapshot | None]:
        """(hit, snapshot)."""
        entry = self._entries.get(key)
        if entry is None or time.monotonic() - entry[0] > self._ttl_s:
            return False, None
        return True, entry[1]

    def put(self, key: tuple[str, str], snapshot: RegistrySnapshot | None) -> None:
        if len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = (time.monotonic(), snapshot)

    async def fetch(
        self, conn: Any, *, project_id: str, graph_id: str
    ) -> RegistrySnapshot | None:
        """Cached :func:`fetch_snapshot`."""
        key = (project_id, graph_id)
        hit, snapshot = self.get(key)
        if not hit:
            snapshot = await fetch_snapshot(conn, project_id=project_id, graph_id=graph_id)
            self.put(key, snapshot)
        return snapshot
