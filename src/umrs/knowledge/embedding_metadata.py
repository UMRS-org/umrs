"""Embedding metadata helpers for knowledge artifact rows (stability S10).

The durable re-index pattern (``JobType.KNOWLEDGE_REINDEX``):

1. **sweep** — :func:`select_stale_artifacts` pages active artifacts whose
   embedding metadata is missing or targets another model;
2. **compute** — submissions ride the canonical ``INDEX_BUILD`` durable
   path (EMBEDDING_BATCH dispatch is explicitly unimplemented, #216 —
   embeddings are computed into the RAG index);
3. **stamp** — :func:`stamp_embeddings_current` records model/dim/time per
   completed artifact set; a model bump first calls
   :func:`mark_embeddings_stale` so the next sweep re-selects them.

Registered payload contract for KNOWLEDGE_REINDEX jobs:
``{"target_model": str, "index_name": str}`` — both required; the spec
never re-derives defaults client-side (deliberate, operator-visible).

Dispatcher wire-up (KNOWLEDGE_REINDEX → sweep/submit/stamp) is a
recorded follow-up; the registered type exists so no caller scripts
re-indexing ad hoc (S9 premise).

Chunk scheme (ADR-093 D-11, 2026-10-10): the embed stamp also records the
scheme that cut the artifact's chunks and how many were indexed, in the
artifact's ``metadata`` (``chunk_scheme``, ``chunks_total``,
``chunks_indexed``; no column, no migration). An artifact without
``chunk_scheme`` was cut ``fixed-v0``. The sweep can select by scheme
mismatch and by project, so the same-model reindex re-chunks a project.

Convergence (ADR-093 D-11, wave F, 2026-10-11): the sweep pages by
``artifact_id`` (keyset ``after``), so one reindex pass visits each stale
artifact once, and with ``held_by`` it skips artifacts that a queued or
running embed/understand job already holds.

Permanent failures (ADR-093 D-11, 2026-10-11): an artifact the embed job
failed by name for a reason no retry changes (content this site has no
embedder for, an object that is gone, no file node to hang chunks on)
carries ``metadata.embed_failed = {model, scheme, reason, at}``. A sweep
with ``skip_failed`` passes it over while the site's model and scheme are
the ones that failed it, so a pass converges; a model or scheme change, or
``retry_failed``, tries it again, and a successful stamp clears the record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TARGET_MODEL_KEY = "target_model"
INDEX_NAME_KEY = "index_name"
REINDEX_PAYLOAD_KEYS = frozenset({TARGET_MODEL_KEY, INDEX_NAME_KEY})

VALID_EMBEDDING_STATUSES = frozenset({"pending", "current", "stale"})


class EmbeddingMetadataError(ValueError):
    """Invalid payload, identifier set, or status."""


def validate_reindex_payload(spec: dict) -> dict:
    """Enforce the registered KNOWLEDGE_REINDEX payload contract."""
    if not isinstance(spec, dict):
        raise EmbeddingMetadataError("reindex spec must be a dict")
    missing = REINDEX_PAYLOAD_KEYS - spec.keys()
    if missing:
        raise EmbeddingMetadataError(f"reindex spec missing keys: {sorted(missing)}")
    model = str(spec.get(TARGET_MODEL_KEY) or "").strip()
    index = str(spec.get(INDEX_NAME_KEY) or "").strip()
    if not model:
        raise EmbeddingMetadataError("target_model must be non-empty")
    if not index:
        raise EmbeddingMetadataError("index_name must be non-empty")
    return {TARGET_MODEL_KEY: model, INDEX_NAME_KEY: index}


# An artifact stamped before chunk schemes existed was cut by fixed-v0.
_UNSTAMPED_SCHEME = "fixed-v0"

# The scheme stamp is per artifact, but one artifact can hold chunks in
# several graphs (one re-embed job each): when one graph's job stamps the
# new scheme and another's fails, the artifact row reads current while that
# graph keeps old chunks. So the chunk rows themselves are read too — an
# artifact with any chunk of another scheme, in its own project, is stale
# (review chunk-data-4, 2026-10-10). Non-correlated, so Postgres hashes it
# once per sweep.
_STALE_CHUNKS_SQL = (
    "(project_id, artifact_id) IN (SELECT n.project_id, n.content->>'artifact_id' "
    "FROM knowledge_nodes n WHERE n.content->>'kind' = 'chunk' "
    "AND COALESCE(n.content->>'chunk_scheme', %s) <> %s{project})"
)


def _scheme_clauses(
    chunk_scheme: str | None, project_id: str | None
) -> tuple[list[str], list[Any]]:
    """The chunk-scheme mismatch clauses: the artifact's stamp, then its
    chunk rows (none without a scheme)."""
    if not chunk_scheme:
        return [], []
    stamp = "COALESCE(metadata->>'chunk_scheme', %s) <> %s"
    rows = _STALE_CHUNKS_SQL.format(
        project=" AND n.project_id = %s" if project_id else ""
    )
    params: list[Any] = [
        _UNSTAMPED_SCHEME,
        chunk_scheme,
        _UNSTAMPED_SCHEME,
        chunk_scheme,
    ]
    return [stamp, rows], params + ([project_id] if project_id else [])


def _stale_where(
    model: str, chunk_scheme: str | None, project_id: str | None
) -> tuple[str, list[Any]]:
    """The sweep's WHERE clause and its parameters (placeholders only)."""
    stale = ["embedding_status <> 'current'", "embedding_model <> %s"]
    params: list[Any] = [model]
    clauses, scheme_params = _scheme_clauses(chunk_scheme, project_id)
    stale += clauses
    params += scheme_params
    where = f"status = 'active' AND ({' OR '.join(stale)})"
    if project_id:
        where += " AND project_id = %s"
        params.append(project_id)
    return where, params


# ADR-093 D-11 (wave F): an artifact that an unfinished job already holds
# (a queued or running embed/understand job lists it in spec.sources) is not
# swept again — every re-embed key hashes its parent job, so two passes never
# deduplicated each other's embeds. NOT EXISTS, never NOT IN (one source
# without artifact_id would make NOT IN NULL for every row); the CASE guards
# a spec whose ``sources`` is not an array. PLANNED is left out on purpose:
# a crash between create and enqueue leaves an orphan PLANNED row that must
# not hold its artifacts for ever.
_HELD_SQL = (
    "NOT EXISTS (SELECT 1 FROM jobs j CROSS JOIN LATERAL jsonb_array_elements("
    "CASE WHEN jsonb_typeof(j.spec->'sources') = 'array' THEN j.spec->'sources' "
    "ELSE '[]'::jsonb END) s "
    "WHERE j.project_id = knowledge_artifacts.project_id "
    "AND j.job_type = ANY(%s) AND j.status = ANY(%s) "
    "AND s->>'artifact_id' = knowledge_artifacts.artifact_id)"
)


@dataclass(frozen=True)
class HeldBy:
    """Jobs whose unfinished runs hold their sources: their types and the
    statuses that count as unfinished (strings, as the jobs table stores
    them; the caller in ``umrs.jobs`` names them)."""

    job_types: tuple[str, ...]
    statuses: tuple[str, ...]


# The record a permanent embed failure leaves (see the module docstring),
# matched on the model and scheme that failed it.
_FAILED_SQL = (
    "(COALESCE(metadata->'embed_failed'->>'model', '') = %s "
    "AND COALESCE(metadata->'embed_failed'->>'scheme', '') = %s)"
)


def _failed_clause(model: str, chunk_scheme: str | None) -> tuple[str, list[Any]]:
    return f" AND NOT {_FAILED_SQL}", [model, chunk_scheme or _UNSTAMPED_SCHEME]


def select_stale_artifacts(
    conn,
    target_model: str,
    *,
    limit: int = 500,
    chunk_scheme: str | None = None,
    project_id: str | None = None,
    after: str | None = None,
    held_by: HeldBy | None = None,
    skip_failed: bool = False,
) -> list[str]:
    """Active artifact ids needing (re)embedding under ``target_model``.

    With ``chunk_scheme``, an artifact cut by another scheme is stale too
    (same-model re-chunk, ADR-093 D-11) — by its stamp or by any of its
    chunk rows; with ``project_id``, only that project's artifacts are
    swept.

    Keyset pages (wave F): ids come in ``artifact_id`` order, and ``after``
    starts the page past the previous page's last id, so one pass visits
    each artifact once — an artifact that fails and turns stale again
    during the pass is not visited twice. With ``held_by``, artifacts an
    unfinished job of those types already holds are skipped, and with
    ``skip_failed``, artifacts this model and scheme failed permanently.
    """
    model = str(target_model or "").strip()
    if not model:
        raise EmbeddingMetadataError("target_model must be non-empty")
    where, params = _stale_where(model, chunk_scheme, project_id)
    if after is not None:
        where += " AND artifact_id > %s"
        params.append(after)
    if held_by is not None:
        where += f" AND {_HELD_SQL}"
        params += [list(held_by.job_types), list(held_by.statuses)]
    if skip_failed:
        clause, extra = _failed_clause(model, chunk_scheme)
        where += clause
        params += extra
    cur = conn.cursor()
    cur.execute(
        f"SELECT artifact_id FROM knowledge_artifacts WHERE {where} "
        "ORDER BY artifact_id LIMIT %s",
        (*params, int(limit)),
    )
    return [row[0] for row in cur.fetchall()]


def count_failed_artifacts(
    conn,
    target_model: str,
    *,
    chunk_scheme: str | None = None,
    project_id: str | None = None,
) -> int:
    """Stale artifacts a sweep with ``skip_failed`` passes over: those this
    model and scheme failed permanently (reported, never silently dropped)."""
    model = str(target_model or "").strip()
    if not model:
        raise EmbeddingMetadataError("target_model must be non-empty")
    where, params = _stale_where(model, chunk_scheme, project_id)
    params += [model, chunk_scheme or _UNSTAMPED_SCHEME]
    cur = conn.cursor()
    cur.execute(
        f"SELECT count(*) FROM knowledge_artifacts WHERE {where} AND {_FAILED_SQL}",
        tuple(params),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def stamp_embeddings_current(
    conn,
    artifact_ids: Sequence[str],
    *,
    model: str,
    dim: int | None = None,
    embedded_at: str | None = None,
) -> int:
    """Mark a completed batch's artifacts current. Returns updated count.

    ``dim`` may be None (unknown/not recorded); when given it must be
    positive (the column stores NULL otherwise — never a silent 0).
    """
    ids = [str(a).strip() for a in artifact_ids if str(a).strip()]
    if not ids:
        return 0
    model = str(model or "").strip()
    if not model:
        raise EmbeddingMetadataError("model must be non-empty")
    if dim is not None and int(dim) <= 0:
        raise EmbeddingMetadataError("dim must be positive when given")
    dim_value = None if dim is None else int(dim)
    ts = embedded_at or datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        "UPDATE knowledge_artifacts "
        "SET embedding_model = %s, embedding_dim = %s, embedded_at = %s, "
        "embedding_status = 'current', updated_at = %s "
        "WHERE artifact_id = ANY(%s)",
        (model, dim_value, ts, ts, ids),
    )
    return int(getattr(cur, "rowcount", 0) or 0)


def mark_embeddings_stale(conn, *, previous_model: str | None = None) -> int:
    """Flag embeddings as stale after a model change.

    With ``previous_model`` set, only rows embedding under OTHER models are
    flagged (the just-bumped-away-from model's rows stay current); without
    it, every active current row is flagged (full re-index).
    """
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    if previous_model is not None:
        cur.execute(
            "UPDATE knowledge_artifacts SET embedding_status = 'stale', updated_at = %s "
            "WHERE status = 'active' AND embedding_model <> %s",
            (ts, previous_model),
        )
    else:
        cur.execute(
            "UPDATE knowledge_artifacts SET embedding_status = 'stale', updated_at = %s "
            "WHERE status = 'active' AND embedding_status = 'current'",
            (ts,),
        )
    return int(getattr(cur, "rowcount", 0) or 0)


# A successful stamp clears a permanent-failure record.
_STAMP_SQL = (
    "UPDATE knowledge_artifacts "
    "SET embedding_model = $1, embedding_dim = $2, embedded_at = now(), "
    "embedding_status = 'current', updated_at = now(), "
    "metadata = COALESCE(metadata, '{}'::jsonb) - 'embed_failed' "
    "WHERE artifact_id = $3"
)
# ADR-093 D-11: the scheme and the chunk counts ride with the stamp.
_STAMP_CHUNKS_SQL = (
    "UPDATE knowledge_artifacts "
    "SET embedding_model = $1, embedding_dim = $2, embedded_at = now(), "
    "embedding_status = 'current', updated_at = now(), "
    "metadata = (COALESCE(metadata, '{}'::jsonb) - 'embed_failed') || jsonb_build_object("
    "'chunk_scheme', $4::text, 'chunks_total', $5::int, 'chunks_indexed', $6::int) "
    "WHERE artifact_id = $3"
)
_FAILED_STAMP_SQL = (
    "UPDATE knowledge_artifacts "
    "SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object("
    "'embed_failed', jsonb_build_object('model', $1::text, 'scheme', $2::text, "
    "'reason', $3::text, 'at', now()::text)), updated_at = now() "
    "WHERE artifact_id = $4"
)
_FAILED_REASON_MAX = 300


@dataclass(frozen=True)
class ChunkStamp:
    """How an artifact was cut: its scheme, its chunks, the ones indexed
    (fewer than ``total`` when the per-artifact cap truncated it)."""

    scheme: str
    total: int
    indexed: int


async def stamp_artifact_embedded(
    pool: Any,
    artifact_id: str,
    *,
    model: str,
    dim: int,
    chunks: ChunkStamp | None = None,
) -> None:
    """Mark ONE artifact current after its vectors were actually written
    (KNOWLEDGE_EMBED — the only honest stamper; asyncpg pool). With
    ``chunks``, the artifact's metadata also records its chunk scheme and
    counts."""
    if not model or dim <= 0:
        raise EmbeddingMetadataError(
            "stamp needs the producing model and a positive dim"
        )
    args: tuple[Any, ...] = (model, dim, artifact_id)
    sql = _STAMP_SQL
    if chunks is not None:
        sql = _STAMP_CHUNKS_SQL
        args += (chunks.scheme, int(chunks.total), int(chunks.indexed))
    async with pool.acquire() as conn:
        await conn.execute(sql, *args)


async def record_embed_failures(
    pool: Any, failures: Sequence[tuple[str, str]], *, model: str, scheme: str
) -> int:
    """Record ``(artifact_id, reason)`` permanent embed failures under the
    model and scheme that failed them (asyncpg pool). Returns the count."""
    rows = [
        (model, scheme, str(reason)[:_FAILED_REASON_MAX], str(artifact_id))
        for artifact_id, reason in failures
        if str(artifact_id).strip()
    ]
    if not rows or not model:
        return 0
    async with pool.acquire() as conn:
        await conn.executemany(_FAILED_STAMP_SQL, rows)
    return len(rows)
