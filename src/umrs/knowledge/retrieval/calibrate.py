"""The dense lane's null distribution, computed in process (ADR-093 D-3, C2).

ADR-083 D-5's sufficiency rule needs to know what an UNRELATED query's best
cosine looks like in this scope: under e5-base-v2 every cosine sits in
0.7-1.0, so an absolute threshold cannot tell nonsense (0.67-0.70) from
evidence (0.87+). Thirty fixed probes, none about anything a project holds,
are encoded once per space through the batch ``/rag/embed``; their exact
top-1 cosines in the caller's scope give ``mu`` and ``sigma``, and
``z_null = (cos_1 - mu) / sigma`` is what ``cutoff.sufficiency`` reads.

An interim until ADR-083 D-1's space registry holds a durable null: the
values live in an LRU of 256 keyed by (project, graph-set hash, space,
vector count, last re-embed), so every worker computes identical values and
a re-embed invalidates them — also one that keeps the count (a re-staged
revision, an edited chunk), since every embedding upsert sets
``updated_at``. The probes are re-encoded on each miss rather than cached
per space: a cached probe vector would outlive an encoder re-stage that the
null cannot see. With fewer than 30 vectors in scope there is no null (only
the lexical rule applies). No table is written.

The key cannot see the encoder's QUERY encoding, which the rag service may
change while UMRS runs (``UMRS_RAG_QUERY_PREFIX=fixed|b0`` recreates only
the rag container; a re-staged encoder keeps its name). So each cached null
keeps the vector it encoded :data:`ENCODING_PROBE` to, and a hit is served
only while the encoder still encodes that probe the same way (cosine at
least :data:`SAME_ENCODING_COSINE`); otherwise the null is recomputed
(review of ADR-093, 2026-10-10). An encoder from :class:`FingerprintedEncoder`
carries the probe's vector from the query batch it just encoded (no extra
call); any other encoder is asked for that one probe.

``python -m umrs.knowledge.retrieval.calibrate --project P [--graph G]``
prints the null for a scope; it is a report.

The probes are disjoint from the eval's ``NONSENSE_EVAL_V1``
(``benchmarks/retrieval/scifact.py``): tuning on the probes the gate scores
would be circular. Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from umrs.knowledge.text_embedder import embedding_space

from .dense import query_rows, scope_clause, vector_literal
from .types import BatchEncoder, RetrieveSpec

PROBES_VERSION = "null-v1"
MIN_VECTORS = 30
CACHE_SIZE = 256

#: 20 seeded word salads and 10 off-domain questions. Never edit in place:
#: a new set is NULL_PROBES_V2 with its own PROBES_VERSION.
NULL_PROBES_V1: tuple[str, ...] = (
    "amber dominoes whisper recipes to a reluctant drawbridge",
    "the felt tram rehearses its apology for the silver pond",
    "nine crooked kettles lease the orchard to a patient storm",
    "a linen hurricane catalogues the hiccups of old chimneys",
    "the plywood choir borrows thursday from a drowsy almanac",
    "saffron ladders negotiate parking with the tidal piano",
    "every crimson doorknob mails a postcard to the wrong fog",
    "the cobalt mattress counts sheep on behalf of the canal",
    "a frugal comet knits mittens for the municipal echo",
    "the rusty sonnet balances its checkbook beside a tame volcano",
    "pewter giraffes rehearse the alphabet of a quiet quarry",
    "the bashful tractor writes limericks about lukewarm lanterns",
    "a gingham planet forgets the password to its own garden",
    "the clockwork pear auditions for the role of november",
    "wicker submarines trade riddles with an exhausted weathervane",
    "the jade tuba sublets the basement to seven polite moths",
    "a velvet avalanche files grievances against the morning bus",
    "the hesitant ferry alphabetizes the freckles of a cliff",
    "marmalade satellites tutor the hedgerow in long division",
    "the corduroy glacier sends regrets to a punctual meadow",
    "Who painted the ceiling of the Sistine Chapel?",
    "What time does the downtown farmers market open on Saturday?",
    "How do I fold a fitted bed sheet neatly?",
    "Which country hosted the 2016 Summer Olympic Games?",
    "What is a good name for a golden retriever puppy?",
    "How many players are on the field in a rugby union team?",
    "Where can I buy cheap concert tickets this weekend?",
    "What is the plot of the opera The Magic Flute?",
    "How do I remove a red wine stain from a carpet?",
    "Which chess opening begins with the move e4 e5 Nf3 Nc6 Bb5?",
)

#: The probe whose vector fingerprints the encoder's query encoding: a prefix
#: switch or a re-staged encoder moves it, float noise does not.
ENCODING_PROBE = NULL_PROBES_V1[0]
#: Two vectors of :data:`ENCODING_PROBE` are one encoding at this cosine or above.
SAME_ENCODING_COSINE = 0.9999

#: The scope is read once and each probe cast once; a correlated subquery
#: re-ran the join and the text->vector cast per row (1.1 s for 2 probes on
#: 8.4k vectors, 2026-10-10; this shape: 0.73 s for all 30).
_PROBE_SQL = (
    "WITH scope AS MATERIALIZED (SELECT e.embedding FROM knowledge_node_embeddings e "
    "JOIN knowledge_nodes n ON n.node_id = e.node_id WHERE {where}), "
    "probes AS MATERIALIZED (SELECT t.i, t.v::vector AS v "
    "FROM unnest(%s::int[], %s::text[]) AS t(i, v)) "
    "SELECT p.i, (SELECT min(s.embedding <=> p.v) FROM scope s) FROM probes p ORDER BY p.i"
)
_COUNT_SQL = (
    "SELECT count(*), max(e.updated_at) FROM knowledge_node_embeddings e "
    "JOIN knowledge_nodes n ON n.node_id = e.node_id WHERE {where}"
)


@dataclass(frozen=True)
class NullStats:
    """Mean and spread of the probes' top-1 cosines in one scope."""

    mu: float
    sigma: float
    n: int
    model: str
    probes_version: str = PROBES_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: key -> (the null, the vector its encoder gave ENCODING_PROBE)
_CACHE: OrderedDict[tuple, tuple[NullStats, tuple[float, ...]]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def scope_key(
    spec: RetrieveSpec, model: str, vectors: int, stamp: str | None = None
) -> tuple:
    """(project, graph-set hash, space, vector count, last re-embed) — the
    cache key."""
    graphs = "*" if spec.graph_ids is None else ",".join(sorted(spec.graph_ids))
    excluded = ",".join(sorted(spec.exclude_graph_ids))
    raw = f"{graphs}|-{excluded}|ambient={int(spec.include_ambient)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return (spec.project_id, digest, embedding_space(model), int(vectors), stamp)


def scope_signature(
    pool: Any, spec: RetrieveSpec, model: str
) -> tuple[int, str | None]:
    """``(vectors in scope, their latest updated_at)`` in one statement."""
    where, params = scope_clause(spec, model)
    rows = query_rows(pool, _COUNT_SQL.format(where=where), params)
    if not rows:
        return 0, None
    stamp = rows[0][1] if len(rows[0]) > 1 else None
    return int(rows[0][0]), None if stamp is None else str(stamp)


def vector_count(pool: Any, spec: RetrieveSpec, model: str) -> int:
    return scope_signature(pool, spec, model)[0]


def _cached(key: tuple) -> tuple[NullStats, tuple[float, ...]] | None:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
        return hit


def _store(key: tuple, entry: tuple[NullStats, tuple[float, ...]]) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)


# ── the encoder's query encoding ────────────────────────────────────────────


class FingerprintedEncoder:
    """A :data:`BatchEncoder` that also encodes :data:`ENCODING_PROBE` in the
    same request (appended unless the batch holds it) and keeps its vector
    as ``encoding_probe``: :func:`null_for` reads it to tell whether a
    cached null was computed under the encoding the query just got, without
    another call. The caller gets exactly its own texts' vectors."""

    def __init__(self, encode: BatchEncoder) -> None:
        self._encode = encode
        self.encoding_probe: list[float] | None = None

    def __call__(
        self, texts: Sequence[str]
    ) -> tuple[str, list[list[float]] | None, str]:
        batch = [str(t) for t in texts]
        sent = batch if ENCODING_PROBE in batch else [*batch, ENCODING_PROBE]
        model, vectors, reason = self._encode(sent)
        if not vectors or len(vectors) != len(sent):
            got = len(vectors or ())
            return (
                model,
                None,
                reason or f"query encoder returned {got} vectors for {len(sent)} texts",
            )
        self.encoding_probe = [float(v) for v in vectors[sent.index(ENCODING_PROBE)]]
        return model, vectors[: len(batch)], reason


def encoding_probe(encode: BatchEncoder) -> list[float] | None:
    """The encoder's vector for :data:`ENCODING_PROBE` now: the one a
    :class:`FingerprintedEncoder` kept from its last batch, else one call."""
    carried = getattr(encode, "encoding_probe", None)
    if carried:
        return list(carried)
    _model, vectors, _reason = encode([ENCODING_PROBE])
    return [float(v) for v in vectors[0]] if vectors else None


def same_encoding(cached: Sequence[float], current: Sequence[float] | None) -> bool:
    """Both vectors encode the probe alike (cosine >= SAME_ENCODING_COSINE)."""
    if not cached or not current or len(cached) != len(current):
        return False
    dot = sum(a * b for a, b in zip(cached, current))
    norms = math.sqrt(sum(a * a for a in cached) * sum(b * b for b in current))
    return norms > 0 and dot / norms >= SAME_ENCODING_COSINE


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def probe_cosines(
    pool: Any, spec: RetrieveSpec, model: str, vectors: Sequence[Sequence[float]]
) -> list[float]:
    """Each probe's exact top-1 cosine in the scope, in ONE statement."""
    where, params = scope_clause(spec, model)
    sql = _PROBE_SQL.format(where=where)
    literals = [vector_literal(v) for v in vectors]
    rows = query_rows(pool, sql, [*params, list(range(len(literals))), literals])
    return [1.0 - float(distance) for _i, distance in rows if distance is not None]


def _compute(
    pool: Any, spec: RetrieveSpec, encode: BatchEncoder, model: str
) -> tuple[NullStats, tuple[float, ...]] | None:
    """The null and the vector its encoder gave :data:`ENCODING_PROBE`."""
    probe_model, vectors, _reason = encode(list(NULL_PROBES_V1))
    if not vectors or embedding_space(probe_model) != embedding_space(model):
        return None
    cosines = probe_cosines(pool, spec, model, vectors)
    if len(cosines) < 2:
        return None
    stats = NullStats(
        mu=statistics.fmean(cosines),
        sigma=statistics.stdev(cosines),
        n=len(cosines),
        model=embedding_space(model),
    )
    return stats, tuple(float(v) for v in vectors[NULL_PROBES_V1.index(ENCODING_PROBE)])


def null_for(
    pool: Any,
    spec: RetrieveSpec,
    encode: BatchEncoder,
    model: str,
    *,
    vectors: int | None = None,
    stamp: str | None = None,
) -> NullStats | None:
    """The scope's null, from the cache or computed once; ``None`` with fewer
    than 30 vectors in scope or when the encoder cannot serve the probes.

    ``vectors`` and ``stamp`` are the dense lane's in-scope count and latest
    ``updated_at`` when the caller has them (saves one read of the scope). A
    cached null is served only while the encoder encodes
    :data:`ENCODING_PROBE` as it did when the null was computed.
    """
    if vectors is None:
        vectors, stamp = scope_signature(pool, spec, model)
    count = int(vectors)
    if count < MIN_VECTORS:
        return None
    key = scope_key(spec, model, count, stamp)
    hit = _cached(key)
    if hit is not None and same_encoding(hit[1], encoding_probe(encode)):
        return hit[0]
    computed = _compute(pool, spec, encode, model)
    if computed is None:
        return None
    _store(key, computed)
    return computed[0]


# ── the report CLI ──────────────────────────────────────────────────────────


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m umrs.knowledge.retrieval.calibrate",
        description="Print the dense null (mu, sigma) of a project scope (ADR-093 C2).",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--graph", action="append", default=[], help="graph id or name (repeatable)"
    )
    parser.add_argument(
        "--encoder-url", default="", help="rag service base (default: env)"
    )
    return parser.parse_args(argv)


def _graph_ids(pool: Any, tokens: list[str], project_id: str) -> tuple[str, ...] | None:
    """Graph ids for names or ids; prefers a graph of ``project_id``."""
    if not tokens:
        return None
    from umrs.knowledge.graph_registry import GraphRegistryStore

    registry = GraphRegistryStore(pool)
    chosen = []
    for token in tokens:
        candidates = registry.graph_candidates(token)
        own = [g for g in candidates if g.project_id == project_id] or candidates
        chosen.extend(g.graph_id for g in own[:1])
    return tuple(chosen)


def _env_encoder(url: str) -> BatchEncoder:
    import os

    from .dense import http_batch_encoder

    base = (
        url
        or os.environ.get("UMRS_KNOWLEDGE_QUERY_ENCODER_URL")
        or os.environ.get("UMRS_RAG_DIRECT_URL", "")
    )
    return http_batch_encoder(base, timeout=30.0)


def _default_pool() -> Any:
    from psycopg2.pool import ThreadedConnectionPool

    from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

    return ThreadedConnectionPool(0, 2, _resolve_knowledge_dsn())


def report(
    pool: Any, encode: BatchEncoder, project_id: str, graphs: list[str]
) -> dict[str, Any]:
    """The null of a scope, or why there is none, as one JSON-ready dict."""
    spec = RetrieveSpec(
        queries=("",),
        project_id=project_id,
        graph_ids=_graph_ids(pool, graphs, project_id),
    )
    model, vectors, reason = encode([NULL_PROBES_V1[0]])
    if not vectors:
        return {"project": project_id, "null": None, "reason": reason}
    count, stamp = scope_signature(pool, spec, model)
    stats = null_for(pool, spec, encode, model, vectors=count, stamp=stamp)
    out = {
        "project": project_id,
        "graphs": list(spec.graph_ids or ()),
        "space": embedding_space(model),
    }
    out.update({"vectors": count, "null": stats.to_dict() if stats else None})
    if stats is None:
        out["reason"] = (
            f"fewer than {MIN_VECTORS} vectors in scope"
            if count < MIN_VECTORS
            else "probes not encoded"
        )
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    pool: Any = None,
    encode: BatchEncoder | None = None,
) -> int:
    args = _parse(argv)
    out = report(
        pool or _default_pool(),
        encode or _env_encoder(args.encoder_url),
        args.project,
        args.graph,
    )
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0 if out.get("null") else 1


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
