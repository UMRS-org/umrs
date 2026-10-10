"""K3: unified knowledge query facade (read-side, additive).

Route: ``GET /umrs/knowledge/query``

One search entry across UMRS knowledge backends, replacing per-route
fan-out by agents/R4 (catalog / knowledge / provenance / artifact /
graph). Backends are composed via the ``search_api`` UnifiedSearch
port protocols. v1 wires the catalog metadata search; backends that
are not wired yet (knowledge graph, embeddings, RAG) are reported in
``unavailable_backends`` with reasons instead of silently returning
empty results — the response is honest about coverage.

WP-15 (2026-09-07, ADR-027 P1b facade leg): the ``rag`` scope is now
wired over the NODE-KEYED central corpus (knowledge_nodes + 0075
embeddings) — never the endpoint-coupled mini-RiVault corpus. Hits
carry ``metadata.corpus_venue: central-pg`` so callers see the venue.

ADR-093 C7 (2026-10-10): ``scope=all`` is fused by weighted RRF in
``search_api``; the ``rag`` lane is the IDF-weighted lexical chunk lane of
the retrieval core (``knowledge.retrieval.lexical``), no longer a whole-phrase
ILIKE ordered by recency; the embeddings and rag lanes honour ``graphs=``
with the legacy ``graph_id ''`` rule, and without a selection read the
whole project minus ``exclude_graphs=`` and the graphs the caller may not
read — the retrieve verb's scope (``api/rag_retrieve.py``); the graph lane
matches query tokens OR-wise and scores the matched fraction; and the
response carries the retrieval core's ``sufficiency`` verdict, so a consumer
can tell "found nothing that answers this" from "found rows".

Existing subsystem routes stay untouched (aliases per K3 guardrail).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request

from umrs.knowledge.access_scope import visibility_predicate
from umrs.knowledge.graph_registry import (
    GraphRegistryStore,
    KnowledgeGraphMeta,
    default_graph_id,
    graph_id_is_default,
)
from umrs.search_api import SearchScope, UnifiedSearch

from ._access import graph_meta_readable
from ._scope_identity import request_caller_scope
from .catalog_store import get_optional_catalog_meta_store
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-query"])

_VALID_SCOPES = {scope.value for scope in SearchScope}

_UNWIRED_BACKENDS: dict[str, str] = {
    "embeddings": "query-embedding generation requires the embedding service",
    "rag": "central knowledge corpus unavailable (knowledge nodes not present)",
}


def _hit_to_dict(hit: Any, *, project_id: str) -> dict[str, Any]:
    """Map a CatalogHit to the ``search_api`` conversion dict shape."""
    meta = hit.meta
    return {
        "title": meta.s3_key.rsplit("/", 1)[-1],
        "content": meta.summary or meta.deterministic_summary or "",
        "score": float(hit.score),
        "source_id": meta.s3_key,
        "source_type": meta.detected_modality or "object",
        "metadata": {
            "owner": meta.owner,
            "tags": list(meta.tags),
            "acl_scope": meta.acl_scope,
            "project_id": project_id,
        },
    }


class _CatalogSearchBackend:
    """``search_api`` CatalogSearchable port over the live catalog store."""

    def __init__(self, meta_store: Any, *, include_ambient: bool = False) -> None:
        self._store = meta_store
        self._include_ambient = include_ambient

    def search_catalog(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if not project_id:
            # Catalog ACL is project-scoped; the facade requires it anyway.
            return []
        from umrs.data_catalog.query import search_catalog

        hits = search_catalog(
            self._store,
            query=query,
            project_id=project_id,
            limit=limit,
            include_ambient=self._include_ambient,
        )
        return [_hit_to_dict(hit, project_id=project_id) for hit in hits]


@dataclass(frozen=True)
class _GraphScope:
    """A graph selection as the ownership half of a node predicate.

    Legacy rows (graph_id '') belong to their project's DEFAULT graph only —
    the rule graph_content and list_nodes use. Mapping every project-bound
    graph let a shared non-default graph expose its owner's legacy rows
    (review of KG-05, 2026-10-09). Every K3 lane that reads nodes renders
    this one clause, so ``graphs=`` means the same thing on each of them.
    """

    graph_ids: tuple[str, ...]
    default_projects: tuple[str, ...]

    @classmethod
    def of(cls, selected: Sequence[KnowledgeGraphMeta]) -> _GraphScope:
        defaults = (
            g.project_id
            for g in selected
            if g.project_id and graph_id_is_default(g.graph_id, g.project_id)
        )
        return cls(
            graph_ids=tuple(g.graph_id for g in selected),
            default_projects=tuple(dict.fromkeys(defaults)),
        )

    def owned_sql(self, alias: str = "") -> str:
        col = f"{alias}." if alias else ""
        return (
            f"({col}graph_id = ANY(%s) OR "
            f"({col}graph_id = '' AND {col}project_id = ANY(%s)))"
        )

    def params(self) -> tuple[list[str], list[str]]:
        return list(self.graph_ids), list(self.default_projects)


def _exclusion_sql(
    graph_ids: Sequence[str], project_id: str | None
) -> tuple[str, list[Any]]:
    """`` AND`` clauses leaving ``graph_ids`` out of a whole-project read.

    Legacy rows (``graph_id ''``) are the project's default graph: they go
    with it — the retrieve verb's rule (``retrieval.dense``).
    """
    if not graph_ids:
        return "", []
    sql = " AND n.graph_id <> ALL(%s)"
    if project_id and default_graph_id(project_id) in graph_ids:
        sql += " AND n.graph_id <> ''"
    return sql, [list(graph_ids)]


# The graph lane matches whole query words, not the whole phrase: a claim is
# rarely a substring of a node. Short words, stop words and duplicates carry
# no signal; the cap bounds the scan (one LIKE per token per node).
_GRAPH_TOKEN_MIN = 3
_GRAPH_TOKEN_MAX = 8
_TOKEN_EDGE = ".,;:!?()[]{}<>\"'`"
_STOP_WORD_TEXT = (
    "about after all also and are been before being between both but can could "
    "did does doing during each few for from had has have having her here his "
    "how into its itself just more most not now off once only other our out "
    "over own same she should some such than that the their them then there "
    "these they this those through too under until very was were what when "
    "where which while who whom why will with would you your"
)
_STOP_WORDS = frozenset(_STOP_WORD_TEXT.split())

# The node's text without its JSON keys: a token like "chunk" or "kind" must
# not match every node through the key names of its payload. Lax mode never
# raises on an odd payload shape (it may repeat a string; matching is a set).
_NODE_TEXT_SQL = (
    "lower(coalesce(label, '') || ' ' || coalesce("
    "jsonb_path_query_array(content, '$.** ? (@.type() == \"string\")')::text, ''))"
)


def _graph_tokens(query: str) -> list[str]:
    """Distinct lowercase query words worth matching, or the whole phrase."""
    words = (w.strip(_TOKEN_EDGE).lower() for w in query.split())
    kept = [w for w in words if len(w) >= _GRAPH_TOKEN_MIN and w not in _STOP_WORDS]
    tokens = list(dict.fromkeys(kept))[:_GRAPH_TOKEN_MAX]
    return tokens or [query.strip().lower()]


def _like_contains(token: str) -> str:
    """``%token%`` with LIKE metacharacters escaped (backslash is the escape)."""
    escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _graph_lane_sql(gate: str, n_tokens: int) -> str:
    """Count matched tokens per node once, then rank by that count.

    The haystack is lowered once per row (MATERIALIZED), so each token costs
    one LIKE; the gate's parameters stay first, then the token patterns.
    """
    any_text = " + ".join(["(s.body LIKE %s)::int"] * n_tokens)
    in_label = " + ".join(["(s.lab LIKE %s)::int"] * n_tokens)
    return (
        "WITH s AS MATERIALIZED ("
        "SELECT node_id, project_id, label, node_type, content, graph_id, "
        f"lower(coalesce(label, '')) AS lab, {_NODE_TEXT_SQL} AS body "
        f"FROM knowledge_nodes WHERE {gate}), "
        "m AS (SELECT node_id, project_id, label, node_type, content, graph_id, "
        f"{any_text} AS hits, {in_label} AS label_hits FROM s) "
        "SELECT node_id, project_id, label, node_type, content, graph_id, hits "
        "FROM m WHERE hits > 0 "
        "ORDER BY hits DESC, label_hits DESC, node_id LIMIT %s"
    )


class _GraphSearchBackend:
    """``search_api`` KnowledgeGraphSearchable port over the canonical store.

    Searches ``knowledge_nodes`` scoped to the selected graphs. Legacy rows
    (``graph_id = ''``) are visible through their project's default graph —
    the same fallback ``KnowledgeStore.list_nodes`` uses.

    ADR-093 C7: a node matches when any query token occurs in its label or
    text values; its score is the fraction of the tokens it contains, so a
    node naming every term outranks one that shares a single word.
    """

    def __init__(
        self,
        pool: Any,
        selected: list[KnowledgeGraphMeta],
        *,
        include_ambient: bool = False,
    ) -> None:
        self._include_ambient = include_ambient
        self._pool = pool
        self._scope = _GraphScope.of(selected)
        self._by_id = {g.graph_id: g for g in selected}
        self._default_by_project = {
            g.project_id: g
            for g in selected
            if g.project_id and graph_id_is_default(g.graph_id, g.project_id)
        }

    def search_knowledge_graph(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        tokens = _graph_tokens(query)
        patterns = [_like_contains(t) for t in tokens]
        # ADR-050 tier gate rendered from the single predicate; ownership
        # half is the graph selection (explicit graphs + legacy '' rows).
        gate = visibility_predicate(
            owned_sql=self._scope.owned_sql(), include_ambient=self._include_ambient
        )
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        _graph_lane_sql(gate, len(patterns)),
                        (*self._scope.params(), *patterns, *patterns, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(len(tokens), row) for row in rows]

    def _hit(self, n_tokens: int, row: tuple) -> dict[str, Any]:
        node_id, project_id, label, node_type, content, graph_id, hits = row
        meta = self._by_id.get(graph_id) or self._default_by_project.get(project_id)
        return {
            "title": label or node_id,
            "content": str(content)[:200] if content else "",
            "score": round(int(hits) / max(1, n_tokens), 6),
            "source_id": node_id,
            "source_type": "knowledge_graph_node",
            "metadata": {
                "graph_id": graph_id or (meta.graph_id if meta else ""),
                "graph_name": meta.name if meta else "",
                "node_type": node_type,
                "project_id": project_id,
                "matched_tokens": int(hits),
                "query_tokens": n_tokens,
            },
        }


def _resolve_exclusions(
    registry: GraphRegistryStore, exclude_csv: str | None
) -> tuple[set[str], list[str]]:
    """Map exclude tokens to graph ids (unknown tokens only recorded by name)."""
    excluded_ids: set[str] = set()
    excluded_names: list[str] = []
    for token in _split_csv(exclude_csv):
        # Every graph the token can name: includes pick the first readable
        # candidate, so excluding only one would let a name slip through and
        # reveal whether another project holds a graph of that name.
        excluded_ids.update(g.graph_id for g in registry.graph_candidates(token))
        excluded_names.append(token)
    return excluded_ids, excluded_names


GraphReadable = Callable[[KnowledgeGraphMeta], bool]


def _resolve_includes(
    registry: GraphRegistryStore,
    tokens: list[str],
    excluded_ids: set[str],
    readable: GraphReadable,
) -> list[KnowledgeGraphMeta]:
    """Resolve include tokens, dropping excluded, duplicate and unreadable graphs.

    KG-05: ``resolve_graph`` matches an id, or a name across every project
    (a default graph is named after its project), so a caller could read
    another project's nodes by naming its graph. A graph the caller may not
    read is skipped exactly like an unknown token: existence is not disclosed.
    """
    selected: list[KnowledgeGraphMeta] = []
    for token in tokens:
        graph = next((g for g in registry.graph_candidates(token) if readable(g)), None)
        if graph is None or graph.graph_id in excluded_ids:
            continue
        if graph.graph_id in {g.graph_id for g in selected}:
            continue
        selected.append(graph)
    return selected


def _resolve_selected_graphs(
    registry: GraphRegistryStore,
    project_id: str,
    graphs_csv: str | None,
    exclude_csv: str | None,
    readable: GraphReadable | None = None,
) -> tuple[list[KnowledgeGraphMeta], list[str]]:
    """Resolve explicit include/exclude tokens into selected graph metas.

    Explicit exclusion beats inclusion. Without explicit inclusion the
    project's default graph is the single candidate (registry discovery /
    candidate ranking lives client-side in the selection policy).
    """
    excluded_ids, excluded_names = _resolve_exclusions(registry, exclude_csv)

    tokens = _split_csv(graphs_csv)
    if tokens:
        # No predicate, no named graphs: fail closed.
        allow = readable or (lambda _graph: False)
        return _resolve_includes(registry, tokens, excluded_ids, allow), excluded_names

    default = registry.ensure_default_graph(project_id)
    selected = [default] if default.graph_id not in excluded_ids else []
    return selected, excluded_names


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


# Query-time text encoder (R6.1): the RAG service owns it, this facade only
# asks. Unset → UMRS_RAG_DIRECT_URL (the same service), else no encoder.
_QUERY_ENCODER_URL_ENV = "UMRS_KNOWLEDGE_QUERY_ENCODER_URL"
_QUERY_ENCODER_TIMEOUT_ENV = "UMRS_KNOWLEDGE_QUERY_ENCODER_TIMEOUT"

# (model_name, vector or None, reason when None)
QueryEncoder = Callable[[str], tuple[str, list[float] | None, str]]
# The retrieval core's BatchEncoder shape: texts -> (model, vectors | None, reason)
BatchQueryEncoder = Callable[[Sequence[str]], tuple[str, list[list[float]] | None, str]]


def _query_encoder_url() -> str:
    url = os.environ.get(_QUERY_ENCODER_URL_ENV) or os.environ.get(
        "UMRS_RAG_DIRECT_URL"
    )
    return (url or "").strip().rstrip("/")


def _query_encoder_timeout() -> float:
    try:
        return max(0.5, float(os.environ.get(_QUERY_ENCODER_TIMEOUT_ENV, "5")))
    except ValueError:
        return 5.0


def _encode_query(query: str) -> tuple[str, list[float] | None, str]:
    """Embed a query through the RAG service's ``POST /rag/embed``.

    The service applies the model's query prefix (e5 ``query:``) and names
    the model, so the caller can rank only vectors from that space (ADR-083).
    """
    base = _query_encoder_url()
    if not base:
        return "", None, "no query encoder configured"
    try:
        response = httpx.post(
            f"{base}/rag/embed", json={"text": query}, timeout=_query_encoder_timeout()
        )
        response.raise_for_status()
        body = response.json()
        return str(body["model"]), [float(v) for v in body["vector"]], ""
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("query encoder at %s unavailable: %s", base, exc)
        return "", None, f"query encoder unavailable ({type(exc).__name__})"


class _EmbeddingsSearchBackend:
    """``search_api`` EmbeddingsSearchable port over migration-0075 rows.

    Two honest lanes (hits carry ``metadata.lane``):

    - ``table-reader`` (default): exposes the canonical node-vector rows
      as honest hits — model/dim/content hash + the owning node — filtered
      by the query against node identity and model name, recency-ranked.
    - ``similarity`` (P3, 2026-08-23): real cosine ANN ranking when the
      caller supplies a PRE-COMPUTED ``query_embedding``. Query-time text
      embedding stays owned by
      ``umrs.api.rag_service.PrefixedSentenceTransformerEmbeddings``
      (R6.1 boundary) — this facade never loads a model; it only ranks
      vectors. Scale note: exact scan (`<=>`) is fine well past the
      current corpus size; the typed-shadow-column + HNSW migration is
      the documented follow-up when rows cross ~100k.

    No mock/fake fallback is ever synthesized: without a query vector the
    lane stays table-reader.

    ADR-093 C7: with a ``graph_scope`` (the caller named graphs) both
    sub-lanes read only the selected graphs' rows, under the legacy
    ``graph_id ''`` rule; without one they keep the project-wide rule
    (``rag_spine`` legacy parity) minus ``exclude_graph_ids``.
    """

    def __init__(
        self,
        pool: Any,
        *,
        include_ambient: bool = False,
        encoder: QueryEncoder | None = None,
        graph_scope: _GraphScope | None = None,
    ) -> None:
        self._pool = pool
        self._include_ambient = include_ambient
        self._encoder = encoder
        self._graph_scope = graph_scope
        # Graphs a whole-project read leaves out (query_knowledge binds them).
        self.exclude_graph_ids: tuple[str, ...] = ()
        # Which lane the last search ran and why — reported to the caller.
        self.lane_report: dict[str, str] | None = None
        # The last encoder-driven similarity search (the sufficiency input):
        # its space and its hits, best first. None when it did not run.
        self.dense_model: str = ""
        self.dense_hits: list[dict[str, Any]] | None = None

    def _gate(self, project_id: str | None) -> tuple[str, tuple[Any, ...]]:
        """ADR-050 tier gate on the owning node, owned by graph or by project."""
        if self._graph_scope is None:
            exclude, params = _exclusion_sql(self.exclude_graph_ids, project_id)
            gate = visibility_predicate("n", include_ambient=self._include_ambient)
            return gate + exclude, (project_id, *params)
        gate = visibility_predicate(
            "n",
            owned_sql=self._graph_scope.owned_sql("n"),
            include_ambient=self._include_ambient,
        )
        return gate, self._graph_scope.params()

    def search_embeddings(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
        query_embedding: list[float] | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if query_embedding:
            return self._similarity_hits(
                query_embedding,
                model_name=model_name,
                project_id=project_id,
                limit=limit,
            )
        if self._encoder is None:
            self.lane_report = {"lane": "table-reader", "reason": "no query encoder"}
            return self._table_reader_hits(query, project_id=project_id, limit=limit)
        model, vector, reason = self._encoder(query)
        if not vector:
            # ADR-083 D-4: an encoder outage is reported, never answered by
            # a lexical match labelled "embeddings".
            self.lane_report = {"lane": "unavailable", "reason": reason}
            return []
        hits = self._similarity_hits(
            vector, model_name=model, project_id=project_id, limit=limit
        )
        self.dense_model, self.dense_hits = model, hits
        self.lane_report = {"lane": "similarity", "model": model}
        if not hits:
            # ADR-083 D-5: a space with no vectors here is no evidence.
            self.lane_report["reason"] = (
                f"no vectors in embedding space {model!r} for this project"
            )
        return hits

    def _table_reader_hits(
        self, query: str, *, project_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        # ADR-050 tier gate on the owning node (R-02 wave 2: this lane was
        # previously ungated — project-only, ambient rows still served).
        gate, gate_params = self._gate(project_id)
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT e.node_id, e.model_name, e.dimension, e.created_at, "
                        "n.label, n.project_id "
                        "FROM knowledge_node_embeddings e "
                        "JOIN knowledge_nodes n ON n.node_id = e.node_id "
                        f"WHERE {gate} "
                        "AND (e.model_name ILIKE %s OR n.label ILIKE %s OR e.node_id ILIKE %s) "
                        "ORDER BY e.created_at DESC LIMIT %s",
                        (*gate_params, pattern, pattern, pattern, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(row) for row in rows]

    def _similarity_hits(
        self,
        query_embedding: list[float],
        *,
        model_name: str | None,
        project_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Cosine-distance ANN over 0075 rows using a pre-computed vector.

        Ranks one embedding space only (ADR-083 D-2): vectors from another
        model are not comparable, and a same-dimension model would mix
        silently instead of failing. A space is the model's basename
        (text_embedder.embedding_space), so the bundled model and a copy staged
        at another path rank together.
        """
        if not model_name:
            raise ValueError("a query vector needs the model_name of its space")
        from umrs.knowledge.text_embedder import embedding_space

        qv = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        # Same space = same model name, or the same model staged elsewhere.
        space = embedding_space(model_name).replace("\\", "\\\\")
        same_space = "%/" + space.replace("%", "\\%").replace("_", "\\_")
        gate, gate_params = self._gate(project_id)
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT e.node_id, e.model_name, e.dimension, e.created_at, "
                        "n.label, n.project_id, (e.embedding <=> %s::vector) AS distance, "
                        "n.content::text "
                        "FROM knowledge_node_embeddings e "
                        "JOIN knowledge_nodes n ON n.node_id = e.node_id "
                        f"WHERE {gate} "
                        "AND (e.model_name = %s OR e.model_name LIKE %s) "
                        "ORDER BY distance ASC LIMIT %s",
                        (qv, *gate_params, model_name, same_space, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(row) for row in rows]

    @staticmethod
    def _hit(row: tuple) -> dict[str, Any]:
        node_id, model_name, dimension, _created, label, project_id = row[:6]
        content = f"vector row for {node_id} ({model_name}, {dimension}d)"
        if len(row) > 6:  # similarity lane: cosine distance in position 7
            distance = float(row[6])
            score, lane = max(0.0, 1.0 - distance), "similarity"
            extra = {"distance": distance}
            # The matched text, so a semantic hit is readable on its own.
            content = _first_text_field(row[7] if len(row) > 7 else "") or content
        else:
            score, lane, extra = 0.5, "table-reader", {}
        return {
            "title": label or node_id,
            "content": content,
            "score": score,
            "source_id": node_id,
            "source_type": "knowledge_node_embedding",
            "metadata": {
                "model_name": model_name,
                "dimension": dimension,
                "project_id": project_id,
                "lane": lane,
                **extra,
            },
        }


def _embeddings_backend_for(
    pool: Any,
    *,
    include_ambient: bool = False,
    encoder: QueryEncoder | None = None,
    graph_scope: _GraphScope | None = None,
) -> tuple[_EmbeddingsSearchBackend | None, str | None]:
    """Wire the 0075 backend iff the table exists; honesty reason otherwise."""
    if pool is None:
        return None, "embeddings search requires the database"
    try:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('knowledge_node_embeddings')")
                    found = cur.fetchone()
                present = bool(found and found[0])
        finally:
            pool.putconn(conn)
    except Exception as exc:  # noqa: BLE001 — honesty over hard failure
        logger.warning("0075 probe failed, embeddings backend unwired: %s", exc)
        return None, "embeddings search unavailable (table probe failed)"
    if not present:
        return None, "embeddings search unavailable (migration 0075 not applied)"
    return (
        _EmbeddingsSearchBackend(
            pool,
            include_ambient=include_ambient,
            encoder=encoder,
            graph_scope=graph_scope,
        ),
        None,
    )


def _first_text_field(content_text: str) -> str | None:
    """First non-empty text-ish value of a JSONB payload, or None."""
    try:
        parsed = json.loads(content_text) if content_text else {}
    except (ValueError, TypeError):
        parsed = {}
    return _first_text_value(parsed)


def _first_text_value(parsed: Any) -> str | None:
    """First non-empty text-ish value of a parsed payload, or None."""
    if not isinstance(parsed, dict):
        return None
    for key in ("text", "preview", "chunk", "excerpt", "summary"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:240]
    return None


def _retrieve_spec(
    query: str,
    project_id: str,
    *,
    graph_scope: _GraphScope | None,
    include_ambient: bool,
    exclude_graph_ids: Sequence[str] = (),
) -> Any:
    """The retrieval core's request for this K3 query (``None``: core absent).

    ``graph_ids`` is ``None`` without a selection: the whole project under
    the legacy rule minus ``exclude_graph_ids``, exactly as the other K3
    lanes read it.
    """
    try:
        from umrs.knowledge.retrieval.types import RetrieveSpec
    except ImportError:
        return None
    return RetrieveSpec(
        queries=(query,),
        project_id=project_id,
        graph_ids=graph_scope.graph_ids if graph_scope is not None else None,
        exclude_graph_ids=tuple(exclude_graph_ids),
        include_ambient=include_ambient,
        k_max=_site_k_max(),
    )


def _site_k_max() -> int:
    """The retrieve verb's k_max (``UMRS_RAG_FINAL_K_MAX``): K3's verdict cuts
    the dense ranking as the verb does before judging it."""
    from .rag_retrieve import site_k_max

    return site_k_max()


def _run_lexical_lane(pool: Any, spec: Any, limit: int) -> tuple[Any, dict[str, Any]]:
    """``(LaneResult | None, report)`` — a missing or failing lane is reported."""
    if spec is None:
        return None, _lane_unavailable("retrieval core not installed")
    try:
        from umrs.knowledge.retrieval.lexical import lexical_lane
    except ImportError as exc:
        return None, _lane_unavailable(f"lexical lane not installed ({exc})")
    try:
        lane = lexical_lane(pool, spec.query, spec, limit=limit)
    except Exception as exc:  # noqa: BLE001 — honesty over hard failure
        logger.warning("lexical lane failed: %s", exc)
        return None, _lane_unavailable(f"lexical lane failed ({type(exc).__name__})")
    return lane, {"lane": "lexical", **lane.report()}


def _lane_unavailable(reason: str) -> dict[str, Any]:
    return {"lane": "lexical", "status": "unavailable", "n": 0, "reason": reason}


def _lexical_hit(candidate: Any, lane_name: str) -> dict[str, Any]:
    """A lexical-lane chunk in the ``search_api`` conversion dict shape."""
    content = candidate.content if isinstance(candidate.content, dict) else {}
    artifact_id = candidate.artifact_id or str(content.get("artifact_id") or "")
    title = f"chunk {candidate.chunk_index} of {artifact_id}" if artifact_id else ""
    return {
        "title": title or candidate.node_id,
        "content": (candidate.text or "")[:240] or _first_text_value(content) or "",
        "score": float(candidate.lane_scores.get(lane_name, 0.0)),
        "source_id": candidate.node_id,
        "source_type": "knowledge_corpus_node",
        "metadata": {
            "lane": "lexical",
            "corpus_venue": "central-pg",
            "artifact_id": artifact_id,
            "chunk_index": candidate.chunk_index,
        },
    }


class _RAGSearchBackend:
    """``search_api`` RAGSearchable port over the node-keyed central corpus.

    ADR-027 P1b facade leg (WP-15): the ``rag`` scope answers from
    ``knowledge_nodes`` in central Postgres — the corpus the upload/intake
    paths mint into — never a deployed mini-RiVault endpoint, with
    ``metadata.corpus_venue`` naming the venue.

    ADR-093 C7: the lane is the retrieval core's lexical chunk lane (IDF-
    weighted OR full-text search over ``search_text``, C6), scored by its
    IDF sum. It replaced a whole-phrase ILIKE ordered by recency with a fixed
    score. When the core or its column is absent the lane is reported
    unavailable with the reason (``lane_report``), never answered by a
    weaker match under the same name (ADR-083 D-4).
    """

    def __init__(
        self,
        pool: Any,
        *,
        include_ambient: bool = False,
        graph_scope: _GraphScope | None = None,
    ) -> None:
        self._pool = pool
        self._include_ambient = include_ambient
        self._graph_scope = graph_scope
        # Graphs a whole-project read leaves out (query_knowledge binds them).
        self.exclude_graph_ids: tuple[str, ...] = ()
        self.lane_report: dict[str, Any] | None = None
        # The last lane's candidates when it ran ok (the sufficiency input).
        self.lexical_candidates: list[Any] | None = None

    def search_rag(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        spec = _retrieve_spec(
            query,
            project_id or "",
            graph_scope=self._graph_scope,
            include_ambient=self._include_ambient,
            exclude_graph_ids=self.exclude_graph_ids,
        )
        lane, self.lane_report = _run_lexical_lane(self._pool, spec, limit)
        if lane is None or lane.status != "ok":
            self.lexical_candidates = None
            return []
        self.lexical_candidates = list(lane.candidates)
        return [_lexical_hit(c, lane.name) for c in lane.candidates[:limit]]


def _rag_backend_for(
    pool: Any,
    *,
    include_ambient: bool = False,
    graph_scope: _GraphScope | None = None,
) -> tuple[_RAGSearchBackend | None, str | None]:
    """Wire the central-corpus RAG backend iff knowledge_nodes exists."""
    if pool is None:
        return None, "RAG search requires the database"
    try:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('knowledge_nodes')")
                    found = cur.fetchone()
                present = bool(found and found[0])
        finally:
            pool.putconn(conn)
    except Exception as exc:  # noqa: BLE001 — honesty over hard failure
        logger.warning("corpus probe failed, RAG backend unwired: %s", exc)
        return None, "RAG search unavailable (corpus probe failed)"
    if not present:
        return (
            None,
            "central knowledge corpus unavailable (knowledge nodes not present)",
        )
    return (
        _RAGSearchBackend(
            pool, include_ambient=include_ambient, graph_scope=graph_scope
        ),
        None,
    )


def _build_unified_search(
    meta_store: Any,
    graph_backend: _GraphSearchBackend | None,
    graph_unavailable_reason: str | None,
    embeddings_backend: _EmbeddingsSearchBackend | None = None,
    embeddings_unavailable_reason: str | None = None,
    rag_backend: _RAGSearchBackend | None = None,
    rag_unavailable_reason: str | None = None,
    include_ambient: bool = False,
) -> tuple[UnifiedSearch, list[str], dict[str, str]]:
    """Compose pluggable backends.

    Returns ``(search, wired_backend_names, unwired_backend_reasons)``.
    """
    wired = ["catalog"]
    unwired = dict(_UNWIRED_BACKENDS)
    kwargs: dict[str, Any] = {
        "catalog": _CatalogSearchBackend(meta_store, include_ambient=include_ambient)
    }
    if graph_backend is not None:
        kwargs["knowledge_graph"] = graph_backend
        wired.append("knowledge_graph")
    else:
        unwired["knowledge_graph"] = graph_unavailable_reason or "unavailable"
    if embeddings_backend is not None:
        kwargs["embeddings"] = embeddings_backend
        unwired.pop("embeddings", None)
        wired.append("embeddings")
    elif embeddings_unavailable_reason is not None:
        unwired["embeddings"] = embeddings_unavailable_reason
    if rag_backend is not None:
        kwargs["rag"] = rag_backend
        unwired.pop("rag", None)
        wired.append("rag")
    elif rag_unavailable_reason is not None:
        unwired["rag"] = rag_unavailable_reason
    return UnifiedSearch(**kwargs), wired, unwired


# ── sufficiency (ADR-093 D-3 / C2, reported by K3 since C7) ─────────────────


def _sufficiency_unknown(detail: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "basis": "",
        "value": None,
        "threshold": None,
        "detail": detail,
    }


def _dense_candidates(emb_backend: _EmbeddingsSearchBackend | None) -> list[Any] | None:
    """The similarity hits as retrieval-core candidates, or None if it did not run."""
    hits = emb_backend.dense_hits if emb_backend is not None else None
    if hits is None:
        return None
    from umrs.knowledge.retrieval.types import Candidate

    ordered = sorted(hits, key=lambda h: float(h["score"]), reverse=True)
    return [
        Candidate(
            node_id=str(hit["source_id"]),
            cosine=float(hit["score"]),
            lane_scores={"dense": float(hit["score"])},
            lane_ranks={"dense": rank},
            text=str(hit.get("content") or ""),  # the S1 term-coverage rule reads it
        )
        for rank, hit in enumerate(ordered, start=1)
    ]


def _null_encoder() -> BatchQueryEncoder:
    """The batch query encoder for null calibration.

    The retrieval verb's ``batch_encode`` (``api/rag_retrieve.py``, one
    ``/rag/embed`` call per batch) when present; else one ``/rag/embed`` call
    per text through this facade's encoder — the same vectors, cold only once
    per scope (the null is cached by the core).
    """
    try:
        module = importlib.import_module(".rag_retrieve", __package__)
    except ImportError:
        return _encode_each
    return getattr(module, "batch_encode", _encode_each)


def _encode_each(texts: Sequence[str]) -> tuple[str, list[list[float]] | None, str]:
    vectors: list[list[float]] = []
    model = ""
    for text in texts:
        model, vector, reason = _encode_query(text)
        if not vector:
            return model, None, reason
        vectors.append(vector)
    return model, vectors, ""


def _null_stats(pool: Any, spec: Any, model: str) -> Any:
    """The calibrated null of this scope's space, or None (lexical rule only)."""
    try:
        from umrs.knowledge.retrieval.calibrate import null_for
    except ImportError:
        return None
    return null_for(pool, spec, _null_encoder(), model)


def _lexical_strength(lexical: list[Any]) -> float | None:
    """The S3 rule's input: the largest IDF of one matched query lexeme, the
    basis the retrieve pipeline uses (C2), or None without the lex lane."""
    if not lexical:
        return None
    try:
        from umrs.knowledge.retrieval.lexical import lexical_hit_strength
    except ImportError:
        return None
    return float(lexical_hit_strength(lexical, per_term=True))


@dataclass(frozen=True)
class _SufficiencyInputs:
    pool: Any
    spec: Any
    emb_backend: _EmbeddingsSearchBackend | None
    rag_backend: _RAGSearchBackend | None


_NO_QUERY_VECTOR = (
    "no query vector: the dense lane did not run, and lexical rows alone are "
    "no evidence (the retrieval core's rule)"
)


def _sufficiency_block(inputs: _SufficiencyInputs) -> dict[str, Any]:
    """The retrieval core's verdict on this query's dense and lexical lanes.

    ``unknown`` (with the reason) when the core is not installed or the
    dense lane did not run — the core judges nothing without a query vector
    and never from lexical rows alone (review core-4/core-5) — never a guess.
    """
    rag = inputs.rag_backend
    lexical = rag.lexical_candidates if rag is not None else None
    if inputs.spec is None:
        return _sufficiency_unknown("retrieval core not installed")
    dense = _dense_candidates(inputs.emb_backend)
    if dense is None:
        none_ran = "no dense or lexical lane ran for this query"
        return _sufficiency_unknown(none_ran if lexical is None else _NO_QUERY_VECTOR)
    try:
        cutoff = importlib.import_module("umrs.knowledge.retrieval.cutoff")
    except ImportError as exc:
        return _sufficiency_unknown(f"sufficiency unavailable ({exc})")
    return _verdict(inputs, cutoff, dense, lexical or [])


def _kept_strength(kept: list[Any], lexical: list[Any]) -> float | None:
    """The largest matched-lexeme IDF among the KEPT passages the lex lane
    also found — never the whole lexical candidate list, where any rare word
    of a nonsense query lent its IDF (review core-1, 2026-10-10)."""
    ids = {c.node_id for c in kept}
    return _lexical_strength([c for c in lexical if c.node_id in ids])


def _verdict(
    inputs: _SufficiencyInputs, cutoff: Any, dense: list[Any], lexical: list[Any]
) -> dict[str, Any]:
    """The pipeline's rule (C2): cut the dense ranking, read the lexical
    evidence over the kept passages only, and judge against the null."""
    emb = inputs.emb_backend
    try:
        kept = cutoff.cut(dense, inputs.spec, mode="dense")
        null = _null_stats(inputs.pool, inputs.spec, emb.dense_model) if kept else None
        evidence = cutoff.lexical_evidence(
            inputs.spec.queries, kept, strength=_kept_strength(kept, lexical)
        )
        verdict = cutoff.sufficiency(
            kept,
            mode="fused" if lexical else "dense",
            null=null,
            lexical_hit=evidence,
            searched=len(dense),
        )
    except Exception as exc:  # noqa: BLE001 — a verdict failure never fails the search
        logger.warning("K3 sufficiency failed: %s", exc)
        return _sufficiency_unknown(f"sufficiency failed ({type(exc).__name__})")
    return verdict.to_dict()


def _parse_scope(raw: str) -> SearchScope:
    """Validate the scope query parameter against ``SearchScope`` names."""
    if raw in _VALID_SCOPES:
        return SearchScope(raw)
    raise_typed(
        PlatformError.knowledgequery(
            "bad_request",
            f"Unknown scope '{raw}'. Valid values: {sorted(_VALID_SCOPES)}",
        ),
    )


def _resolve_project_id(identity: CallerIdentity, project_id: str | None) -> str:
    """Verified project context wins; fall back to the explicit parameter."""
    resolved = identity.project_id or (project_id or "")
    if not resolved:
        raise_typed(
            PlatformError.knowledgequery(
                "bad_request", "project_id is required (or a verified project header)"
            ),
        )
    return resolved


@router.get("/query")
def query_knowledge(
    request: Request,
    q: str = Query(..., min_length=1, max_length=512, description="Free-text query"),
    project_id: str = Query(default=None, description="Project scope"),
    scope: str = Query(
        default="all",
        description="catalog | knowledge_graph | embeddings | rag | all",
    ),
    graphs: str = Query(
        default=None,
        description="Comma-separated graph ids/names to query (default: project graph)",
    ),
    exclude_graphs: str = Query(
        default=None,
        description="Comma-separated graph ids/names to exclude (wins over includes)",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """K3 facade: one search entry across UMRS knowledge backends.

    The knowledge-graph backend is wired when the durable pool exists and
    the graph selection is non-empty; every hit carries graph provenance
    (``metadata.graph_id`` / ``metadata.graph_name``). With ``graphs=`` the
    embeddings and rag lanes read the same selection; without it they read
    the whole project minus ``exclude_graphs=`` and the graphs the caller may
    not read, as ``/rag/retrieve`` does. ``sufficiency`` is the retrieval
    core's verdict on the dense and lexical lanes (ADR-093 C7).
    """
    meta_store = get_optional_catalog_meta_store()
    if meta_store is None:
        raise_typed(
            PlatformError.knowledgequery(
                "unavailable", "Catalog metadata store is unavailable"
            ),
        )
    resolved_project = _resolve_project_id(identity, project_id)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    # ADR-050 wave-2.1: the ambient read lane is admin-only; regular callers
    # never see ambient rows on any lane.
    include_ambient = bool(identity.is_admin)
    scope_for_graphs = request_caller_scope(request)
    backend, reason, selected, excluded = _graph_backend_for(
        pool,
        resolved_project,
        graphs,
        exclude_graphs,
        include_ambient=include_ambient,
        readable=lambda graph: graph_meta_readable(request, graph, scope_for_graphs),
    )
    # ADR-093 C7: a named selection binds every lane, not only the graph lane.
    graph_scope = _GraphScope.of(selected) if _split_csv(graphs) else None
    emb_backend, emb_reason = _embeddings_backend_for(
        pool,
        include_ambient=include_ambient,
        encoder=_encode_query if _query_encoder_url() else None,
        graph_scope=graph_scope,
    )
    rag_backend, rag_reason = _rag_backend_for(
        pool, include_ambient=include_ambient, graph_scope=graph_scope
    )
    lane_excluded = _bind_lane_exclusions(
        _LaneScope(request, pool, resolved_project, graphs, exclude_graphs),
        emb_backend,
        rag_backend,
    )
    search, wired, unwired = _build_unified_search(
        meta_store,
        backend,
        reason,
        emb_backend,
        emb_reason,
        rag_backend,
        rag_reason,
        include_ambient=include_ambient,
    )
    results = search.search(
        q,
        scope=_parse_scope(scope),
        project_id=resolved_project,
        limit=limit,
    )
    payload = results.to_dict()
    payload["wired_backends"] = wired
    payload["unavailable_backends"] = unwired
    _add_lane_reports(payload, emb_backend, rag_backend)
    spec = _retrieve_spec(
        q,
        resolved_project,
        graph_scope=graph_scope,
        include_ambient=include_ambient,
        exclude_graph_ids=lane_excluded,
    )
    payload["sufficiency"] = _sufficiency_block(
        _SufficiencyInputs(pool, spec, emb_backend, rag_backend)
    )
    payload["selected_graphs"] = [g.graph_id for g in selected]
    payload["excluded_graphs"] = excluded
    payload["ontologies"] = _ontology_blocks_for(
        selected, getattr(request.app.state, "umrs_db_pool", None)
    )
    return {"status": "ok", "query": payload}


@dataclass(frozen=True)
class _LaneScope:
    """The request facts the passage lanes' whole-project scope is built from."""

    request: Request
    pool: Any
    project_id: str
    graphs: str | None
    exclude_graphs: str | None


def _bind_lane_exclusions(
    scope: _LaneScope,
    emb_backend: _EmbeddingsSearchBackend | None,
    rag_backend: _RAGSearchBackend | None,
) -> tuple[str, ...]:
    """Without a named selection the embeddings and rag lanes read the whole
    project minus the excluded graphs and the graphs the caller may not read
    (another member's personal graph, KG-05) — the retrieve verb's scope.

    An exclusion alone used to bind both lanes to the default graph, dropping
    every other graph of the project (review of ADR-093, 2026-10-10).
    """
    wired = [b for b in (emb_backend, rag_backend) if b is not None]
    if _split_csv(scope.graphs) or not wired or scope.pool is None:
        return ()
    from .rag_retrieve import hidden_graphs

    excluded = set(hidden_graphs(scope.request, scope.pool, scope.project_id))
    if _split_csv(scope.exclude_graphs):
        named, _names = _resolve_exclusions(
            GraphRegistryStore(scope.pool), scope.exclude_graphs
        )
        excluded |= named
    ids = tuple(sorted(excluded))
    for backend in wired:
        backend.exclude_graph_ids = ids
    return ids


def _add_lane_reports(
    payload: dict[str, Any],
    emb_backend: _EmbeddingsSearchBackend | None,
    rag_backend: _RAGSearchBackend | None,
) -> None:
    """Which sub-lane each wired backend ran, and why (ADR-083 D-4)."""
    if emb_backend is not None and emb_backend.lane_report is not None:
        payload["embeddings_lane"] = emb_backend.lane_report
    if rag_backend is not None and rag_backend.lane_report is not None:
        payload["rag_lane"] = rag_backend.lane_report


def _graph_backend_for(
    pool: Any,
    project_id: str,
    graphs_csv: str | None,
    exclude_csv: str | None,
    *,
    include_ambient: bool = False,
    readable: GraphReadable | None = None,
) -> tuple[_GraphSearchBackend | None, str | None, list, list]:
    """Build the graph backend or an honesty reason, plus selection truth."""
    if pool is None:
        return (
            None,
            "knowledge graph search requires the database",
            [],
            _split_csv(exclude_csv),
        )
    registry = GraphRegistryStore(pool)
    selected, excluded = _resolve_selected_graphs(
        registry, project_id, graphs_csv, exclude_csv, readable
    )
    if not selected:
        return None, "no graphs selected (empty or fully excluded)", selected, excluded
    return (
        _GraphSearchBackend(pool, selected, include_ambient=include_ambient),
        None,
        selected,
        excluded,
    )


def _ontology_registry_for(pool: Any) -> tuple[Any | None, Any | None]:
    """Open a registry sync conn branched off a psycopg-shaped pool.

    Returns (registry, conn); both None when the registry is unavailable —
    callers then resolve via inline provenance only.
    """
    from umrs.knowledge.ontology_registry import OntologyRegistryStore

    if pool is None or not hasattr(pool, "getconn"):
        return None, None
    try:
        conn = pool.getconn()
    except Exception:  # noqa: BLE001 — registry degrades to inline only
        return None, None
    return OntologyRegistryStore(conn), conn


def _ontology_blocks_for(selected: list, pool: Any) -> list[dict[str, Any]]:
    """ADR-049 Leg B/A: the bound ontology per selected graph, registry
    (0091) resolving before inline provenance; unresolvable references are
    named via ontology_error, never smudged."""
    from umrs.knowledge.ontology import ontology_for_graph

    registry, conn = _ontology_registry_for(pool)
    blocks: list[dict[str, Any]] = []
    try:
        for graph in selected:
            meta = (
                graph
                if isinstance(graph, dict)
                else (graph.to_dict() if hasattr(graph, "to_dict") else {})
            )
            block = {"graph_id": meta.get("graph_id", "?")}
            try:
                blocks.append(
                    {
                        **block,
                        "ontology": ontology_for_graph(
                            meta, registry=registry
                        ).to_dict(),
                    }
                )
            except ValueError as exc:
                blocks.append({**block, "ontology_error": str(exc)})
    finally:
        if conn is not None and hasattr(pool, "putconn"):
            try:
                pool.putconn(conn)
            except Exception:  # noqa: BLE001 — pool return must not mask the response
                pass
    return blocks
