"""ADR-093 wave B end to end: sentence-aware chunks from the embed lane to the
passage, through the bundle, and the re-chunk that leaves nothing stale.

Everything is the real code except the database, the object store and the
model:

- the in-process embed lane (``InProcessChunkEmbedder`` ->
  ``embed_worker.chunk_and_embed``) chosen by ``_chunk_embedder_for_site``
  from ``UMRS_KNOWLEDGE_CHUNK_SCHEME``, the header title read from the
  artifact row, the KNOWLEDGE_EMBED batch path (``_embed_sources``) and its
  chunk writer, the real ``KnowledgeStore`` upserts (``search_text``
  COALESCE), the orphan delete, the stale-vector delete and the stamp;
- hydrate (the ``search_text`` md5 rule after ``header_chars``), the
  neighbour merge (node ids ``node-chunk-{scope}{artifact}-{i}``) and
  ``pipeline.retrieve``;
- the export engine with ``chunk_text`` (r4-kb v1.3), the import engine and
  the ``search_text`` backfill.

The fake database evaluates exactly the statements those modules send over
in-memory tables (nodes, edges, vectors, artifacts) with the 0055/0075
``ON DELETE CASCADE`` rules. The abstract is shaped like a SciFact document
as the eval corpus stores it (``# title``, then the abstract), long enough
that sent-v1 cuts it in three overlapping chunks (integrator, 2026-10-10).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import io
import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import bundle_export, chunking, embedding_metadata
from umrs.knowledge import store as store_mod
from umrs.knowledge.bundle_export import KnowledgeExportEngine
from umrs.knowledge.bundle_import import KnowledgeImportEngine
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.retrieval import backfill, expand, hydrate, pipeline
from umrs.knowledge.retrieval.types import Candidate, RetrieveSpec
from umrs.knowledge.store import KnowledgeArtifact, KnowledgeNode, KnowledgeStore

PROJECT = "scifact-v1"
ART = "art-5f1a"
OTHER = "art-77c0"
BUCKET = "kb"
OLD_MODEL = "/models/e5-base-v2"  # the path the corpus was first embedded under
MODEL = "/opt/umrs/models/e5-base-v2"  # the site's path today: the same space
DIM = 256
TITLE = "Vitamin D supplementation and fracture risk in older adults"
ABSTRACT = (
    "BACKGROUND Low serum 25-hydroxyvitamin D is common in adults over 65 years, "
    "and observational cohorts link it to hip fracture. Whether daily cholecalciferol "
    "prevents fractures remains uncertain, e.g. trials differ in dose, adherence and "
    "baseline status. Previous meta-analyses (Bischoff-Ferrari et al. 2012) pooled "
    "heterogeneous regimens. We therefore designed a pragmatic trial with a fixed dose. "
    "METHODS We randomly assigned 2,256 community-dwelling adults aged 70 years or older "
    "to 2,000 IU of cholecalciferol per day or to matching placebo. Randomisation was "
    "stratified by sex, prior fracture and study centre. The primary outcome was any "
    "non-vertebral fracture confirmed by radiograph over 3.0 years. Secondary outcomes "
    "included hip fracture, falls, bone mineral density and serum calcium. Analyses "
    "followed the intention-to-treat principle with a Cox model. "
    "RESULTS Mean serum 25-hydroxyvitamin D rose from 22.4 to 37.6 ng/mL in the "
    "supplemented group and was unchanged with placebo. Non-vertebral fractures occurred "
    "in 9.1% of participants receiving cholecalciferol and in 9.6% receiving placebo "
    "(hazard ratio 0.94; 95% CI 0.71 to 1.25; P = 0.67). Hip fractures were rare in both "
    "groups. Falls did not differ between groups (rate ratio 1.02). Bone mineral density "
    "at the femoral neck increased by 0.8% with supplementation. Hypercalcaemia was "
    "recorded in 11 participants, nine of them in the supplemented group. "
    "Nephrolithiasis was reported by 14 participants receiving cholecalciferol and by "
    "12 receiving placebo. "
    "CONCLUSIONS Among older adults without marked deficiency, daily cholecalciferol did "
    "not lower the incidence of non-vertebral fracture over three years. Routine "
    "supplementation for fracture prevention is not supported by these data, although "
    "subgroups with severe deficiency were too small for firm conclusions."
)
DOC = f"# {TITLE}\n\n{ABSTRACT}"
OTHER_DOC = "# Zeolite cracking\n\nZeolite catalysts crack methanol into olefins at 450 C."


# ── the model and the object store ──────────────────────────────────────────

_PIECE = re.compile(r"\w{1,5}|[^\w\s]")  # word pieces: about e5's count on prose
_WORD = re.compile(r"[a-z][a-z0-9-]{3,}")


class _Tokenizer:
    """A fast-tokenizer stand-in: ids and character offsets of word pieces."""

    def __call__(self, text: str, **_kw: Any) -> dict[str, list]:
        spans = [m.span() for m in _PIECE.finditer(text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def _bucket(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % DIM


def bag_vector(text: str) -> list[float]:
    """A deterministic bag-of-words vector (the e5 prefixes carry no word)."""
    vector = np.zeros(DIM)
    for word in _WORD.findall(text.lower()):
        if word not in ("passage", "query"):
            vector[_bucket(word)] += 1.0
    norm = float(np.linalg.norm(vector)) or 1.0
    return list(vector / norm)


class _Embedder:
    """``SentenceTransformerEmbeddings``' surface: ``client.tokenizer`` and
    ``embed_documents``; records what it was asked to embed."""

    def __init__(self) -> None:
        self.client = type("Client", (), {"tokenizer": _Tokenizer()})()
        self.inputs: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        return [bag_vector(t) for t in texts]


def encode(texts: Sequence[str]) -> tuple[str, list[list[float]], str]:
    """The query encoder (a ``BatchEncoder``) in the site's space."""
    return MODEL, [bag_vector(t) for t in texts], ""


class _S3:
    def __init__(self, objects: dict[str, str]) -> None:
        self.objects = objects

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 — boto3's names
        return {"Body": io.BytesIO(self.objects[f"{Bucket}/{Key}"].encode("utf-8"))}


# ── the database ────────────────────────────────────────────────────────────


def _like(pattern: str) -> re.Pattern[str]:
    """A SQL ``LIKE`` pattern (backslash escapes) as a regex."""
    out, chars = "", iter(pattern)
    for char in chars:
        if char == "\\":
            out += re.escape(next(chars, ""))
        else:
            out += {"%": ".*", "_": "."}.get(char, re.escape(char))
    return re.compile(f"^{out}$", re.DOTALL)


class Db:
    """In-memory knowledge tables with the FK cascades of 0055/0075."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.vectors: dict[tuple[str, str], dict[str, Any]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}

    # writes of the embed writer and the store

    def upsert_node(self, args: tuple) -> str:
        node_id, project, _label, node_type, content, content_hash = args[:6]
        text = args[12] if len(args) > 12 else None
        row = self.nodes.get(node_id)
        if row is None:
            self.nodes[node_id] = {
                "project_id": project,
                "graph_id": args[9],
                "node_type": node_type,
                "content": copy.deepcopy(content),
                "content_hash": content_hash,
                "search_text": text,
            }
            return "INSERT 0 1"
        row.update(content=copy.deepcopy(content), content_hash=content_hash)
        row["search_text"] = text if text is not None else row["search_text"]
        return "INSERT 0 1"

    def upsert_edge(self, args: tuple) -> str:
        edge_id, project, source, target, relation = args[:5]
        self.edges[edge_id] = {
            "project_id": project,
            "source_id": source,
            "target_id": target,
            "relation_type": relation,
        }
        return "INSERT 0 1"

    def upsert_vector(self, args: tuple) -> str:
        node_id, model, _dim, literal, content_hash = args
        self.vectors[(node_id, model)] = {
            "embedding": json.loads(literal),
            "content_hash": content_hash,
        }
        return "INSERT 0 1"

    def drop_nodes(self, ids: set[str]) -> int:
        for node_id in ids:
            del self.nodes[node_id]
        self.edges = {
            k: e
            for k, e in self.edges.items()
            if e["source_id"] not in ids and e["target_id"] not in ids
        }
        self.vectors = {k: v for k, v in self.vectors.items() if k[0] not in ids}
        return len(ids)

    def delete_orphans(self, args: tuple) -> str:
        """``ka._DELETE_ORPHAN_CHUNKS_SQL``, predicate by predicate."""
        file_id, exact, project, artifact, prefix, keep = args
        candidates = set(exact) | {
            e["source_id"]
            for e in self.edges.values()
            if e["target_id"] == file_id and e["relation_type"] == "part_of"
        }
        doomed = {
            nid
            for nid in candidates
            if _orphan(self.nodes.get(nid), nid, project, artifact, prefix, keep)
        }
        return f"DELETE {self.drop_nodes(doomed)}"

    def delete_stale_vectors(self, args: tuple) -> str:
        """``ka._DELETE_STALE_VECTORS_SQL``."""
        ids, md5s, model, space, pattern = args
        wanted, like = dict(zip(ids, md5s, strict=True)), _like(pattern)
        doomed = [
            key
            for key, row in self.vectors.items()
            if key[0] in wanted
            and key[1] != model
            and _stale(key[1], row["content_hash"], wanted[key[0]], space, like)
        ]
        for key in doomed:
            del self.vectors[key]
        return f"DELETE {len(doomed)}"

    def stamp(self, args: tuple) -> str:
        model, dim, artifact, scheme, total, indexed = args
        row = self.artifacts[artifact]
        row.update(embedding_model=model, embedding_dim=dim, embedding_status="current")
        row["metadata"] = {
            **row["metadata"],
            "chunk_scheme": scheme,
            "chunks_total": total,
            "chunks_indexed": indexed,
        }
        return "UPDATE 1"

    # reads

    def chunks(self, prefix: str) -> list[tuple[str, dict[str, Any]]]:
        """The chunk rows whose id starts with ``prefix``, in index order."""
        rows = [
            (nid, row)
            for nid, row in self.nodes.items()
            if nid.startswith(prefix) and row["content"].get("kind") == "chunk"
        ]
        return sorted(rows, key=lambda pair: pair[1]["content"]["chunk_index"])

    def file_nodes(self, project: str, artifact: str, graphs: list[str]) -> list[tuple]:
        """``ka._FILE_NODES_SQL``: the artifact's file nodes in the graphs."""
        return [
            (nid,)
            for nid, row in self.nodes.items()
            if (row["project_id"], row["node_type"]) == (project, "file")
            and row["content"].get("artifact_id") == artifact
            and row["graph_id"] in graphs
        ]

    def label_rows(self, ids: list[str]) -> list[tuple]:
        return [
            (aid, a["metadata"].get("title"), a["title"], a["metadata"].get("source_name"), None)
            for aid, a in self.artifacts.items()
            if aid in ids
        ]


def _orphan(
    row: dict[str, Any] | None, nid: str, project: str, artifact: str, prefix: str, keep: int
) -> bool:
    if row is None:
        return False
    content = row["content"]
    index = str(content.get("chunk_index"))
    same = (row["project_id"], content.get("kind"), content.get("artifact_id")) == (
        project,
        "chunk",
        artifact,
    )
    return same and nid == prefix + index and index.isdigit() and int(index) >= keep


def _stale(model: str, held: str, md5: str, space: str, like: re.Pattern[str]) -> bool:
    return model == space or bool(like.match(model)) or (held != "" and held != md5)


class _AsyncConn:
    """asyncpg's surface over :class:`Db`: the statements the writer, the
    store, the stamp and the bundle engines send — nothing else."""

    def __init__(self, db: Db) -> None:
        self.db = db
        self.writes = {
            store_mod._UPSERT_NODE_SQL: db.upsert_node,
            store_mod._UPSERT_NODE_TEXT_SQL: db.upsert_node,
            ka._DELETE_ORPHAN_CHUNKS_SQL: db.delete_orphans,
            ka._DELETE_STALE_VECTORS_SQL: db.delete_stale_vectors,
            embedding_metadata._STAMP_CHUNKS_SQL: db.stamp,
        }

    async def execute(self, sql: str, *args: Any) -> str:
        if sql in self.writes:
            return self.writes[sql](args)
        if sql.lstrip().startswith("INSERT INTO knowledge_edges"):
            return self.db.upsert_edge(args)
        if sql.lstrip().startswith("INSERT INTO knowledge_node_embeddings"):
            return self.db.upsert_vector(args)
        raise AssertionError(f"unexpected write: {sql[:120]}")

    @contextlib.asynccontextmanager
    async def transaction(self):
        yield self  # the writer's one transaction per artifact (wave F)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if sql == ka._ARTIFACT_ROW_LOCK_SQL:  # the artifact's own row (wave F)
            row = self.db.artifacts.get(args[0])
            return 1 if row is not None and row["project_id"] == args[1] else None
        if sql == ka._FILE_NODE_SQL:  # the writer's own file node, by id
            row = self.db.nodes.get(args[0])
            return 1 if row is not None and row["project_id"] == args[1] else None
        assert sql == store_mod._SEARCH_TEXT_PROBE_SQL, sql[:120]
        return True

    async def fetch(self, sql: str, *args: Any) -> list[tuple]:
        if sql == ka._ARTIFACT_LABELS_SQL:
            return self.db.label_rows(list(args[0]))
        if sql == ka._FILE_NODES_SQL:
            return self.db.file_nodes(*args)
        if sql == bundle_export._CHUNK_SEARCH_TEXT_SQL:
            return [
                (nid, row["search_text"])
                for nid, row in self.db.nodes.items()
                if row["project_id"] == args[0] and row["search_text"] is not None
            ]
        if sql == bundle_export._ARTIFACT_REFS_SQL:
            return [
                (aid, a["source_ref"], a["metadata"].get("source_name"))
                for aid, a in self.db.artifacts.items()
            ]
        return []  # the export's other optional tables: empty here

    async def fetchrow(self, sql: str, *_args: Any) -> Any:
        if sql == bundle_export._PROJECT_FACTS_SQL:
            return {"owner_id": "alice", "data_class": "non_sensitive"}
        return None  # no catalog row, no identity collision


class AsyncPool:
    def __init__(self, db: Db) -> None:
        self.conn = _AsyncConn(db)

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self.conn

    async def fetch(self, sql: str, *args: Any) -> list[tuple]:
        return await self.conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return await self.conn.fetchrow(sql, *args)


_HYDRATE_MARKER = "AND a.project_id = n.project_id WHERE n.node_id = ANY(%s)"


class _Cursor:
    """psycopg2's cursor over :class:`Db`: hydrate, the dense lane and the
    backfill."""

    def __init__(self, db: Db) -> None:
        self.db, self.rows, self.rowcount = db, [], 0

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        handler = next((h for marker, h in self._handlers() if marker in sql), None)
        assert handler is not None, f"unexpected SQL: {sql[:160]}"
        self.rows = handler(list(params))

    def _handlers(self) -> list[tuple[str, Any]]:
        return [
            (hydrate._PROBE_SQL, lambda _p: [(1,)]),
            (_HYDRATE_MARKER, self._hydrate_rows),
            (expand._SCOPE_SQL, self._scope_rows),
            ("GROUP BY 1, 2", self._chunk_counts),
            ("count(*) OVER () AS in_scope", self._dense),
            ("/* backfill:counts */", self._backfill_counts),
            ("/* backfill:rows */", self._backfill_rows),
            ("/* backfill:write */", self._backfill_write),
        ]

    def fetchall(self) -> list[tuple]:
        return list(self.rows)

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None

    def _artifact(self, row: dict[str, Any]) -> dict[str, Any]:
        """The join: the node's artifact, of the node's own project only."""
        art = self.db.artifacts.get(row["content"].get("artifact_id"), {})
        return art if art.get("project_id") == row["project_id"] else {}

    def _scope_rows(self, params: list[Any]) -> list[tuple]:
        """``expand._SCOPE_SQL``: (node, project, graph, tier, artifact)."""
        rows = (self.db.nodes.get(nid) for nid in params[0])
        return [
            (nid, r["project_id"], r["graph_id"], "project", r["content"].get("artifact_id"))
            for nid, r in zip(params[0], rows, strict=True)
            if r is not None and r["content"].get("kind") == "chunk"
        ]

    def _hydrate_rows(self, params: list[Any]) -> list[tuple]:
        out = []
        for nid in params[0]:
            row = self.db.nodes.get(nid)
            if row is not None:
                art = self._artifact(row)
                meta = art.get("metadata", {})
                content = json.dumps(row["content"])
                labels = (meta.get("source_name"), None, art.get("title"))
                ref = art.get("source_ref")
                out.append((nid, row["graph_id"], content, ref, *labels, row["search_text"]))
        return out

    def _chunk_counts(self, params: list[Any]) -> list[tuple]:
        counts: dict[tuple[str, str], int] = {}
        for row in self.db.nodes.values():
            key = (row["graph_id"], row["content"].get("artifact_id"))
            if row["content"].get("kind") == "chunk" and key[0] in params[0]:
                counts[key] = counts.get(key, 0) + 1
        return [(g, a, n) for (g, a), n in counts.items() if a in params[1]]

    def _dense(self, params: list[Any]) -> list[tuple]:
        query, project, model, pattern = np.asarray(json.loads(params[0])), *params[1:4]
        like = _like(pattern)
        scored = sorted(
            (1.0 - float(np.dot(query, v["embedding"])), nid)
            for (nid, m), v in self.db.vectors.items()
            if (m == model or like.match(m))
            and self.db.nodes[nid]["project_id"] == project
            and self.db.nodes[nid]["content"].get("kind") == "chunk"
        )
        return [
            (nid, json.dumps(self.db.nodes[nid]["content"]), d, len(scored))
            for d, nid in scored[: int(params[-1])]
        ]

    def _pending(self, project: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (nid, row)
            for nid, row in self.db.nodes.items()
            if row["project_id"] == project and row["content"].get("kind") == "chunk"
        ]

    def _backfill_counts(self, params: list[Any]) -> list[tuple]:
        rows = self._pending(params[0])
        return [(len(rows), sum(1 for _n, r in rows if r["search_text"] is not None))]

    def _backfill_rows(self, params: list[Any]) -> list[tuple]:
        out = []
        for nid, row in sorted(self._pending(params[0])):
            if row["search_text"] is None:
                art = self._artifact(row)
                meta = json.dumps(art.get("metadata", {}))
                ids = (art.get("artifact_id"), art.get("source_ref"))
                out.append((nid, json.dumps(row["content"]), *ids, "document", meta))
        return out

    def _backfill_write(self, params: list[Any]) -> list[tuple]:
        text, nid, md5 = params
        row = self.db.nodes.get(nid)
        writable = row is not None and row["search_text"] is None
        self.rowcount = int(writable and row["content"].get("chunk_md5") == md5)
        if self.rowcount:
            row["search_text"] = text
        return []


class _SyncConn:
    def __init__(self, db: Db) -> None:
        self.db = db

    def __enter__(self) -> _SyncConn:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self) -> _Cursor:
        return _Cursor(self.db)


class SyncPool:
    def __init__(self, db: Db) -> None:
        self.db = db

    def getconn(self) -> _SyncConn:
        return _SyncConn(self.db)

    def putconn(self, _conn: Any) -> None:
        return None


class Store(KnowledgeStore):
    """The real store's writes over :class:`Db`; the reads the bundle
    engines use come straight from the tables."""

    def __init__(self, db: Db) -> None:
        super().__init__(AsyncPool(db))
        self.db = db

    async def list_artifacts(self, **_kw: Any) -> tuple[list[KnowledgeArtifact], int]:
        rows = [
            KnowledgeArtifact(
                artifact_id=aid,
                content_hash=hashlib.sha256(a["text"].encode()).hexdigest(),
                kind="document",
                project_id=a["project_id"],
                source_ref=a["source_ref"],
                size_bytes=len(a["text"].encode()),
                mime_type="text/markdown",
            )
            for aid, a in self.db.artifacts.items()
        ]
        return rows, len(rows)

    async def list_irs(self, **_kw: Any) -> list:
        return []

    async def list_nodes(self, **_kw: Any) -> list[KnowledgeNode]:
        return [
            KnowledgeNode(
                node_id=nid,
                project_id=row["project_id"],
                label=nid,
                node_type=row["node_type"],
                content=copy.deepcopy(row["content"]),
                content_hash=row["content_hash"],
                graph_id=row["graph_id"],
            )
            for nid, row in self.db.nodes.items()
        ]

    async def list_edges(self, **_kw: Any) -> list:
        return []

    async def insert_artifact(self, **_kw: Any) -> bool:
        return True

    async def insert_ir(self, **_kw: Any) -> bool:
        return True

    async def rebind_artifact_source(self, *_a: Any) -> bool:
        return True


# ── the site ────────────────────────────────────────────────────────────────


@pytest.fixture
def site(monkeypatch):
    """A site whose embedder is :class:`_Embedder`; hydrate re-probes 0120."""
    embedder = _Embedder()
    monkeypatch.setattr(ka, "_get_node_embedder", lambda: embedder)
    monkeypatch.setattr(hydrate, "_SEARCH_TEXT", None)
    monkeypatch.delenv("UMRS_EMBEDDING_SANDBOX", raising=False)
    monkeypatch.delenv("UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("UMRS_RAG_MERGE", raising=False)
    return embedder


def _configure(monkeypatch, *, model: str, scheme: str, window: int = 256) -> None:
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", model)
    monkeypatch.setenv(chunking.CHUNK_SCHEME_ENV, scheme)
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_CHUNK_TOKENS", str(window))


def _seed(db: Db, artifact: str, text: str, graphs: Sequence[str] = ("",)) -> None:
    """The artifact row and its file nodes, as upload + understand leave them."""
    db.artifacts[artifact] = {
        "artifact_id": artifact,
        "project_id": PROJECT,
        "title": "",
        "metadata": {"source_name": f"{artifact}.md"},
        "source_ref": f"s3://{BUCKET}/{artifact}.md",
        "text": text,
    }
    for graph in graphs:
        scope = f"{graph}-" if graph else ""
        db.nodes[f"node-file-{scope}{artifact}"] = {
            "project_id": PROJECT,
            "graph_id": graph or default_graph_id(PROJECT),
            "node_type": "file",
            "content": {"artifact_id": artifact},
            "content_hash": "h",
            "search_text": None,
        }


def _embed(db: Db, artifacts: Sequence[str], graph: str = "") -> ka.KnowledgeEmbedResult:
    """One KNOWLEDGE_EMBED run over ``artifacts`` with the site's lane."""
    lane, reason = ka._chunk_embedder_for_site()
    assert lane is not None, reason
    sources = [{"artifact_id": a, "source_ref": db.artifacts[a]["source_ref"]} for a in artifacts]
    objects = {f"{BUCKET}/{a}.md": db.artifacts[a]["text"] for a in artifacts}
    result = ka.KnowledgeEmbedResult()
    graph_id = graph or default_graph_id(PROJECT)
    asyncio.run(
        ka._embed_sources(
            Store(db), sources, PROJECT, graph_id, lane, (_S3(objects), BUCKET), result
        )
    )
    assert result.failed_items == []
    return result


def _expected() -> list[dict[str, Any]]:
    return chunking.sentence_chunks(DOC, title=TITLE, tokenizer=_Tokenizer())


def _ingested(monkeypatch) -> Db:
    db = Db()
    _seed(db, ART, DOC)
    _configure(monkeypatch, model=MODEL, scheme="sent-v1")
    _embed(db, [ART])
    return db


# ── 1. sent-v1 from the lane to the stored rows ─────────────────────────────


def test_a_scifact_abstract_is_cut_into_overlapping_headed_sent_v1_chunks(monkeypatch, site):
    db = _ingested(monkeypatch)
    contents = [row["content"] for _nid, row in db.chunks(f"node-chunk-{ART}-")]
    cuts = [(c["chunk_index"], c["chunk_start"], c["chunk_end"], c["header"]) for c in contents]
    expected = [(i, c["start"], c["end"], TITLE) for i, c in enumerate(_expected())]
    assert cuts == expected and len(cuts) == 3  # the header title is the '# ' heading
    assert contents[1]["chunk_start"] < contents[0]["chunk_end"]  # the overlap sentence
    assert site.inputs[0].startswith(f"passage: {TITLE}\n# {TITLE}")


def test_every_sent_v1_row_verifies_through_hydrate_and_carries_its_scheme(monkeypatch, site):
    db = _ingested(monkeypatch)
    for nid, row in db.chunks(f"node-chunk-{ART}-"):
        content = row["content"]
        body = hydrate.stored_text(content, row["search_text"])
        assert body == DOC[content["chunk_start"] : content["chunk_end"]]
        assert row["search_text"] == f"{content['header']}\n{body}"
        assert (content["chunk_scheme"], content["header_chars"]) == (
            "sent-v1",
            len(TITLE) + 1,
        )
        assert db.vectors[(nid, MODEL)]["content_hash"] == content["chunk_md5"]


def test_the_stamp_records_the_scheme_and_the_counts(monkeypatch, site):
    db = _ingested(monkeypatch)
    art = db.artifacts[ART]
    assert (art["embedding_model"], art["embedding_status"]) == (MODEL, "current")
    assert {k: art["metadata"][k] for k in ("chunk_scheme", "chunks_total", "chunks_indexed")} == {
        "chunk_scheme": "sent-v1",
        "chunks_total": 3,
        "chunks_indexed": 3,
    }


# ── 2. hydrate, the neighbour merge, the pipeline ───────────────────────────


def _hydrated(db: Db, index: int) -> Candidate:
    candidate = Candidate(node_id=f"node-chunk-{ART}-{index}", chunk_index=index)
    hydrate.hydrate(SyncPool(db), [candidate], None)
    return candidate


def test_hydrate_reads_the_body_and_the_trusted_labels(monkeypatch, site):
    db = _ingested(monkeypatch)
    middle = _hydrated(db, 1)
    start, end = middle.span
    assert (middle.text, middle.text_source) == (DOC[start:end], "search_text")
    assert (middle.source_name, middle.chunk_count, middle.artifact_id) == (
        f"{ART}.md",
        3,
        ART,
    )
    assert not middle.text.startswith(TITLE)  # the header is searched, never shown


def test_the_neighbour_merge_stitches_overlapping_chunks_by_their_ids(monkeypatch, site):
    db = _ingested(monkeypatch)
    middle = _hydrated(db, 1)
    (merged,) = expand.merge_neighbours(SyncPool(db), [middle], max_passage_tokens=2000)
    assert merged.span == (0, len(DOC))  # chunks 0 and 2 joined through the overlap
    assert merged.text == DOC
    assert merged.section == "part 1-3/3"


def test_the_pipeline_returns_the_verified_sent_v1_passage(monkeypatch, site):
    db = _ingested(monkeypatch)
    spec = RetrieveSpec(
        queries=("How often was nephrolithiasis reported with cholecalciferol?",),
        project_id=PROJECT,
        lexical="off",
        rerank="off",
        merge="on",
    )
    result = pipeline.retrieve(SyncPool(db), spec, encode=encode)
    top = result.passages[0]
    assert (top.source_id, top.text_source) == (f"node-chunk-{ART}-2", "search_text")
    assert top.text == DOC[top.span[0] : top.span[1]] and "Nephrolithiasis" in top.text
    assert (result.searched, result.space, result.lanes["merge"]["status"]) == (
        3,
        "e5-base-v2",
        "ok",
    )


# ── 3. the bundle: export --chunk-text, import, backfill ────────────────────


def _export(db: Db, tmp_path, monkeypatch) -> Any:
    monkeypatch.setattr(KnowledgeExportEngine, "_resolve_export_snapshot", lambda *_a: None)
    objects = {a["source_ref"]: a["text"].encode() for a in db.artifacts.values()}
    engine = KnowledgeExportEngine(
        store=Store(db),
        output_dir=tmp_path,
        allow_unsigned=True,
        artifact_reader=objects.get,
        chunk_text=True,
    )
    asyncio.run(engine.export_project(PROJECT))
    return tmp_path / PROJECT


def _import(bundle: Any) -> tuple[Db, Any]:
    target = Db()
    engine = KnowledgeImportEngine(store=Store(target))
    result = asyncio.run(
        engine.import_project(bundle, target_project_id="scifact-import", allow_unsigned=True)
    )
    return target, result


def _search_texts(db: Db) -> dict[str, str | None]:
    return {nid: row["search_text"] for nid, row in db.chunks("node-chunk-")}


def test_export_chunk_text_then_import_restores_the_headed_search_text(tmp_path, monkeypatch, site):
    db = _ingested(monkeypatch)
    bundle = _export(db, tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (bundle / "nodes.jsonl").read_text().splitlines()]
    chunk_rows = [r for r in rows if r["content"].get("kind") == "chunk"]
    assert json.loads((bundle / "manifest.json").read_text())["version"] == "1.3"
    assert {r["text_header"] for r in chunk_rows} == {f"{TITLE}\n"}
    target, result = _import(bundle)
    assert _search_texts(target) == _search_texts(db)
    assert (result.search_text_from_node, result.search_text_pending) == (3, 0)


def test_a_chunk_exported_through_the_artifact_slice_regains_its_header(
    tmp_path, monkeypatch, site
):
    """No stored text at the source (an un-backfilled row): the body travels
    as the md5-verified slice WITHOUT ``text_header``; the import rebuilds
    the header from ``content.header`` (the writer's rule), so hydrate's
    ``search_text[header_chars:]`` still verifies at the destination."""
    db = _ingested(monkeypatch)
    expected = _search_texts(db)
    for _nid, row in db.chunks("node-chunk-"):
        row["search_text"] = None
    bundle = _export(db, tmp_path, monkeypatch)
    target, result = _import(bundle)
    assert _search_texts(target) == expected
    assert (result.search_text_from_node, result.search_text_pending) == (3, 0)


def test_the_backfill_restores_header_plus_body(monkeypatch, site):
    db = _ingested(monkeypatch)
    expected = _search_texts(db)
    for _nid, row in db.chunks("node-chunk-"):
        row["search_text"] = None
    report = backfill.run_backfill(SyncPool(db), PROJECT, load_document=lambda _a: DOC)
    assert _search_texts(db) == expected
    assert (report.outcomes["from_artifact"], report.written) == (3, 3)


# ── 4. the re-chunk: fixed-v0 -> sent-v1 leaves no stale index row ──────────


def _snapshot(db: Db, prefixes: Sequence[str]) -> dict[str, Any]:
    def keep(key: str) -> bool:
        return key.startswith(tuple(prefixes))

    return {
        "nodes": {k: v for k, v in db.nodes.items() if keep(k)},
        "edges": {k: v for k, v in db.edges.items() if keep(v["source_id"])},
        "vectors": {k: v for k, v in db.vectors.items() if keep(k[0])},
    }


BYSTANDERS = (f"node-chunk-{OTHER}-", f"node-chunk-g-named-{ART}-")


def _fixed_v0_corpus(monkeypatch) -> tuple[Db, dict[str, Any]]:
    """ART in the default and a named graph, OTHER beside it: all cut
    fixed-v0 with a 96-token window under the old model path."""
    db = Db()
    _seed(db, ART, DOC, graphs=("", "g-named"))
    _seed(db, OTHER, OTHER_DOC)
    _configure(monkeypatch, model=OLD_MODEL, scheme="fixed-v0", window=96)
    _embed(db, [ART, OTHER])
    _embed(db, [ART], graph="g-named")
    return db, copy.deepcopy(_snapshot(db, BYSTANDERS))


def _dangling(db: Db) -> list[str]:
    edges = [
        e["source_id"]
        for e in db.edges.values()
        if e["source_id"] not in db.nodes or e["target_id"] not in db.nodes
    ]
    return edges + [k[0] for k in db.vectors if k[0] not in db.nodes]


def test_a_rechunk_from_fixed_v0_to_sent_v1_leaves_no_stale_index_rows(monkeypatch, site):
    db, bystanders = _fixed_v0_corpus(monkeypatch)
    assert len(db.chunks(f"node-chunk-{ART}-")) == 6
    _configure(monkeypatch, model=MODEL, scheme="sent-v1")
    result = _embed(db, [ART])  # what KNOWLEDGE_REINDEX enqueues for the default graph
    rows = db.chunks(f"node-chunk-{ART}-")
    assert [nid for nid, _r in rows] == [f"node-chunk-{ART}-{i}" for i in range(3)]
    assert (result.chunks_removed, result.vectors_removed) == (3, 3)
    assert _dangling(db) == []
    assert _snapshot(db, BYSTANDERS) == bystanders  # other scopes untouched
    _assert_one_current_vector_each(db, rows)


def _assert_one_current_vector_each(db: Db, rows: list[tuple[str, dict[str, Any]]]) -> None:
    for nid, row in rows:
        held = {model: v["content_hash"] for (n, model), v in db.vectors.items() if n == nid}
        assert held == {MODEL: row["content"]["chunk_md5"]}, nid
        assert row["content"]["chunk_scheme"] == "sent-v1"
        assert hydrate.stored_text(row["content"], row["search_text"]) is not None


def test_after_the_rechunk_the_read_side_sees_only_the_new_cut(monkeypatch, site):
    db, _bystanders = _fixed_v0_corpus(monkeypatch)
    _configure(monkeypatch, model=MODEL, scheme="sent-v1")
    _embed(db, [ART])
    last = _hydrated(db, 2)
    meta = db.artifacts[ART]["metadata"]
    assert (last.chunk_count, last.text_source) == (3, "search_text")
    assert (meta["chunk_scheme"], meta["chunks_total"], meta["chunks_indexed"]) == (
        "sent-v1",
        3,
        3,
    )
    beyond = Candidate(node_id=f"node-chunk-{ART}-3", chunk_index=3)
    hydrate.hydrate(SyncPool(db), [beyond], None)
    assert beyond.text == ""  # the old chunk 3 is gone, not merely unranked
