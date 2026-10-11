"""Bundle import restores no row of an artifact it deduplicated (ADR-093
D-11, wave F, 2026-10-11).

``insert_artifact`` stores no row for a bundle artifact whose bytes the
target project already holds under another id. The import still restored
that artifact's IR, file and chunk nodes, part_of edges and trusted node
vectors under the bundle's id: orphans with vectors and search_text, which
the dense and lexical lanes would read. And every node exported without an
IR carried ``source_ir = ''``, which violates knowledge_nodes_source_ir_fkey
on a real database and aborted the import mid-way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from umrs.knowledge.bundle_import import KnowledgeImportEngine
from umrs.knowledge.store import ArtifactWriteError

MODEL = "/srv/models/e5-base-v2"


class _Conn:
    def __init__(self, pool: _Pool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _Conn:
        return self

    async def __aexit__(self, *_a: Any) -> bool:
        return False

    async def execute(self, sql: str, *params: Any) -> str:
        if "INSERT INTO knowledge_node_embeddings" in sql:
            self.pool.vectors.append(params[0])
        return "INSERT 0 1"

    async def fetchval(self, *_a: Any) -> Any:
        return True  # the search_text column exists

    async def fetchrow(self, *_a: Any) -> Any:
        return None  # no identity collision


class _Pool:
    def __init__(self) -> None:
        self.vectors: list[str] = []

    def acquire(self) -> _Conn:
        return _Conn(self)


class _Store:
    """KnowledgeStore's write contract: insert_artifact returns the id that
    holds the bytes (``held``: hash -> stored id) or raises by name."""

    def __init__(self, held: dict[str, str], refused: frozenset[str] = frozenset()) -> None:
        self._pool = _Pool()
        self.held, self.refused = held, refused
        self.artifacts: list[str] = []
        self.irs: list[str] = []
        self.nodes: list[Any] = []
        self.edges: list[str] = []

    async def insert_artifact(self, **kw: Any) -> str:
        if kw["artifact_id"] in self.refused:
            raise ArtifactWriteError(f"artifact {kw['artifact_id']} exists in another project")
        if kw["content_hash"] in self.held:
            return self.held[kw["content_hash"]]
        self.artifacts.append(kw["artifact_id"])
        return kw["artifact_id"]

    async def insert_ir(self, **kw: Any) -> bool:
        self.irs.append(kw["ir_id"])
        return True

    async def upsert_node(self, node: Any) -> None:
        self.nodes.append(node)

    async def upsert_edge(self, edge: Any) -> None:
        self.edges.append(edge.edge_id)

    async def rebind_artifact_source(self, *_a: Any) -> bool:
        return True


def _nodes(art: str, ir: str) -> list[dict[str, Any]]:
    tag = art[-1]
    return [
        {"node_id": f"node-file-{art}", "node_type": "file", "label": art,
         "content": {"artifact_id": art}, "content_hash": f"f{tag}", "source_ir": ir},
        {"node_id": f"node-chunk-{art}-0", "node_type": "entity", "label": "chunk",
         "content": {"kind": "chunk", "artifact_id": art}, "content_hash": f"c{tag}",
         "source_ir": None},
        {"node_id": f"node-sem-{tag}", "node_type": "concept", "label": "zeolite",
         "content": {}, "content_hash": f"s{tag}", "source_ir": ir},
    ]  # fmt: skip


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    tables = {
        "artifacts": [
            {"artifact_id": "art-A", "content_hash": "hA", "kind": "document"},
            {"artifact_id": "art-B", "content_hash": "hB", "kind": "document"},
        ],
        "documents": [
            {"ir_id": "ir-A", "artifact_id": "art-A", "content_hash": "iA"},
            {"ir_id": "ir-B", "artifact_id": "art-B", "content_hash": "iB"},
        ],
        "nodes": _nodes("art-A", "ir-A") + _nodes("art-B", "ir-B"),
        "edges": [
            {"edge_id": "e-A", "source_id": "node-chunk-art-A-0", "target_id": "node-file-art-A"},
            {"edge_id": "e-B", "source_id": "node-chunk-art-B-0", "target_id": "node-file-art-B"},
            {"edge_id": "e-BA", "source_id": "node-sem-B", "target_id": "node-sem-A"},
        ],
        "node_embeddings": [
            {"node_id": f"node-chunk-{a}-0", "model_name": MODEL, "dimension": 2,
             "embedding": [1.0, 0.0], "content_hash": "c"}
            for a in ("art-A", "art-B")
        ],
    }  # fmt: skip
    for name, rows in tables.items():
        (bundle / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    manifest = {"format": "r4-kb", "version": "1.2", "project_id": "p", "tables": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


async def _import(tmp_path: Path, store: _Store) -> Any:
    engine = KnowledgeImportEngine(store=store)
    return await engine.import_project(_bundle(tmp_path), verify=False, allow_unsigned=True)


@pytest.fixture(autouse=True)
def _site(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", MODEL)


@pytest.mark.asyncio
async def test_no_node_edge_or_vector_under_a_deduplicated_bundle_id(tmp_path: Path) -> None:
    store = _Store(held={"hA": "art-stored"})
    result = await _import(tmp_path, store)
    assert store.artifacts == ["art-B"]
    assert store.irs == ["ir-B"]
    assert [n.node_id for n in store.nodes] == [
        "node-file-art-B",
        "node-chunk-art-B-0",
        "node-sem-B",
    ]
    assert store.edges == ["e-B"]  # e-A and e-BA touch art-A's nodes
    assert store._pool.vectors == ["node-chunk-art-B-0"]
    assert result.deduped_artifacts == {"art-A": "art-stored"}


@pytest.mark.asyncio
async def test_the_import_result_names_what_it_skipped(tmp_path: Path) -> None:
    result = await _import(tmp_path, _Store(held={"hA": "art-stored"}))
    # ir-A, 3 nodes, e-A, e-BA, 1 vector
    assert result.rows_skipped_deduped == 7
    assert result.to_dict()["deduped_artifacts"] == {"art-A": "art-stored"}


@pytest.mark.asyncio
async def test_a_refused_artifact_row_restores_nothing_of_its_own(tmp_path: Path) -> None:
    """An id held by another project raises ArtifactWriteError: the row is
    skipped by name and its derived rows go with it, the import goes on."""
    store = _Store(held={}, refused=frozenset({"art-B"}))
    result = await _import(tmp_path, store)
    assert store.artifacts == ["art-A"]
    assert result.deduped_artifacts == {"art-B": ""}
    assert {n.node_id for n in store.nodes} == {
        "node-file-art-A",
        "node-chunk-art-A-0",
        "node-sem-A",
    }


@pytest.mark.asyncio
async def test_a_node_exported_without_an_ir_is_restored_with_none(tmp_path: Path) -> None:
    store = _Store(held={})
    await _import(tmp_path, store)
    by_id = {n.node_id: n for n in store.nodes}
    assert by_id["node-chunk-art-A-0"].source_ir is None
    assert by_id["node-file-art-A"].source_ir == "ir-A"
