"""Unit tests for W2/W3 knowledge extraction + W1 durable-graph union helpers.

- W3: graph-JSON artifacts import as semantic nodes/edges
- W2: prose with no code structure falls back to site-LLM extraction
- inline_text ingest source stores client-supplied content server-side
- W1: durable KnowledgeStore rows map to the Studio/webview graph shape
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from umrs.api import knowledge as api_knowledge
from umrs.jobs import knowledge_adapters as ka

_GRAPH_JSON = json.dumps(
    {
        "nodes": [
            {"id": "sem", "name": "Spectral element method", "type": "method"},
            {"id": "fem", "name": "Finite element method", "type": "method"},
            {"id": "nek5000", "name": "Nek5000", "type": "software"},
            {},  # no id/name → dropped
        ],
        "edges": [
            {"source": "sem", "target": "fem", "label": "extends"},
            {"from": "nek5000", "to": "sem", "relation": "implements"},
            {"source": "sem", "target": "ghost", "label": "dangling"},
            {"source": "sem", "target": "sem", "label": "self"},
        ],
    }
)


class TestNormalizeGraphPayload:
    def test_accepts_node_edge_aliases(self):
        payload = ka._normalize_graph_payload(json.loads(_GRAPH_JSON))
        assert payload is not None
        names = {n["name"] for n in payload["nodes"]}
        assert "Spectral element method" in names
        assert "junk-no-name" not in [n["id"] for n in payload["nodes"]]
        rels = {(e["source"], e["target"], e["relation"]) for e in payload["edges"]}
        assert ("sem", "fem", "extends") in rels
        assert ("nek5000", "sem", "implements") in rels

    def test_rejects_non_graph_dicts(self):
        assert ka._normalize_graph_payload({"a": 1}) is None
        assert ka._normalize_graph_payload({"nodes": []}) is None
        assert ka._normalize_graph_payload([1, 2]) is None
        assert ka._normalize_graph_payload("text") is None

    def test_detect_graph_payload(self):
        payload = ka._detect_graph_payload(_GRAPH_JSON)
        assert payload is not None
        assert ka._detect_graph_payload("not json {") is None
        assert ka._detect_graph_payload("[1, 2]") is None
        assert ka._detect_graph_payload("plain prose") is None


class TestSemanticDbType:
    def test_entity_types(self):
        for t in ("paper", "dataset", "software", "person", "model"):
            assert ka._semantic_db_type(t) == "entity"

    def test_concept_fallback(self):
        for t in ("method", "technique", "hypothesis", "unknown", ""):
            assert ka._semantic_db_type(t) == "concept"


class TestSemanticNodesEdges:
    def _payload(self):
        return ka._normalize_graph_payload(json.loads(_GRAPH_JSON))

    def _built(self):
        payload = self._payload()
        assert payload is not None
        return ka._semantic_nodes_edges("art-1", "ir-1", "proj", "", payload)

    def test_node_types_mapped(self):
        nodes, _ = self._built()
        assert {n.node_type for n in nodes} <= {"entity", "concept"}
        # semantic type preserved raw for UI coloring
        software = next(n for n in nodes if n.label == "Nek5000")
        assert software.node_type == "entity"
        assert software.content["semantic_type"] == "software"

    def test_relations_resolved_dangling_and_self_dropped(self):
        _, edges = self._built()
        relations = [e for e in edges if e.relation_type != "mentions"]
        assert len(relations) == 2
        assert any(e.relation_type == "extends" for e in relations)

    def test_every_node_gets_mentions_edge(self):
        nodes, edges = self._built()
        mentions = [e for e in edges if e.relation_type == "mentions"]
        assert len(mentions) == len(nodes)
        assert all(e.source_id == "node-file-art-1" for e in mentions)


class _FakeStorage:
    def __init__(self):
        self.stored = {}

    def put_artifact(self, project_id, content):
        digest = "deadbeef" + content[:2].hex()
        self.stored[digest] = content
        return digest, len(content), b""

    def key(self, project_id, content_hash):
        return f"projects/{project_id}/knowledge/artifacts/{content_hash}"


class TestInlineTextIngest:
    def test_inline_text_stored(self):
        storage = _FakeStorage()
        result = ka.KnowledgeIngestResult()
        ids: list[str] = []
        descs: list[dict] = []
        ka._put_inline_text_sync(
            "job-1",
            "att-1",
            storage,
            "proj",
            {"kind": "inline_text", "uri": "sem_graph.json", "content": _GRAPH_JSON},
            "art-9",
            result,
            ids,
            descs,
        )
        assert result.failed_items == []
        assert ids == ["art-9"]
        assert descs[0]["kind"] == "schema"  # .json suffix heuristic
        assert descs[0]["content_hash"] in storage.stored

    def test_inline_text_rejects_missing_or_huge(self):
        storage = _FakeStorage()
        result = ka.KnowledgeIngestResult()
        ka._put_inline_text_sync(
            "job-1",
            "att-1",
            storage,
            "proj",
            {"kind": "inline_text", "uri": "x.md", "content": ""},
            "art-1",
            result,
            [],
            [],
        )
        huge = ka.KnowledgeIngestResult()
        ka._put_inline_text_sync(
            "job-1",
            "att-1",
            storage,
            "proj",
            {"kind": "inline_text", "uri": "x.md", "content": "x" * (3 * 1024 * 1024)},
            "art-2",
            huge,
            [],
            [],
        )
        assert result.failed_items and "no content" in result.failed_items[0]["error"]
        assert huge.failed_items and "2 MiB" in huge.failed_items[0]["error"]


class TestNodesEdgesForArtifact:
    def _parsed_empty(self):
        return {k: [] for k in ("modules", "classes", "functions", "algorithms", "dependencies", "kernels")}

    def test_w3_graph_json_imported(self, monkeypatch):
        monkeypatch.setenv("UMRS_KNOWLEDGE_LLM_EXTRACT", "0")
        nodes, edges = ka._nodes_edges_for_artifact(
            "art-1", "ir-1", "document", _GRAPH_JSON.encode(), self._parsed_empty(), "proj", ""
        )
        types = {n.node_type for n in nodes}
        assert "file" in types
        assert {"entity", "concept"} & types  # semantic nodes joined
        assert any(n.label == "Nek5000" for n in nodes)
        assert any(e.relation_type == "extends" for e in edges)

    def test_w2_llm_called_only_when_structure_empty(self, monkeypatch):
        calls: list[str] = []
        payload = ka._normalize_graph_payload(json.loads(_GRAPH_JSON))

        def _fake_llm(text: str):
            calls.append(text)
            return payload

        monkeypatch.setattr(ka, "_llm_extract_document", _fake_llm)
        # prose, empty structure → LLM extraction
        nodes, _ = ka._nodes_edges_for_artifact(
            "art-1", "ir-1", "document", b"wikipedia prose", self._parsed_empty(), "proj", ""
        )
        assert calls, "LLM extractor should run for structure-free prose"
        assert any(n.label == "Nek5000" for n in nodes)
        # code artifacts never reach LLM extraction
        calls.clear()
        parsed_code = {**self._parsed_empty(), "functions": ["f"]}
        ka._nodes_edges_for_artifact("art-2", "ir-2", "code", b"def f(): pass", parsed_code, "proj", "")
        assert calls == []

    def test_llm_disabled_env(self, monkeypatch):
        monkeypatch.setenv("UMRS_KNOWLEDGE_LLM_EXTRACT", "0")
        nodes, _ = ka._nodes_edges_for_artifact(
            "art-1", "ir-1", "document", b"plain prose", self._parsed_empty(), "proj", ""
        )
        assert len(nodes) == 1
        assert nodes[0].node_type == "file"


class TestDurableRowsPayload:
    def test_rows_map_to_studio_shape(self, monkeypatch):
        node = SimpleNamespace(
            node_id="n1",
            content={"excerpt": "x"},
            label="Spectral element method",
            node_type="concept",
            graph_id="",
            created_at=None,
        )
        edge = SimpleNamespace(
            edge_id="e1",
            source_id="n1",
            target_id="n2",
            relation_type="extends",
            created_at=None,
        )

        class _FakeStore:
            def __init__(self, _pool):
                pass

            async def list_nodes(self, project_id):
                return [node]

            async def list_edges(self, project_id):
                return [edge]

        async def _fake_pool(_request):
            return object()

        monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _FakeStore)
        monkeypatch.setattr(api_knowledge, "knowledge_asyncpg_pool", _fake_pool)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(umrs_db_pool=None)))

        nodes, edges = asyncio.run(api_knowledge._durable_graph_rows(request, "proj"))

        assert nodes[0]["id"] == "n1"
        assert nodes[0]["type"] == "concept"
        assert nodes[0]["label"] == "Spectral element method"
        assert edges[0]["from"] == "n1"
        assert edges[0]["to"] == "n2"
        assert edges[0]["label"] == "extends"

    def test_rows_empty_without_pool(self, monkeypatch):
        async def _no_pool(_request):
            return None

        monkeypatch.setattr(api_knowledge, "knowledge_asyncpg_pool", _no_pool)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(umrs_db_pool=None)))
        assert asyncio.run(api_knowledge._durable_graph_rows(request, "proj")) == ([], [])
