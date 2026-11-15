"""ADR-116 cross-lane contracts, with the real modules on both sides.

The lane tests stub the other side of each contract: the windowed extractor
is tested with a fake vocabulary, the vocabulary without the extractor, and
entity resolution with a fake store. These tests join them: the built-in
``r4-science`` vocabulary drives the extractor, the extractor's payload
becomes the understand job's rows ([C-ING]), and, on a real Postgres, those
rows are what ``propose_same_as`` pairs ([C-ER]). The model is a fake chat
callable; no network, no model.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import llm_extract as lx
from umrs.knowledge.ontology_vocab import ExtractionVocabulary, default_vocabulary

_NODE_TYPES = {"class", "function", "file", "module", "concept", "entity", "relationship"}

_REPLY = {
    "entities": [
        {
            "name": "Zirconia",
            "type": "Materials",
            "aliases": ["ZrO2"],
            "quote": "Zirconia (ZrO2) was sintered at 1400 C.",
        },
        {"name": "Spark plasma sintering", "type": "method", "quote": "spark plasma"},
        {"name": "Flux widget", "type": "gadget"},
    ],
    "relations": [
        {"source": "Spark plasma sintering", "target": "Zirconia", "relation": "applied to"},
        {"source": "Flux widget", "target": "Zirconia", "relation": "zaps"},
    ],
}


def _chat(messages: list[dict[str, Any]], **_kw: Any) -> str:
    return json.dumps(_REPLY)


def _payload() -> dict[str, Any]:
    request = lx.ExtractRequest(
        chat=_chat,
        model="m",
        vocab=default_vocabulary(),
        settings=lx.ExtractSettings(window_chars=2000, windows_max=4, concurrency=1, budget_s=30.0),
    )
    notes: list[str] = []
    payload = lx.extract_document("Zirconia (ZrO2) was sintered at 1400 C.", request, notes)
    assert notes == []
    assert payload is not None
    return payload


def _rows(artifact_id: str, project_id: str = "p-int", graph_id: str = "") -> tuple:
    return ka._semantic_nodes_edges(artifact_id, "ir-1", project_id, graph_id, _payload())


def _by_label(nodes: list) -> dict[str, Any]:
    return {n.label: n for n in nodes}


# ── [C-ONT] → extractor ─────────────────────────────────────────────────────


def test_the_extractor_falls_back_to_the_real_science_vocabulary():
    vocab = lx.default_vocabulary()
    assert isinstance(vocab, ExtractionVocabulary)
    assert (vocab.ontology_id, vocab.version) == ("r4-science", "1")


def test_the_prompt_lists_the_vocabulary_classes_and_relations():
    prompt = lx.system_prompt(default_vocabulary())
    assert "material" in prompt and "applied_to" in prompt


def test_the_default_graph_resolves_to_the_science_vocabulary_without_notes():
    vocab, notes = lx.resolve_vocabulary("p-int", "")
    assert isinstance(vocab, ExtractionVocabulary) and notes == []


def test_a_named_graph_without_a_database_falls_back_with_a_note(monkeypatch):
    for name in ("DB_HOST", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    vocab, notes = lx.resolve_vocabulary("p-int", "g-unknown")
    assert isinstance(vocab, ExtractionVocabulary)
    assert [n for n in notes if n.startswith("ontology_fallback:")] == notes != []


# ── extractor → understand rows ([C-ING]) ────────────────────────────────────


def test_matched_types_are_canonical_and_buckets_are_db_node_types():
    nodes, _edges = _rows("art-1")
    nodes = _by_label(nodes)
    assert nodes["Zirconia"].content["semantic_type"] == "material"
    assert nodes["Zirconia"].node_type == "entity"
    assert nodes["Spark plasma sintering"].content["semantic_type"] == "method"
    assert {n.node_type for n in nodes.values()} <= _NODE_TYPES


def test_node_content_carries_aliases_evidence_and_the_extractor():
    zirconia = _by_label(_rows("art-1")[0])["Zirconia"]
    content = zirconia.content
    assert content["artifact_id"] == "art-1"
    assert content["aliases"] == ["ZrO2"]
    assert content["extractor"] == "llm-windowed@2"
    assert content["evidence"] == [
        {"window": 0, "quote": "Zirconia (ZrO2) was sintered at 1400 C.", "parser": "raw"}
    ]
    assert "type_raw" not in content


def test_an_unmatched_type_is_unclassified_with_its_raw_type():
    widget = _by_label(_rows("art-1")[0])["Flux widget"]
    assert widget.content["semantic_type"] == "unclassified"
    assert widget.content["type_raw"] == "gadget"
    assert widget.node_type == "concept"


def test_relations_map_to_the_vocabulary_or_related_to_with_the_raw_phrase():
    _nodes, edges = _rows("art-1")
    semantic = {e.relation_type: e for e in edges if e.relation_type != "mentions"}
    assert set(semantic) == {"applied_to", "related_to"}
    assert semantic["related_to"].content["rel_raw"] == "zaps"
    assert "rel_raw" not in semantic["applied_to"].content


# ── understand rows → entity resolution, on real Postgres ([C-ER]) ──────────


def _pg_url() -> str:
    """``UMRS_TEST_DB_URL``, else the CI job's ``POSTGRES_*`` (the
    ``integration`` marker skipped this test in CI, which sets only those)."""
    url = os.environ.get("UMRS_TEST_DB_URL", "")
    host = os.environ.get("POSTGRES_HOST")
    if not url and host:
        user = os.environ.get("POSTGRES_USER", "test")
        secret = os.environ.get("POSTGRES_PASSWORD", "test")
        port = os.environ.get("POSTGRES_PORT", "5432")
        name = os.environ.get("POSTGRES_DB", "umrs_test")
        url = f"postgresql://{user}:{secret}@{host}:{port}/{name}"
    if not url:
        pytest.skip("needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
    return url


async def _write_semantic_rows(store: Any, project: str, artifact_id: str) -> list[str]:
    """One artifact's semantic nodes and the edges between them (no IR or
    file node exists here, so ``source_ir`` is cleared and the mentions edges
    are left out)."""
    nodes, edges = _rows(artifact_id, project_id=project)
    for node in nodes:
        await store.upsert_node(replace(node, source_ir=None))
    for edge in edges:
        if edge.relation_type != "mentions":
            await store.upsert_edge(edge)
    return [n.node_id for n in nodes]


async def _scenario(url: str, project: str) -> dict[str, Any]:
    import asyncpg
    from umrs.knowledge.store import KnowledgeStore

    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, init=ka._init_json_codecs)
    store = KnowledgeStore(pool)
    try:
        await _write_semantic_rows(store, project, f"art-a-{project}")
        ids = await _write_semantic_rows(store, project, f"art-b-{project}")
        ctx = ka._UnderstandContext(sem_node_ids=ids)
        result = ka.KnowledgeUnderstandResult()
        job = SimpleNamespace(job_id="job-int", spec={})
        await ka._propose_same_as(store, job, project, "", ctx, result)
        rows = await pool.fetch(
            "SELECT edge_id, source_id, target_id, status, content FROM knowledge_edges "
            "WHERE project_id = $1 AND relation_type = 'same_as' ORDER BY edge_id",
            project,
        )
        return {"result": result, "rows": [dict(r) for r in rows]}
    finally:
        await pool.execute("DELETE FROM knowledge_nodes WHERE project_id = $1", project)
        await pool.close()


def test_the_same_entity_in_two_documents_gets_one_proposed_same_as_per_node(
    monkeypatch,
):
    url = _pg_url()
    monkeypatch.setattr(ka, "_label_embedder", lambda: (None, ""))
    project = f"p-int-{uuid.uuid4().hex[:10]}"
    seen = asyncio.run(_scenario(url, project))
    rows, result = seen["rows"], seen["result"]
    # each of the three entities appears in both documents: three pairs
    assert result.same_as_proposed == 3
    assert [_edge_shape(row) for row in rows] == [_expected_shape(row) for row in rows]
    assert len(rows) == 3
    assert "same_as:proposed=3,candidates=3" in result.extract_notes


def _edge_shape(row: dict[str, Any]) -> tuple:
    content = row["content"]
    return (
        row["edge_id"],
        row["source_id"],
        row["target_id"],
        row["status"],
        content["method"],
        content["job_id"],
        content["origin_actor"],
    )


def _expected_shape(row: dict[str, Any]) -> tuple:
    a, b = sorted([row["source_id"], row["target_id"]])
    return (f"edge-sameas-{a}-{b}", a, b, "proposed", "label", "job-int", "extracted")
