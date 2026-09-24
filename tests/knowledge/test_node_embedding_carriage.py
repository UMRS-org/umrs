"""D-40: canonical node vectors (knowledge_node_embeddings, 0075) travel in the
bundle; import trusts only the destination's own embedder (ADR-032 #3)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from umrs.knowledge import bundle_import as bi
from umrs.knowledge.bundle_export import KnowledgeExportEngine, _parse_vector_text
from umrs.knowledge.bundle_reader import ExportBundleWriter


class _ExportPool:
    async def fetch(self, query, *_args):
        assert "knowledge_node_embeddings" in query
        return [
            {
                "node_id": "n-keep",
                "model_name": "/site-a/e5-base-v2",
                "dimension": 3,
                "embedding": "[0.1,0.2,0.3]",
                "content_hash": "h1",
            },
            {
                "node_id": "n-partial",
                "model_name": "/site-a/e5-base-v2",
                "dimension": 3,
                "embedding": "[0.4,0.5,0.6]",
                "content_hash": "h2",
            },
        ]


@pytest.mark.asyncio
async def test_export_writes_eligible_node_vectors(tmp_path):
    engine = KnowledgeExportEngine(
        store=SimpleNamespace(_pool=_ExportPool()), output_dir=tmp_path, secret_key="k"
    )
    writer = ExportBundleWriter(tmp_path / "b")
    n = await engine._dump_node_embeddings("p", writer, frozenset({"n-partial"}))
    rows = [
        json.loads(x) for x in (tmp_path / "b" / "node_embeddings.jsonl").read_text().splitlines()
    ]
    assert n == 1
    assert rows == [
        {
            "node_id": "n-keep",
            "model_name": "/site-a/e5-base-v2",
            "dimension": 3,
            "embedding": [0.1, 0.2, 0.3],
            "content_hash": "h1",
        }
    ]


def test_vector_text_parsing():
    assert _parse_vector_text("[1,2.5, -3]") == [1.0, 2.5, -3.0]
    assert _parse_vector_text([1, 2]) == [1.0, 2.0]
    assert _parse_vector_text("[]") == []


@pytest.mark.parametrize(
    ("row", "dest", "trusted"),
    [
        (
            {"model_name": "/site-a/e5-base-v2", "dimension": 2, "embedding": [0.1, 0.2]},
            "/opt/models/e5-base-v2",
            True,
        ),
        (
            {"model_name": "/site-a/all-MiniLM", "dimension": 2, "embedding": [0.1, 0.2]},
            "/opt/models/e5-base-v2",
            False,
        ),
        (
            {"model_name": "/site-a/e5-base-v2", "dimension": 3, "embedding": [0.1, 0.2]},
            "/opt/models/e5-base-v2",
            False,
        ),
        ({"model_name": "/site-a/e5-base-v2", "dimension": 2, "embedding": [0.1, 0.2]}, "", False),
    ],
)
def test_destination_trust_rule(row, dest, trusted):
    assert bi._trusted_node_vector(row, dest) is trusted


class _Conn:
    def __init__(self, fail_ids=()):
        self.inserted = []
        self._fail = set(fail_ids)

    async def execute(self, _sql, *args):
        if args[0] in self._fail:
            raise RuntimeError("fk violation")
        self.inserted.append(args)


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *_):
                return False

        return _Ctx()


def _bundle(tmp_path, rows):
    (tmp_path / "node_embeddings.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


@pytest.mark.asyncio
async def test_import_lands_trusted_rows_under_destination_model(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/opt/models/e5-base-v2")
    rows = [
        {
            "node_id": "n1",
            "model_name": "/site-a/e5-base-v2",
            "dimension": 2,
            "embedding": [0.1, 0.2],
            "content_hash": "h",
        },
        {
            "node_id": "n2",
            "model_name": "/site-a/other-model",
            "dimension": 2,
            "embedding": [0.3, 0.4],
        },
        {
            "node_id": "n-missing",
            "model_name": "/site-a/e5-base-v2",
            "dimension": 2,
            "embedding": [0.5, 0.6],
        },
    ]
    conn = _Conn(fail_ids={"n-missing"})
    engine = bi.KnowledgeImportEngine(store=SimpleNamespace(_pool=_Pool(conn)), secret_key="")
    result = bi.KnowledgeImportResult()
    await engine._restore_node_embeddings(_bundle(tmp_path, rows), result)
    assert (result.node_embedding_count, result.node_embeddings_pending) == (1, 2)
    node_id, model, dim, literal, chash = conn.inserted[0]
    assert (node_id, model, dim, literal, chash) == (
        "n1",
        "/opt/models/e5-base-v2",
        2,
        "[0.1,0.2]",
        "h",
    )


@pytest.mark.asyncio
async def test_no_destination_embedder_leaves_everything_pending(tmp_path, monkeypatch):
    monkeypatch.delenv("UMRS_KNOWLEDGE_EMBED_MODEL", raising=False)
    rows = [{"node_id": "n1", "model_name": "e5", "dimension": 1, "embedding": [0.1]}]
    conn = _Conn()
    engine = bi.KnowledgeImportEngine(store=SimpleNamespace(_pool=_Pool(conn)), secret_key="")
    result = bi.KnowledgeImportResult()
    await engine._restore_node_embeddings(_bundle(tmp_path, rows), result)
    assert (result.node_embedding_count, result.node_embeddings_pending, conn.inserted) == (
        0,
        1,
        [],
    )
    assert result.to_dict()["node_embeddings_pending"] == 1
