"""Legacy corpus canonization runner (ADR-027 G5, live-survey wave).

Fakes mirror the embeddings-backend test style: fake source pool for the
legacy table, recording cursor for canonical writes. Verifies grouping,
synthetic document-id derivation, executor-mint SQL reuse with
``rag.legacy-canonize`` provenance, and embedding rows from the same rows.
"""

from __future__ import annotations

from umrs.knowledge.legacy_corpus_canonize import (
    _SELECT_ROWS_SQL,
    _legacy_document_id,
    canonize_document,
    read_corpus,
)

_CORPUS_ROWS = [
    # s3_object, checksum, chunk_start, chunk_end, chunk_md5, model, embedding, _content
    # checksum = per-DOCUMENT key (live survey: doc identity), s3_object is per-chunk
    (
        "s3://bucket/docA.txt#chunk0",
        "sumA",
        0,
        100,
        "md5-a0",
        "intfloat/e5-base-v2",
        "[0.1,0.2]",
        "alpha chunk zero",
    ),
    (
        "s3://bucket/docA.txt#chunk1",
        "sumA",
        100,
        200,
        "md5-a1",
        "intfloat/e5-base-v2",
        "[0.3,0.4]",
        "alpha chunk one",
    ),
    (
        "s3://bucket/9f8e7d.txt",
        "sumB",
        0,
        50,
        "md5-b0",
        "intfloat/e5-base-v2",
        "[0.9,0.9]",
        "beta chunk zero",
    ),
]


class _Cur:
    def __init__(self, rows=None, capture=None) -> None:
        self._rows = rows or []
        self._capture = capture

    def execute(self, sql, params=None):
        if self._capture is not None:
            self._capture.append((sql, params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Conn:
    def __init__(self, rows=None, capture=None) -> None:
        self._cur = _Cur(rows, capture)
        self.commits = 0

    def cursor(self):
        return self._cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class _Pool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        pass


def test_read_corpus_groups_by_checksum_key():
    pool = _Pool(_Conn(rows=_CORPUS_ROWS))
    docs = read_corpus(pool, "text")
    assert sorted(docs) == ["sumA", "sumB"]
    doc_a = docs["sumA"]
    assert doc_a["uri"] == "s3://bucket/docA.txt"  # fragment stripped
    assert len(doc_a["rows"]) == 2
    first = doc_a["rows"][0]
    assert first["chunk_md5"] == "md5-a0"
    assert first["embedding"] == [0.1, 0.2]
    assert first["s3_chunk"] == "s3://bucket/docA.txt#chunk0"
    assert "ORDER BY checksum" in _SELECT_ROWS_SQL


def _doc(doc_key: str) -> dict:
    """Materialize read_corpus's grouped entry for one fake document."""
    pool = _Pool(_Conn(rows=_CORPUS_ROWS))
    return read_corpus(pool, "text")[doc_key]


def test_canonize_document_mints_mint_shaped_rows_with_legacy_provenance():
    capture: list = []
    conn = _Conn(capture=capture)
    entry = _doc("sumA")
    minted = canonize_document(
        conn,
        doc_key="sumA",
        doc_uri=entry["uri"],
        chunk_rows=entry["rows"],
        project_id="default",
        max_per_doc=500,
    )
    assert minted == 2
    wrote = [sql for sql, _ in capture]

    doc_id = "legacy-" + _legacy_document_id("sumA")
    assert any("knowledge_artifacts" in sql for sql in wrote)
    node_inserts = [p for sql, p in capture if "knowledge_nodes" in sql and "INSERT" in sql]
    node_ids = {p[0] for p in node_inserts}
    assert f"ragdoc-{doc_id}" in node_ids
    assert f"ragdoc-{doc_id}-chunk0" in node_ids
    assert f"ragdoc-{doc_id}-chunk1" in node_ids

    # provenance: every JSON content param carries the legacy origin
    import json

    doc_params = next(
        p for sql, p in capture if "knowledge_nodes" in sql and p[0] == f"ragdoc-{doc_id}"
    )
    doc_content = json.loads(doc_params[4])
    assert doc_content["origin"] == "rag.legacy-canonize"
    assert doc_content["s3_object"] == "s3://bucket/docA.txt"

    chunk_params = next(
        p for sql, p in capture if "knowledge_nodes" in sql and p[0] == f"ragdoc-{doc_id}-chunk0"
    )
    chunk_content = json.loads(chunk_params[4])
    assert chunk_content["s3_chunk"] == "s3://bucket/docA.txt#chunk0"

    edge_params = next(p for sql, p in capture if "knowledge_edges" in sql)
    edge_content = json.loads(edge_params[4])
    assert edge_content["origin"] == "rag.legacy-canonize"

    emb_params = next(p for sql, p in capture if "knowledge_node_embeddings" in sql)
    assert emb_params[0] == f"ragdoc-{doc_id}-chunk0"
    assert emb_params[1] == "intfloat/e5-base-v2"
    assert emb_params[2] == 2  # dimension from [0.1, 0.2]
    assert emb_params[3] == "[0.1,0.2]"
    assert emb_params[4] == "md5-a0"


def test_max_per_doc_caps_chunks():
    capture: list = []
    conn = _Conn(capture=capture)
    minted = canonize_document(
        conn,
        doc_key="sumA",
        doc_uri="s3://bucket/docA.txt",
        chunk_rows=_doc("sumA")["rows"],
        project_id="default",
        max_per_doc=1,
    )
    assert minted == 1


def test_chunk_nodes_carry_their_text():
    """Spine answers need the passage text; legacy chunks have no artifact to
    slice it from, so a text-less chunk node was dropped from every answer."""
    import json

    capture: list = []
    entry = _doc("sumA")
    canonize_document(
        _Conn(capture=capture),
        doc_key="sumA",
        doc_uri=entry["uri"],
        chunk_rows=entry["rows"],
        project_id="default",
        max_per_doc=500,
    )
    doc_id = "legacy-" + _legacy_document_id("sumA")
    chunk_params = next(
        p for sql, p in capture if "knowledge_nodes" in sql and p[0] == f"ragdoc-{doc_id}-chunk0"
    )
    assert json.loads(chunk_params[4])["text"] == entry["rows"][0]["content"]
