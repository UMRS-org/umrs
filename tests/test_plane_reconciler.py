"""Plane reconciliation checker (ADR-031 §Operational enforcement).

Classification per Invariant 4, read-only; this pins the class rules and
the scan behavior (doc/chunk/vector aggregation + catalogue-presence gating
+ orphaned-row detection) using a fake pool.
"""

from __future__ import annotations

import pytest

from umrs.knowledge.plane_reconciler import (
    Asset,
    classify,
    scan_project,
    STATE_COMPLETE,
    STATE_EMBEDDING_PENDING,
    STATE_ORPHANED,
    STATE_PARTIAL,
)


def _asset(
    *,
    docs: int = 1,
    chunks: int = 2,
    vectors: bool = True,
    catalogue: bool = True,
) -> Asset:
    asset = Asset(document_id="doc-x")
    asset.doc_node_ids = [f"n-doc-{i}" for i in range(docs)]
    asset.chunk_node_ids = [f"n-ch-{i}" for i in range(chunks)]
    asset.vector_rows_by_chunk = {cid: (1 if vectors else 0) for cid in asset.chunk_node_ids}
    if catalogue:
        asset.catalogue_keys = {"proj/file.txt"}
    return asset


class TestClassify:
    def test_complete(self):
        assert classify(_asset()).state == STATE_COMPLETE

    def test_embedding_pending_when_no_vectors(self):
        asset = classify(_asset(vectors=False))
        assert asset.state == STATE_EMBEDDING_PENDING
        assert "non-error" in asset.reasons[0]

    def test_partial_when_catalogue_absent(self):
        asset = classify(_asset(catalogue=False))
        assert asset.state == STATE_PARTIAL
        assert "without catalogue" in asset.reasons[0]

    def test_partial_when_canonical_absent(self):
        asset = _asset(docs=0, chunks=0)
        asset.doc_node_ids = []
        asset.chunk_node_ids = []
        asset.catalogue_keys = {"proj/file.txt"}
        asset = classify(asset)
        assert asset.state == STATE_PARTIAL
        assert "without canonical" in asset.reasons[0]

    def test_partial_when_doc_has_zero_chunks(self):
        asset = Asset(document_id="doc-e")
        asset.doc_node_ids = ["n-doc-0"]
        asset.chunk_node_ids = []
        asset.vector_rows_by_chunk = {}
        asset.catalogue_keys = {"file.txt"}
        asset = classify(asset)
        assert asset.state == STATE_PARTIAL
        assert "zero chunk" in asset.reasons[0]


class _Cur:
    def __init__(self, script: dict[str, list]) -> None:
        self._script = script
        self._rows: list = []

    def execute(self, sql, params=None):
        for probe, rows in self._script.items():
            if probe in sql:
                self._rows = rows
                return
        self._rows = []

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Conn:
    def __init__(self, script) -> None:
        self._cur = _Cur(script)

    def cursor(self):
        return self._cur


class _Pool:
    def __init__(self, script) -> None:
        self._conn = _Conn(script)

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        pass


_SCRIPT_HEALTHY = {
    "kind' = 'document'": [("n-doc-0", "doc-1")],
    "kind' = 'chunk'": [
        ("n-ch-0", "doc-1", "proj/file.txt"),
        ("n-ch-1", "doc-1", "proj/file.txt"),
    ],
    "knowledge_node_embeddings": [("n-ch-0", 1), ("n-ch-1", 1)],
    "to_regclass": [("catalog_objects",)],
    "FROM catalog_objects": [("proj/file.txt",)],
}


def test_scan_project_fails_closed_without_pool():
    with pytest.raises(RuntimeError):
        scan_project(None, "proj")


def test_healthy_asset_is_complete():
    report = scan_project(_Pool(_SCRIPT_HEALTHY), "proj")
    assert report.catalogue_available is True
    assert len(report.assets) == 1
    asset = report.assets[0]
    assert asset.state == STATE_COMPLETE
    counts = report.counts
    assert counts[STATE_COMPLETE] == 1


def test_orphaned_vector_and_chunk_counted_separately():
    script = {
        "kind' = 'document'": [("n-doc-0", "doc-1")],
        "kind' = 'chunk'": [
            ("n-ch-0", "doc-1", "proj/file.txt"),
            ("n-ch-X", "doc-MISSING", "proj/other.txt"),
        ],
        "knowledge_node_embeddings": [("n-ch-0", 1), ("n-ch-Z", 2)],
        "to_regclass": [("catalog_objects",)],
        "FROM catalog_objects": [("proj/file.txt",)],
    }
    report = scan_project(_Pool(script), "proj")
    assert report.orphaned_chunks == ["n-ch-X"]
    assert report.orphaned_vectors == ["n-ch-Z"]
    counts = report.counts
    assert counts[STATE_ORPHANED] == 2
    assert counts[STATE_COMPLETE] == 1


def test_catalogue_unavailable_still_classifies_without_mislabeling():
    script = {
        "kind' = 'document'": [("n-doc-0", "doc-1")],
        "kind' = 'chunk'": [("n-ch-0", "doc-1", "proj/file.txt")],
        "knowledge_node_embeddings": [("n-ch-0", 1)],
        "to_regclass": [(None,)],
    }
    report = scan_project(_Pool(script), "proj")
    assert report.catalogue_available is False
    assert report.assets[0].state == STATE_COMPLETE  # no false PARTIAL storm


def test_embedding_pending_asset_counts():
    script = {
        "kind' = 'document'": [("n-doc-0", "doc-1")],
        "kind' = 'chunk'": [("n-ch-0", "doc-1", "proj/file.txt")],
        "knowledge_node_embeddings": [],
        "to_regclass": [("catalog_objects",)],
        "FROM catalog_objects": [("proj/file.txt",)],
    }
    report = scan_project(_Pool(script), "proj")
    assert report.assets[0].state == STATE_EMBEDDING_PENDING
    assert report.counts[STATE_EMBEDDING_PENDING] == 1
