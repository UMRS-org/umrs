"""Catalogue→canonical links hydration (ADR-031 plane-bridging, UX §2/§4).

Server-side read projection: `catalog/search` + `catalog/read` now carry a
``links`` block when the shared canonical pool resolves a document, and
silently omit it otherwise (fail-open — links absent is never an error).
"""

from __future__ import annotations

from umrs.data_catalog.links import hydrate_links


class _Cur:
    def __init__(self, script: list[tuple[str, list]], chunk_rows: list) -> None:
        self._script = script
        self._chunk_rows = chunk_rows
        self._rows: list = []
        self.queries: list[tuple[str, tuple]] = []

    def _chunk_probe(self, sql) -> list | None:
        if "count(*)" in sql and "chunk" in sql and "GROUP BY" not in sql:
            return self._chunk_rows
        return None

    def _vector_probe(self, sql) -> list | None:
        if "knowledge_node_embeddings" not in sql:
            return None
        return self._probe_rows(sql, "LIKE %s") or []

    def _probe_rows(self, sql, only: str | None = None):
        for probe, rows in self._script:
            if only is not None and probe != only:
                continue
            if only is None and probe == "LIKE %s":
                continue
            if probe in sql:
                return rows
        return None

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        rows = self._chunk_probe(sql)
        if rows is None:
            vector_rows = self._vector_probe(sql)
        else:
            vector_rows = None
        self._rows = row if (row := (rows if rows is not None else vector_rows)) is not None else (self._probe_rows(sql) or [])

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Conn:
    def __init__(self, script, chunk_rows) -> None:
        self._cur = _Cur(script, chunk_rows)

    def cursor(self):
        return self._cur


class _Pool:
    def __init__(self, script, chunk_rows=None) -> None:
        self._conn = _Conn(script, chunk_rows or [])
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        self.returned += 1


def test_hydrate_links_requires_pool_or_key():
    assert hydrate_links(None, project_id="p", s3_key="a/b.txt") is None
    assert hydrate_links(_Pool([]), project_id="p", s3_key="") is None


def test_s3_object_lane_wins_over_filename():
    pool = _Pool([
        ("s3_object", [("doc-xyz",)]),
        ("LIKE %s", [("intfloat/e5-base-v2", 768, 3)]),
    ], chunk_rows=[(7,)])
    # Sanity guard: probes must NEVER confuse doc-resolution with vector stats.
    assert "knowledge_node_embeddings" not in "GROUP BY content->>'document_id'"
    links = hydrate_links(pool, project_id="spring8", s3_key="spring8/scan1.h5")
    assert links is not None
    assert links["document_id"] == "doc-xyz"
    assert links["link_method"] == "s3_object"
    assert links["chunk_nodes"] == 7
    assert links["vector_rows"] == [
        {"model": "intfloat/e5-base-v2", "dimension": 768, "rows": 3}
    ]
    sql = pool._conn._cur.queries[0][0]
    assert "s3_object" in sql
    params = pool._conn._cur.queries[0][1]
    assert params == ("spring8", "spring8/scan1.h5")


def test_filename_lane_matches_exactly_one_doc_node():
    pool = _Pool([
        ("s3_object", []),
        ("GROUP BY", [("doc-abc", 1)]),
        ("LIKE %s", []),
    ], chunk_rows=[(2,)])
    links = hydrate_links(pool, project_id="default", s3_key="default/data/scan.h5")
    assert links is not None
    assert links["document_id"] == "doc-abc"
    assert links["link_method"] == "filename"
    assert links["chunk_nodes"] == 2
    assert links["vector_rows"] == []


def test_filename_is_only_basename_of_the_key():
    pool = _Pool([
        ("GROUP BY", [("doc-t", 1)]),
        ("LIKE %s", []),
    ], chunk_rows=[(0,)])
    hydrate_links(pool, project_id="default", s3_key="deep/nested/area/report.pdf")
    filename_sql = next(sql for sql, _ in pool._conn._cur.queries if "filename" in sql)
    params = next(p for sql, p in pool._conn._cur.queries if sql == filename_sql)
    assert params[1] == "report.pdf"


def test_multiple_filename_matches_yield_no_link():
    pool = _Pool([
        ("s3_object", []),
        ("GROUP BY", [("doc-a", 1), ("doc-b", 1)]),
    ], chunk_rows=[])
    assert hydrate_links(pool, project_id="default", s3_key="default/ambiguous.txt") is None


def test_no_match_returns_none():
    pool = _Pool([("s3_object", []), ("GROUP BY", [])], chunk_rows=[])
    assert hydrate_links(pool, project_id="default", s3_key="default/x.txt") is None


def test_query_failure_fails_open():
    class _BadPool:
        def getconn(self):
            raise RuntimeError("db down")

        def putconn(self, _conn):
            pass

    assert hydrate_links(_BadPool(), project_id="p", s3_key="a/b") is None


def test_no_links_for_catalogue_only_objects():
    pool = _Pool([
        ("s3_object", []),
        ("GROUP BY", []),
    ])
    assert hydrate_links(pool, project_id="default", s3_key="default/unseen.bin") is None
