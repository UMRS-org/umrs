"""G5 wave 3 (ADR-027 G1'): executor-side canonical mint for RAG uploads.

Covers: SQL shape (upserts/0075 cast), cap-at-100, id scheme parity with the
proxy P1a mint (ragdoc-<id> / -chunk<i>), best-effort skip paths.
"""

from __future__ import annotations

from umrs.knowledge import rag_upload_mint


class _Cur:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Conn:
    def __init__(self, cur: _Cur) -> None:
        self.cur = cur
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> _Cur:
        return self.cur

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self.returned = 0

    def getconn(self) -> _Conn:
        return self._conn

    def putconn(self, _conn: _Conn) -> None:
        self.returned += 1


def _mk_pool() -> tuple[_Cur, _Conn, _Pool]:
    cur = _Cur()
    conn = _Conn(cur)
    return cur, conn, _Pool(conn)


_CHUNKS = [
    {"start": 0, "end": 10, "md5": "aaa"},
    {"start": 10, "end": 20, "md5": "bbb"},
]


def _mint_two(pool):
    return rag_upload_mint.mint_rag_upload(
        pool,
        project_id="spring8",
        document_id="doc-9",
        filename="scan.h5",
        mime_type="application/x-hdf5",
        chunks=_CHUNKS,
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        model_name="e5-large",
    )


def test_mint_commits_all_tables_and_path():
    cur, conn, pool = _mk_pool()
    assert _mint_two(pool) == 2
    assert conn.committed and pool.returned == 1
    sql_all = "\n".join(s for s, _ in cur.calls)
    assert "knowledge_artifacts" in sql_all
    assert "knowledge_nodes" in sql_all
    assert "knowledge_edges" in sql_all
    assert "knowledge_node_embeddings" in sql_all


def test_mint_node_ids_match_proxy_scheme():
    cur, _, pool = _mk_pool()
    _mint_two(pool)
    node_params = [p for s, p in cur.calls if "INSERT INTO knowledge_nodes" in s]
    ids = {p[0] for p in node_params}
    assert ids == {"ragdoc-doc-9", "ragdoc-doc-9-chunk0", "ragdoc-doc-9-chunk1"}


def _embedding_calls(cur):
    return [(s, p) for s, p in cur.calls if "knowledge_node_embeddings" in s]


def test_mint_embedding_rows_cast_sql():
    cur, _, pool = _mk_pool()
    _mint_two(pool)
    calls = _embedding_calls(cur)
    assert len(calls) == 2
    assert "%s::vector" in calls[0][0]


def test_mint_embedding_row_params():
    cur, _, pool = _mk_pool()
    _mint_two(pool)
    _, (node_id, model, dim, vec, md5) = _embedding_calls(cur)[0]
    assert node_id == "ragdoc-doc-9-chunk0" and model == "e5-large"
    assert dim == 2 and vec == "[0.1,0.2]" and md5 == "aaa"


def test_mint_part_of_edges_target_doc_node():
    cur, _, pool = _mk_pool()
    _mint_two(pool)
    edge_params = [p for s, p in cur.calls if "INSERT INTO knowledge_edges" in s]
    assert edge_params[0][0] == "ragdoc-doc-9-chunk0-part-of"
    assert edge_params[0][3] == "ragdoc-doc-9"


def test_mint_caps_at_100_chunks():
    cur, _, pool = _mk_pool()
    chunks = [{"start": i, "end": i + 1, "md5": f"m{i}"} for i in range(150)]
    minted = rag_upload_mint.mint_rag_upload(
        pool,
        project_id="p",
        document_id="doc-big",
        filename="big.txt",
        mime_type="text/plain",
        chunks=chunks,
        embeddings=[[0.0]] * 150,
        model_name="m",
    )
    assert minted == 100


def test_mint_skips_without_pool_or_model():
    cur, _, pool = _mk_pool()
    assert (
        rag_upload_mint.mint_rag_upload(
            None,
            project_id="p",
            document_id="d",
            filename="f",
            mime_type="",
            chunks=[],
            embeddings=[],
            model_name="m",
        )
        == 0
    )
    assert (
        rag_upload_mint.mint_rag_upload(
            pool,
            project_id="p",
            document_id="d",
            filename="f",
            mime_type="",
            chunks=[],
            embeddings=[],
            model_name=None,
        )
        == 0
    )
    assert not cur.calls


def test_mint_rolls_back_and_raises_on_sql_error():
    cur = _Cur()

    def _boom(sql, params=()):
        raise RuntimeError("db down")

    cur.execute = _boom  # type: ignore[method-assign]
    conn = _Conn(cur)
    pool = _Pool(conn)
    try:
        rag_upload_mint.mint_rag_upload(
            pool,
            project_id="p",
            document_id="d",
            filename="f",
            mime_type="",
            chunks=_CHUNKS,
            embeddings=[[0.1]],
            model_name="m",
        )
        failed = False
    except RuntimeError:
        failed = True
    assert failed
    assert conn.rolled_back and not conn.committed
    assert pool.returned == 1
