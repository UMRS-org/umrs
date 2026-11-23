"""K4 reads-that-write (2026-11-24): a knowledge query writes no graph row.

Without ``graphs=`` the query facade resolved the project's default graph
through ``GraphRegistryStore.ensure_default_graph``, an INSERT, so every
``GET /umrs/knowledge/query`` on a project materialized a
``knowledge_graphs`` row. The resolver now looks the row up and, when there
is none, selects an unsaved stand-in with the same deterministic id (legacy
nodes written without a registry row stay searchable).

The hermetic half uses a registry double whose write fails the test; the
Postgres half runs when ``UMRS_TEST_DB_URL`` (or the CI job's
``POSTGRES_*``) names a migrated database.
"""

from __future__ import annotations

import os
import uuid

import pytest
from umrs.api.knowledge_query import _resolve_selected_graphs
from umrs.knowledge.graph_registry import KnowledgeGraphMeta, default_graph_id


class _ReadOnlyRegistry:
    def __init__(self, graphs: dict[str, KnowledgeGraphMeta]) -> None:
        self.graphs = graphs

    def get_graph(self, graph_id: str) -> KnowledgeGraphMeta | None:
        return self.graphs.get(graph_id)

    def graph_candidates(self, token: str) -> list[KnowledgeGraphMeta]:
        return [g for g in self.graphs.values() if token in (g.graph_id, g.name)]

    def ensure_default_graph(self, *_args, **_kwargs):
        raise AssertionError("a read wrote the default graph row")


def test_a_fresh_project_selects_an_unsaved_default():
    selected, excluded = _resolve_selected_graphs(_ReadOnlyRegistry({}), "p-new", None, None)
    assert [(g.graph_id, g.project_id, g.kind) for g in selected] == [
        (default_graph_id("p-new"), "p-new", "project")
    ]
    assert excluded == []


def test_a_registered_default_is_the_stored_row():
    gid = default_graph_id("p-old")
    stored = KnowledgeGraphMeta(graph_id=gid, name="p-old", project_id="p-old", ontology_id="o-1")
    selected, _ = _resolve_selected_graphs(_ReadOnlyRegistry({gid: stored}), "p-old", None, None)
    assert selected == [stored]


def test_excluding_the_unsaved_default_by_id_selects_nothing():
    gid = default_graph_id("p-new")
    selected, excluded = _resolve_selected_graphs(_ReadOnlyRegistry({}), "p-new", None, gid)
    assert selected == []
    assert excluded == [gid]


# ── real Postgres ────────────────────────────────────────────────────────────


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


def _graph_rows(pool, project: str) -> int:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM knowledge_graphs WHERE project_id = %s", (project,))
            return int(cur.fetchone()[0])
    finally:
        pool.putconn(conn)


@pytest.mark.skipif(not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
def test_a_query_on_a_fresh_project_leaves_no_graph_row():
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.api.knowledge_query import _graph_backend_for

    pool = ThreadedConnectionPool(1, 2, _pg_url())
    project = f"p-k4-{uuid.uuid4().hex[:8]}"
    try:
        backend, reason, selected, _excluded = _graph_backend_for(pool, project, None, None)
        assert reason is None and backend is not None
        assert [g.graph_id for g in selected] == [default_graph_id(project)]
        assert _graph_rows(pool, project) == 0
    finally:
        pool.closeall()
