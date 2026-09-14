"""ADR-050 visibility tiering pins (shape level; live rows in
``test_visibility_tiering_live.py``).

Contracts locked:

- ONE predicate renders the family rule for every read lane
  (``knowledge.access_scope.visibility_predicate``, 2026-09-13 wave 2):
  ``((<a>.project_id = %s OR <a>.visibility = 'system') AND <a>.visibility != 'ambient')``.
- Every K3 facade lane (rag, graph, embeddings table-reader + similarity)
  and the catalog store search render their gate from it — project rows +
  system rows are visible, ambient rows are never served regardless of
  ownership (kept, not deleted — user decision). The gate is additive along
  the existing ownership clause, never replacing it.
- Migrations 0092 (knowledge planes) and 0093 (catalog_objects, R-02) add
  the column with default ``'project'`` — no existing row changes tier.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

from umrs.api.knowledge_query import (
    _EmbeddingsSearchBackend,
    _GraphSearchBackend,
    _RAGSearchBackend,
)
from umrs.data_catalog.pg_store import CatalogMetaStore, _search_filters, _visibility_clause
from umrs.knowledge.access_scope import (
    VISIBILITY_TIERS,
    visibility_clause,
    visibility_predicate,
)

_MIGRATIONS = Path(__file__).resolve().parents[2] / "src" / "umrs" / "migrations"

ADR_050_RULE = "((project_id = %s OR visibility = 'system') AND visibility != 'ambient')"


class TestVisibilityPredicate:
    def test_default_rule_is_the_adr_050_family_rule(self) -> None:
        assert visibility_predicate() == ADR_050_RULE

    def test_alias_prefixes_every_column(self) -> None:
        assert visibility_predicate("n") == (
            "((n.project_id = %s OR n.visibility = 'system') AND n.visibility != 'ambient')"
        )

    def test_asyncpg_placeholder_variant(self) -> None:
        assert visibility_predicate("k", placeholder="$1") == (
            "((k.project_id = $1 OR k.visibility = 'system') AND k.visibility != 'ambient')"
        )

    def test_owned_sql_keeps_wider_ownership_clause_intact(self) -> None:
        owned = "(project_id = %s OR s3_key LIKE %s)"
        assert visibility_predicate(owned_sql=owned) == (
            f"(({owned} OR visibility = 'system') AND visibility != 'ambient')"
        )

    def test_clause_binds_exactly_the_caller_project(self) -> None:
        sql, params = visibility_clause("proj-a", "n")
        assert sql == visibility_predicate("n")
        assert params == ["proj-a"]

    def test_tier_vocabulary(self) -> None:
        assert VISIBILITY_TIERS == ("project", "system", "ambient")


class TestVisibilityGatingShape:
    """Every lane renders its gate from the single predicate (no hand-rolled copies)."""

    def test_rag_lane_uses_shared_predicate(self) -> None:
        src = inspect.getsource(_RAGSearchBackend.search_rag)
        assert "visibility_predicate('n', include_ambient=self._include_ambient)" in src
        assert "visibility = 'system'" not in src  # no second copy of the rule

    def test_graph_lane_uses_shared_predicate_with_graph_ownership(self) -> None:
        src = inspect.getsource(_GraphSearchBackend.search_knowledge_graph)
        assert "visibility_predicate(" in src
        assert "graph_id = ANY(%s)" in src  # ownership half intact

    def test_embeddings_lanes_use_shared_predicate(self) -> None:
        for fn in (
            _EmbeddingsSearchBackend.search_embeddings,
            _EmbeddingsSearchBackend._similarity_hits,
        ):
            src = inspect.getsource(fn)
            assert "visibility_predicate('n', include_ambient=self._include_ambient)" in src, (
                fn.__name__
            )
            assert "WHERE n.project_id = %s" not in src, fn.__name__  # old ungated form gone

    def test_catalog_store_uses_shared_predicate_with_s3_ownership(self) -> None:
        src = inspect.getsource(CatalogMetaStore.search)
        assert "_search_filters" in src  # one filter-builder, not hand-rolled gates
        filters_src = inspect.getsource(_search_filters)
        assert "_visibility_clause" in filters_src
        helper = inspect.getsource(_visibility_clause)
        assert "visibility_predicate(" in helper
        assert 'owned_sql="(project_id = %s OR s3_key LIKE %s)"' in helper
        # system rows bypass the project ACL re-check (visible to every caller)
        assert 'row.get("visibility") != VISIBILITY_SYSTEM' in src

    def test_rag_lane_executes_atomically_per_caller(self) -> None:
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cursor
        # RAG backend does `with pool.getconn() as conn:` — getconn IS the CM.
        pool.getconn.return_value.__enter__.return_value = conn
        backend = _RAGSearchBackend(pool)
        out = backend.search_rag("x", project_id="proj-a")
        assert out == []
        sql, params = cursor.execute.call_args.args
        assert params[0] == "proj-a"  # caller project stays the first bind
        assert visibility_predicate("n") in sql

    def test_embeddings_table_reader_binds_caller_project_first(self) -> None:
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cursor
        pool.getconn.return_value.__enter__.return_value = conn
        out = _EmbeddingsSearchBackend(pool).search_embeddings("m", project_id="proj-a")
        assert out == []
        sql, params = cursor.execute.call_args.args
        assert params[0] == "proj-a"
        assert visibility_predicate("n") in sql


class TestVisibilityTierConfig:
    def test_migration_0092_shape(self) -> None:
        migration = (_MIGRATIONS / "0092_visibility_tiers.sql").read_text()
        assert "knowledge_nodes_visibility_check" in migration
        assert "sat_knowledge_objects_visibility_check" in migration
        assert "'project', 'system', 'ambient'" in migration
        # default remains project — existing rows never flipped by mig
        assert "DEFAULT 'project'" in migration

    def test_migration_0093_adds_catalog_plane(self) -> None:
        """R-02: CatalogMetaStore.search filters visibility on catalog_objects,
        which 0092 never altered — 0093 closes the gap, same idempotent style."""
        migration = (_MIGRATIONS / "0093_catalog_objects_visibility.sql").read_text()
        assert "ALTER TABLE catalog_objects" in migration
        assert "ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'project'" in migration
        assert "catalog_objects_visibility_check" in migration
        assert "'project', 'system', 'ambient'" in migration
        assert "catalog_objects_visibility_idx" in migration
        assert "ON catalog_objects (project_id, visibility)" in migration
        assert "COMMENT ON COLUMN catalog_objects.visibility" in migration


class TestIncludeAmbient:
    def test_default_predicate_excludes_ambient(self) -> None:
        sql = visibility_predicate("n")
        assert "visibility != 'ambient'" in sql

    def test_include_ambient_drops_the_exclusion(self) -> None:
        sql = visibility_predicate("n", include_ambient=True)
        assert "visibility != 'ambient'" not in sql

    def test_route_passes_include_ambient_from_admin_identity(self) -> None:
        src = inspect.getsource(
            query_knowledge := __import__(
                "umrs.api.knowledge_query", fromlist=["query_knowledge"]
            ).query_knowledge
        )
        assert "identity.is_admin" in src
        assert "include_ambient=include_ambient" in src
