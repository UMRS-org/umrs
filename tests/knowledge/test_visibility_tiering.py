"""ADR-050 visibility tiering pins.

Contracts locked:

- The two K3 facade lanes now carry the tier gate: project rows + system
  rows are visible to the caller; ambient rows are never served regardless
  of project ownership (kept, not deleted — user decision).
- The gate is additive along the existing project ACL, never replacing it:
  system rows are visible to every caller, ambient rows are visible to
  no regular read path.

These pins are SQL-shape pins for the query lanes plus plan pins for the
catalog store. Live row behavior is rehearsed against the umrs_test bench
(none of this changes an existing row's visibility class by itself).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from umrs.api.knowledge_query import _RAGSearchBackend


class TestVisibilityGatingShape:
    def test_rag_lane_sql_carries_gate_and_system_widening(self) -> None:
        src = inspect.getsource(_RAGSearchBackend.search_rag)
        assert "visibility = 'system'" in src
        assert "visibility != 'ambient'" in src

    def test_graph_lane_sql_carries_gate(self) -> None:
        from umrs.api.knowledge_query import _GraphSearchBackend

        src = inspect.getsource(_GraphSearchBackend.search_knowledge_graph)
        assert "visibility != 'ambient'" in src
        assert "visibility = 'system'" in src

    def test_catalog_store_sql_carries_gate(self) -> None:
        from umrs.data_catalog.pg_store import CatalogMetaStore as _Store

        src = inspect.getsource(_Store.search)
        assert "visibility = 'system'" in src
        assert "visibility != 'ambient'" in src

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
        assert params[0] == "proj-a"          # caller project stays the first bind
        assert "visibility" in sql


class TestVisibilityTierConfig:
    def test_migration_shape(self) -> None:
        from pathlib import Path

        migration = (
            Path(__file__).resolve().parents[2]
            / "src" / "umrs" / "migrations" / "0092_visibility_tiers.sql"
        ).read_text()
        assert "knowledge_nodes_visibility_check" in migration
        assert "sat_knowledge_objects_visibility_check" in migration
        assert "'project', 'system', 'ambient'" in migration
        # default remains project — existing rows never flipped by mig
        assert "DEFAULT 'project'" in migration


class TestVisibilityTierConfig:
    def test_migration_shape(self) -> None:
        migration = open(
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "src" / "umrs" / "migrations" / "0092_visibility_tiers.sql"
        ).read()
        assert "knowledge_nodes_visibility_check" in migration
        assert "sat_knowledge_objects_visibility_check" in migration
        assert "'project', 'system', 'ambient'" in migration
        # default remains project — existing rows never flipped by mig
        assert "DEFAULT 'project'" in migration
