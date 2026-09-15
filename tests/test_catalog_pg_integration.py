"""Postgres integration tests for CatalogMetaStore (requires --db-url)."""

from __future__ import annotations

from pathlib import Path

import pytest

from umrs.data_catalog.pg_store import CatalogMetaStore
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

from test_catalog_pg_store import _ready_meta

_MIGRATION_SQL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "migrations"
    / "0054_catalog_objects.sql"
)


@pytest.mark.integration
class TestCatalogMetaStoreIntegration:
    """Exercise real Postgres FTS and schema against migration 0054."""

    @pytest.fixture(autouse=True)
    def setup_store(self, db_url):
        from psycopg2.pool import SimpleConnectionPool

        pool = SimpleConnectionPool(1, 4, db_url)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS catalog_objects CASCADE")
                cur.execute(_MIGRATION_SQL.read_text())
            conn.commit()
        finally:
            pool.putconn(conn)

        self.pool = pool
        self.store = CatalogMetaStore(pool)
        yield
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS catalog_objects CASCADE")
            conn.commit()
        finally:
            pool.putconn(conn)
            pool.closeall()

    def test_upsert_get_delete_round_trip(self):
        meta = _ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha dataset")
        self.store.upsert(meta, "proj-1")

        loaded = self.store.get("proj-1", "proj-1/uploads/a.csv")
        assert loaded is not None
        assert loaded.summary == "alpha dataset"

        assert self.store.delete("proj-1", "proj-1/uploads/a.csv")
        assert self.store.get("proj-1", "proj-1/uploads/a.csv") is None

    def test_search_matches_summary_via_fts(self):
        self.store.upsert(
            _ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha dataset"),
            "proj-1",
        )
        self.store.upsert(
            _ready_meta("proj-1/uploads/b.csv", "proj-1", "unrelated"),
            "proj-1",
        )

        hits = self.store.search("proj-1", "alpha", limit=5)

        assert hits
        assert hits[0].meta.s3_key == "proj-1/uploads/a.csv"

    def test_search_matches_tags_via_fts(self):
        meta = CatalogObjectMeta(
            s3_key="proj-1/uploads/tagged.csv",
            object_etag="etag-1",
            state=CatalogState.READY,
            owner="user-1",
            acl_scope="proj-1",
            updated_at="2026-01-01T00:00:00",
            summary="batch results",
            deterministic_summary="batch results",
            detected_modality="structured",
            tags=["yield", "protein"],
            filename_signals=[],
        )
        self.store.upsert(meta, "proj-1")

        hits = self.store.search("proj-1", "protein", limit=5)

        assert hits
        assert hits[0].meta.s3_key == "proj-1/uploads/tagged.csv"

    def test_get_requires_matching_project_id(self):
        meta = _ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha")
        self.store.upsert(meta, "proj-1")

        assert self.store.get("proj-1", "proj-1/uploads/a.csv") is not None
        assert self.store.get("proj-2", "proj-1/uploads/a.csv") is None
