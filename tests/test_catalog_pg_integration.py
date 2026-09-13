"""Postgres integration tests for CatalogMetaStore.

Two entry points, same cases:

- ``TestCatalogMetaStoreIntegration`` — legacy ``integration`` mark, runs
  with ``--db-url`` / ``UMRS_TEST_DB_URL`` (marker semantics unchanged).
- ``TestCatalogMetaStoreCIShape`` — env-gated variant (R-02, 2026-09-13):
  runs whenever the CI job's ``POSTGRES_*`` variables are present, so the
  catalog lane is exercised on real Postgres in the DB-backed CI job instead
  of being permanently skipped.

The table is rebuilt from 0054 AND 0093 — the ``visibility`` column the
search predicate filters on lives in 0093 (the R-02 defect was exactly a
DB shaped by 0054 alone).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_catalog_pg_store import _ready_meta
from umrs.data_catalog.pg_store import CatalogMetaStore
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_MIGRATION_SQL = _MIGRATIONS / "0054_catalog_objects.sql"
_VISIBILITY_SQL = _MIGRATIONS / "0093_catalog_objects_visibility.sql"


def ci_shape_db_url() -> str | None:
    """DSN from the CI job's POSTGRES_* env (or UMRS_TEST_DB_URL); None when absent."""
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


class _CatalogMetaStoreCases:
    """Exercise real Postgres FTS and schema against migrations 0054 + 0093."""

    @pytest.fixture(autouse=True)
    def setup_store(self, catalog_db_url):
        from psycopg2.pool import SimpleConnectionPool

        pool = SimpleConnectionPool(1, 4, catalog_db_url)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS catalog_objects CASCADE")
                cur.execute(_MIGRATION_SQL.read_text())
                cur.execute(_VISIBILITY_SQL.read_text())
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
                # Restore the migrated shape for later suites on a shared DB.
                cur.execute(_MIGRATION_SQL.read_text())
                cur.execute(_VISIBILITY_SQL.read_text())
            conn.commit()
        finally:
            pool.putconn(conn)
            pool.closeall()

    def _set_visibility(self, project_id: str, s3_key: str, tier: str) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE catalog_objects SET visibility = %s "
                    "WHERE project_id = %s AND s3_key = %s",
                    (tier, project_id, s3_key),
                )
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def test_search_widens_system_rows_and_hides_ambient(self):
        """ADR-050 on real Postgres: project + system visible, ambient never (R-02)."""
        self.store.upsert(_ready_meta("proj-1/uploads/own.csv", "proj-1", "alpha own"), "proj-1")
        self.store.upsert(
            _ready_meta("proj-1/uploads/hidden.csv", "proj-1", "alpha hidden"), "proj-1"
        )
        self.store.upsert(
            _ready_meta("proj-2/uploads/shared.csv", "proj-2", "alpha shared"), "proj-2"
        )
        self.store.upsert(
            _ready_meta("proj-2/uploads/private.csv", "proj-2", "alpha private"), "proj-2"
        )
        self._set_visibility("proj-1", "proj-1/uploads/hidden.csv", "ambient")
        self._set_visibility("proj-2", "proj-2/uploads/shared.csv", "system")

        keys = {hit.meta.s3_key for hit in self.store.search("proj-1", "alpha", limit=10)}

        assert keys == {"proj-1/uploads/own.csv", "proj-2/uploads/shared.csv"}

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


@pytest.mark.integration
class TestCatalogMetaStoreIntegration(_CatalogMetaStoreCases):
    """Legacy entry: ``--db-url`` / ``UMRS_TEST_DB_URL`` (integration mark)."""

    @pytest.fixture
    def catalog_db_url(self, db_url):
        return db_url


class TestCatalogMetaStoreCIShape(_CatalogMetaStoreCases):
    """Env-gated entry: runs under the CI job's ``POSTGRES_*`` variables."""

    @pytest.fixture
    def catalog_db_url(self):
        url = ci_shape_db_url()
        if not url:
            pytest.skip("POSTGRES_* env not set — CI-shaped Postgres absent")
        return url
