"""
Pytest configuration and fixtures for UMRS tests.
"""

import os
from collections.abc import Generator

import pytest

# Belt-and-braces: the ADR-026 sat_kg import hook already detects pytest via
# sys.modules; keep the env flag for embedded/interpreter runners.
os.environ.setdefault("UMRS_SAT_KG_QUIET", "1")

# Auth mode (audit B12): the UMRS DEFAULT is strict — an unconfigured process
# must not run wide open. The test suite, however, was written against the
# historic mvp default (anonymous TestClient requests, unsigned verified
# headers), so tests run under an explicitly declared mvp baseline; strict-
# mode cases opt in per-test via monkeypatch.setenv("UMRS_AUTH_MODE", "strict").
os.environ.setdefault("UMRS_AUTH_MODE", "mvp")

# Demo/onboarding files that wrap argparse at module scope: collecting them
# at test time makes pytest's argument parser abort with SystemExit — see the
# `usage: __main__.py [-h] [-c CONFIG]` INTERNALERROR in full-suite runs.
collect_ignore = [
    "s3_test",
    "S3_Postgres_PGVector_demo",
]


def pytest_addoption(parser):
    """Add command-line options for integration tests."""
    parser.addoption(
        "--db-url",
        action="store",
        default=None,
        help="PostgreSQL URL for integration tests (e.g., postgresql://user:pass@host:5432/db)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires database)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if no database URL is provided."""
    db_url = config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")

    if db_url:
        # Database URL provided, don't skip integration tests
        return

    skip_integration = pytest.mark.skip(
        reason="Need --db-url or UMRS_TEST_DB_URL env var for integration tests"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _restore_environ_between_tests() -> Generator:
    """Re-apply the pre-test environment after every test.

    Some production paths (e.g. site_config ``inject_env``) write straight
    into ``os.environ`` outside monkeypatch control. Left unrestored, a
    leaked ``DB_HOST``/``DATABASE_URL`` silently flips auto-mode store
    factories onto the PG backend for every later test in the worker
    (observed: dashboard + knowledge suites failing only when run after
    test_rag_service_llm_resolution). Snapshot/restore per test.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


# Tables the knowledge premise suites (adapters / blank_project_e2e /
# demo08_operational) read-after-write against. When a knowledge DSN
# resolves (CI self-hosted, or any host with deployment creds), leftover
# rows from previous runs break counts (rehearsal 2026-09-13: 100 != 21).
_KNOWLEDGE_PREMISE_TABLES = (
    "knowledge_node_embeddings",
    "knowledge_edges",
    "knowledge_nodes",
    "sat_knowledge_object_versions",
    "sat_knowledge_tombstones",
    "sat_knowledge_objects",
    "scientific_memory_entries",
    "hypothesis_history",
    "hypotheses",
    "evidence_items",
    "experiment_results",
    "experiments",
)


def truncate_knowledge_premise_tables() -> None:
    """TRUNCATE premise tables when — and only when — a knowledge DSN
    resolves. No-op on hosts without PG (tests then ride in-memory state)."""
    try:
        from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

        dsn = _resolve_knowledge_dsn()
        if not dsn:
            return
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                for table in _KNOWLEDGE_PREMISE_TABLES:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
        finally:
            conn.commit()
            conn.close()
    except Exception:  # noqa: BLE001 — isolation helper must not break collection
        pass


@pytest.fixture(autouse=True)
def _reset_knowledge_pool_between_tests() -> Generator:
    """The global knowledge pool is loop-bound at build time; with a
    knowledge DSN configured (self-hosted CI) surviving it kills the next
    test with cross-loop asyncpg errors. Rebuild under each test's loop.
    No-op when no DSN resolves (local runs)."""
    yield
    try:
        from umrs.knowledge.store import reset_knowledge_pool

        reset_knowledge_pool()
    except Exception:  # noqa: BLE001 — isolation fixture must not raise
        pass


@pytest.fixture(scope="session")
def db_url(request) -> str:
    """Get database URL from command line or environment."""
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("No database URL provided")
    return url


@pytest.fixture(scope="function")
def db_connection(db_url) -> Generator:
    """
    Create a database connection for integration tests.

    Yields a psycopg2 connection and rolls back after each test.
    """
    import psycopg2

    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
