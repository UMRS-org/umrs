"""
Pytest configuration and fixtures for UMRS tests.
"""

import os
from typing import Generator

import pytest

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
