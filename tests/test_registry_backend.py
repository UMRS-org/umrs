"""Unit tests for registry backend auto-detection."""

from unittest.mock import patch

from umrs.orchestrator.registry_backend import InMemoryBackend, get_backend


def test_get_backend_auto_detects_postgres_from_dsn(monkeypatch):
    monkeypatch.delenv("UMRS_REGISTRY_BACKEND", raising=False)
    monkeypatch.setenv(
        "UMRS_REGISTRY_POSTGRES_DSN",
        "postgres://umrs:umrs_rccs@localhost:5432/umrs_db",
    )

    with patch("umrs.orchestrator.registry_backend.PostgresBackend") as mock_backend:
        get_backend()

    mock_backend.assert_called_once_with("postgres://umrs:umrs_rccs@localhost:5432/umrs_db")


def test_get_backend_explicit_memory_overrides_dsn(monkeypatch):
    monkeypatch.setenv("UMRS_REGISTRY_BACKEND", "memory")
    monkeypatch.setenv(
        "UMRS_REGISTRY_POSTGRES_DSN",
        "postgres://umrs:umrs_rccs@localhost:5432/umrs_db",
    )

    backend = get_backend()

    assert isinstance(backend, InMemoryBackend)


# ── Postgres row shape (regression 2026-10-01) ─────────────────────────────
# list_all() selected 15 columns while _row_to_endpoint reads 16, so every
# startup load raised IndexError, the registry came up empty after each
# restart, and the janitor stopped every live endpoint as an orphan.


class _FakeCursor:
    """Answers any SELECT with one row as wide as its column list."""

    def __init__(self):
        self.width = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, _params=None):
        import re

        match = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.S | re.I)
        self.width = len([c for c in match.group(1).split(",") if c.strip()]) if match else 0

    def _row(self):
        from datetime import datetime

        now = datetime(2026, 10, 1, 9, 57)
        base = [
            "391365",
            "default",
            "Bespoke-Nimble-9B",
            "127.0.0.1",
            12680,
            "ready",
            "slurm-remote",
            None,
            now,
            now,
            None,
            None,
            None,
            0,
            {"access": ["system"]},
            "391365",
        ]
        return tuple(base[: self.width])

    def fetchall(self):
        return [self._row()]

    def fetchone(self):
        return self._row()


class _FakeConn:
    closed = False

    def cursor(self):
        return _FakeCursor()


def _pg_backend():
    from umrs.orchestrator.registry_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)  # no DSN, no connect
    backend._conn = _FakeConn()
    backend._get_connection = lambda: backend._conn
    return backend


def test_postgres_list_all_rows_parse_with_backend_job_id():
    (row,) = _pg_backend().list_all()
    assert (row.job_id, row.backend_job_id, row.metadata["access"]) == (
        "391365",
        "391365",
        ["system"],
    )


def test_every_postgres_read_selects_the_columns_the_parser_reads():
    backend = _pg_backend()
    assert backend.load("default", "Bespoke-Nimble-9B").backend_job_id == "391365"
    assert backend.load_by_job_id("391365").backend_job_id == "391365"
    assert backend.list_all(project_id="default", status_filter=["ready"])[0].port == 12680


def test_registry_restores_rows_from_the_postgres_backend():
    from umrs.orchestrator.endpoint_registry import EndpointRegistry

    registry = EndpointRegistry(backend=_pg_backend())
    restored = registry.get_by_job_id("391365")
    assert restored is not None and restored.status.value == "ready"
