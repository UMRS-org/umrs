"""Unit tests for registry backend auto-detection."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
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


# ── ADR-060 U-1: one hub registry, venue-scoped ────────────────────────────


class _RecordingCursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._conn.queries.append((" ".join(str(query).split()), params))
        self.rowcount = 1

    def fetchone(self):
        return self._conn.one

    def fetchall(self):
        return self._conn.many


class _RecordingConn:
    closed = False

    def __init__(self, one=None, many=()):
        self.queries: list = []
        self.one = one
        self.many = list(many)

    def cursor(self):
        return _RecordingCursor(self)


def _venue_conn(venue="rccs-front", one=None, many=()):
    from umrs.orchestrator.registry_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._venue = venue
    backend._conn = _RecordingConn(one=one, many=many)
    backend._get_connection = lambda: backend._conn
    return backend


def test_venue_save_stamps_the_site_name():
    backend = _venue_conn("rccs-front")
    from umrs.orchestrator.endpoint_registry import EndpointRegistry

    registry = EndpointRegistry(backend=backend)
    registry.register(
        project_id="default",
        model="m",
        job_id="j1",
        host="node",
        port=1,
        backend="slurm-remote",
    )
    sql, params = backend._conn.queries[-1]
    assert "INSERT INTO endpoint_registry" in sql and "venue" in sql
    assert params[-1] == "rccs-front"


def test_venue_reads_filter_to_the_site():
    backend = _venue_conn("rccs-front")
    backend.load("default", "m")
    backend.load_by_job_id("j1")
    backend.list_all(project_id="default")
    backend.delete("j1")
    backend.clear_all()
    for sql, params in backend._conn.queries:
        assert "venue = %s" in sql, sql
        assert "rccs-front" in params, params


def test_unscoped_backend_keeps_the_legacy_shape():
    backend = _venue_conn("")
    backend.load_by_job_id("j1")
    sql, params = backend._conn.queries[-1]
    assert "venue" not in sql and params == ["j1"]


def test_migrate_schema_adds_venue_column_and_index():
    conn = _RecordingConn(one=None)
    conn.one = None  # information_schema misses → ALTER fires
    from umrs.orchestrator.registry_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._conn = conn
    backend._get_connection = lambda: conn
    backend._migrate_schema()
    sqls = [sql for sql, _ in conn.queries]
    assert any("ADD COLUMN venue" in sql for sql in sqls)
    assert any("CREATE INDEX IF NOT EXISTS idx_registry_venue" in sql for sql in sqls)


def test_get_backend_shared_branch_wins_with_flag(monkeypatch):
    monkeypatch.setenv("UMRS_ENDPOINT_REGISTRY_SHARED", "1")
    monkeypatch.setenv("DB_HOST", "db0")
    # Pinned: CI's DB-backed job exports POSTGRES_PORT=55432 (DB_PORT wins).
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_SUPERUSER", "umrs")
    monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "pw")
    monkeypatch.setenv("UMRS_JOBS_DB", "umrs_db")

    with (
        patch("umrs.orchestrator.registry_backend.PostgresBackend") as mock_backend,
        patch(
            "umrs.orchestrator.registry_backend._shared_registry_venue",
            return_value="rccs-front",
        ),
    ):
        from umrs.orchestrator.registry_backend import get_backend as gb

        gb()

    args, kwargs = mock_backend.call_args
    assert args[0] == "postgres://umrs:pw@db0:5432/umrs_db"
    assert kwargs == {"venue": "rccs-front"}


def test_get_backend_explicit_memory_still_wins_over_shared(monkeypatch):
    monkeypatch.setenv("UMRS_REGISTRY_BACKEND", "memory")
    monkeypatch.setenv("UMRS_ENDPOINT_REGISTRY_SHARED", "1")
    assert isinstance(get_backend(), InMemoryBackend)


# ── S-D47a (DEC-D47): registry keyed (venue, job_id), venue-guarded upsert ──
# 0117 keyed the hub table on job_id alone and save() upserted with
# ``ON CONFLICT (job_id) DO UPDATE SET ..., venue = EXCLUDED.venue``: a second
# venue registering the same Slurm id took the first venue's row over, and
# the first venue's janitor then reaped its live job as an orphan.

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_M0117 = _MIGRATIONS / "0117_hub_endpoint_registry.sql"
_M0141 = _MIGRATIONS / "0141_endpoint_registry_venue_key.sql"


def _endpoint(job_id="4242", host="node-a"):
    from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint

    return RegisteredEndpoint(
        job_id=job_id,
        project_id="default",
        model="m",
        host=host,
        port=8000,
        status=EndpointStatus.READY,
        backend="slurm-remote",
        created_at=datetime(2026, 11, 24, 9, 0),
    )


def test_save_names_the_primary_key_and_never_rewrites_another_venue():
    backend = _venue_conn("rccs-front")
    backend.save(_endpoint())
    sql, _ = backend._conn.queries[-1]
    assert "ON CONFLICT ON CONSTRAINT endpoint_registry_pkey" in sql
    assert "WHERE endpoint_registry.venue = EXCLUDED.venue" in sql
    assert "venue = EXCLUDED.venue," not in sql and "ON CONFLICT (job_id)" not in sql


class _ZeroRowCursor(_RecordingCursor):
    """The guarded upsert met another venue's row: nothing written."""

    def execute(self, query, params=None):
        super().execute(query, params)
        self.rowcount = 0


def test_save_that_touches_no_row_is_refused_by_name():
    from umrs.orchestrator import registry_backend

    backend = _venue_conn("octopus-front")
    backend._conn.cursor = lambda: _ZeroRowCursor(backend._conn)
    refused = getattr(registry_backend, "RegistryVenueConflictError", None)
    assert refused is not None  # old code: silently took the row over
    with pytest.raises(refused, match="job_id='4242'"):
        backend.save(_endpoint())


def test_migration_0141_swaps_the_key_after_a_duplicate_precheck():
    sql = _M0141.read_text()
    assert "PRIMARY KEY (venue, job_id)" in sql
    assert "ADD CONSTRAINT endpoint_registry_pkey" in sql
    assert "HAVING count(*) > 1" in sql and "RAISE EXCEPTION" in sql
    assert "CONCURRENTLY" not in sql.upper()  # one migration transaction


# ── real Postgres (UMRS_TEST_DB_URL; CI's DB-backed job sets it) ───────────

_SCHEMA = f"s_d47a_{os.getpid()}"


def _pg_connect(schema: str = _SCHEMA):
    import psycopg2

    return psycopg2.connect(
        os.environ["UMRS_TEST_DB_URL"], options=f"-c search_path={schema}"
    )


def _run_sql(conn, text: str) -> list:
    with conn.cursor() as cur:
        cur.execute(text)
        rows = cur.fetchall() if cur.description is not None else []
    conn.commit()
    return rows


@pytest.fixture()
def pg_schema():
    if not os.environ.get("UMRS_TEST_DB_URL"):
        pytest.skip("UMRS_TEST_DB_URL is not set")
    admin = _pg_connect("public")
    _run_sql(admin, f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE; CREATE SCHEMA {_SCHEMA}")
    opened: list = []

    def connect():
        conn = _pg_connect()
        opened.append(conn)
        return conn

    yield connect
    for conn in opened:
        conn.close()
    _run_sql(admin, f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    admin.close()


def _venue_backend(connect, venue: str):
    from umrs.orchestrator.registry_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._venue, backend.read_only = venue, False
    backend._conn = connect()
    backend._conn.autocommit = True
    backend._get_connection = lambda: backend._conn
    return backend


def _hub_table(connect, *, keyed: bool):
    conn = connect()
    _run_sql(conn, _M0117.read_text())
    if keyed:
        _run_sql(conn, _M0141.read_text())
    return conn


def _primary_key(conn) -> list:
    return _run_sql(
        conn,
        "SELECT a.attname FROM pg_constraint c "
        "CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(n, o) "
        "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.n "
        "WHERE c.conrelid = 'endpoint_registry'::regclass AND c.contype = 'p' "
        "ORDER BY k.o",
    )


def test_pg_two_venues_with_one_job_id_keep_both_rows(pg_schema):
    conn = _hub_table(pg_schema, keyed=True)
    rccs = _venue_backend(pg_schema, "rccs-front")
    octo = _venue_backend(pg_schema, "octopus-front")
    rccs.save(_endpoint(host="node-rccs"))
    octo.save(_endpoint(host="node-octo"))
    rccs.save(_endpoint(host="node-rccs-2"))  # same-venue upsert still updates
    rows = _run_sql(conn, "SELECT venue, host FROM endpoint_registry ORDER BY venue")
    assert rows == [("octopus-front", "node-octo"), ("rccs-front", "node-rccs-2")]


def test_pg_janitor_of_one_venue_never_selects_or_removes_the_other(pg_schema):
    from umrs.orchestrator.endpoint_registry import EndpointRegistry

    conn = _hub_table(pg_schema, keyed=True)
    _venue_backend(pg_schema, "octopus-front").save(_endpoint(host="node-octo"))
    rccs_backend = _venue_backend(pg_schema, "rccs-front")
    rccs_backend.save(_endpoint(host="node-rccs"))
    rccs = EndpointRegistry(backend=rccs_backend)  # boot load, venue-scoped
    assert [e.host for e in rccs.list_all()] == ["node-rccs"]
    rccs.remove("4242")
    assert _run_sql(conn, "SELECT venue, host FROM endpoint_registry") == [
        ("octopus-front", "node-octo")
    ]


def test_pg_unkeyed_table_refuses_the_cross_venue_takeover(pg_schema):
    conn = _hub_table(pg_schema, keyed=False)  # code deployed, 0141 not yet
    _venue_backend(pg_schema, "rccs-front").save(_endpoint(host="node-rccs"))
    with pytest.raises(RuntimeError, match="another venue"):
        _venue_backend(pg_schema, "octopus-front").save(_endpoint(host="node-octo"))
    rows = _run_sql(conn, "SELECT venue, host FROM endpoint_registry")
    assert rows == [("rccs-front", "node-rccs")]


def test_pg_migration_0141_is_idempotent_and_keeps_rows(pg_schema):
    conn = _hub_table(pg_schema, keyed=False)
    _venue_backend(pg_schema, "rccs-front").save(_endpoint())
    _run_sql(conn, _M0141.read_text())
    _run_sql(conn, _M0141.read_text())
    assert _primary_key(conn) == [("venue",), ("job_id",)]
    assert _run_sql(conn, "SELECT venue, job_id FROM endpoint_registry") == [
        ("rccs-front", "4242")
    ]


def test_pg_migration_0141_precheck_aborts_on_duplicates_and_changes_nothing(pg_schema):
    import psycopg2

    conn = _hub_table(pg_schema, keyed=False)
    _run_sql(conn, "ALTER TABLE endpoint_registry DROP CONSTRAINT endpoint_registry_pkey")
    insert = (
        "INSERT INTO endpoint_registry (job_id, project_id, model, host, port, "
        "status, backend, created_at, venue) VALUES "
        "('7', 'p', 'm', 'h', 1, 'ready', 'b', now(), 'v')"
    )
    _run_sql(conn, f"{insert}; {insert}")
    with pytest.raises(psycopg2.Error, match="S-D47a precheck"):
        _run_sql(conn, _M0141.read_text())
    conn.rollback()
    assert _primary_key(conn) == []
