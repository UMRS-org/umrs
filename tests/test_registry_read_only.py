"""Read-only endpoint registry pins (ADR-060 U-1 follow-up, 2026-10-03).

umrs-rag reads the hub registry the umrs servers own. Its old writer
registry upserted every column from its boot snapshot on mark_used, venue
included ('' without a site.yaml), so endpoints would vanish from their
owning server's venue-filtered view. A reader now never runs DDL, never
issues INSERT/UPDATE/DELETE, and re-reads the owners' rows on list_all.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.registry_backend import (
    PostgresBackend,
    RegistryReadOnlyError,
    drop_read_only_flag_in_owner,
    get_backend,
)

_WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "DROP")


def _row(job_id, project="default", model="m", status="ready"):
    now = datetime(2026, 10, 3, 9, 0)
    return (
        job_id,
        project,
        model,
        "node",
        8000,
        status,
        "slurm-remote",
        None,
        now,
        now,
        now,
        None,
        None,
        0,
        {},
        f"b-{job_id}",
    )


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._conn.queries.append((" ".join(str(query).split()), params))
        if self._conn.fail:
            raise RuntimeError("db down")

    def fetchone(self):
        return self._conn.rows[0] if self._conn.rows else None

    def fetchall(self):
        return list(self._conn.rows)


class _Conn:
    closed = False

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.queries: list = []
        self.fail = False

    def cursor(self):
        return _Cursor(self)


def _reader(venue="rccs-front", rows=()):
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._venue = venue
    backend.read_only = True
    backend._conn = _Conn(rows)
    backend._get_connection = lambda: backend._conn
    return backend


def _writes(backend):
    return [sql for sql, _ in backend._conn.queries if sql.split(" ", 1)[0].upper() in _WRITE_VERBS]


def test_reader_constructor_runs_no_ddl_and_opens_no_connection():
    with (
        patch.object(PostgresBackend, "_ensure_database") as ensure,
        patch.object(PostgresBackend, "_init_schema") as init,
        patch.object(PostgresBackend, "_migrate_schema") as migrate,
    ):
        backend = PostgresBackend("postgres://u:p@h/umrs_db", venue="rccs-front", read_only=True)
    assert backend.read_only is True and backend._conn is None
    ensure.assert_not_called()
    init.assert_not_called()
    migrate.assert_not_called()


def test_reader_session_is_read_only_server_side():
    fake = MagicMock()
    fake.closed = False
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._dsn, backend._venue, backend.read_only, backend._conn = (
        "postgres://h/db",
        "",
        True,
        None,
    )
    with patch("psycopg2.connect", return_value=fake):
        backend._get_connection()
    fake.set_session.assert_called_once_with(readonly=True, autocommit=True)


@pytest.mark.parametrize(
    ("operation", "args"),
    [("save", (MagicMock(),)), ("delete", ("j1",)), ("clear_all", ())],
)
def test_reader_backend_refuses_every_write_before_any_sql(operation, args):
    backend = _reader()
    with pytest.raises(RegistryReadOnlyError, match=operation):
        getattr(backend, operation)(*args)
    assert backend._conn.queries == []


def test_registry_over_a_reader_never_writes_the_shared_table():
    backend = _reader(rows=[_row("j1")])
    registry = EndpointRegistry(backend=backend)
    registry.mark_used("j1")  # the RAG candidate path
    registry.update_status("j1", EndpointStatus.DEGRADED)
    registry.register(project_id="default", model="m2", job_id="j2", host="n", port=1, backend="x")
    registry.remove("j2")
    registry.clear_all()
    assert _writes(backend) == []
    for sql, params in backend._conn.queries:  # reads only, own venue only
        assert sql.startswith("SELECT") and "venue = %s" in sql, sql
        assert params[0] == "rccs-front"


def test_mark_used_on_a_reader_stays_in_memory():
    backend = _reader(rows=[_row("j1")])
    registry = EndpointRegistry(backend=backend)
    before = registry.get_by_job_id("j1").last_used_at
    endpoint = registry.mark_used("j1")
    assert endpoint is not None and endpoint.last_used_at != before
    assert _writes(backend) == []


def test_reader_list_all_follows_the_owners_rows():
    backend = _reader(rows=[_row("j1"), _row("j2")])
    registry = EndpointRegistry(backend=backend)
    assert {e.job_id for e in registry.list_all()} == {"j1", "j2"}
    backend._conn.rows = [_row("j2"), _row("j3")]  # owner removed j1, added j3
    assert {e.job_id for e in registry.list_all()} == {"j2", "j3"}
    assert registry.get_by_job_id("j1") is None


def test_reader_refresh_failure_keeps_the_last_view(caplog):
    backend = _reader(rows=[_row("j1")])
    registry = EndpointRegistry(backend=backend)
    backend._conn.fail = True
    with caplog.at_level(logging.WARNING):
        endpoints = registry.list_all()
    assert [e.job_id for e in endpoints] == ["j1"]
    assert "refresh failed" in caplog.text


def test_writer_registry_keeps_its_boot_snapshot_and_its_writes():
    backend = _reader(rows=[_row("j1")])
    backend.read_only = False  # an owning server
    registry = EndpointRegistry(backend=backend)
    reads_after_boot = len(backend._conn.queries)
    registry.list_all()
    assert len(backend._conn.queries) == reads_after_boot  # no re-read
    registry.mark_used("j1")
    assert any(sql.startswith("INSERT INTO endpoint_registry") for sql in _writes(backend))


def test_a_mock_backend_is_not_mistaken_for_a_reader():
    backend = MagicMock()
    backend.list_all.return_value = []
    registry = EndpointRegistry(backend=backend)
    assert registry._read_only is False


class _RecordingReader:
    """A read-only backend that records writes instead of refusing them, so
    only the registry's own guard keeps them away (the backend's refusal is
    raised before any SQL and swallowed by _persist, which hides a missing
    registry guard from the SQL-level pins above)."""

    read_only = True

    def __init__(self):
        self.writes: list[str] = []

    def list_all(self):
        return []

    def save(self, endpoint):
        self.writes.append(f"save {endpoint.job_id}")

    def delete(self, job_id):
        self.writes.append(f"delete {job_id}")
        return True

    def clear_all(self):
        self.writes.append("clear_all")
        return 0


def test_the_registry_itself_never_calls_a_readers_write_methods(caplog):
    backend = _RecordingReader()
    registry = EndpointRegistry(backend=backend)
    with caplog.at_level(logging.WARNING):
        registry.register(
            project_id="default", model="m", job_id="j1", host="n", port=1, backend="x"
        )
        registry.mark_used("j1")
        registry.update_status("j1", EndpointStatus.DEGRADED)
        registry.remove("j1")
        registry.clear_all()
    assert backend.writes == []
    # No "Failed to persist endpoint" noise on every RAG mark_used either.
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


# ── get_backend wiring ─────────────────────────────────────────────────────


def _shared_env(monkeypatch):
    monkeypatch.delenv("UMRS_REGISTRY_BACKEND", raising=False)
    monkeypatch.setenv("UMRS_ENDPOINT_REGISTRY_SHARED", "1")
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    # Hermetic: the self-hosted CI runner exports POSTGRES_PORT=55432.
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setenv("DB_SUPERUSER", "umrs")
    monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "pw")
    monkeypatch.setenv("UMRS_JOBS_DB", "umrs_db")


def test_get_backend_reader_uses_the_explicit_venue_without_site_yaml(monkeypatch):
    _shared_env(monkeypatch)
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "1")
    monkeypatch.setenv("UMRS_REGISTRY_VENUE", "rccs-front")
    with (
        patch("umrs.orchestrator.registry_backend.PostgresBackend") as backend_cls,
        patch("umrs.orchestrator.registry_backend._shared_registry_venue") as site_venue,
    ):
        get_backend()
    args, kwargs = backend_cls.call_args
    assert args[0] == "postgres://umrs:pw@127.0.0.1:5432/umrs_db"
    assert kwargs == {"venue": "rccs-front", "read_only": True}
    site_venue.assert_not_called()


def test_get_backend_reader_without_any_venue_warns_it_reads_every_venue(monkeypatch, caplog):
    _shared_env(monkeypatch)
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "1")
    monkeypatch.delenv("UMRS_REGISTRY_VENUE", raising=False)
    with (
        patch("umrs.orchestrator.registry_backend.PostgresBackend") as backend_cls,
        patch(
            "umrs.orchestrator.registry_backend._shared_registry_venue",
            return_value="",
        ),
        caplog.at_level(logging.WARNING),
    ):
        get_backend()
    assert backend_cls.call_args.kwargs == {"venue": "", "read_only": True}
    assert "EVERY venue" in caplog.text


def test_get_backend_writer_ignores_the_reader_venue(monkeypatch):
    _shared_env(monkeypatch)
    monkeypatch.delenv("UMRS_REGISTRY_READ_ONLY", raising=False)
    monkeypatch.setenv("UMRS_REGISTRY_VENUE", "elsewhere")
    with (
        patch("umrs.orchestrator.registry_backend.PostgresBackend") as backend_cls,
        patch(
            "umrs.orchestrator.registry_backend._shared_registry_venue",
            return_value="bdrt-front",
        ),
    ):
        get_backend()
    assert backend_cls.call_args.kwargs == {"venue": "bdrt-front"}


def test_get_backend_reader_on_the_per_cluster_dsn(monkeypatch):
    monkeypatch.delenv("UMRS_REGISTRY_BACKEND", raising=False)
    monkeypatch.delenv("UMRS_ENDPOINT_REGISTRY_SHARED", raising=False)
    monkeypatch.setenv("UMRS_REGISTRY_POSTGRES_DSN", "postgres://u:p@h/front_rccs")
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "true")
    with patch("umrs.orchestrator.registry_backend.PostgresBackend") as backend_cls:
        get_backend()
    backend_cls.assert_called_once_with("postgres://u:p@h/front_rccs", read_only=True)


# ── the standalone RAG service is a reader by default ─────────────────────


def test_rag_service_main_defaults_the_registry_to_read_only(monkeypatch):
    from umrs.api import rag_service

    # setenv first so teardown removes the value main() sets (a bare delenv
    # of an absent variable records nothing and the "1" would leak into the
    # tests that follow, turning their get_backend() into a reader).
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "")
    monkeypatch.delenv("UMRS_REGISTRY_READ_ONLY")
    with patch("uvicorn.run") as run:
        rag_service.main()
    run.assert_called_once()
    assert os.environ["UMRS_REGISTRY_READ_ONLY"] == "1"


def test_rag_service_main_keeps_an_explicit_registry_mode(monkeypatch):
    from umrs.api import rag_service

    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "0")
    with patch("uvicorn.run"):
        rag_service.main()
    assert os.environ["UMRS_REGISTRY_READ_ONLY"] == "0"


# ── an owning server never runs read-only (the flag is umrs-rag's) ────────


def test_an_owning_server_drops_a_read_only_flag_leaked_from_env_file(monkeypatch, caplog):
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "1")
    with caplog.at_level(logging.ERROR):
        assert drop_read_only_flag_in_owner() is True
    assert "UMRS_REGISTRY_READ_ONLY" not in os.environ
    assert "IGNORED" in caplog.text


def test_an_owning_server_leaves_a_falsy_flag_alone(monkeypatch, caplog):
    monkeypatch.setenv("UMRS_REGISTRY_READ_ONLY", "0")
    with caplog.at_level(logging.ERROR):
        assert drop_read_only_flag_in_owner() is False
    assert os.environ["UMRS_REGISTRY_READ_ONLY"] == "0"
    assert caplog.text == ""


def test_the_server_entrypoint_drops_the_flag_before_it_boots():
    src = (Path(__file__).resolve().parents[1] / "src" / "server_umrs.py").read_text()
    main_body = src.split("\ndef main():", 1)[1].split("\ndef ", 1)[0]
    assert "drop_read_only_flag_in_owner()" in main_body
    assert main_body.index("drop_read_only_flag_in_owner()") < main_body.index(
        "parser.parse_args()"
    )
