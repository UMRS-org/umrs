"""Regression: api-side asyncpg DSN resolution must not fabricate a DSN.

Before this fix, ``_asyncpg_pool._resolve_knowledge_dsn`` always returned a
localhost-fabricated DSN even in a fully unconfigured environment, defeating
the designed 503 ("no DSN") path and risking a connection to an ambient
localhost postgres with default credentials. Job-side resolution had been
correct; this brings api-side to parity.
"""

from __future__ import annotations

import pytest
from umrs.api import _asyncpg_pool
from umrs.jobs import knowledge_adapters as ka

_KEYS = (
    "UMRS_DATA_CATALOG_DB",
    "DATABASE_URL",
    "DB_HOST",
    "DB_PORT",
    "UMRS_JOBS_DB",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "DB_SUPERUSER",
    "DB_SUPERUSER_PASSWORD",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in _KEYS:
        monkeypatch.delenv(var, raising=False)
    yield


def test_unconfigured_env_returns_blank_and_no_fabrication():
    dsn, _is_local = _asyncpg_pool._resolve_knowledge_dsn()
    assert dsn == ""
    assert "postgres:postgres@127.0.0.1" not in dsn  # never fabricate


def test_unconfigured_env_parity_with_jobs_side():
    api_dsn, _ = _asyncpg_pool._resolve_knowledge_dsn()
    jobs_dsn = ka._resolve_knowledge_dsn()
    assert api_dsn == jobs_dsn == ""


def test_explicit_dsn_still_wins(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://a:b@h:5433/kdb")
    dsn, _ = _asyncpg_pool._resolve_knowledge_dsn()
    assert dsn == "postgresql://a:b@h:5433/kdb"


def test_component_vars_build_dsn_and_local_flag(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.local")
    monkeypatch.setenv("DB_PORT", "5443")
    monkeypatch.setenv("UMRS_JOBS_DB", "knowdb")
    monkeypatch.setenv("DB_SUPERUSER", "op")
    monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "pw")
    dsn, is_local = _asyncpg_pool._resolve_knowledge_dsn()
    assert dsn == "postgresql://op:pw@db.local:5443/knowdb"
    assert is_local is False


def test_component_localhost_flags_local(monkeypatch):
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    dsn, is_local = _asyncpg_pool._resolve_knowledge_dsn()
    assert dsn
    assert is_local is True


def test_pool_created_with_json_codecs(monkeypatch):
    """Codec parity: pool must register JSON codecs or dict params break."""

    async def run():
        import types

        from fastapi import FastAPI, Request
        from umrs.api import _asyncpg_pool as pool_mod

        app = FastAPI()
        request = Request(scope={"type": "http", "app": app, "headers": []})
        monkeypatch.setenv("DATABASE_URL", "postgresql://a:b@h:5433/kdb")

        captured: dict = {}

        class _FakePool:
            pass

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return _FakePool()

        fake_asyncpg = types.SimpleNamespace(create_pool=fake_create_pool)
        import sys

        monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)

        pool = await pool_mod.knowledge_asyncpg_pool(request)
        assert pool is not None
        assert captured["init"] is pool_mod._init_json_codecs
        assert captured["dsn"] == "postgresql://a:b@h:5433/kdb"

    import anyio

    anyio.run(run)


def test_no_dsn_503_is_typed_database_unavailable():
    """Cycle-3 (2026-10-22): the no-DSN 503 carries database.unavailable.

    Untyped, r4 (every version) read it as "No active endpoint available. Run
    `r4 endpoint up`"; the typed code reads as the platform's database
    service. The detail string is unchanged.
    """
    from fastapi import FastAPI, HTTPException, Request

    request = Request(scope={"type": "http", "app": FastAPI(), "headers": []})
    with pytest.raises(HTTPException) as info:
        import anyio

        anyio.run(_asyncpg_pool.knowledge_asyncpg_pool, request)
    assert info.value.status_code == 503
    assert info.value.detail == "Database unavailable (no DSN)"
    assert info.value.headers == {"x-platform-error-code": "database.unavailable"}
