"""R-14 Postgres session hygiene pins (2026-09-13).

Incident 2026-09-09: blank ``application_name`` backends and 10–23 minute
idle-in-transaction holders exhausted the ai4s Postgres. These tests pin:

- the helper output shape (``application_name`` + ``options`` for psycopg2,
  ``server_settings`` for asyncpg);
- the env overrides (``UMRS_PG_APP_NAME`` prefix, ``UMRS_PG_IDLE_IN_TX_TIMEOUT_MS``);
- ``0`` disables the idle timeout entirely;
- that EVERY psycopg2 connect / pool and asyncpg pool construction site in
  ``umrs/src`` passes the helper (source-level AST scan with an explicit,
  justified allowlist);
- ``statement_timeout`` is deliberately NOT set (long migrations / index
  builds must survive).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.shared import pg_settings
from umrs.shared.pg_settings import (
    DEFAULT_IDLE_IN_TX_TIMEOUT_MS,
    application_name,
    asyncpg_pool_kwargs,
    asyncpg_server_settings,
    idle_in_transaction_timeout_ms,
    pg_connect_kwargs,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "umrs"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(pg_settings.ENV_APP_NAME_PREFIX, raising=False)
    monkeypatch.delenv(pg_settings.ENV_IDLE_IN_TX_TIMEOUT_MS, raising=False)


# ── helper output shape ─────────────────────────────────────────────────────


def test_psycopg2_kwargs_default_shape():
    kwargs = pg_connect_kwargs("durable-jobs")
    assert kwargs == {
        "application_name": "umrs-durable-jobs",
        "options": f"-c idle_in_transaction_session_timeout={DEFAULT_IDLE_IN_TX_TIMEOUT_MS}",
    }
    assert DEFAULT_IDLE_IN_TX_TIMEOUT_MS == 600_000


def test_asyncpg_settings_default_shape():
    assert asyncpg_server_settings("knowledge-asyncpg") == {
        "application_name": "umrs-knowledge-asyncpg",
        "idle_in_transaction_session_timeout": str(DEFAULT_IDLE_IN_TX_TIMEOUT_MS),
    }
    kwargs = asyncpg_pool_kwargs("knowledge-asyncpg", dsn="postgresql://x", init=None)
    assert kwargs["dsn"] == "postgresql://x" and kwargs["init"] is None
    assert kwargs["server_settings"]["application_name"] == "umrs-knowledge-asyncpg"


def test_asyncpg_caller_server_settings_merge_and_win():
    kwargs = asyncpg_pool_kwargs(
        "a2l-store", server_settings={"application_name": "custom", "search_path": "x"}
    )
    assert kwargs["server_settings"]["application_name"] == "custom"
    assert kwargs["server_settings"]["search_path"] == "x"
    assert "idle_in_transaction_session_timeout" in kwargs["server_settings"]


def test_statement_timeout_is_never_set():
    """Long migrations / CONCURRENTLY index builds must not be capped globally."""
    assert "statement_timeout" not in pg_connect_kwargs("x").get("options", "")
    assert "statement_timeout" not in asyncpg_server_settings("x")


# ── env overrides ───────────────────────────────────────────────────────────


def test_app_name_prefix_override(monkeypatch):
    monkeypatch.setenv(pg_settings.ENV_APP_NAME_PREFIX, "umrs-front")
    assert application_name("server-pool") == "umrs-front-server-pool"
    assert pg_connect_kwargs("server-pool")["application_name"] == "umrs-front-server-pool"


def test_app_name_is_sanitised_and_bounded():
    assert application_name("odd name/with spaces") == "umrs-odd-name-with-spaces"
    assert application_name("") == "umrs-unnamed"
    assert len(application_name("x" * 200)) == 63


def test_idle_timeout_env_override(monkeypatch):
    monkeypatch.setenv(pg_settings.ENV_IDLE_IN_TX_TIMEOUT_MS, "120000")
    assert idle_in_transaction_timeout_ms() == 120_000
    assert pg_connect_kwargs("x")["options"] == "-c idle_in_transaction_session_timeout=120000"
    assert asyncpg_server_settings("x")["idle_in_transaction_session_timeout"] == "120000"


def test_idle_timeout_zero_disables(monkeypatch):
    monkeypatch.setenv(pg_settings.ENV_IDLE_IN_TX_TIMEOUT_MS, "0")
    assert idle_in_transaction_timeout_ms() == 0
    assert pg_connect_kwargs("x") == {"application_name": "umrs-x"}
    assert asyncpg_server_settings("x") == {"application_name": "umrs-x"}


@pytest.mark.parametrize("raw", ["", "  ", "abc", "-5"])
def test_idle_timeout_invalid_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv(pg_settings.ENV_IDLE_IN_TX_TIMEOUT_MS, raw)
    assert idle_in_transaction_timeout_ms() == DEFAULT_IDLE_IN_TX_TIMEOUT_MS


# ── every construction site passes the helper (source-level pin) ───────────

_CONNECT_NAMES = {"connect", "ThreadedConnectionPool", "SimpleConnectionPool", "create_pool"}
# Module-local ``_pg_kwargs()`` wrappers (api/context.py, api/egress_audit.py)
# delegate to pg_connect_kwargs — pinned by test_local_wrappers_delegate.
_HELPERS = {"pg_connect_kwargs", "asyncpg_pool_kwargs", "_pg_kwargs"}
_LOCAL_WRAPPERS = ("api/context.py", "api/egress_audit.py")

# Sites that legitimately do not carry the hygiene kwargs — each with a reason.
ALLOWLIST: dict[str, str] = {
    # Personal dev script with hard-coded local credentials (user="ejeannot"),
    # never imported by the server; not a platform connection site.
    "inference/embeddings.py": "dev-only script, not a platform connection site",
}


def _is_pg_connect_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _CONNECT_NAMES:
        return False
    base = func.value
    # psycopg2.connect / pg_pool.X / pool.X / asyncpg.create_pool /
    # self._psycopg2.connect — anything psycopg2/pool/asyncpg shaped.
    name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
    return any(token in name for token in ("psycopg2", "pool", "asyncpg", "pg_pool"))


def _carries_helper(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call):  # **helper(...)
            fn = kw.value.func
            if getattr(fn, "id", None) in _HELPERS or getattr(fn, "attr", None) in _HELPERS:
                return True
        if kw.arg == "server_settings":
            return True
    # asyncpg.create_pool(**kwargs) where kwargs came from asyncpg_pool_kwargs(...)
    return any(kw.arg is None and isinstance(kw.value, ast.Name) for kw in node.keywords)


def _connect_sites() -> list[tuple[str, int, bool]]:
    sites: list[tuple[str, int, bool]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_pg_connect_call(node):
                sites.append((rel, node.lineno, _carries_helper(node)))
    return sites


def test_scan_finds_the_known_construction_sites():
    """Guard the scanner itself: it must see the pools the incident named."""
    files = {rel for rel, _line, _ok in _connect_sites()}
    for expected in (
        "shared/pg.py",
        "server/sql.py",
        # ADR-053: builder's home moved to shared/pg.py (site list tracks truth).
        "shared/pg.py",
        "api/_asyncpg_pool.py",
        "knowledge/store.py",
        "migrations/__main__.py",
    ):
        assert expected in files, f"scanner lost sight of {expected}"


def test_every_connect_site_passes_hygiene_kwargs():
    missing = [
        f"{rel}:{line}" for rel, line, ok in _connect_sites() if not ok and rel not in ALLOWLIST
    ]
    assert missing == [], (
        "psycopg2/asyncpg construction sites without umrs.shared.pg_settings kwargs "
        "(R-14): add **pg_connect_kwargs(<component>) / **asyncpg_pool_kwargs(...) "
        "or allowlist with a reason:\n" + "\n".join(missing)
    )


def test_local_wrappers_delegate_to_pg_connect_kwargs():
    for rel in _LOCAL_WRAPPERS:
        tree = ast.parse((SRC_ROOT / rel).read_text())
        wrappers = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_pg_kwargs"
        ]
        assert len(wrappers) == 1, f"{rel}: expected one _pg_kwargs wrapper"
        assert "pg_connect_kwargs" in ast.unparse(wrappers[0]), f"{rel}: wrapper must delegate"


def test_allowlist_entries_still_exist():
    files = {rel for rel, _line, _ok in _connect_sites()}
    stale = sorted(rel for rel in ALLOWLIST if rel not in files)
    assert stale == [], f"allowlist entries with no connect site left: {stale}"


# ── call-capture: the shared sync pool builder forwards the kwargs ─────────


def test_build_sync_pool_from_env_forwards_hygiene(monkeypatch):
    captured: dict = {}

    class _FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import psycopg2.pool as real_pool

    monkeypatch.setattr(real_pool, "ThreadedConnectionPool", _FakePool)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")
    from umrs.shared.pg import build_sync_pool_from_env

    build_sync_pool_from_env(min_conn=1, max_conn=2)
    assert captured["dsn"] == "postgresql://u:p@h:5432/db"
    assert captured["application_name"] == "umrs-sync-pool"
    assert captured["options"].startswith("-c idle_in_transaction_session_timeout=")


def test_knowledge_asyncpg_pool_forwards_server_settings(monkeypatch):
    async def run():
        import sys
        import types

        from fastapi import FastAPI, Request
        from umrs.api import _asyncpg_pool as pool_mod

        app = FastAPI()
        request = Request(scope={"type": "http", "app": app, "headers": []})
        monkeypatch.setenv("DATABASE_URL", "postgresql://a:b@h:5433/kdb")
        captured: dict = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setitem(
            sys.modules, "asyncpg", types.SimpleNamespace(create_pool=fake_create_pool)
        )
        await pool_mod.knowledge_asyncpg_pool(request)
        assert captured["server_settings"]["application_name"] == "umrs-knowledge-asyncpg"
        assert captured["server_settings"]["idle_in_transaction_session_timeout"] == "600000"

    import anyio

    anyio.run(run)
