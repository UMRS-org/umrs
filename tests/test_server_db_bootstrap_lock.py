"""The servers' startup bootstrap is serialized by one advisory lock (2026-10-16).

Every ``umrs-server-<front>`` on umrs-front shares one Postgres cluster and one
``umrs_server_db_user`` role. At start each runs
``create_umrs_server_db_user``: ensure its database, ``ALTER USER`` the shared
role, ``GRANT CREATE ON SCHEMA public``. A full converge starts all four at
once; concurrent catalog updates of the one role row failed with
``ERROR: tuple concurrently updated`` and the losing server crashed in
``create_umrs_server_db_user`` before it restarted (ops-watch #469 and #472,
2026-10-16). The bootstrap now holds a session advisory lock (one constant key)
around those statements, so the servers take turns.
"""

from __future__ import annotations

import pytest
from umrs.server import sql as server_sql


class _Cursor:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self._log = log
        self._last = ""

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, query: object, params: object = None) -> None:
        self._last = str(query)
        self._log.append((self._last, params))

    def fetchone(self) -> tuple[int] | None:
        exists = "pg_database" in self._last or "pg_roles" in self._last
        return (1,) if exists else None

    def close(self) -> None:
        pass


class _Conn:
    def __init__(self, log: list[tuple[str, object]], dbname: str) -> None:
        self._log = log
        self.dbname = dbname
        self.autocommit = False

    def cursor(self) -> _Cursor:
        return _Cursor(self._log)

    def close(self) -> None:
        self._log.append((f"close {self.dbname}", None))

    def commit(self) -> None:
        self._log.append(("commit", None))

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def statements(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    log: list[tuple[str, object]] = []
    cfg = {
        "host": "127.0.0.1",
        "port": 5432,
        "superuser": "umrs",
        "superuser_password": "x",
        "umrs_server_password": "y",
    }
    monkeypatch.setattr(server_sql, "get_db_config", lambda: cfg)
    monkeypatch.setattr(server_sql, "get_umrs_server_db_name", lambda: "umrs_ai4s")
    monkeypatch.setattr(server_sql, "get_umrs_server_db_user", lambda: "umrs_server_db_user")
    monkeypatch.setattr(server_sql, "pg_connect_kwargs", lambda _role: {})
    monkeypatch.setattr(
        server_sql.psycopg2, "connect", lambda **kw: _Conn(log, kw.get("dbname", ""))
    )
    server_sql.create_umrs_server_db_user()
    return log


def _index(log: list[tuple[str, object]], fragment: str) -> int:
    hits = [i for i, (q, _) in enumerate(log) if fragment in q]
    assert hits, (fragment, [q for q, _ in log])
    return hits[0]


def test_the_lock_is_taken_before_any_catalog_statement(statements) -> None:
    lock = _index(statements, "pg_advisory_lock(")
    assert lock < _index(statements, "pg_database")  # _ensure_umrs_db
    assert lock < _index(statements, "ALTER USER")


def test_the_lock_is_released_after_the_grant(statements) -> None:
    unlock = _index(statements, "pg_advisory_unlock(")
    assert unlock > _index(statements, "GRANT CREATE ON SCHEMA public")
    assert unlock < _index(statements, "close postgres")


def test_every_server_uses_the_same_lock_key(statements) -> None:
    keys = {p for q, p in statements if "pg_advisory_" in q}
    assert keys == {(server_sql.SERVER_BOOTSTRAP_LOCK_KEY,)}
    assert 0 < server_sql.SERVER_BOOTSTRAP_LOCK_KEY < 2**63


@pytest.fixture
def role_statements(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    log: list[tuple[str, object]] = []
    cfg = {
        "host": "127.0.0.1",
        "port": 5432,
        "superuser": "umrs",
        "superuser_password": "x",
        "user_password": "z",
    }
    monkeypatch.setattr(server_sql, "get_db_config", lambda: cfg)
    monkeypatch.setattr(server_sql, "pg_connect_kwargs", lambda _role: {})
    monkeypatch.setattr(
        server_sql.psycopg2, "connect", lambda **kw: _Conn(log, kw.get("dbname", ""))
    )
    server_sql.create_db_user("demo_vision")
    return log


def test_experience_role_ddl_runs_under_the_bootstrap_lock(role_statements) -> None:
    """#559 (2026-10-23): every front re-passwords the shared demo roles at
    start; unlocked, two fronts' ALTER USER collided and octopus crashed."""
    lock = _index(role_statements, "pg_advisory_lock(")
    alter = _index(role_statements, "ALTER USER")
    unlock = _index(role_statements, "pg_advisory_unlock(")
    assert lock < _index(role_statements, "pg_roles") < alter < unlock
    assert unlock < _index(role_statements, "close postgres")
    keys = {p for q, p in role_statements if "pg_advisory_" in q}
    assert keys == {(server_sql.SERVER_BOOTSTRAP_LOCK_KEY,)}
