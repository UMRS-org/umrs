"""PooledModelStore: the model store survives a reaped Postgres session.

Live rccs/bdrt 2026-10-07 (Ops Watch, ADR-089): every ``Exception in ASGI
application`` and the model-store lookups ended in
``psycopg2.OperationalError: server closed the connection unexpectedly``. The
sidecar leased one connection at boot; reads never ended their implicit
transaction, and the R-14 idle-in-transaction reaper killed the session. The
fakes die the way psycopg2 does: the failing call raises OperationalError and
the connection then reports ``closed``.
"""

from __future__ import annotations

import psycopg2
import pytest
from umrs.model_registry.store import ModelStore, PooledModelStore
from umrs.model_registry.store.pooled import READ_METHODS


class _Conn:
    def __init__(self, *, die: bool = False) -> None:
        self.closed = 0
        self.die = die
        self.in_transaction = False
        self.rollbacks = 0
        self.commits = 0

    def cursor(self):
        return _Cursor(self)

    def rollback(self) -> None:
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        self.rollbacks += 1
        self.in_transaction = False

    def commit(self) -> None:
        self.commits += 1
        self.in_transaction = False


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def execute(self, *_args) -> None:
        if self._conn.die:
            self._conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        self._conn.in_transaction = True


class _Pool:
    def __init__(self, *conns: _Conn) -> None:
        self._free = list(conns)
        self.returned: list[tuple[_Conn, bool]] = []

    def getconn(self) -> _Conn:
        return self._free.pop(0)

    def putconn(self, conn: _Conn, close: bool = False) -> None:
        self.returned.append((conn, close))


@pytest.fixture
def methods(monkeypatch):
    """Stand-in read and write bodies that use the store's connection."""

    def list_models(self, ctx=None, limit=500):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1")
        return ["m"]

    def update_model(self, *_args, **_kwargs):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE model_catalog_entries SET x = 1")
        self.conn.commit()
        return "updated"

    monkeypatch.setattr(ModelStore, "list_models", list_models)
    monkeypatch.setattr(ModelStore, "update_model", update_model)


def test_a_read_ends_its_transaction_and_returns_the_connection(methods):
    conn = _Conn()
    pool = _Pool(conn)
    assert PooledModelStore(pool).list_models(None) == ["m"]
    assert (conn.in_transaction, conn.rollbacks) == (False, 1)
    assert pool.returned == [(conn, False)]


def test_a_write_commits_and_is_still_released(methods):
    conn = _Conn()
    pool = _Pool(conn)
    assert PooledModelStore(pool).update_model("id") == "updated"
    assert conn.commits == 1 and pool.returned == [(conn, False)]


def test_a_read_on_a_reaped_session_is_retried_once_on_a_fresh_one(methods):
    dead, fresh = _Conn(die=True), _Conn()
    pool = _Pool(dead, fresh)
    assert PooledModelStore(pool).list_models(None) == ["m"]
    assert pool.returned == [(dead, True), (fresh, False)]  # the dead one is discarded


def test_a_write_on_a_reaped_session_is_not_retried(methods):
    dead, spare = _Conn(die=True), _Conn()
    pool = _Pool(dead, spare)
    with pytest.raises(psycopg2.OperationalError):
        PooledModelStore(pool).update_model("id")
    assert pool.returned == [(dead, True)]
    assert spare.commits == 0


def test_a_second_loss_is_raised(methods):
    pool = _Pool(_Conn(die=True), _Conn(die=True))
    with pytest.raises(psycopg2.OperationalError):
        PooledModelStore(pool).list_models(None)
    assert [close for _conn, close in pool.returned] == [True, True]


def test_non_session_errors_are_not_retried(monkeypatch):
    def boom(self, *_a, **_k):
        raise ValueError("bad input")

    monkeypatch.setattr(ModelStore, "get_model", boom)
    conn = _Conn()
    pool = _Pool(conn, _Conn())
    with pytest.raises(ValueError):
        PooledModelStore(pool).get_model("x")
    assert pool.returned == [(conn, False)]


def test_only_public_store_methods_are_exposed():
    store = PooledModelStore(_Pool())
    assert store.acl is not None and store.path_validator is not None
    for name in ("_conn", "conn_missing", "nonexistent"):
        with pytest.raises(AttributeError):
            getattr(store, name)


def test_read_allowlist_names_real_read_methods():
    for name in READ_METHODS:
        assert callable(getattr(ModelStore, name))
    assert not READ_METHODS & {
        "create_model",
        "update_model",
        "delete_model",
        "update_policy_by_model_name",  # delegates to a write
        "activate_version",
    }


def test_the_sidecar_wires_a_pooled_store_and_keeps_no_boot_lease(monkeypatch):
    from types import SimpleNamespace

    import umrs.model_registry as model_registry
    from umrs.api import models as models_api
    from umrs.server import http_sidecar

    monkeypatch.setattr(model_registry, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(models_api, "_model_store", None)
    boot = _Conn()
    pool = _Pool(boot)
    http_sidecar._init_model_store(SimpleNamespace(state=SimpleNamespace()), pool)
    assert isinstance(models_api.get_model_store(), PooledModelStore)
    assert pool.returned == [(boot, False)]  # the DDL connection went back


def test_release_connection_rolls_back_or_discards():
    from umrs.model_registry.store import release_connection

    live, dead = _Conn(), _Conn()
    live.in_transaction = True
    dead.closed = 2
    pool = _Pool()
    release_connection(pool, live)
    release_connection(pool, dead)
    assert live.in_transaction is False
    assert pool.returned == [(live, False), (dead, True)]


def test_grant_write_check_returns_its_connection_clean(monkeypatch):
    from types import SimpleNamespace

    from umrs.api import grants

    def get_model(self, model_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1")
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(ModelStore, "get_model", get_model)
    conn = _Conn()
    pool = _Pool(conn)
    caller = SimpleNamespace(user_id="u", is_admin=True, project_id="p")
    assert grants._model_writable(pool, "m-1", caller) in (True, False)
    assert (conn.in_transaction, pool.returned) == (False, [(conn, False)])
