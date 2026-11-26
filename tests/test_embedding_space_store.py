"""Admin-side registry store (slice KN-S5, task kn-s5-02).

A small psycopg2-shaped fake keeps rows by ``space_id`` and answers the
statement shapes the store's docstrings fix; the Postgres tests at the end
run the same verbs against the real migration (env-gated).
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.knowledge import embedding_space_store as store
from umrs.knowledge import embedding_spaces as es
from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    CODE_EXISTS,
    CODE_INVALID,
    CODE_NAME_TAKEN,
    EmbeddingSpace,
    EmbeddingSpaceError,
)

ME5B = EmbeddingSpace(
    space_id="me5b",
    model_name="/models/me5b",
    dimension=768,
    legacy_names=("me5b",),
    source="admin",
)
CLIP = EmbeddingSpace(
    space_id="clip_b16",
    model_name="/models/clip-b16",
    dimension=512,
    modalities=("image", "text"),
)
_STATUS = es.COLUMNS.index("status")


class _Db:
    """Rows keyed by space_id; every statement recorded with its params."""

    def __init__(self, *spaces: EmbeddingSpace) -> None:
        self.rows = {s.space_id: list(es.space_to_row(s)) for s in spaces}
        self.statements: list[tuple[str, tuple]] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        raise AssertionError("the store never commits: the caller owns the transaction")

    def rollback(self) -> None:
        raise AssertionError("the store never rolls back")

    def space(self, space_id: str) -> EmbeddingSpace:
        return es.space_from_row(dict(zip(es.COLUMNS, self.rows[space_id], strict=True)))

    def writes(self) -> list[str]:
        return [sql for sql, _ in self.statements if not sql.lstrip().upper().startswith("SELECT")]


class _Cursor:
    def __init__(self, db: _Db) -> None:
        self.db = db
        self.rowcount = -1
        self._result: list[tuple] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        params = tuple(params or ())
        self.db.statements.append((sql, params))
        for value in params:
            if isinstance(value, str) and value in ("me5b", "/models/me5b", "clip_b16"):
                assert value not in sql, "values must be parameters, never formatted"
        verb = sql.lstrip().split()[0].upper()
        getattr(self, f"_{verb.lower()}")(sql, params)

    def _select(self, _sql: str, params: tuple) -> None:
        rows = [self.db.rows[k] for k in sorted(self.db.rows)]
        if params:
            rows = [r for r in rows if r[0] == params[0]]
        self._result = [tuple(r) for r in rows]

    def _insert(self, sql: str, params: tuple) -> None:
        if params[0] in self.db.rows:
            assert "ON CONFLICT" in sql.upper(), "duplicate INSERT without ON CONFLICT"
            self.rowcount = 0
            return
        self.db.rows[params[0]] = list(params)
        self.rowcount = 1

    def _update(self, _sql: str, params: tuple) -> None:
        space_id = params[-1]
        if space_id not in self.db.rows:
            self.rowcount = 0
            return
        if len(params) == 2:
            self.db.rows[space_id][_STATUS] = params[0]
        else:
            assert len(params) == len(es.COLUMNS), "replace sets every column but space_id"
            self.db.rows[space_id] = [space_id, *params[:-1]]
        self.rowcount = 1

    def fetchall(self) -> list[tuple]:
        return list(self._result)

    def fetchone(self) -> tuple | None:
        return self._result[0] if self._result else None


# ── reads ───────────────────────────────────────────────────────────────────


def test_list_returns_active_spaces_in_id_order():
    retired = dataclasses.replace(CLIP, status="retired")
    db = _Db(ME5B, BUNDLED_SPACE, retired)
    assert store.list_spaces(db) == [BUNDLED_SPACE, ME5B]
    assert store.list_spaces(db, include_retired=True) == [retired, BUNDLED_SPACE, ME5B]


def test_list_runs_one_unparameterized_select():
    db = _Db(BUNDLED_SPACE)
    store.list_spaces(db)
    ((sql, params),) = db.statements
    assert sql.lstrip().upper().startswith("SELECT") and params == ()
    assert es.TABLE in sql


def test_get_finds_a_space_in_any_status():
    retired = dataclasses.replace(ME5B, status="retired")
    db = _Db(BUNDLED_SPACE, retired)
    assert store.get_space(db, "me5b") == retired
    assert store.get_space(db, "nope") is None
    assert db.statements[-1][1] == ("nope",)


# ── register ────────────────────────────────────────────────────────────────


def test_register_inserts_a_new_space():
    db = _Db(BUNDLED_SPACE)
    assert store.register_space(db, ME5B) is True
    assert db.space("me5b") == ME5B


def test_register_refuses_an_existing_id_without_replace():
    db = _Db(BUNDLED_SPACE, ME5B)
    changed = dataclasses.replace(ME5B, dimension=1024)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.register_space(db, changed)
    assert err.value.code == CODE_EXISTS and "me5b" in str(err.value)
    assert db.writes() == [] and db.space("me5b") == ME5B


def test_register_replace_overwrites_the_row():
    db = _Db(BUNDLED_SPACE, ME5B)
    changed = dataclasses.replace(ME5B, dimension=1024, revision="r2")
    assert store.register_space(db, changed, replace=True) is False
    assert db.space("me5b") == changed
    (update,) = db.writes()
    assert update.lstrip().upper().startswith("UPDATE") and "updated_at" in update


def test_register_validates_before_any_write():
    db = _Db(BUNDLED_SPACE)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.register_space(db, dataclasses.replace(ME5B, space_id="Bad-Id"))
    assert err.value.code == CODE_INVALID and db.writes() == []


@pytest.mark.parametrize(
    "taker",
    [
        # its model name is a legacy name of the bundled space
        EmbeddingSpace(space_id="x", model_name="intfloat/e5-base-v2", dimension=8),
        # its path ends in the bundled space's bare name
        EmbeddingSpace(space_id="x", model_name="/srv/e5-base-v2", dimension=8),
        # its bare legacy name matches the bundled model's basename
        EmbeddingSpace(space_id="x", model_name="/m/x", dimension=8, legacy_names=("e5-base-v2",)),
    ],
)
def test_register_refuses_a_name_another_space_owns(taker):
    db = _Db(BUNDLED_SPACE)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.register_space(db, taker)
    assert err.value.code == CODE_NAME_TAKEN
    assert "e5_base_v2" in str(err.value) and "x" in str(err.value)
    assert db.writes() == []


def test_a_retired_space_still_owns_its_names():
    db = _Db(BUNDLED_SPACE, dataclasses.replace(ME5B, status="retired"))
    with pytest.raises(EmbeddingSpaceError) as err:
        store.register_space(db, dataclasses.replace(ME5B, space_id="me5b_v2"))
    assert err.value.code == CODE_NAME_TAKEN


def test_replacing_a_space_does_not_collide_with_its_own_names():
    db = _Db(BUNDLED_SPACE, ME5B)
    assert store.register_space(db, ME5B, replace=True) is False


# ── seed and status ─────────────────────────────────────────────────────────


def test_seed_inserts_only_absent_spaces_in_input_order():
    retired = dataclasses.replace(ME5B, status="retired")
    db = _Db(BUNDLED_SPACE, retired)
    inserted = store.seed_spaces(db, [CLIP, ME5B])
    assert inserted == ["clip_b16"]
    assert db.space("clip_b16") == CLIP
    assert db.space("me5b") == retired  # never resurrected
    assert all("ON CONFLICT" in sql.upper() for sql in db.writes())


def test_seed_refuses_a_space_reusing_another_rows_name():
    # Review KN-S5: seed-site inserted a space whose legacy name another row
    # owns; every later lookup of that name then failed as ambiguous.
    steals = dataclasses.replace(CLIP, legacy_names=("e5-base-v2",))
    db = _Db(BUNDLED_SPACE)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.seed_spaces(db, [steals])
    assert err.value.code == CODE_NAME_TAKEN and "clip_b16" in str(err.value)
    assert db.writes() == []


def test_seed_refuses_a_retired_rows_name_under_a_new_id():
    retired = dataclasses.replace(ME5B, status="retired")
    again = dataclasses.replace(ME5B, space_id="me5b_again", model_name="/opt/me5b")
    db = _Db(retired)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.seed_spaces(db, [again])
    assert err.value.code == CODE_NAME_TAKEN
    assert db.writes() == []


def test_seed_validates_each_space():
    with pytest.raises(EmbeddingSpaceError):
        store.seed_spaces(_Db(), [dataclasses.replace(CLIP, dimension=0)])


def test_set_status_retires_and_reactivates():
    db = _Db(BUNDLED_SPACE, ME5B)
    assert store.set_status(db, "me5b", "retired") is True
    assert db.space("me5b").status == "retired"
    assert store.set_status(db, "me5b", "active") is True
    assert db.space("me5b").active
    assert db.statements[-1][1] == ("active", "me5b")


def test_set_status_on_a_missing_space_is_false():
    assert store.set_status(_Db(BUNDLED_SPACE), "nope", "retired") is False


def test_set_status_refuses_an_unknown_status_before_sql():
    db = _Db(BUNDLED_SPACE)
    with pytest.raises(EmbeddingSpaceError) as err:
        store.set_status(db, "e5_base_v2", "deleted")
    assert err.value.code == CODE_INVALID and db.statements == []


# ── Postgres (env-gated): the store against the real migration ──────────────

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src/umrs/migrations/0145_knowledge_embedding_spaces.sql"
)


def _db_url() -> str | None:
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


@pytest.fixture()
def pg():
    if _db_url() is None:
        pytest.skip("no CI-shape Postgres")
    import psycopg2

    schema = f"kns5s_{uuid.uuid4().hex[:12]}"
    conn = psycopg2.connect(_db_url())
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        cur.execute(_MIGRATION.read_text())
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


def test_pg_register_retire_seed_round_trip(pg):
    assert store.list_spaces(pg) == [BUNDLED_SPACE]
    assert store.register_space(pg, ME5B) is True
    assert store.set_status(pg, "me5b", "retired") is True
    assert [s.space_id for s in store.list_spaces(pg)] == ["e5_base_v2"]
    assert store.seed_spaces(pg, [ME5B, CLIP]) == ["clip_b16"]


def test_pg_seed_keeps_retired_and_replace_overwrites(pg):
    store.register_space(pg, ME5B)
    store.set_status(pg, "me5b", "retired")
    store.seed_spaces(pg, [ME5B, CLIP])
    assert store.get_space(pg, "me5b").status == "retired"
    changed = dataclasses.replace(CLIP, dimension=640)
    assert store.register_space(pg, changed, replace=True) is False
    assert store.get_space(pg, "clip_b16") == changed
