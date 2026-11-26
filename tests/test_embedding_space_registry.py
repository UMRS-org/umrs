"""Embedding-space registry (ADR-083 D-1, decision KN-3, slice KN-S5).

Acceptance: an unregistered model is refused by the embed writer; the three
e5 names resolve to one space; a stand-alone install has an active space
with no admin step (the migration seeds the bundled row).

The Postgres tests are env-gated like ``test_companion_snapshot_revision``
(``UMRS_TEST_DB_URL`` or ``POSTGRES_*``) and run in a scratch schema.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.knowledge import embedding_space_guard as guard
from umrs.knowledge import embedding_spaces as es
from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    CODE_AMBIGUOUS,
    CODE_INVALID,
    CODE_UNREGISTERED,
    EmbeddingRegistryUnavailable,
    EmbeddingSpace,
    EmbeddingSpaceError,
    UnregisteredEmbeddingModel,
)

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "migrations"
    / "0144_knowledge_embedding_spaces.sql"
)
ME5B = EmbeddingSpace(
    space_id="me5b",
    model_name="/models/me5b",
    dimension=768,
    legacy_names=("intfloat/multilingual-e5-base",),
    source="site",
)


# ── validation ──────────────────────────────────────────────────────────────


def test_the_bundled_space_is_valid_and_names_its_index():
    assert es.validate_space(BUNDLED_SPACE) is BUNDLED_SPACE
    assert BUNDLED_SPACE.index_name == "hnsw_e5_base_v2"
    assert BUNDLED_SPACE.dimension == 768 and BUNDLED_SPACE.active


@pytest.mark.parametrize(
    "change",
    [
        {"space_id": "Bad-ID"},
        {"space_id": "x;drop table t"},
        {"space_id": "a" * 49},
        {"space_id": ""},
        {"space_id": "abc\n"},
        {"model_name": "/models/x\n"},
        {"model_name": "/models/x'; --"},
        {"model_name": "has space"},
        {"model_name": ""},
        {"legacy_names": ("ok", "bad name")},
        {"dimension": 0},
        {"dimension": True},
        {"dimension": 16001},
        {"modalities": ()},
        {"modalities": ("smell",)},
        {"metric": "l2"},
        {"locality": "public"},
        {"status": "deleted"},
        {"source": "user"},
        {"query_prefix": "q\n"},
        {"document_prefix": "p" * 65},
    ],
)
def test_validation_refuses_every_bad_field(change):
    with pytest.raises(EmbeddingSpaceError) as err:
        es.validate_space(dataclasses.replace(BUNDLED_SPACE, **change))
    assert err.value.code == CODE_INVALID
    assert str(err.value).startswith(CODE_INVALID)


def test_the_migration_checks_use_the_module_patterns():
    sql = _MIGRATION.read_text()
    assert f"space_id ~ '{es.SPACE_ID_PATTERN}'" in sql
    assert f"model_name ~ '{es.MODEL_NAME_PATTERN}'" in sql


def test_the_migration_seeds_exactly_the_bundled_space():
    sql = _MIGRATION.read_text()
    values = sql.split("VALUES (", 1)[1]
    for value in (
        BUNDLED_SPACE.space_id,
        BUNDLED_SPACE.model_name,
        BUNDLED_SPACE.model_ref,
        BUNDLED_SPACE.revision,
        BUNDLED_SPACE.query_prefix,
        BUNDLED_SPACE.document_prefix,
        *BUNDLED_SPACE.legacy_names,
    ):
        assert f"'{value}'" in values, value
    assert re.search(r"\n\s*768,\n", values)


# ── resolution ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "/opt/umrs/models/e5-base-v2",
        "/models/e5-base-v2",
        "intfloat/e5-base-v2",
        "/srv/models/e5-base-v2/",
        " e5-base-v2 ",
    ],
)
def test_the_three_e5_names_resolve_to_one_space(name):
    assert es.resolve_space(name, [BUNDLED_SPACE, ME5B]) is BUNDLED_SPACE


def test_an_unregistered_model_is_refused_by_name():
    with pytest.raises(UnregisteredEmbeddingModel) as err:
        es.require_space("/models/bge-m3", [BUNDLED_SPACE])
    assert str(err.value).startswith(CODE_UNREGISTERED)
    assert "/models/bge-m3" in str(err.value)
    assert err.value.permanent is False


def test_a_retired_space_never_resolves():
    retired = dataclasses.replace(BUNDLED_SPACE, status="retired")
    assert es.resolve_space("/opt/umrs/models/e5-base-v2", [retired]) is None


def test_only_bare_names_match_by_basename():
    # me5b's only bare name would be its legacy hub id's tail — it has none.
    assert es.resolve_space("/elsewhere/me5b", [ME5B]) is None
    bare = dataclasses.replace(ME5B, legacy_names=("me5b",))
    assert es.resolve_space("/elsewhere/me5b", [bare]) is bare


def test_an_exact_match_beats_a_basename_match():
    other = EmbeddingSpace(space_id="other", model_name="/x/e5-base-v2", dimension=8)
    assert es.resolve_space("/x/e5-base-v2", [BUNDLED_SPACE, other]) is other


def test_two_spaces_claiming_one_name_is_an_error():
    twin = dataclasses.replace(BUNDLED_SPACE, space_id="twin", model_name="/t/e5")
    with pytest.raises(EmbeddingSpaceError) as err:
        es.resolve_space("/models/e5-base-v2", [BUNDLED_SPACE, twin])
    assert err.value.code == CODE_AMBIGUOUS


def test_names_are_normalized_and_deduplicated():
    space = dataclasses.replace(ME5B, legacy_names=("/models/me5b/", "", "me5b"))
    assert es.space_names(space) == ("/models/me5b", "me5b")


def test_rows_round_trip():
    row = dict(zip(es.COLUMNS, es.space_to_row(BUNDLED_SPACE), strict=True))
    assert es.space_from_row(row) == BUNDLED_SPACE


# ── the writer-side guard (asyncpg-shaped fake connection) ──────────────────


class _Conn:
    def __init__(self, spaces=(), *, missing_table: bool = False):
        self.rows = [dict(zip(es.COLUMNS, es.space_to_row(s), strict=True)) for s in spaces]
        self.missing_table = missing_table
        self.inserted: list[tuple] = []

    async def fetch(self, sql, *args):
        assert sql == guard.SELECT_SPACES_SQL and not args
        if self.missing_table:
            raise type("UndefinedTableError", (Exception,), {})("no table")
        return self.rows

    async def execute(self, sql, *args):
        assert sql == guard.INSERT_IF_ABSENT_SQL
        self.inserted.append(args)
        return "INSERT 0 1"


def _require(conn, name, site=()):
    calls: list[int] = []

    def site_spaces():
        calls.append(1)
        return list(site)

    space = asyncio.run(guard.require_writer_space(conn, name, site_spaces=site_spaces))
    return space, calls


def test_a_registered_model_is_admitted_without_reading_the_site_file():
    space, calls = _require(_Conn([BUNDLED_SPACE]), "/models/e5-base-v2")
    assert space == BUNDLED_SPACE and calls == []


def test_an_empty_registry_stands_for_the_bundled_space():
    space, _ = _require(_Conn([]), "/opt/umrs/models/e5-base-v2")
    assert space == BUNDLED_SPACE


def test_the_guard_refuses_an_unregistered_model():
    with pytest.raises(UnregisteredEmbeddingModel):
        _require(_Conn([BUNDLED_SPACE]), "/models/bge-m3")


def test_a_retired_bundled_space_is_refused_even_by_its_own_name():
    retired = dataclasses.replace(BUNDLED_SPACE, status="retired")
    with pytest.raises(UnregisteredEmbeddingModel):
        _require(_Conn([retired]), "/opt/umrs/models/e5-base-v2")


def test_a_site_declared_space_is_seeded_on_first_write():
    conn = _Conn([BUNDLED_SPACE])
    space, calls = _require(conn, "/models/me5b", site=[ME5B])
    assert calls == [1] and space.space_id == "me5b"
    assert space.source == "site" and space.active
    assert conn.inserted == [es.space_to_row(space)]


def test_a_site_declaration_never_resurrects_a_retired_row():
    conn = _Conn([BUNDLED_SPACE, dataclasses.replace(ME5B, status="retired")])
    with pytest.raises(UnregisteredEmbeddingModel):
        _require(conn, "/models/me5b", site=[ME5B])
    assert conn.inserted == []


# Review KN-S5 (2026-11-27): seeding checked only space_id and model_name,
# so a declared space reusing another row's name (legacy or bare) was seeded.
ACME_STEALS_E5 = EmbeddingSpace(
    space_id="acme",
    model_name="/models/acme",
    dimension=768,
    legacy_names=("e5-base-v2",),
    source="site",
)


def test_a_site_space_reusing_a_registered_name_is_refused_not_seeded():
    conn = _Conn([BUNDLED_SPACE])
    with pytest.raises(EmbeddingSpaceError) as err:
        _require(conn, "/models/acme", site=[ACME_STEALS_E5])
    assert err.value.code == es.CODE_NAME_TAKEN
    assert conn.inserted == []
    # the bundled space still resolves (the seed would have made it ambiguous)
    assert _require(conn, "/opt/umrs/models/e5-base-v2")[0] == BUNDLED_SPACE


def test_a_site_space_never_resurrects_a_retired_name_under_a_new_id():
    retired = dataclasses.replace(BUNDLED_SPACE, status="retired")
    again = EmbeddingSpace(
        space_id="e5_again", model_name="intfloat/e5-base-v2", dimension=768, source="site"
    )
    conn = _Conn([retired])
    with pytest.raises(EmbeddingSpaceError) as err:
        _require(conn, "intfloat/e5-base-v2", site=[again])
    assert err.value.code == es.CODE_NAME_TAKEN
    assert conn.inserted == []


def test_a_site_space_reusing_a_bundled_name_is_refused_on_an_empty_registry():
    conn = _Conn([])
    with pytest.raises(EmbeddingSpaceError) as err:
        _require(conn, "/models/acme", site=[ACME_STEALS_E5])
    assert err.value.code == es.CODE_NAME_TAKEN
    assert conn.inserted == []


def test_a_missing_registry_table_is_named():
    with pytest.raises(EmbeddingRegistryUnavailable) as err:
        _require(_Conn(missing_table=True), "/models/e5-base-v2")
    assert "admin migrate" in str(err.value)


# ── the embed writer (fail-before: the old writer stored any model) ─────────


class _WriterConn:
    def __init__(self):
        self.executed: list[str] = []

    @contextlib.asynccontextmanager
    async def transaction(self):
        yield self

    async def fetch(self, _sql, *_args):
        return []

    async def fetchval(self, *_args):
        return 1

    async def execute(self, sql, *_args):
        self.executed.append(sql)
        return "DELETE 0"


class _WriterStore:
    def __init__(self):
        self.conn = _WriterConn()
        self._pool = self
        self.nodes: list[Any] = []
        self.vectors: list[Any] = []

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self.conn

    async def upsert_node(self, node, **_k):
        self.nodes.append(node)

    async def upsert_edge(self, *_a, **_k):
        return None

    async def upsert_node_embedding(self, *args, **_k):
        self.vectors.append(args)


def _write(monkeypatch, model: str) -> tuple[_WriterStore, Any]:
    from umrs.knowledge import embedding_space_site

    monkeypatch.setattr(embedding_space_site, "site_embedding_spaces", lambda: [])
    return _run_writer(monkeypatch, model)


def _run_writer(monkeypatch, model: str) -> tuple[_WriterStore, Any]:
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", model)
    monkeypatch.setattr(ka, "_resolve_target", _same_target)
    store, result = _WriterStore(), ka.KnowledgeEmbedResult()
    embedded = {"chunks": [{"text": "a body", "start": 0, "end": 6}], "vectors": [[0.1, 0.2]]}
    asyncio.run(ka._write_isolated(store, {"artifact_id": "art-1"}, "P", "", embedded, result))
    return store, result


async def _same_target(_conn, wanted):
    return wanted


def test_the_embed_writer_refuses_an_unregistered_model(monkeypatch):
    store, result = _write(monkeypatch, "/models/bge-m3")
    assert store.nodes == [] and store.vectors == []
    (failed,) = result.failed_items
    assert failed["artifact_id"] == "art-1"
    assert failed["error"].startswith(CODE_UNREGISTERED)
    assert failed["permanent"] is False


def test_the_embed_writer_admits_the_bundled_space(monkeypatch):
    store, result = _write(monkeypatch, "/models/e5-base-v2")
    assert result.failed_items == []
    assert [v[1] for v in store.vectors] == ["/models/e5-base-v2"]


@pytest.fixture
def me5b_site_file(tmp_path, monkeypatch):
    """A real site.yaml declaring me5b, read through the real site reader."""
    from umrs.orchestrator import site_config

    site = tmp_path / "site.yaml"
    site.write_text(
        "name: t\nknowledge:\n  embedding_spaces:\n"
        "    - {space_id: me5b, model_name: /models/me5b, dimension: 768}\n"
    )
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(site))
    site_config.reset_site_config()
    yield site
    site_config.reset_site_config()


def test_the_embed_writer_seeds_a_space_from_the_real_site_file(monkeypatch, me5b_site_file):
    store, result = _run_writer(monkeypatch, "/models/me5b")
    assert result.failed_items == []
    assert [v[1] for v in store.vectors] == ["/models/me5b"]
    assert any(sql.startswith("INSERT INTO knowledge_embedding_spaces") for sql in store.conn.executed)


def test_the_real_site_file_does_not_admit_an_undeclared_model(monkeypatch, me5b_site_file):
    store, result = _run_writer(monkeypatch, "/models/bge-m3")
    assert store.vectors == []
    (failed,) = result.failed_items
    assert failed["error"].startswith(CODE_UNREGISTERED)


# ── the site file key ───────────────────────────────────────────────────────


def test_the_site_config_keeps_the_knowledge_mapping(tmp_path):
    from umrs.orchestrator.site_config import load_site_config

    site = tmp_path / "site.yaml"
    site.write_text(
        "name: t\nknowledge:\n  embedding_spaces:\n"
        "    - {space_id: me5b, model_name: /models/me5b, dimension: 768}\n"
    )
    cfg = load_site_config(str(site))
    assert cfg.knowledge["embedding_spaces"][0]["space_id"] == "me5b"


def test_a_knowledge_key_that_is_not_a_mapping_is_an_error(tmp_path):
    from umrs.orchestrator.site_config import load_site_config

    site = tmp_path / "site.yaml"
    site.write_text("name: t\nknowledge: [1]\n")
    with pytest.raises(TypeError, match="knowledge"):
        load_site_config(str(site))


def test_the_site_schema_accepts_a_declared_space():
    import json

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src/umrs/orchestrator/site-config.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    good = {"name": "t", "knowledge": {"embedding_spaces": [{"space_id": "me5b", "model_name": "/m/me5b", "dimension": 768}]}}
    jsonschema.validate(good, schema)
    bad = {"name": "t", "knowledge": {"embedding_spaces": [{"space_id": "me5b", "model_name": "/m", "dimension": 8, "status": "retired"}]}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


# ── the admin CLI plumbing (parsers and connection) ─────────────────────────


def test_the_admin_parser_routes_embedding_space_verbs():
    from umrs.cli.admin import embedding_spaces as cli
    from umrs.cli.admin.main import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["embedding-space", "register", "--space-id", "me5b", "--model-name",
         "/models/me5b", "--dimension", "768", "--legacy-name", "me5b"]
    )
    assert args.func is cli.cmd_register and args.legacy_name == ["me5b"]
    assert parser.parse_args(["embedding-space", "retire", "me5b"]).func is cli.cmd_retire
    assert parser.parse_args(["embedding-space", "list", "--all"]).func is cli.cmd_list
    seed = parser.parse_args(["embedding-space", "seed-site", "--dry-run"])
    assert seed.func is cli.cmd_seed_site and seed.dry_run


class _CliPool:
    def __init__(self):
        self.events: list[str] = []

    def getconn(self):
        pool = self

        class _C:
            def commit(self):
                pool.events.append("commit")

            def rollback(self):
                pool.events.append("rollback")

        return _C()

    def putconn(self, _conn):
        self.events.append("put")


def test_the_registry_connection_commits_or_rolls_back(monkeypatch):
    from umrs.cli.admin import cascade
    from umrs.cli.admin import embedding_spaces as cli

    pool = _CliPool()
    monkeypatch.setattr(cascade, "cli_db_pool", lambda: pool)
    with cli.registry_conn():
        pass
    with pytest.raises(RuntimeError), cli.registry_conn():
        raise RuntimeError("boom")
    assert pool.events == ["commit", "put", "rollback", "put"]
    monkeypatch.setattr(cascade, "cli_db_pool", lambda: None)
    with pytest.raises(cli.RegistryUnavailable), cli.registry_conn():
        pass


# ── Postgres: the migration (stand-alone needs no admin step) ───────────────


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


_needs_pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def pg():
    import psycopg2

    schema = f"kns5_{uuid.uuid4().hex[:12]}"
    conn = psycopg2.connect(_db_url())
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        cur.execute(_MIGRATION.read_text())
        cur.execute(_MIGRATION.read_text())  # idempotent
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@_needs_pg
def test_the_migration_seeds_one_active_bundled_space(pg):
    with pg.cursor() as cur:
        cur.execute(f"SELECT {', '.join(es.COLUMNS)} FROM knowledge_embedding_spaces")
        rows = cur.fetchall()
    assert [es.space_from_row(dict(zip(es.COLUMNS, r, strict=True))) for r in rows] == [BUNDLED_SPACE]


@_needs_pg
@pytest.mark.parametrize(
    ("column", "value"),
    [("space_id", "Bad-Id"), ("model_name", "x'y"), ("status", "gone"), ("metric", "l2")],
)
def test_the_table_refuses_unsafe_identifiers(pg, column, value):
    import psycopg2

    row = dict(zip(es.COLUMNS, es.space_to_row(ME5B), strict=True))
    row[column] = value
    sql = (
        f"INSERT INTO knowledge_embedding_spaces ({', '.join(es.COLUMNS)}) "
        f"VALUES ({', '.join(['%s'] * len(es.COLUMNS))})"
    )
    with pytest.raises(psycopg2.errors.CheckViolation), pg.cursor() as cur:
        cur.execute(sql, [row[c] for c in es.COLUMNS])


# ── Postgres: the writer guard on a real asyncpg connection (review) ────────


async def _guard_on_pg(schema: str, steps):
    import asyncpg

    conn = await asyncpg.connect(_db_url(), server_settings={"search_path": schema})
    try:
        return [await step(conn) for step in steps]
    finally:
        await conn.close()


def _pg_require(name, site=()):
    async def step(conn):
        try:
            space = await guard.require_writer_space(conn, name, site_spaces=lambda: list(site))
        except (UnregisteredEmbeddingModel, EmbeddingSpaceError) as exc:
            return type(exc).__name__
        return space.space_id

    return step


async def _pg_ids(conn):
    return [r["space_id"] for r in await conn.fetch("SELECT space_id FROM knowledge_embedding_spaces ORDER BY 1")]


@_needs_pg
def test_pg_the_guard_admits_seeds_and_refuses_on_asyncpg(pg):
    with pg.cursor() as cur:
        cur.execute("SELECT current_schema()")
        (schema,) = cur.fetchone()
    steps = [
        _pg_require("/models/e5-base-v2"),
        _pg_require("/models/me5b", site=[ME5B]),
        _pg_require("/models/me5b", site=[ME5B]),
        _pg_require("/models/acme", site=[ACME_STEALS_E5]),
        _pg_require("/models/bge-m3"),
        _pg_ids,
    ]
    got = asyncio.run(_guard_on_pg(schema, steps))
    assert got == [
        "e5_base_v2",
        "me5b",
        "me5b",
        "EmbeddingSpaceError",
        "UnregisteredEmbeddingModel",
        ["e5_base_v2", "me5b"],
    ]
