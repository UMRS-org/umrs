"""Ontology registry spec round trip on real Postgres (ADR-116 D-4, migration 0139).

Before 0139, ``register()`` stored only entity/relation/transitive types, so
``subtype_of``, ``extraction_fields`` and every ADR-116 field were lost on
the way back. Runs when ``UMRS_TEST_DB_URL`` (or the CI job's
``POSTGRES_*``) names a migrated database, and skips cleanly otherwise; the
connection rolls back after each test (``commit`` is a no-op here).

Review fixes pinned here: each project (and the global scope) has its own
id@version namespace, and lookups read the project's record before the
global one; a graph never resolves another project's record; a retired
version keeps resolving for graphs already bound to it.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from umrs.knowledge.linkml_import import linkml_to_ontology
from umrs.knowledge.ontology import ontology_for_graph
from umrs.knowledge.ontology_registry import (
    OntologyCoordinateConflict,
    OntologyRegistryStore,
)


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port, name = os.environ.get("POSTGRES_PORT", "5432"), os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


pytestmark = pytest.mark.skipif(
    not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*"
)


@pytest.fixture
def pg_conn() -> Iterator[Any]:
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(_pg_url())
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


_SCHEMA = """\
id: https://example.org/{name}
name: {name}
version: "{version}"
classes:
  Thing:
    abstract: true
  Alloy:
    is_a: Thing
    aliases: [metal mix]
    annotations:
      r4_bucket: entity
  Furnace:
    is_a: Thing
slots:
  heated_in:
    domain: Alloy
    range: Furnace
    transitive: true
"""


class _NoCommit:
    """Delegates to the test connection; commit stays inside the rollback."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> Any:
        return self._conn.cursor()

    def commit(self) -> None:
        pass


@pytest.fixture
def store(pg_conn) -> OntologyRegistryStore:
    return OntologyRegistryStore(_NoCommit(pg_conn))


def _name() -> str:
    return f"pg-onto-{uuid.uuid4().hex[:8]}"


def test_0139_columns_exist(pg_conn) -> None:
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'knowledge_ontologies'"
    )
    columns = {row[0] for row in cur.fetchall()}
    assert {"spec", "source_format", "source_text", "source_sha256"} <= columns


def test_plain_register_keeps_subtype_and_signatures(store: OntologyRegistryStore) -> None:
    """KG-04: 0091 register() dropped subtype_of; the spec column keeps it."""
    name = _name()
    onto = linkml_to_ontology(_SCHEMA.format(name=name, version="1"))
    store.register(onto, owner_id="alice")
    back = store.live_coordinate(name, "1")
    assert back is not None
    assert dict(back.ontology.subtype_of) == {"alloy": "thing", "furnace": "thing"}
    assert back.ontology == onto


def test_full_spec_and_source_round_trip(store: OntologyRegistryStore) -> None:
    name = _name()
    text = _SCHEMA.format(name=name, version="1")
    onto = linkml_to_ontology(text)
    assert onto.subtype_of  # the field 0091 dropped
    store.register(
        onto, owner_id="alice", project_id="proj-a", source_format="linkml", source_text=text
    )
    back = store.live_coordinate(name, "1", "proj-a")
    assert back is not None
    assert back.ontology == onto
    assert dict(back.ontology.subtype_of) == {"alloy": "thing", "furnace": "thing"}
    assert ("heated_in", "alloy", "furnace") in back.ontology.relation_signatures
    assert back.source_text == text
    assert back.source_format == "linkml"
    assert back.source_sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_pre_0139_rows_read_from_the_0091_columns(store, pg_conn) -> None:
    name = _name()
    cur = pg_conn.cursor()
    cur.execute(
        """INSERT INTO knowledge_ontologies
           (record_id, ontology_id, version, entity_types, relation_types,
            transitive_relations, description, owner_id)
           VALUES (%s, %s, '1', '["a"]', '["r"]', '[]', 'old', 'bob')""",
        (f"onto:{name}:1:old", name),
    )
    back = store.live_coordinate(name, "1")
    assert back is not None
    assert back.ontology.entity_types == {"a"}
    assert back.ontology.subtype_of == ()
    assert back.source_format == "" and back.source_text is None


def test_conflict_and_latest_live_version(store: OntologyRegistryStore) -> None:
    name = _name()
    for version in ("2", "10", "9"):
        store.register(
            linkml_to_ontology(_SCHEMA.format(name=name, version=version)),
            owner_id="alice",
            project_id="proj-a",
        )
    with pytest.raises(OntologyCoordinateConflict, match="project 'proj-a'"):
        _register(store, name, "2", "proj-a")
    assert store.latest_live_version(name, "proj-a") == "10"
    assert store.latest_live_version(name, "proj-b") is None  # project-scoped rows
    assert store.latest_live_version(name) is None  # None reads global rows only
    store.retire(name, "10", "proj-a")
    assert store.latest_live_version(name, "proj-a") == "9"
    graph = {"graph_id": "g", "ontology_id": name, "project_id": "proj-a"}
    assert ontology_for_graph(graph, registry=store).version == "9"


def _register(store: OntologyRegistryStore, name: str, version: str, project: Any) -> None:
    onto = linkml_to_ontology(_SCHEMA.format(name=name, version=version))
    store.register(onto, owner_id="alice", project_id=project)


def test_each_project_has_its_own_coordinate_namespace(store) -> None:
    """The 0091 index made id@version global: project B got a 409 for a
    coordinate it could neither see nor retire."""
    name = _name()
    _register(store, name, "1", "proj-a")
    _register(store, name, "1", "proj-b")
    _register(store, name, "1", None)
    with pytest.raises(OntologyCoordinateConflict) as caught:
        _register(store, name, "1", "proj-b")
    assert "proj-a" not in str(caught.value)
    assert store.live_coordinate(name, "1", "proj-a").project_id == "proj-a"
    assert store.live_coordinate(name, "1", "proj-b").project_id == "proj-b"
    assert store.live_coordinate(name, "1", "proj-c").project_id is None  # global
    assert store.live_coordinate(name, "1").project_id is None


def test_the_live_index_is_per_scope(pg_conn) -> None:
    """The database refuses a second live row in one scope, not across scopes."""
    psycopg2 = pytest.importorskip("psycopg2")
    name = _name()
    insert = (
        "INSERT INTO knowledge_ontologies (record_id, ontology_id, version, "
        "entity_types, relation_types, owner_id, project_id) "
        "VALUES (%s, %s, '1', '[]', '[]', 'x', %s)"
    )
    cur = pg_conn.cursor()
    for n, project in enumerate(("proj-a", "proj-b", None)):
        cur.execute(insert, (f"{name}-{n}", name, project))
    cur.execute("SAVEPOINT dup")
    for n, project in enumerate(("proj-a", None)):
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(insert, (f"{name}-dup-{n}", name, project))
        cur.execute("ROLLBACK TO SAVEPOINT dup")


def test_0139_reapplies_cleanly(pg_conn) -> None:
    from umrs import migrations

    sql = (migrations.MIGRATIONS_DIR / "0139_knowledge_ontology_spec.sql").read_text()
    cur = pg_conn.cursor()
    for statement in migrations._split_statements(sql):
        cur.execute(statement)
    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'knowledge_ontologies' "
        "AND indexname LIKE 'knowledge_ontologies_live%%'"
    )
    assert [row[0] for row in cur.fetchall()] == ["knowledge_ontologies_live_scoped_coordinate"]


@pytest.mark.parametrize(
    "graph",
    [
        {"ontology_version": "1", "project_id": "proj-b"},
        {"project_id": "proj-b"},
        {"project_id": ""},
    ],
)
def test_a_graph_never_resolves_another_projects_record(store, graph) -> None:
    """A coordinate only project A holds must not feed B's extraction prompt."""
    name = _name()
    _register(store, name, "1", "proj-a")
    store.retire(name, "1", "proj-a")
    _register(store, name, "1", "proj-a")
    with pytest.raises(ValueError, match="unknown ontology"):
        ontology_for_graph({"graph_id": "g-b", "ontology_id": name, **graph}, registry=store)


def test_a_graph_reads_its_projects_record_before_the_global_one(store) -> None:
    name = _name()
    store.register(
        linkml_to_ontology(_SCHEMA.format(name=name, version="1")),
        owner_id="root",
        project_id=None,
    )
    mine = linkml_to_ontology(_SCHEMA.format(name=name, version="1").replace("Furnace", "Kiln"))
    store.register(mine, owner_id="alice", project_id="proj-a")
    graph = {"graph_id": "g", "ontology_id": name, "ontology_version": "1"}
    assert ontology_for_graph({**graph, "project_id": "proj-a"}, registry=store) == mine
    other = ontology_for_graph({**graph, "project_id": "proj-b"}, registry=store)
    assert "kiln" not in other.entity_types


def test_a_retired_version_keeps_resolving_for_bound_graphs(store) -> None:
    """Retiring switched graphs pinned to it to r4-science with only a job
    note; it now blocks new bindings (live_coordinate) only."""
    name = _name()
    _register(store, name, "1", "proj-a")
    store.retire(name, "1", "proj-a")
    assert store.live_coordinate(name, "1", "proj-a") is None
    graph = {"graph_id": "g", "ontology_id": name, "ontology_version": "1", "project_id": "proj-a"}
    resolved = ontology_for_graph(graph, registry=store)
    assert (resolved.ontology_id, resolved.version) == (name, "1")
    assert "alloy" in resolved.entity_types
