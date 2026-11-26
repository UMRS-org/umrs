"""ADR-081 KN-2: the bundled-ontology seed SQL (``ontologies.seed_sql``).

Task kn-s7-05 implements the literal helpers and ``render_seed_sql``. The
hermetic half pins the text; the Postgres half applies a rendered seed to a
migrated database inside a transaction that is rolled back.
"""

from __future__ import annotations

import os
import uuid

import pytest
from _kn_s7_pending import pending
from umrs.knowledge.ontologies.seed_sql import (
    HEADER,
    TERM_COLUMNS,
    render_seed_sql,
    sql_bool,
    sql_jsonb,
    sql_text,
    sql_text_array,
)
from umrs.knowledge.term_types import OntologyDoc, TermSpec

pytestmark = pending(sql_text, sql_bool, sql_text_array, sql_jsonb, render_seed_sql)


# ── literals ─────────────────────────────────────────────────────────────────


def test_sql_text():
    assert sql_text(None) == "NULL"
    assert sql_text("") == "''"
    assert sql_text("O'Brien") == "'O''Brien'"
    assert sql_text("a\\b") == "'a\\b'"
    assert sql_text("名前") == "'名前'"


def test_sql_bool():
    assert (sql_bool(True), sql_bool(False)) == ("true", "false")


def test_sql_text_array():
    assert sql_text_array([]) == "'{}'::text[]"
    assert sql_text_array(()) == "'{}'::text[]"
    assert sql_text_array(("b", "a'x")) == "ARRAY['b', 'a''x']::text[]"


def test_sql_jsonb():
    assert sql_jsonb([]) == "'[]'::jsonb"
    assert sql_jsonb(()) == "'[]'::jsonb"
    assert sql_jsonb({}) == "'{}'::jsonb"
    assert sql_jsonb({"b": 1, "a": "it's"}) == """'{"a": "it''s", "b": 1}'::jsonb"""
    assert sql_jsonb(("x", "y")) == """'["x", "y"]'::jsonb"""
    assert sql_jsonb({"k": "接近"}) == """'{"k": "接近"}'::jsonb"""


# ── render ───────────────────────────────────────────────────────────────────


def _doc(prefix: str = "x", base: str = "http://x/") -> OntologyDoc:
    a = TermSpec(f"{prefix}:A", f"{base}A", "class", "A", (), "entity", source=f"{prefix}@1")
    b = TermSpec(
        f"{prefix}:B", f"{base}B", "class", "B", (f"{prefix}:A",), "file",
        aliases=("bee",), source=f"{prefix}@1",
    )
    r = TermSpec(f"{prefix}:r", f"{base}r", "relation", "r", transitive=True, source=f"{prefix}@1")
    return OntologyDoc(prefix, "1", base, {prefix: base}, (), "demo", (b, a, r))


def _closure(prefix: str = "x") -> dict[tuple[str, str], int]:
    a, b, r = f"{prefix}:A", f"{prefix}:B", f"{prefix}:r"
    return {(b, b): 0, (a, a): 0, (a, b): 1, (r, r): 0}


def _admitted(prefix: str = "x") -> dict[str, dict[str, str]]:
    return {f"{prefix}@1": {f"{prefix}:B": "own", f"{prefix}:A": "own", f"{prefix}:r": "own"}}


def _lines() -> list[str]:
    text = render_seed_sql([_doc()], _closure(), _admitted())
    assert text.startswith(HEADER)
    return text[len(HEADER):].splitlines()


def test_render_ontology_row():
    assert _lines()[0] == (
        "INSERT INTO knowledge_ontologies (record_id, ontology_id, version, entity_types, "
        "relation_types, transitive_relations, description, owner_id, project_id, iri_base, "
        "prefixes, imports, profile) VALUES ('bundled:x@1', 'x', '1', '[\"x:A\", \"x:B\"]'::jsonb, "
        "'[\"x:r\"]'::jsonb, '[\"x:r\"]'::jsonb, 'demo', 'system', NULL, 'http://x/', "
        "'{\"x\": \"http://x/\"}'::jsonb, '[]'::jsonb, '{}'::jsonb) ON CONFLICT DO NOTHING;"
    )


def test_render_term_rows_sorted_with_every_column():
    lines = [line for line in _lines() if line.startswith("INSERT INTO knowledge_terms")]
    assert len(lines) == 3
    assert lines[0] == (
        f"INSERT INTO knowledge_terms ({', '.join(TERM_COLUMNS)}) VALUES ('x:A', 'http://x/A', "
        "'class', 'A', '[]'::jsonb, '', '{}'::text[], 'entity', '{}'::text[], '{}'::text[], "
        "NULL, NULL, false, false, NULL, '{}'::text[], '{}'::jsonb, '{}'::text[], 'minted', "
        "true, 'x@1', 'active', NULL) ON CONFLICT (term_id) DO NOTHING;"
    )
    assert "ARRAY['x:A']::text[], 'file'" in lines[1]
    assert "ARRAY['bee']::text[]" in lines[1]
    assert lines[2].startswith("INSERT INTO knowledge_terms") and "'x:r'" in lines[2]


def test_render_closure_and_admitted_rows_sorted():
    lines = _lines()
    closure = [line for line in lines if "knowledge_term_closure" in line]
    assert closure == [
        "INSERT INTO knowledge_term_closure (ancestor, descendant, depth) VALUES "
        f"({a}, {d}, {n}) ON CONFLICT DO NOTHING;"
        for a, d, n in (("'x:A'", "'x:A'", 0), ("'x:A'", "'x:B'", 1), ("'x:B'", "'x:B'", 0),
                        ("'x:r'", "'x:r'", 0))
    ]
    admitted = [line for line in lines if "knowledge_ontology_terms" in line]
    assert admitted == [
        "INSERT INTO knowledge_ontology_terms (record_id, term_id, via) VALUES "
        f"('bundled:x@1', '{t}', 'own') ON CONFLICT DO NOTHING;"
        for t in ("x:A", "x:B", "x:r")
    ]


def test_render_order_is_ontologies_terms_closure_admitted():
    tables = [line.split()[2] for line in _lines()]
    order = ["knowledge_ontologies", "knowledge_terms", "knowledge_term_closure",
             "knowledge_ontology_terms"]
    assert [t for i, t in enumerate(tables) if i == 0 or tables[i - 1] != t] == order


def test_render_documents_sorted_by_ref():
    docs = [_doc("y", "http://y/"), _doc("x", "http://x/")]
    closure = {**_closure("x"), **_closure("y")}
    text = render_seed_sql(docs, closure, {**_admitted("x"), **_admitted("y")})
    rows = [line for line in text.splitlines() if "INTO knowledge_ontologies" in line]
    assert ["bundled:x@1" in rows[0], "bundled:y@1" in rows[1]] == [True, True]


def test_render_refuses_a_term_defined_twice():
    with pytest.raises(ValueError, match=r"x:[AB]"):
        render_seed_sql([_doc(), _doc()], _closure(), _admitted())


def test_render_is_deterministic():
    assert render_seed_sql([_doc()], _closure(), _admitted()) == render_seed_sql(
        [_doc()], dict(reversed(list(_closure().items()))), _admitted()
    )


# ── real Postgres ────────────────────────────────────────────────────────────


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


def _count(cur, sql: str, *params: object) -> int:
    cur.execute(sql, params)
    return int(cur.fetchone()[0])


@pytest.mark.skipif(not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
def test_rendered_seed_applies_to_a_migrated_database():
    import psycopg2

    prefix = f"t{uuid.uuid4().hex[:8]}"
    base = f"http://{prefix}/"
    sql = render_seed_sql([_doc(prefix, base)], _closure(prefix), _admitted(prefix))
    conn = psycopg2.connect(_pg_url())
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(sql)  # idempotent
            like = f"{prefix}:%"
            assert _count(cur, "SELECT count(*) FROM knowledge_terms WHERE term_id LIKE %s", like) == 3
            assert _count(
                cur, "SELECT count(*) FROM knowledge_term_closure WHERE descendant LIKE %s", like
            ) == 4
            assert _count(
                cur, "SELECT count(*) FROM knowledge_ontology_terms WHERE record_id = %s",
                f"bundled:{prefix}@1",
            ) == 3
            cur.execute("SELECT parents, aliases FROM knowledge_terms WHERE term_id = %s",
                        (f"{prefix}:B",))
            assert cur.fetchone() == ([f"{prefix}:A"], ["bee"])
    finally:
        conn.rollback()
        conn.close()
