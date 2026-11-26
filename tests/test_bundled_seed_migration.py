"""ADR-081 KN-2: the checked-in seed migration equals the package data.

``std@1`` and ``r4-core@2`` reach a stand-alone or air-gapped install through
one generated migration, ``<NNNN>_knowledge_terms_seed.sql``, so the first
migration run seeds the registry with no fetch and no admin step. The YAML in
``umrs/knowledge/ontologies/`` is the one source: regenerate the file with

    python -m umrs.knowledge.ontologies > src/umrs/migrations/<NNNN>_knowledge_terms_seed.sql

Renumbering the seed (a later migration number at merge time) keeps its
text; only a YAML change needs a regeneration.
"""

from __future__ import annotations

from umrs.knowledge.ontologies.__main__ import bundled_seed_sql
from umrs.migrations import get_migration_files


def _seed_files() -> list[tuple[str, str]]:
    return [
        (v, p.read_text())
        for v, p in get_migration_files()
        if p.name.endswith("_knowledge_terms_seed.sql")
    ]


def _version_of(suffix: str) -> str:
    return next(v for v, p in get_migration_files() if p.name.endswith(suffix))


def test_exactly_one_seed_migration_exists_after_the_typed_columns():
    seeds = _seed_files()
    assert len(seeds) == 1, (
        "generate it: python -m umrs.knowledge.ontologies > <NNNN>_knowledge_terms_seed.sql"
    )
    assert (
        seeds[0][0]
        > _version_of("_knowledge_typed_columns.sql")
        > _version_of("_knowledge_terms.sql")
    )


def test_seed_migration_equals_the_package_data():
    seeds = _seed_files()
    assert seeds and seeds[0][1] == bundled_seed_sql()


def test_seed_needs_no_network():
    text = bundled_seed_sql()
    assert "COPY" not in text and "dblink" not in text and "http_get" not in text
