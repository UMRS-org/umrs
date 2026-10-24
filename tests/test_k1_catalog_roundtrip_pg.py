"""K1.1(f) / Ops Watch #545: bundle export and import use ``catalog_objects``.

Before 2026-10-24 the export read, and the import wrote, a
``knowledge_catalog`` table no migration creates: every export logged
``relation "knowledge_catalog" does not exist`` and shipped an empty
``catalog.jsonl``, and an import carrying catalog rows failed. Runs on the
CI-shape Postgres (migrated ``umrs_test``); skipped without one.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.knowledge.bundle_export import KnowledgeExportEngine
from umrs.knowledge.bundle_import import KnowledgeImportEngine, KnowledgeImportResult
from umrs.knowledge.store import KnowledgeStore


def _db_url() -> str | None:
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


async def _codecs(conn: Any) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


_SEED = """INSERT INTO catalog_objects
    (project_id, s3_key, state, summary, deterministic_summary, tags,
     filename_signals, detected_modality, owner, acl_scope, visibility,
     extracted_text_ref)
VALUES ($1, $2, 'ready', $3, 'det', $4, '[]'::jsonb, 'text', 'alice',
        'scope-a', $5, 's3://src/extracted')"""


async def _roundtrip(tmp_path: Path) -> dict[str, Any]:
    import asyncpg

    src, dst = f"k1cat-src-{uuid.uuid4().hex[:8]}", f"k1cat-dst-{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(_db_url(), min_size=1, max_size=2, init=_codecs)
    try:
        async with pool.acquire() as conn:
            await conn.execute(_SEED, src, "data/run1.csv", "run one", ["yield"], "project")
            await conn.execute(_SEED, src, "data/sys.csv", "system row", [], "system")
            await conn.execute(_SEED, src, "data/hidden.csv", "ambient row", [], "ambient")
        store = KnowledgeStore(pool)
        exported = await KnowledgeExportEngine(store=store, output_dir=tmp_path)._list_catalog(
            src
        )
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "catalog.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in exported), encoding="utf-8"
        )
        result = KnowledgeImportResult()
        importer = KnowledgeImportEngine(store=store)
        count = await importer._restore_catalog(bundle, dst, result)
        again = await importer._restore_catalog(bundle, dst, KnowledgeImportResult())
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT s3_key, summary, tags, visibility, owner, acl_scope, "
                "extracted_text_ref, state FROM catalog_objects WHERE project_id = $1 "
                "ORDER BY s3_key",
                dst,
            )
            await conn.execute(
                "DELETE FROM catalog_objects WHERE project_id = ANY($1::text[])", [src, dst]
            )
        return {"exported": exported, "count": count, "again": again, "rows": rows}
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = asyncio.run(_roundtrip(tmp_path_factory.mktemp("catalog")))
    out["by_key"] = {row["s3_key"]: row for row in out["exported"]}
    out["imported"] = {r["s3_key"]: r for r in out["rows"]}
    return out


def test_export_reads_catalog_objects_without_ambient_rows(roundtrip) -> None:
    # Ambient rows are read-gated: they stay home.
    assert set(roundtrip["by_key"]) == {"data/run1.csv", "data/sys.csv"}
    run1 = roundtrip["by_key"]["data/run1.csv"]
    assert (run1["summary"], run1["tags"]) == ("run one", ["yield"])


def test_export_leaves_owner_scope_and_source_refs_home(roundtrip) -> None:
    run1 = roundtrip["by_key"]["data/run1.csv"]
    assert not {"owner", "acl_scope", "extracted_text_ref"} & set(run1)


def test_import_writes_the_rows_into_the_importing_project(roundtrip) -> None:
    assert roundtrip["count"] == 2
    row = roundtrip["imported"]["data/run1.csv"]
    assert (row["summary"], row["tags"], row["state"]) == ("run one", ["yield"], "ready")


def test_import_never_widens_visibility(roundtrip) -> None:
    # A system row lands as a project row; owner and source refs stay empty.
    rows = list(roundtrip["imported"].values())
    assert {r["visibility"] for r in rows} == {"project"}
    assert {(r["owner"], r["extracted_text_ref"]) for r in rows} == {("", None)}


def test_reimport_converges(roundtrip) -> None:
    assert len(roundtrip["rows"]) == 2


def test_import_accepts_the_older_row_shape(tmp_path: Path) -> None:
    async def _run() -> list[Any]:
        import asyncpg

        dst = f"k1cat-old-{uuid.uuid4().hex[:8]}"
        pool = await asyncpg.create_pool(_db_url(), min_size=1, max_size=2, init=_codecs)
        try:
            bundle = tmp_path / "old"
            bundle.mkdir()
            (bundle / "catalog.jsonl").write_text(
                json.dumps({"id": "1", "name": "legacy.csv", "metadata": {}}) + "\n",
                encoding="utf-8",
            )
            importer = KnowledgeImportEngine(store=KnowledgeStore(pool))
            await importer._restore_catalog(bundle, dst, KnowledgeImportResult())
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT s3_key, state FROM catalog_objects WHERE project_id = $1", dst
                )
                await conn.execute("DELETE FROM catalog_objects WHERE project_id = $1", dst)
            return list(rows)
        finally:
            await pool.close()

    rows = asyncio.run(_run())
    assert [(r["s3_key"], r["state"]) for r in rows] == [("legacy.csv", "pending")]
