"""S-D39a (DEC-D39, decisions-2026-11-24 §5 wave 2): a bundle carries the
full catalog identity and the import records where it came from.

Acceptance: the round trip keeps the version and remaps member refs by
content identity; ``origin_catalog_id`` is the source id; a re-import is
idempotent (one row, same local id). Same on a hosted front and on a
self-hosted stand-alone UMRS: nothing here depends on a deployment.

Layers:

- import wiring with hand-written blocks (always runs);
- an in-memory export -> import round trip (runs once tasks s-d39a-01..03
  have landed);
- the same round trip and the resolver's project scoping on a real Postgres
  scratch schema (``UMRS_TEST_DB_URL`` or the CI ``POSTGRES_*``).
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from umrs.knowledge.bundle_export import KnowledgeExportEngine
from umrs.knowledge.bundle_import import KnowledgeImportEngine, KnowledgeImportResult
from umrs.knowledge.graph_registry import default_graph_id

_SRC_FP = "inst-" + "a" * 32

# ── fakes ──────────────────────────────────────────────────────────────────


class _RecordingCatalogs:
    """Catalog adapter double: records specs, ids by (project, name, version)."""

    def __init__(self, resolver: Any = None) -> None:
        self.specs: list[dict[str, Any]] = []
        self._rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._resolver = resolver
        if resolver is not None:
            self.member_resolver = self._member_resolver

    def _member_resolver(self, project_id: str) -> Any:
        self._resolver.project_id = project_id
        return self._resolver

    def register_from_bundle(self, spec: dict, *, owner_id: str) -> dict:
        self.specs.append({**spec, "_owner": owner_id})
        key = (spec["project_id"], spec["name"], spec["version"])
        row = self._rows.setdefault(key, {**spec, "catalog_id": f"scat-local-{len(self._rows)}"})
        return row


class _Resolver:
    """Target-project lookups (the PG leg uses the real SqlMemberResolver)."""

    project_id = ""

    def default_graph_id(self) -> str:
        return default_graph_id(self.project_id)

    def graph_id_by_name(self, name: str) -> str | None:
        return {"spectra": "g-dst-spectra"}.get(name)

    def skill_id(self, scope: str, name: str, version: str) -> str | None:
        return "sk-dst-fit" if (scope, name, version) == ("project", "fit", "1.0.0") else None

    def mcp_server(self, name: str, version: str) -> tuple[str, str] | None:
        return ("mcp-dst-1", "front-end") if (name, version) == ("microscope", "0.2.0") else None


def _block(**over: Any) -> dict[str, Any]:
    block = {
        "catalog_id": "scat-0123456789abcdef",
        "origin_fingerprint": _SRC_FP,
        "name": "mof-defects",
        "version": "0.2.0",
        "licence": "CC-BY-4.0",
        "contributors": ["alice"],
        "provenance": {"source": "lab"},
        "change_notes": [
            {"version": "0.1.0", "change_note": "first"},
            {"version": "0.2.0", "change_note": "added spectra"},
        ],
        "members": {
            "graphs": [{"default": True, "name": ""}, {"default": False, "name": "spectra"}],
            "skills": [{"scope": "project", "name": "fit", "version": "1.0.0"}],
            "mcp_servers": [{"name": "microscope", "version": "0.2.0"}],
            "embedding_models": ["intfloat/e5-base-v2"],
            "unresolved": 0,
        },
    }
    block.update(over)
    return block


def _register(catalogs: Any, block: dict[str, Any], target: str = "dst-proj") -> Any:
    engine = KnowledgeImportEngine(store=MagicMock(), catalog_store=catalogs)
    result = KnowledgeImportResult()
    engine._register_catalog_row({"catalog": block, "ownership": {}}, target, result)
    return result


# ── import wiring (always runs) ──────────────────────────────────────────


def test_import_records_origin_and_mints_its_own_id() -> None:
    catalogs = _RecordingCatalogs()
    result = _register(catalogs, _block())
    spec = catalogs.specs[0]
    assert _pick(spec, "origin_catalog_id", "origin_fingerprint", "project_id") == {
        "origin_catalog_id": "scat-0123456789abcdef",
        "origin_fingerprint": _SRC_FP,
        "project_id": "dst-proj",
    }
    assert "catalog_id" not in spec  # the bundle's id is a label, never our key
    assert spec["_owner"] == "bundle-import"
    assert _pick(spec, "version", "change_note") == {
        "version": "0.2.0",
        "change_note": "added spectra",
    }
    origin = spec["provenance"]["origin"]
    assert (spec["provenance"]["source"], origin["catalog_id"]) == (
        "lab",
        "scat-0123456789abcdef",
    )
    assert _versions(origin["change_notes"]) == ["0.1.0", "0.2.0"]
    assert (result.catalog_registered, result.catalog_id) == (True, "scat-local-0")
    assert result.to_dict()["catalog_id"] == "scat-local-0"


def _pick(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: row.get(k) for k in keys}


def _versions(chain: list[dict[str, Any]]) -> list[str]:
    return [e["version"] for e in chain]


def test_malformed_origin_is_dropped_not_stored() -> None:
    catalogs = _RecordingCatalogs()
    _register(catalogs, _block(catalog_id="../../x y", origin_fingerprint="inst-zz"))
    spec = catalogs.specs[0]
    assert (spec["origin_catalog_id"], spec["origin_fingerprint"]) == ("", "")
    assert "origin" not in spec["provenance"]


def test_bundle_supplied_provenance_origin_is_replaced() -> None:
    forged = {"source": "lab", "origin": {"catalog_id": "scat-forged"}}
    catalogs = _RecordingCatalogs()
    _register(catalogs, _block(provenance=forged))
    assert catalogs.specs[0]["provenance"]["origin"]["catalog_id"] == "scat-0123456789abcdef"
    catalogs = _RecordingCatalogs()
    _register(catalogs, _block(provenance=forged, catalog_id=None))
    assert "origin" not in catalogs.specs[0]["provenance"]


def test_adapter_without_resolver_registers_no_members() -> None:
    catalogs = _RecordingCatalogs()
    result = _register(catalogs, _block())
    assert "member_graph_ids" not in catalogs.specs[0]
    assert result.catalog_members_unmapped == []


def test_mapping_failure_is_named_and_the_row_still_registers() -> None:
    class _Broken:
        def __init__(self) -> None:
            self.member_resolver = self._boom

        @staticmethod
        def _boom(project_id: str) -> Any:
            raise RuntimeError(f"no resolver for {project_id}")

        def register_from_bundle(self, spec: dict, *, owner_id: str) -> dict:
            self.spec, self.owner = spec, owner_id
            return {"catalog_id": "scat-local"}

    catalogs = _Broken()
    result = _register(catalogs, _block())
    assert result.catalog_members_unmapped == ["members:mapping_failed"]
    assert result.catalog_registered is True
    assert "member_graph_ids" not in catalogs.spec


def test_unmapped_labels_from_the_bundle_are_bounded() -> None:
    graphs = [{"default": False, "name": f"g{i}"} for i in range(300)]
    catalogs = _RecordingCatalogs(_Resolver())
    result = _register(catalogs, _block(members={"graphs": graphs}))
    assert len(result.catalog_members_unmapped) == 257
    assert result.catalog_members_unmapped[-1] == "+44 more"
    origin = catalogs.specs[0]["provenance"]["origin"]
    assert origin["unmapped_members"] == result.catalog_members_unmapped


def test_pre_identity_bundle_still_registers() -> None:
    legacy = {"name": "mof-defects", "version": "0.3.0", "licence": "CC-BY-4.0"}
    catalogs = _RecordingCatalogs()
    result = _register(catalogs, legacy)
    spec = catalogs.specs[0]
    assert (spec["origin_catalog_id"], spec["change_note"]) == ("", "")
    assert result.catalog_registered is True


# ── in-memory round trip (after s-d39a-01..03) ───────────────────────────


class _SourcePool:
    """asyncpg-shaped reads of a source installation, answered by marker."""

    def __init__(self) -> None:
        self.default = default_graph_id("src-proj")

    async def fetchrow(self, query: str, *_args: Any) -> dict[str, Any] | None:
        if "FROM umrs_installation" in query:
            return {"installation_id": "src-installation"}
        if "FROM scientific_catalogs" in query:
            return {
                "catalog_id": "scat-src-v2",
                "name": "mof-defects",
                "version": "0.2.0",
                "description": "d",
                "ontology": "",
                "licence": "CC-BY-4.0",
                "contributors": '["alice"]',
                "provenance": "{}",
                "redaction_rules": "[]",
                "member_graph_ids": [self.default, "g-src-spectra"],
                "skill_ids": ["sk-src-fit"],
                "mcp_server_ids": ["mcp-src-1"],
                "embedding_model_ids": ["intfloat/e5-base-v2"],
            }
        return None

    async def fetch(self, query: str, *_args: Any) -> list[dict[str, Any]]:
        answers = {
            "SELECT version, change_note": [
                {"version": "0.1.0", "change_note": "first"},
                {"version": "0.2.0", "change_note": "added spectra"},
                {"version": "0.3.0", "change_note": "later"},
            ],
            "FROM knowledge_graphs": [{"graph_id": "g-src-spectra", "name": "spectra"}],
            "FROM skills": [
                {"skill_id": "sk-src-fit", "scope": "project", "name": "fit", "version": "1.0.0"}
            ],
            "FROM mcp_servers": [
                {"server_id": "mcp-src-1", "name": "microscope", "version": "0.2.0"}
            ],
        }
        return next((rows for marker, rows in answers.items() if marker in query), [])


async def _export_block(pool: Any, project_id: str, tmp_path: Path) -> dict[str, Any]:
    store = MagicMock()
    store._pool = pool
    engine = KnowledgeExportEngine(store=store, output_dir=str(tmp_path), secret_key="")
    record = await engine._resolve_catalog_record(project_id)
    manifest = engine._build_manifest(project_id, {}, {}, catalog_record=record)
    return manifest


_MAPPED_MEMBERS = {
    "member_graph_ids": [default_graph_id("dst-proj"), "g-dst-spectra"],
    "skill_ids": ["sk-dst-fit"],
    "mcp_server_ids": ["mcp-dst-1"],
    "mcp_trust_classes": {"mcp-dst-1": "front-end"},
}


async def test_in_memory_export_carries_identity_not_local_ids(tmp_path) -> None:
    block = (await _export_block(_SourcePool(), "src-proj", tmp_path))["catalog"]
    assert (block["catalog_id"], block["origin_fingerprint"][:5]) == ("scat-src-v2", "inst-")
    assert _versions(block["change_notes"]) == ["0.1.0", "0.2.0"]
    assert "member_graph_ids" not in block  # local ids never travel
    assert block["members"]["graphs"][0] == {"default": True, "name": ""}


async def test_in_memory_round_trip_keeps_version_and_remaps_members(tmp_path) -> None:
    manifest = await _export_block(_SourcePool(), "src-proj", tmp_path)
    catalogs = _RecordingCatalogs(_Resolver())
    engine = KnowledgeImportEngine(store=MagicMock(), catalog_store=catalogs)
    result = KnowledgeImportResult()
    engine._register_catalog_row(manifest, "dst-proj", result)
    spec = catalogs.specs[0]
    assert _pick(spec, "version", "change_note", "origin_catalog_id") == {
        "version": "0.2.0",
        "change_note": "added spectra",
        "origin_catalog_id": "scat-src-v2",
    }
    assert spec["origin_fingerprint"] == manifest["catalog"]["origin_fingerprint"]
    assert _pick(spec, *_MAPPED_MEMBERS) == _MAPPED_MEMBERS
    assert spec["embedding_model_ids"] == ["intfloat/e5-base-v2"]
    assert result.catalog_members_unmapped == []

    again = KnowledgeImportResult()
    engine._register_catalog_row(manifest, "dst-proj", again)
    assert again.catalog_id == result.catalog_id  # idempotent re-import


# ── real Postgres (scratch schema) ───────────────────────────────────────

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit or not os.environ.get("POSTGRES_HOST"):
        return explicit
    env = os.environ.get
    return (
        f"postgresql://{env('POSTGRES_USER', 'test')}:{env('POSTGRES_PASSWORD', 'test')}"
        f"@{env('POSTGRES_HOST')}:{env('POSTGRES_PORT', '5432')}/{env('POSTGRES_DB', 'umrs_test')}"
    )


def _statement(file: str, pattern: str) -> list[str]:
    text = (_MIGRATIONS / file).read_text()
    return re.findall(pattern, text, flags=re.DOTALL)


def _ddl() -> list[str]:
    """The catalog table (0101..0144) and the member registries it maps onto."""
    ddl = [
        (_MIGRATIONS / n).read_text()
        for n in (
            "0101_scientific_catalogs.sql",
            "0102_scientific_catalogs_changelog.sql",
            "0103_scientific_catalogs_redaction.sql",
            "0108_catalog_mcp_trust_classes.sql",
        )
    ]
    ddl += _statement("0116_resource_access.sql", r"ALTER TABLE scientific_catalogs\s+ADD COLUMN[^;]*;")
    # By suffix: the migration number is set at merge (decisions-2026-11-24 §3).
    ddl.append(next(_MIGRATIONS.glob("*_catalog_origin.sql")).read_text())
    ddl += _statement("0069_knowledge_graphs.sql", r"CREATE TABLE IF NOT EXISTS knowledge_graphs \(.*?\n\);")
    ddl += _statement("0052_skills_registry.sql", r"CREATE TABLE IF NOT EXISTS skills \(.*?\n\);")
    ddl += _statement("0001_mcp_registry.sql", r"CREATE TABLE IF NOT EXISTS mcp_servers \(.*?\n\);")
    ddl += _statement("0105_mcp_placement_trust.sql", r"ALTER TABLE mcp_servers[^;]*;")
    ddl.append((_MIGRATIONS / "0140_mcp_servers_project_name.sql").read_text())
    ddl.append(
        "CREATE TABLE IF NOT EXISTS model_catalog_entries"
        " (model_id TEXT, huggingface_id TEXT, name TEXT)"
    )
    return ddl


@pytest.fixture()
def pg_conn():
    url = _db_url()
    if not url:
        pytest.skip("no Postgres (UMRS_TEST_DB_URL / POSTGRES_HOST)")
    psycopg2 = pytest.importorskip("psycopg2")
    schema = f"sd39a_{uuid.uuid4().hex[:10]}"
    conn = psycopg2.connect(url, options=f"-c search_path={schema}")
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        for sql in _ddl():
            cur.execute(sql)
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


def _seed(conn: Any) -> None:
    rows = [
        "INSERT INTO knowledge_graphs (graph_id, name, project_id) VALUES"
        " ('g-src-spectra', 'spectra', 'src-proj'),"
        " ('g-dst-spectra', 'spectra', 'dst-proj'),"
        " ('g-other-secret', 'secret', 'other-proj')",
        "INSERT INTO skills (skill_id, name, version, scope, owner_id, project_id) VALUES"
        " ('sk-src-fit', 'fit', '1.0.0', 'project', 'a', 'src-proj'),"
        " ('sk-dst-fit', 'fit', '1.0.0', 'project', 'b', 'dst-proj'),"
        " ('sk-other', 'only-other', '1.0.0', 'project', 'c', 'other-proj'),"
        " ('sk-user', 'mine', '1.0.0', 'user', 'd', NULL),"
        " ('sk-sys', 'plot', '2.0.0', 'system', '', NULL)",
        "INSERT INTO mcp_servers (server_id, name, version, url, project_id, trust_class) VALUES"
        " ('mcp-src-1', 'microscope', '0.2.0', 'http://x', 'src-proj', 'local-safe'),"
        " ('mcp-dst-1', 'microscope', '0.2.0', 'http://x', 'dst-proj', 'front-end'),"
        " ('mcp-global', 'microscope', '0.2.0', 'http://x', NULL, 'local-safe'),"
        " ('mcp-other', 'other-only', '1.0', 'http://x', 'other-proj', 'local-safe')",
    ]
    with conn.cursor() as cur:
        for sql in rows:
            cur.execute(sql)
    conn.commit()


class _AsyncPg:
    """asyncpg-shaped pool over a psycopg2 connection ($n -> %s)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def _run(self, query: str, args: tuple[Any, ...]) -> list[Any]:
        from psycopg2.extras import RealDictCursor

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(re.sub(r"\$\d+", "%s", query), args)
            return list(cur.fetchall()) if cur.description else []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return self._run(query, args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        rows = self._run(query, args)
        return rows[0] if rows else None


def _resolver(conn: Any, project_id: str) -> Any:
    from umrs.knowledge.catalog_member_resolver import SqlMemberResolver

    return SqlMemberResolver(conn, project_id)


def test_pg_resolver_maps_graphs_and_skills_inside_the_project_only(pg_conn) -> None:
    _seed(pg_conn)
    dst = _resolver(pg_conn, "dst-proj")
    assert (dst.graph_id_by_name("spectra"), dst.graph_id_by_name("secret")) == (
        "g-dst-spectra",
        None,  # another project's graph
    )
    assert dst.default_graph_id() == default_graph_id("dst-proj")
    skills = [
        dst.skill_id("project", "fit", "1.0.0"),
        dst.skill_id("project", "only-other", "1.0.0"),
        dst.skill_id("user", "mine", "1.0.0"),  # user skills never map
        dst.skill_id("system", "plot", "2.0.0"),
    ]
    assert skills == ["sk-dst-fit", None, None, "sk-sys"]


def test_pg_resolver_prefers_project_mcp_and_never_another_projects(pg_conn) -> None:
    _seed(pg_conn)
    dst = _resolver(pg_conn, "dst-proj")
    assert dst.mcp_server("microscope", "0.2.0") == ("mcp-dst-1", "front-end")
    assert dst.mcp_server("other-only", "1.0") is None
    fresh = _resolver(pg_conn, "new-proj")
    assert fresh.mcp_server("microscope", "0.2.0") == ("mcp-global", "local-safe")


def test_pg_resolver_failure_reads_as_no_match_and_keeps_the_conn(pg_conn) -> None:
    with pg_conn.cursor() as cur:
        cur.execute("DROP TABLE knowledge_graphs")
    pg_conn.commit()
    dst = _resolver(pg_conn, "dst-proj")
    assert dst.graph_id_by_name("spectra") is None
    with pg_conn.cursor() as cur:
        cur.execute("SELECT 1")  # not left in an aborted transaction
        assert cur.fetchone() == (1,)


def test_pg_resolver_needs_a_project() -> None:
    from umrs.knowledge.catalog_member_resolver import SqlMemberResolver

    with pytest.raises(ValueError):
        SqlMemberResolver(object(), " ")


def _seed_source_catalog(store: Any) -> dict[str, Any]:
    """Two versions of ``mof`` in src-proj; returns the 0.2.0 row."""
    from umrs.scientific_catalogs.store import CatalogSpec

    members = {
        "member_graph_ids": [default_graph_id("src-proj"), "g-src-spectra"],
        "skill_ids": ["sk-src-fit"],
        "mcp_server_ids": ["mcp-src-1"],
        "embedding_model_ids": ["intfloat/e5-base-v2"],
    }
    first = CatalogSpec(project_id="src-proj", name="mof", version="0.1.0", change_note="first")
    store.create(spec=first, owner_id="alice")
    second = CatalogSpec(
        project_id="src-proj", name="mof", version="0.2.0", change_note="spectra", **members
    )
    return store.create(spec=second, owner_id="alice")


async def test_pg_round_trip_origin_version_members_idempotent(pg_conn, tmp_path) -> None:
    from umrs.api.bundle_restore import _CatalogRestoreAdapter
    from umrs.scientific_catalogs.store import ScientificCatalogStore

    _seed(pg_conn)
    store = ScientificCatalogStore(pg_conn)
    source = _seed_source_catalog(store)
    manifest = await _export_block(_AsyncPg(pg_conn), "src-proj", tmp_path)
    engine = KnowledgeImportEngine(store=MagicMock(), catalog_store=_CatalogRestoreAdapter(store))
    result = KnowledgeImportResult()
    engine._register_catalog_row(manifest, "dst-proj", result)
    row = store.get(result.catalog_id)
    assert row["catalog_id"] not in ("", source["catalog_id"])
    assert _pick(row, "project_id", "version", "change_note", "origin_catalog_id") == {
        "project_id": "dst-proj",
        "version": "0.2.0",
        "change_note": "spectra",
        "origin_catalog_id": source["catalog_id"],
    }
    assert row["origin_fingerprint"][:5] == "inst-"
    assert _pick(row, *_PG_MAPPED) == _PG_MAPPED
    assert result.catalog_members_unmapped == []

    again = KnowledgeImportResult()
    engine._register_catalog_row(manifest, "dst-proj", again)
    assert again.catalog_id == result.catalog_id
    assert len(store.versions(project_id="dst-proj", name="mof")) == 1


_PG_MAPPED = {
    "member_graph_ids": [default_graph_id("dst-proj"), "g-dst-spectra"],
    "skill_ids": ["sk-dst-fit"],
    "mcp_server_ids": ["mcp-dst-1"],
    "mcp_trust_classes": {"mcp-dst-1": "front-end"},
}
