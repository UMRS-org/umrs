"""D-39 pins: Scientific Catalog registry (Phase 1, wave 2026-09-22).

Contracts locked:

- one row per (project_id, name, version); re-create of the same triple is
  idempotent (existing row returned, never stomped).
- version rows carry the member ids (graphs/skills/MCP servers/embedding
  models) plus contributors, licence, provenance.
- PATCH is metadata-only; retire is a tombstone, never a DELETE; unknown
  ids are a typed 404 (x-platform-error-code header set).
- non-admin writes are project-scoped to the caller's verified project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from umrs.api.scientific_catalogs import (
    CatalogNotFound,
    CatalogPatch,
    CatalogSpec,
    ScientificCatalogStore,
    router,
)


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        self._conn._handle(sql, params, self)

    def fetchone(self):
        if not self._result:
            return None
        return self._result.pop(0)

    def fetchall(self):
        out = self._result[:]
        self._result.clear()
        return out


def _tuple_from_dict(d: dict) -> tuple:
    return (
        d["catalog_id"],
        d["project_id"],
        d["name"],
        d["version"],
        d["description"],
        d["ontology"],
        d["member_graph_ids"],
        d["skill_ids"],
        d["mcp_server_ids"],
        d["embedding_model_ids"],
        d["contributors"],
        d["licence"],
        d["provenance"],
        d["change_note"],
        d["redaction_rules"],
        d["owner_id"],
        d["status"],
        d["created_at"],
        d["updated_at"],
    )


class _Conn:
    """psycopg-shaped fake keyed by catalog_id, matching the 0101 columns."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self._triples: dict[tuple, str] = {}

    def cursor(self):
        return _Cursor(self)

    def _handle_insert(self, params, cur: _Cursor) -> None:
        (
            cid,
            pid,
            name,
            version,
            desc,
            onto,
            graphs,
            skills,
            mcps,
            embeds,
            contribs,
            licence,
            prov,
            change_note,
            redact,
            owner,
        ) = params
        triple = (pid, name, version)
        if triple not in self._triples:
            now = datetime.now(timezone.utc)
            self._triples[triple] = cid
            self.rows[cid] = {
                "catalog_id": cid,
                "project_id": pid,
                "name": name,
                "version": version,
                "description": desc,
                "ontology": onto,
                "member_graph_ids": graphs,
                "skill_ids": skills,
                "mcp_server_ids": mcps,
                "embedding_model_ids": embeds,
                "contributors": contribs,
                "licence": licence,
                "provenance": prov,
                "change_note": change_note,
                "redaction_rules": redact,
                "owner_id": owner,
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        cur._result = []

    def _handle_update(self, sql, params, cur: _Cursor) -> None:
        if "SET status = 'retired'" in sql:
            cid = params[0]
            self.rows[cid]["status"] = "retired"
        elif "jsonb_set" in sql:
            import json as _json

            pin_arr, cid = params
            prov = self.rows[cid]["provenance"]
            prov_dict = _json.loads(prov) if isinstance(prov, str) else dict(prov or {})
            prov_dict.setdefault("pins", [])
            prov_dict["pins"] = prov_dict["pins"] + _json.loads(pin_arr)
            self.rows[cid]["provenance"] = _json.dumps(prov_dict)
        else:
            desc, onto, contribs, licence, prov, redact, cid = params
            row = self.rows[cid]
            if desc is not None:
                row["description"] = desc
            if onto is not None:
                row["ontology"] = onto
            if contribs is not None:
                row["contributors"] = contribs
            if licence is not None:
                row["licence"] = licence
            if prov is not None:
                row["provenance"] = prov
            if redact is not None:
                row["redaction_rules"] = redact
            row["updated_at"] = datetime.now(timezone.utc)
        cur._result = []

    def _handle_select(self, sql, params, cur: _Cursor) -> None:
        if "ORDER BY created_at, catalog_id" in sql:
            rows = [
                v
                for v in self.rows.values()
                if v["project_id"] == params[0] and v["name"] == params[1]
            ]
            rows.sort(key=lambda v: v["created_at"])
            cur._result = [_tuple_from_dict(v) for v in rows]
        elif "WHERE catalog_id = %s" in sql:
            row = self.rows.get(params[0])
            cur._result = [_tuple_from_dict(row)] if row else []
        elif "AND name = %s" in sql:
            cid = self._triples.get((params[0], params[1], params[2]))
            row = self.rows.get(cid) if cid else None
            cur._result = [_tuple_from_dict(row)] if row else []
        elif "ORDER BY project_id, name, created_at" in sql:
            want = params[0] if params else None
            rows = [v for v in self.rows.values() if v["status"] != "retired"]
            if want:
                rows = [v for v in rows if v["project_id"] == want]
            cur._result = [_tuple_from_dict(v) for v in rows]
        else:
            cur._result = []

    def _handle(self, sql, params, cur: _Cursor) -> None:
        verb = sql.lstrip().upper()
        if verb.startswith("INSERT"):
            self._handle_insert(params, cur)
        elif verb.startswith("UPDATE"):
            self._handle_update(sql, params, cur)
        elif verb.startswith("SELECT"):
            self._handle_select(sql, params, cur)
        else:
            cur._result = []

    def commit(self):
        pass


def _spec(**over) -> CatalogSpec:
    base = {
        "project_id": "proj-a",
        "name": "mof-defects",
        "version": "0.1.0",
        "description": "MOF defect analysis catalog",
        "ontology": "mof-v2",
        "member_graph_ids": ["kg-1"],
        "skill_ids": ["skl-1"],
        "mcp_server_ids": ["mcp-1"],
        "embedding_model_ids": ["intfloat/e5-base-v2"],
        "contributors": ["alice"],
        "licence": "CC-BY-4.0",
        "provenance": {"source": "lab-ingest"},
    }
    base.update(over)
    return CatalogSpec(**base)


def _client(store) -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    pool = MagicMock()
    pool.getconn.return_value = store._conn
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _headers(user: str, project: str = "proj-a", admin: bool = False) -> dict[str, str]:
    headers = {"X-User-ID": user, "X-Verified-Project-ID": project}
    if admin:
        headers["X-Verified-Is-Admin"] = "true"
    return headers


class TestStoreLifecycle:
    def test_create_returns_members_and_metadata(self) -> None:
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(), owner_id="alice")
        assert row["catalog_id"].startswith("scat-")
        assert row["member_graph_ids"] == ["kg-1"]
        assert row["skill_ids"] == ["skl-1"]
        assert row["mcp_server_ids"] == ["mcp-1"]
        assert row["embedding_model_ids"] == ["intfloat/e5-base-v2"]
        assert row["contributors"] == ["alice"]
        assert row["licence"] == "CC-BY-4.0"
        assert row["provenance"] == {"source": "lab-ingest"}

    def test_create_same_triple_is_idempotent(self) -> None:
        store = ScientificCatalogStore(_Conn())
        first = store.create(spec=_spec(), owner_id="alice")
        again = store.create(spec=_spec(description="stomped?"), owner_id="bob")
        assert again["catalog_id"] == first["catalog_id"]
        assert again["description"] == first["description"]
        assert again["owner_id"] == "alice"

    def test_new_version_is_a_new_row(self) -> None:
        store = ScientificCatalogStore(_Conn())
        v1 = store.create(spec=_spec(), owner_id="alice")
        v2 = store.create(spec=_spec(version="0.2.0"), owner_id="alice")
        assert v2["catalog_id"] != v1["catalog_id"]
        assert len(store.list(project_id="proj-a")) == 2

    def test_patch_is_metadata_only(self) -> None:
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(), owner_id="alice")
        patched = store.update_metadata(
            row["catalog_id"], patch=CatalogPatch(licence="MIT", description="v")
        )
        assert patched["licence"] == "MIT"
        assert patched["member_graph_ids"] == ["kg-1"]  # membership pinned

    def test_get_unknown_raises_not_found(self) -> None:
        store = ScientificCatalogStore(_Conn())
        with pytest.raises(CatalogNotFound):
            store.get("scat-ghost")

    def test_retire_tombstones_never_deletes(self) -> None:
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(), owner_id="alice")
        dead = store.retire(row["catalog_id"])
        assert dead["status"] == "retired"
        store.get(row["catalog_id"])  # row kept
        assert store.list(project_id="proj-a") == []  # hidden from lists

    def test_json_columns_accept_string_payloads(self) -> None:
        # psycopg2 returns JSONB decoded; sqlite-ish fakes may return strings.
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(), owner_id="alice")
        conn_row = store._conn.rows[row["catalog_id"]]
        assert isinstance(json.loads(conn_row["skill_ids"]), list)


class TestRoutes:
    def test_create_binds_caller_owner_and_scope(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        r = client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("alice"),
        )
        assert r.status_code == 201
        assert r.json()["owner_id"] == "alice"

    def test_create_outside_project_scope_forbidden(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        r = client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("alice", project="proj-other"),
        )
        assert r.status_code == 403

    def test_admin_may_write_any_project(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        r = client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("ops", project="proj-other", admin=True),
        )
        assert r.status_code == 201

    def test_list_is_project_scoped_for_non_admin(self) -> None:
        store = ScientificCatalogStore(_Conn())
        store.create(spec=_spec(), owner_id="alice")
        store.create(spec=_spec(project_id="proj-b", name="other"), owner_id="bob")
        client = _client(store)
        r = client.get("/umrs/scientific-catalogs", headers=_headers("alice"))
        assert r.status_code == 200
        names = [c["name"] for c in r.json()["catalogs"]]
        assert names == ["mof-defects"]

    def test_get_unknown_404_typed(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        r = client.get("/umrs/scientific-catalogs/scat-ghost", headers=_headers("alice"))
        assert r.status_code == 404
        assert "x-platform-error-code" in r.headers

    def test_patch_and_retire_flow(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        created = client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("alice"),
        ).json()
        cid = created["catalog_id"]
        patched = client.patch(
            f"/umrs/scientific-catalogs/{cid}",
            json={"licence": "Apache-2.0"},
            headers=_headers("alice"),
        )
        assert patched.status_code == 200
        assert patched.json()["licence"] == "Apache-2.0"
        retired = client.delete(f"/umrs/scientific-catalogs/{cid}", headers=_headers("alice"))
        assert retired.status_code == 200
        assert retired.json()["status"] == "retired"

    def test_store_unavailable_without_pool(self) -> None:
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        r = client.get("/umrs/scientific-catalogs", headers=_headers("alice"))
        assert r.status_code == 503


# ── D-43: version chain + pins ────────────────────────────────────────────


class TestVersionChainAndPins:
    def test_versions_chain_oldest_first_with_change_notes(self) -> None:
        store = ScientificCatalogStore(_Conn())
        store.create(spec=_spec(change_note="first"), owner_id="alice")
        store.create(
            spec=_spec(version="0.2.0", change_note="added defect class"),
            owner_id="alice",
        )
        chain = store.versions(project_id="proj-a", name="mof-defects")
        assert [r["version"] for r in chain] == ["0.1.0", "0.2.0"]
        assert chain[1]["change_note"] == "added defect class"

    def test_create_carries_change_note(self) -> None:
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(change_note="initial"), owner_id="alice")
        assert row["change_note"] == "initial"

    def test_add_pin_appends_never_rewrites(self) -> None:
        store = ScientificCatalogStore(_Conn())
        row = store.create(spec=_spec(), owner_id="alice")
        cid = row["catalog_id"]
        store.add_pin(cid, kind="job", ref="5432", note="defect run")
        row = store.add_pin(cid, kind="finding", ref="f-9")
        pins = row["provenance"]["pins"]
        assert [p["ref"] for p in pins] == ["5432", "f-9"]
        assert row["member_graph_ids"] == ["kg-1"]  # membership untouched

    def test_pin_unknown_raises_not_found(self) -> None:
        store = ScientificCatalogStore(_Conn())
        with pytest.raises(CatalogNotFound):
            store.add_pin("scat-ghost", kind="job", ref="1")

    def test_pin_route_201(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        created = client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("alice"),
        ).json()
        r = client.post(
            f"/umrs/scientific-catalogs/{created['catalog_id']}/pins",
            json={"kind": "job", "ref": "5434", "note": "n"},
            headers=_headers("alice"),
        )
        assert r.status_code == 201
        assert r.json()["provenance"]["pins"][0]["ref"] == "5434"

    def test_versions_route(self) -> None:
        client = _client(ScientificCatalogStore(_Conn()))
        client.post(
            "/umrs/scientific-catalogs",
            json=_spec().model_dump(),
            headers=_headers("alice"),
        )
        r = client.get(
            "/umrs/scientific-catalogs/versions/list",
            params={"project_id": "proj-a", "name": "mof-defects"},
            headers=_headers("alice"),
        )
        assert r.status_code == 200
        assert [v["version"] for v in r.json()["versions"]] == ["0.1.0"]
