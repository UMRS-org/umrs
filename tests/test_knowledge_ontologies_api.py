"""Route tests for the ontology registry API and graph binding checks (ADR-116 D-5).

The ontology registry is an in-memory fake installed in place of
``open_ontology_registry``; the graph registry rides the scripted FakePool
like the other knowledge graph route tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge_ontologies
from umrs.api.knowledge_graphs import router as graphs_router
from umrs.api.knowledge_ontologies import MAX_BODY_BYTES
from umrs.api.knowledge_ontologies import router as ontologies_router
from umrs.knowledge.ontology import Ontology
from umrs.knowledge.ontology_registry import (
    OntologyCoordinateConflict,
    OntologyRecord,
    _pick,
    source_sha256,
)

_ALICE = {"caller_user_id": "alice", "caller_project_id": "proj-a"}
_CAROL = {"caller_user_id": "carol", "caller_project_id": "proj-b"}
_BOB = {"caller_user_id": "bob", "caller_project_id": "proj-a"}
_ADMIN = {"caller_user_id": "root", "caller_project_id": "ops", "is_admin": "true"}
_AGENT = {"x-r4-actor": "agent"}
_READER = {"X-Project-ID": "proj-a"}
_CODE = "x-platform-error-code"

_RECORD_KEYS = {
    "record_id",
    "ontology_id",
    "version",
    "description",
    "project_id",
    "owner_id",
    "status",
    "created_at",
    "source_sha256",
    "source_format",
}

LAB_SCHEMA = """\
id: https://example.org/lab
name: lab
version: "1"
classes:
  Alloy:
    aliases: [metal mix]
  Furnace: {}
slots:
  heated_in:
    domain: Alloy
    range: Furnace
"""


class _FakeRegistry:
    """In-memory OntologyRegistryStore stand-in (same method surface and the
    same per-scope pick: the project's record, then the global one)."""

    def __init__(self) -> None:
        self.rows: list[OntologyRecord] = []

    @property
    def records(self) -> dict[tuple[str, str], OntologyRecord]:
        """Live records by coordinate (tests register one scope each)."""
        return {
            (r.ontology.ontology_id, r.ontology.version): r
            for r in self.rows
            if r.retired_at is None
        }

    def _at(self, ontology_id: str, version: str) -> list[OntologyRecord]:
        key = (ontology_id, version)
        return [r for r in self.rows if (r.ontology.ontology_id, r.ontology.version) == key]

    def register(self, ontology: Ontology, **kw: Any) -> OntologyRecord:
        key = (ontology.ontology_id, ontology.version)
        scope = kw.get("project_id") or None
        if any(r.retired_at is None and r.project_id == scope for r in self._at(*key)):
            raise OntologyCoordinateConflict(f"{key} already registered live in this scope")
        now = datetime.now(timezone.utc)
        record = OntologyRecord(
            record_id=f"onto:{key[0]}:{key[1]}:{len(self.rows)}",
            ontology=ontology,
            owner_id=kw["owner_id"],
            project_id=kw.get("project_id"),
            status="active",
            retired_at=None,
            created_at=now,
            updated_at=now,
            source_format=kw.get("source_format", ""),
            source_sha256=source_sha256(kw.get("source_text")),
            source_text=kw.get("source_text"),
        )
        self.rows.append(record)
        return record

    def live_coordinate(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        return _pick(self._at(ontology_id, version), project_id, live_only=True)

    def latest_live_version(self, ontology_id: str, project_id: Any = None) -> str | None:
        versions = [
            r.ontology.version
            for r in self.rows
            if r.ontology.ontology_id == ontology_id
            and r.retired_at is None
            and r.project_id in (None, project_id or None)
        ]
        return max(versions, key=int) if versions else None

    def list(self, **_: Any) -> list[OntologyRecord]:
        return list(self.records.values())

    def retire(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        record = self.live_coordinate(ontology_id, version, project_id)
        if record is None:
            return None
        retired = replace(record, status="retired", retired_at=datetime.now(timezone.utc))
        self.rows[self.rows.index(record)] = retired
        return retired


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> _FakeRegistry:
    fake = _FakeRegistry()

    @contextmanager
    def _open(_request: Any) -> Iterator[_FakeRegistry]:
        yield fake

    monkeypatch.setattr(knowledge_ontologies, "open_ontology_registry", _open)
    return fake


def _client(script: list[Any] | None = None) -> tuple[TestClient, FakePool]:
    app = FastAPI()
    app.include_router(graphs_router)
    app.include_router(ontologies_router)
    pool = FakePool(script or [])
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _register(client: TestClient, params: dict, **body: Any) -> Any:
    payload = {"source": LAB_SCHEMA, "format": "linkml", "scope": "project", **body}
    return client.post("/umrs/knowledge/ontologies", json=payload, params=params)


class TestRegister:
    def test_linkml_registration_201(self, registry: _FakeRegistry) -> None:
        client, _ = _client()
        resp = _register(client, _ALICE)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["ontology"]["ontology_id"] == "lab"
        assert body["ontology"]["relation_signatures"] == [
            {"relation": "heated_in", "domain": "alloy", "range": "furnace"}
        ]
        assert set(body["record"]) == _RECORD_KEYS
        assert body["record"]["project_id"] == "proj-a"
        assert body["record"]["owner_id"] == "alice"
        assert body["record"]["source_sha256"] == source_sha256(LAB_SCHEMA)
        assert registry.records[("lab", "1")].source_text == LAB_SCHEMA

    def test_json_registration(self) -> None:
        client, _ = _client()
        source = json.dumps(
            {
                "ontology_id": "lab-json",
                "version": "1",
                "entity_types": ["alloy"],
                "relation_types": [],
            }
        )
        resp = _register(client, _ALICE, source=source, format="json")
        assert resp.status_code == 201, resp.text
        assert resp.json()["ontology"]["source_format"] == "json"

    def test_json_names_must_be_snake_case(self) -> None:
        client, _ = _client()
        source = json.dumps(
            {
                "ontology_id": "lab-json",
                "version": "1",
                "entity_types": ["Alloy"],
                "relation_types": [],
            }
        )
        resp = _register(client, _ALICE, source=source, format="json")
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"

    def test_agent_is_refused(self, registry: _FakeRegistry) -> None:
        client, _ = _client()
        resp = client.post(
            "/umrs/knowledge/ontologies",
            json={"source": LAB_SCHEMA},
            params=_ALICE,
            headers=_AGENT,
        )
        assert resp.status_code == 403
        assert resp.headers[_CODE] == "ontology.forbidden"
        assert registry.records == {}

    def test_conflict_409(self) -> None:
        client, _ = _client()
        assert _register(client, _ALICE).status_code == 201
        resp = _register(client, _BOB)
        assert resp.status_code == 409
        assert resp.headers[_CODE] == "ontology.conflict"

    def test_builtin_id_is_a_conflict(self) -> None:
        client, _ = _client()
        resp = _register(client, _ALICE, source=LAB_SCHEMA.replace("name: lab", "name: r4-science"))
        assert resp.status_code == 409

    def test_invalid_source_400(self) -> None:
        client, _ = _client()
        resp = _register(client, _ALICE, source=LAB_SCHEMA.replace('version: "1"\n', ""))
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"
        assert "version" in resp.json()["detail"]

    def test_body_cap(self) -> None:
        client, _ = _client()
        resp = _register(client, _ALICE, source="#" * (MAX_BODY_BYTES + 1))
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"

    def test_a_source_just_under_the_source_cap_is_not_refused_as_a_body(self) -> None:
        """JSON escaping grew a 255 KiB file past a body cap equal to the
        source cap: 'request body is larger' for a source within its limit."""
        from umrs.knowledge.linkml_import import MAX_SOURCE_BYTES

        client, _ = _client()
        padding = '# "quoted" comment line\n' * ((MAX_SOURCE_BYTES - 1024) // 25)
        source = padding + LAB_SCHEMA
        assert len(source.encode()) < MAX_SOURCE_BYTES
        assert len(json.dumps({"source": source}).encode()) > MAX_SOURCE_BYTES
        resp = _register(client, _ALICE, source=source)
        assert resp.status_code == 201, resp.text[:300]

    def test_yaml_aliases_are_refused_quickly(self) -> None:
        """A 500-byte alias tree expanded to ~16 GB once a value was str()-ed."""
        import time

        lines = ['a0: &a0 ["xxxxxxxxxx"]']
        lines += [f"a{i}: &a{i} [{', '.join([f'*a{i - 1}'] * 10)}]" for i in range(1, 9)]
        source = "\n".join([*lines, "name: *a8", "id: x", 'version: "1"', "classes: {A: {}}"])
        client, _ = _client()
        started = time.monotonic()
        resp = _register(client, _ALICE, source=source)
        assert time.monotonic() - started < 2.0
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"
        assert "aliases" in resp.json()["detail"]
        assert len(resp.text) < 2000

    @pytest.mark.parametrize("fmt", ["linkml", "json"])
    def test_deep_nesting_is_400_not_500(self, fmt: str) -> None:
        client, _ = _client()
        nested = "[" * 5000 + "]" * 5000
        source = f"description: {nested}\n" if fmt == "linkml" else nested
        resp = _register(client, _ALICE, source=source, format=fmt)
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"

    def test_json_registration_meets_the_term_cap(self) -> None:
        from umrs.knowledge.linkml_import import MAX_TERMS

        client, _ = _client()
        classes = [f"c{i}" for i in range(MAX_TERMS + 1)]
        source = json.dumps(
            {"ontology_id": "big", "version": "1", "entity_types": classes, "relation_types": []}
        )
        resp = _register(client, _ALICE, source=source, format="json")
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"]

    def test_each_project_registers_show_and_retires_its_own_version(
        self, registry: _FakeRegistry
    ) -> None:
        """Per-project namespaces: project B's lab@1 is its own record, which
        B can show and retire, while A's stays live (the reads and the retire
        used to look the coordinate up with no project)."""
        client, _ = _client()
        assert [_register(client, who).status_code for who in (_ALICE, _CAROL)] == [201, 201]
        shown = client.get("/umrs/knowledge/ontologies/lab/1", headers={"X-Project-ID": "proj-b"})
        assert (shown.status_code, shown.json()["record"]["owner_id"]) == (200, "carol"), shown.text
        resp = client.post("/umrs/knowledge/ontologies/lab/1/retire", params=_CAROL)
        assert resp.status_code == 200, resp.text
        assert _live_scopes(registry) == [("alice", "proj-a")]

    def test_global_needs_admin(self) -> None:
        client, _ = _client()
        resp = _register(client, _ALICE, scope="global")
        assert resp.status_code == 403
        assert resp.headers[_CODE] == "ontology.forbidden"
        resp = _register(client, _ADMIN, scope="global")
        assert resp.status_code == 201, resp.text
        assert resp.json()["record"]["project_id"] is None


def _live_scopes(registry: _FakeRegistry) -> list[tuple[str, str | None]]:
    return [(r.owner_id, r.project_id) for r in registry.rows if r.retired_at is None]


class TestRead:
    def test_list_includes_builtins_and_visible_records(self, registry: _FakeRegistry) -> None:
        client, _ = _client()
        _register(client, _ALICE)
        registry.register(
            replace(registry.records[("lab", "1")].ontology, ontology_id="other"),
            owner_id="carol",
            project_id="proj-z",
        )
        resp = client.get("/umrs/knowledge/ontologies", headers=_READER)
        assert resp.status_code == 200, resp.text
        rows = {row["ontology_id"]: row for row in resp.json()["ontologies"]}
        assert set(rows) == {"r4-default", "r4-science", "lab"}
        assert rows["r4-science"]["builtin"] is True
        assert rows["r4-science"]["source_format"] == "builtin"
        assert rows["lab"]["builtin"] is False
        assert (rows["lab"]["n_classes"], rows["lab"]["n_relations"]) == (2, 1)
        assert set(rows["lab"]) >= _RECORD_KEYS

    def test_show_builtin_with_source(self) -> None:
        client, _ = _client()
        resp = client.get("/umrs/knowledge/ontologies/r4-science/1", headers=_READER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["record"] is None
        assert body["source_format"] == "builtin"
        assert "name: r4-science" in body["source_text"]

    def test_show_registered_with_source(self) -> None:
        client, _ = _client()
        _register(client, _ALICE)
        body = client.get("/umrs/knowledge/ontologies/lab/1", headers=_READER).json()
        assert body["source_text"] == LAB_SCHEMA
        assert body["source_format"] == "linkml"
        assert body["record"]["owner_id"] == "alice"

    def test_show_unknown_404(self) -> None:
        client, _ = _client()
        for path in ("ghost/1", "r4-science/7"):
            resp = client.get(f"/umrs/knowledge/ontologies/{path}", headers=_READER)
            assert resp.status_code == 404
            assert resp.headers[_CODE] == "ontology.not_found"


class TestRetire:
    def _post(self, client: TestClient, path: str, params: dict, headers=None) -> Any:
        return client.post(
            f"/umrs/knowledge/ontologies/{path}/retire", params=params, headers=headers or {}
        )

    def test_owner_retires(self, registry: _FakeRegistry) -> None:
        client, _ = _client()
        _register(client, _ALICE)
        resp = self._post(client, "lab/1", _ALICE)
        assert resp.status_code == 200
        assert resp.json() == {"retired": True}
        assert registry.records == {}

    def test_a_retired_version_cannot_be_bound_again(self) -> None:
        """Retiring blocks new bindings; bound graphs keep resolving it (see
        test_ontology_registry*.py)."""
        client, _ = _client()
        _register(client, _ALICE)
        assert self._post(client, "lab/1", _ALICE).status_code == 200
        resp = _create(client, ontology_id="lab", ontology_version="1")
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.not_found"

    def test_non_owner_agent_builtin_unknown(self) -> None:
        client, _ = _client()
        _register(client, _ALICE)
        assert self._post(client, "lab/1", _BOB).status_code == 403
        assert self._post(client, "lab/1", _ALICE, _AGENT).status_code == 403
        assert self._post(client, "r4-science/1", _ADMIN).status_code == 403
        assert self._post(client, "ghost/1", _ALICE).status_code == 404
        # An admin retires in the record's project (each project has its own lab@1).
        assert self._post(client, "lab/1", _ADMIN).status_code == 404
        in_proj_a = {**_ADMIN, "caller_project_id": "proj-a"}
        assert self._post(client, "lab/1", in_proj_a).status_code == 200


# ── graph create / patch validate the ontology coordinate ────────────────


def _graph_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "graph_id": "g-abc",
        "name": "materials",
        "description": "",
        "owner_id": "alice",
        "project_id": "proj-a",
        "kind": "research",
        "status": "active",
        "ontology_id": None,
        "ontology_version": None,
        "provenance": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _create(client: TestClient, **body: Any) -> Any:
    payload = {"name": "materials", "project_id": "proj-a", "kind": "research", **body}
    return client.post("/umrs/knowledge/graphs", json=payload, params=_ALICE)


class TestGraphBinding:
    def test_unknown_ontology_is_400_and_nothing_written(self) -> None:
        client, pool = _client([None, _graph_row()])
        resp = _create(client, ontology_id="ghost")
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.not_found"
        assert pool.record == []

    def test_id_without_version_pins_the_latest_live(self) -> None:
        client, pool = _client([None, _graph_row(ontology_id="lab", ontology_version="2")])
        _register(client, _ALICE)
        _register(client, _ALICE, source=LAB_SCHEMA.replace('version: "1"', 'version: "2"'))
        resp = _create(client, ontology_id="lab")
        assert resp.status_code == 201, resp.text
        insert_params = pool.record[0][1]
        assert insert_params[6:8] == ("lab", "2")

    def test_builtin_pins_its_version(self) -> None:
        client, pool = _client([None, _graph_row()])
        assert _create(client, ontology_id="r4-science").status_code == 201
        assert pool.record[0][1][6:8] == ("r4-science", "1")

    def test_wrong_builtin_version_is_400(self) -> None:
        client, _ = _client()
        resp = _create(client, ontology_id="r4-science", ontology_version="5")
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.not_found"

    def test_version_without_id_is_invalid(self) -> None:
        client, _ = _client()
        resp = _create(client, ontology_version="1")
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.invalid"

    def test_other_project_ontology_is_not_bindable(self, registry: _FakeRegistry) -> None:
        client, _ = _client()
        registry.register(
            Ontology("theirs", "1", frozenset({"a"}), frozenset()),
            owner_id="carol",
            project_id="proj-z",
        )
        resp = _create(client, ontology_id="theirs", ontology_version="1")
        assert resp.status_code == 400

    def test_patch_validates_before_any_write(self) -> None:
        client, pool = _client([_graph_row()])
        resp = client.patch(
            "/umrs/knowledge/graphs/g-abc",
            json={"ontology_id": "ghost", "status": "archived"},
            params=_ALICE,
        )
        assert resp.status_code == 400
        assert resp.headers[_CODE] == "ontology.not_found"
        assert len(pool.record) == 1  # the existence read only

    def test_patch_pins_latest(self) -> None:
        client, pool = _client([_graph_row(), None, _graph_row(ontology_id="lab")])
        _register(client, _ALICE)
        resp = client.patch(
            "/umrs/knowledge/graphs/g-abc", json={"ontology_id": "lab"}, params=_ALICE
        )
        assert resp.status_code == 200, resp.text
        sql, params = pool.record[1]
        assert "ontology_id = %s" in sql and "ontology_version = %s" in sql
        assert params[:2] == ("lab", "1")
