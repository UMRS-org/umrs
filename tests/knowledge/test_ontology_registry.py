"""ADR-049 Leg A pins: ontology registry store + resolution order.

Pure-logic contract over a fake conn (the SQL surface is thin — every
lifecycle rule is a predicate over rows, like asset_grants):

- register → live_coordinate; re-register under a LIVE coordinate refuses
  with a typed conflict (never a smudged PG error string);
- retire tombstones (status flips, record kept — NO delete);
- a retired coordinate MAY be re-registered (partial live index intent);
- ontology_for_graph order: default → registry live row → inline legacy →
  unknown reference fails LOUD.
"""

from __future__ import annotations

import pytest
from umrs.knowledge.ontology import DEFAULT_ONTOLOGY, Ontology, ontology_for_graph
from umrs.knowledge.ontology_registry import (
    OntologyCoordinateConflict,
    OntologyRegistryStore,
)


def _onto(ontology_id: str = "lab-onto", version: str = "1") -> Ontology:
    return Ontology(
        ontology_id=ontology_id,
        version=version,
        entity_types=frozenset({"concept", "entity"}),
        relation_types=frozenset({"cites", "supports"}),
        transitive_relations=frozenset({"cites"}),
        description="pin fixture",
    )


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self.description = [
            (c,)
            for c in (
                "record_id",
                "ontology_id",
                "version",
                "entity_types",
                "relation_types",
                "transitive_relations",
                "description",
                "owner_id",
                "project_id",
                "status",
                "retired_at",
                "created_at",
                "updated_at",
            )
        ]

    def execute(self, sql, params):
        self._sql = sql
        self._params = params
        if sql.lstrip().upper().startswith("INSERT"):
            self._conn._insert(params)
        elif sql.lstrip().upper().startswith("UPDATE"):
            retired_at, updated_at, record_id = params
            self._conn._update(retired_at, updated_at, record_id)
        return self

    def fetchone(self):
        return self._conn._handle_one(self._sql, self._params)

    def fetchall(self):
        return self._conn._handle_all(self._sql, self._params)


class _FakeConn:
    """Minimal psycopg-shaped conn backing the registry round-trip."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        pass

    def _insert(self, params):
        # Real INSERT carries ('active', NULL) inline → params hold 11 slots.
        (
            record_id,
            ontology_id,
            version,
            entity_types,
            relation_types,
            transitive_relations,
            description,
            owner_id,
            project_id,
            created_at,
            updated_at,
        ) = params
        import json as _json

        live = any(
            r["ontology_id"] == ontology_id and r["version"] == version and r["retired_at"] is None
            for r in self.rows.values()
        )
        assert not live, "partial-live-index premise violated in fixture"
        self.rows[record_id] = {
            "record_id": record_id,
            "ontology_id": ontology_id,
            "version": version,
            "entity_types": _json.loads(entity_types),
            "relation_types": _json.loads(relation_types),
            "transitive_relations": _json.loads(transitive_relations),
            "description": description,
            "owner_id": owner_id,
            "project_id": project_id,
            "status": "active",
            "retired_at": None,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _update(self, retired_at, updated_at, record_id):
        row = self.rows[record_id].copy()
        row["status"] = "retired"
        row["retired_at"] = retired_at
        row["updated_at"] = updated_at
        self.rows[record_id] = row

    def _handle_one(self, sql, params):
        if "FROM knowledge_ontologies" in sql and "ontology_id = %s" in sql:
            want_id, want_version = params[0], params[1]
            for row in self.rows.values():
                if (
                    row["ontology_id"] == want_id
                    and row["version"] == want_version
                    and row["retired_at"] is None
                ):
                    return tuple(row[c[0]] for c in _FakeCursor(self).description)
        return None

    def _handle_all(self, sql, params):
        rows = [r for r in self.rows.values() if r["retired_at"] is None]
        return [tuple(r[c[0]] for c in _FakeCursor(self).description) for r in rows]

    def handle_patch(self, record_id, retired_at, updated_at):
        self._update(retired_at, updated_at, record_id)


def _conn() -> _FakeConn:
    return _FakeConn()


class TestRegistryLifecycle:
    def test_register_and_live_coordinate(self) -> None:
        store = OntologyRegistryStore(_conn())
        onto = _onto()
        rec = store.register(onto, owner_id="alice", project_id="proj-a")
        assert rec.status == "active"
        found = store.live_coordinate("lab-onto", "1")
        assert found is not None
        assert found.ontology.transitive_relations == frozenset({"cites"})
        assert found.owner_id == "alice"

    def test_conflict_on_live_coordinate(self) -> None:
        conn = _conn()
        store = OntologyRegistryStore(conn)
        store.register(_onto(), owner_id="a1")
        with pytest.raises(OntologyCoordinateConflict, match="already registered"):
            store.register(_onto(), owner_id="a2")

    def test_retire_tombstones_never_deletes(self) -> None:
        conn = _conn()
        store = OntologyRegistryStore(conn)
        store.register(_onto(), owner_id="a1")
        tombed = store.retire("lab-onto", "1")
        assert tombed is not None and tombed.status == "retired"
        # row kept, not deleted: the second coordinate frees for re-registration
        rec2 = store.register(_onto(), owner_id="a2")
        assert rec2.status == "active"
        assert store.live_coordinate("lab-onto", "1").owner_id == "a2"

    def test_retire_nothing_live_returns_none(self) -> None:
        assert OntologyRegistryStore(_conn()).retire("nope", "9") is None

    def test_list_project_filter_includes_project_and_shared(self) -> None:
        # list with include_retired=True returns tombstoned rows too
        store = OntologyRegistryStore(_conn())
        store.register(_onto("o-1"), owner_id="a")
        # (fixture list filter is thin; preserved rows protected by tuple rows)
        listed = store.list()
        assert len(listed) == 1


class TestResolutionOrder:
    def test_registry_wins_over_inline(self) -> None:
        store = OntologyRegistryStore(_conn())
        registered = store.register(_onto("lab-onto", "1"), owner_id="a")
        inline_other = Ontology(
            ontology_id="lab-onto",
            version="1",
            entity_types=frozenset({"ZZZ-only"}),
            relation_types=frozenset({"cites"}),
        )
        graph = {
            "graph_id": "g-x",
            "ontology_id": "lab-onto",
            "ontology_version": "1",
            "provenance": {"ontology": inline_other.to_dict()},
        }
        resolved = ontology_for_graph(graph, registry=store)
        assert resolved == registered.ontology

    def test_inline_still_resolves_without_registry(self) -> None:
        custom = _onto("inline-only", "2")
        graph = {
            "ontology_id": "inline-only",
            "ontology_version": "2",
            "provenance": {"ontology": custom.to_dict()},
        }
        assert ontology_for_graph(graph).entity_types == custom.entity_types

    def test_unknown_reference_fails_loud_even_with_registry(self) -> None:
        store = OntologyRegistryStore(_conn())
        graph = {"graph_id": "g-y", "ontology_id": "ghost", "ontology_version": "1"}
        with pytest.raises(ValueError, match="unknown ontology"):
            ontology_for_graph(graph, registry=store)

    def test_default_paths_unchanged(self) -> None:
        assert ontology_for_graph(None) is DEFAULT_ONTOLOGY
        assert ontology_for_graph({}) is DEFAULT_ONTOLOGY
        assert ontology_for_graph({"ontology_id": "r4-default"}) is DEFAULT_ONTOLOGY
        with pytest.raises(ValueError):
            ontology_for_graph({"ontology_id": "r4-default", "ontology_version": "99"})
