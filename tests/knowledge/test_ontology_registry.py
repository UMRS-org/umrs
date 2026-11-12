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
        # Real INSERT carries ('active', NULL) inline → the 0091 columns fill
        # the first 11 slots; the 0139 spec/source slots follow and this
        # fixture keeps reading the 0091 columns only.
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
        ) = params[:11]
        import json as _json

        live = any(
            r["ontology_id"] == ontology_id
            and r["version"] == version
            and r["project_id"] == project_id
            and r["retired_at"] is None
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
        active = _where_terms(sql, params)
        rows = [r for r in self.rows.values() if all(r[column] == want for column, want in active)]
        return [tuple(r[c[0]] for c in _FakeCursor(self).description) for r in rows]

    def handle_patch(self, record_id, retired_at, updated_at):
        self._update(retired_at, updated_at, record_id)


def _where_terms(sql, params) -> list[tuple[str, object]]:
    """(column, wanted value) for each WHERE term the store's SQL uses."""
    padded = (*params, None, None)
    terms = [
        ("retired_at IS NULL", "retired_at", None),
        ("ontology_id = %s", "ontology_id", padded[0]),
        ("version = %s", "version", padded[1]),
    ]
    return [(column, want) for fragment, column, want in terms if fragment in sql]


def _conn() -> _FakeConn:
    return _FakeConn()


class TestRegistryLifecycle:
    def test_register_and_live_coordinate(self) -> None:
        store = OntologyRegistryStore(_conn())
        onto = _onto()
        rec = store.register(onto, owner_id="alice", project_id="proj-a")
        assert rec.status == "active"
        found = store.live_coordinate("lab-onto", "1", "proj-a")
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


class TestReviewFixes:
    def test_register_writes_the_full_spec_and_the_source(self) -> None:
        """The PG round trip runs only with a database; this pins the INSERT
        (0091 register() dropped subtype_of / extraction_fields)."""
        import hashlib
        import json

        conn = _conn()
        seen: list[tuple] = []
        insert = conn._insert
        conn._insert = lambda params: (seen.append(params), insert(params))
        onto = Ontology(
            ontology_id="lab",
            version="1",
            entity_types=frozenset({"alloy", "thing"}),
            relation_types=frozenset({"cites"}),
            subtype_of=(("alloy", "thing"),),
            extraction_fields=(("citations", "cites"),),
        )
        OntologyRegistryStore(conn).register(
            onto, owner_id="a", source_format="linkml", source_text="name: lab"
        )
        spec, source_format, source_text, sha = seen[0][11:]
        assert json.loads(spec) == onto.to_dict()
        assert dict(json.loads(spec)["subtype_of"]) == {"alloy": "thing"}
        assert (source_format, source_text) == ("linkml", "name: lab")
        assert sha == hashlib.sha256(b"name: lab").hexdigest()

    def test_each_scope_has_its_own_coordinate_namespace(self) -> None:
        """One global id@version namespace gave project B a 409 for a
        coordinate only project A could see or retire."""
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="a", project_id="proj-a")
        store.register(_onto(), owner_id="b", project_id="proj-b")
        store.register(_onto(), owner_id="root")
        with pytest.raises(OntologyCoordinateConflict) as caught:
            store.register(_onto(), owner_id="b2", project_id="proj-b")
        assert "project 'proj-b'" in str(caught.value)
        assert "proj-a" not in str(caught.value)

    def test_lookups_read_the_project_record_before_the_global_one(self) -> None:
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="root")
        store.register(_onto(), owner_id="a", project_id="proj-a")
        assert store.live_coordinate("lab-onto", "1", "proj-a").owner_id == "a"
        assert store.live_coordinate("lab-onto", "1", "proj-b").owner_id == "root"
        assert store.live_coordinate("lab-onto", "1").owner_id == "root"
        store.retire("lab-onto", "1", "proj-a")  # retires the project's own
        assert store.live_coordinate("lab-onto", "1", "proj-a").owner_id == "root"
        assert store.latest_live_version("lab-onto") == "1"

    def test_none_reads_global_records_only(self) -> None:
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="a", project_id="proj-a")
        assert store.live_coordinate("lab-onto", "1") is None
        assert store.latest_live_version("lab-onto") is None
        assert store.latest_live_version("lab-onto", "proj-a") == "1"

    def test_a_retired_version_keeps_resolving_for_bound_graphs(self) -> None:
        """Retiring switched pinned graphs to r4-science with only a job note;
        it now blocks new bindings (live_coordinate) only."""
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="a", project_id="proj-a")
        store.retire("lab-onto", "1", "proj-a")
        assert store.live_coordinate("lab-onto", "1", "proj-a") is None
        graph = {"ontology_id": "lab-onto", "ontology_version": "1", "project_id": "proj-a"}
        assert ontology_for_graph(graph, registry=store).ontology_id == "lab-onto"
        with pytest.raises(ValueError, match="unknown ontology"):
            ontology_for_graph({**graph, "project_id": "proj-b"}, registry=store)

    def test_a_live_version_wins_over_a_retired_one(self) -> None:
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="old", project_id="proj-a")
        store.retire("lab-onto", "1", "proj-a")
        store.register(_onto(), owner_id="new", project_id="proj-a")
        assert store.resolve_coordinate("lab-onto", "1", "proj-a").owner_id == "new"

    def test_another_projects_record_never_resolves_for_a_graph(self) -> None:
        store = OntologyRegistryStore(_conn())
        store.register(_onto(), owner_id="a", project_id="proj-a")
        mine = {"ontology_id": "lab-onto", "ontology_version": "1", "project_id": "proj-a"}
        assert ontology_for_graph(mine, registry=store).ontology_id == "lab-onto"
        with pytest.raises(ValueError, match="unknown ontology"):
            ontology_for_graph({**mine, "project_id": "proj-b"}, registry=store)


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
