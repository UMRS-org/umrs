"""ADR-116 D-4/D-6: built-in r4-science, extraction vocabulary, graph resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from umrs.knowledge import graph_registry, ontology_registry
from umrs.knowledge.graph_registry import KnowledgeGraphMeta
from umrs.knowledge.ontology import (
    DEFAULT_ONTOLOGY,
    Ontology,
    builtin_ontology,
    ontology_for_graph,
    ontology_from_dict,
)
from umrs.knowledge.ontology_vocab import (
    default_vocabulary,
    resolve_graph_vocabulary,
    vocabulary_for,
)
from umrs.shared import pg

# The 13 types the pre-ADR-116 prose extractor listed (_PROSE_EXTRACT_SYSTEM)
# and its entity-bucket set (_ENTITY_TYPES), copied here so a drift in
# either side is a test failure rather than a silent change.
_OLD_PROMPT_TYPES = {
    "concept", "method", "technique", "property", "process", "model", "paper",
    "dataset", "software", "application", "person", "material", "finding",
}  # fmt: skip
_OLD_ENTITY_TYPES = {
    "paper", "dataset", "software", "application", "person", "institution",
    "model", "code", "notebook", "figure", "result", "material",
}  # fmt: skip


class TestR4ScienceBuiltin:
    def test_resolves_without_a_registry(self) -> None:
        onto = ontology_for_graph({"ontology_id": "r4-science"})
        assert onto.ontology_id == "r4-science"
        assert onto.version == "1"
        assert onto.source_format == "builtin"
        assert ontology_for_graph({"ontology_id": "r4-science", "ontology_version": "1"}) is onto

    def test_wrong_version_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="no version '9'"):
            ontology_for_graph({"ontology_id": "r4-science", "ontology_version": "9"})

    def test_superset_of_the_old_prompt_types(self) -> None:
        classes = set(default_vocabulary().class_names())
        assert classes >= _OLD_PROMPT_TYPES
        assert {"instrument", "organization", "task", "metric", "experiment"} <= classes
        assert {"sample", "parameter"} <= classes

    def test_buckets_reproduce_the_old_entity_rule(self) -> None:
        vocab = default_vocabulary()
        for cls in _OLD_PROMPT_TYPES:
            expected = "entity" if cls in _OLD_ENTITY_TYPES else "concept"
            assert vocab.bucket_for(cls) == expected, cls

    def test_relations_and_signatures(self) -> None:
        onto = builtin_ontology("r4-science")
        assert onto is not None
        assert {
            "uses", "produces", "part_of", "evaluated_on", "measures", "improves_on",
            "implements", "depends_on", "cites", "derived_from", "compares_with",
            "applied_to", "authored_by", "related_to",
        } == onto.relation_types  # fmt: skip
        assert ("cites", "paper", "paper") in onto.relation_signatures
        assert ("authored_by", "", "person") in onto.relation_signatures
        assert "depends_on" in onto.transitive_relations

    def test_round_trip(self) -> None:
        onto = builtin_ontology("r4-science")
        assert onto is not None
        assert ontology_from_dict(onto.to_dict()) == onto


class TestMatching:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Method", "method"),
            ("  DATASETS ", "dataset"),
            ("Properties", "property"),
            ("processes", "process"),
            ("Neural Network", "model"),
            ("neural-networks", "model"),
            ("Institution", "organization"),
            ("code", "software"),
            ("data set", "dataset"),
            ("results", "finding"),
            ("unclassified", None),  # abstract: never assigned from the vocabulary
            ("research object", None),
            ("wormhole", None),
            ("", None),
        ],
    )
    def test_match_class(self, raw: str, expected: str | None) -> None:
        assert default_vocabulary().match_class(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("uses", "uses"),
            ("use", "uses"),
            ("Utilizes", "uses"),
            ("is part of", "part_of"),
            ("is based on", "derived_from"),
            ("evaluated-on", "evaluated_on"),
            ("outperforms", "improves_on"),
            ("enchants", None),
            ("", None),
        ],
    )
    def test_match_relation(self, raw: str, expected: str | None) -> None:
        assert default_vocabulary().match_relation(raw) == expected

    def test_unclassified_bucket_is_concept(self) -> None:
        assert default_vocabulary().bucket_for("unclassified") == "concept"
        assert default_vocabulary().bucket_for("never-heard-of") == "entity"

    def test_gliner_labels_are_readable_and_capped(self) -> None:
        labels = default_vocabulary().gliner_labels()
        assert "dataset" in labels
        assert all("_" not in label for label in labels)
        assert len(labels) <= 25

    def test_prompt_block_lists_classes_and_signatures(self) -> None:
        block = default_vocabulary().prompt_block()
        assert "- method: A way of solving a problem" in block
        assert "Also called: approach, algorithm, methodology." in block
        assert "- cites (paper -> paper)" in block
        assert "- uses (any -> any)" in block
        assert "research_object" not in block
        assert "- unclassified" not in block


class TestVocabularyFor:
    def test_r4_default_yields_the_science_vocabulary(self) -> None:
        assert vocabulary_for(DEFAULT_ONTOLOGY) is default_vocabulary()

    def test_custom_ontology_buckets_and_aliases(self) -> None:
        onto = Ontology(
            ontology_id="lab",
            version="1",
            entity_types=frozenset({"alloy", "furnace"}),
            relation_types=frozenset({"heated_in"}),
            aliases=(("alloy", ("metal mix",)),),
            class_buckets=(("alloy", "entity"), ("furnace", "concept")),
        )
        vocab = vocabulary_for(onto)
        assert vocab.ontology_id == "lab"
        assert vocab.match_class("Metal Mixes") == "alloy"
        assert vocab.bucket_for("furnace") == "concept"
        assert vocab.relation_names() == ["heated_in"]


# ── resolve_graph_vocabulary with fakes for both registries ──────────────


_LAB = Ontology(
    ontology_id="lab",
    version="3",
    entity_types=frozenset({"alloy"}),
    relation_types=frozenset({"heated_in"}),
)


@dataclass
class _Record:
    ontology: Ontology
    project_id: str | None = None


class _FakePool:
    def __init__(self) -> None:
        self.taken = 0
        self.returned = 0

    def getconn(self) -> object:
        self.taken += 1
        return object()

    def putconn(self, _conn: object) -> None:
        self.returned += 1


class _FakeOntologyRegistry:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def latest_live_version(self, ontology_id: str, _project_id: Any = None) -> str | None:
        return "3" if ontology_id == "lab" else None

    def live_coordinate(self, ontology_id: str, version: str) -> _Record | None:
        if (ontology_id, version) == ("lab", "3"):
            return _Record(_LAB)
        return None


def _install(monkeypatch: pytest.MonkeyPatch, graph: KnowledgeGraphMeta | None) -> _FakePool:
    pool = _FakePool()

    class _FakeGraphRegistry:
        def __init__(self, _pool: Any) -> None:
            assert _pool is pool

        def get_graph(self, _graph_id: str) -> KnowledgeGraphMeta | None:
            return graph

    monkeypatch.setattr(pg, "pg_configured_from_env", lambda: True)
    monkeypatch.setattr(pg, "get_default_sync_pool", lambda: pool)
    monkeypatch.setattr(graph_registry, "GraphRegistryStore", _FakeGraphRegistry)
    monkeypatch.setattr(ontology_registry, "OntologyRegistryStore", _FakeOntologyRegistry)
    return pool


def _graph(**overrides: Any) -> KnowledgeGraphMeta:
    return KnowledgeGraphMeta(graph_id="g-1", name="g", project_id="proj-a", **overrides)


class TestResolveGraphVocabulary:
    def test_empty_graph_id_is_the_default(self) -> None:
        vocab, notes = resolve_graph_vocabulary("proj-a", "")
        assert vocab is default_vocabulary()
        assert notes == []

    def test_no_database_falls_back_with_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pg, "pg_configured_from_env", lambda: False)
        vocab, notes = resolve_graph_vocabulary("proj-a", "g-1")
        assert vocab is default_vocabulary()
        assert notes == ["ontology_fallback:no_database"]

    def test_unbound_graph_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _graph())
        assert resolve_graph_vocabulary("proj-a", "g-1") == (default_vocabulary(), [])

    def test_registered_ontology_latest_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pool = _install(monkeypatch, _graph(ontology_id="lab"))
        vocab, notes = resolve_graph_vocabulary("proj-a", "g-1")
        assert (vocab.ontology_id, vocab.version) == ("lab", "3")
        assert notes == []
        assert pool.taken == pool.returned == 1

    def test_unknown_ontology_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pool = _install(monkeypatch, _graph(ontology_id="ghost", ontology_version="1"))
        vocab, notes = resolve_graph_vocabulary("proj-a", "g-1")
        assert vocab is default_vocabulary()
        assert notes == ["ontology_fallback:unresolved_ontology"]
        assert pool.taken == pool.returned == 1

    def test_missing_graph_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, None)
        vocab, notes = resolve_graph_vocabulary("proj-a", "g-404")
        assert vocab is default_vocabulary()
        assert notes == ["ontology_fallback:graph_not_found"]

    def test_any_error_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> None:
            raise ConnectionError("db down")

        monkeypatch.setattr(pg, "pg_configured_from_env", lambda: True)
        monkeypatch.setattr(pg, "get_default_sync_pool", _boom)
        vocab, notes = resolve_graph_vocabulary("proj-a", "g-1")
        assert vocab is default_vocabulary()
        assert notes == ["ontology_fallback:ConnectionError"]

    def test_empty_graph_id_reads_the_default_graphs_binding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Workflow-output auto-ingest sends graph_id '' while uploads send the
        default graph's id; both must extract with the ontology bound to it."""
        from umrs.knowledge import graph_registry
        from umrs.knowledge.graph_registry import default_graph_id

        asked: list[str] = []
        bound = KnowledgeGraphMeta(
            graph_id=default_graph_id("proj-a"), name="default", project_id="proj-a",
            ontology_id="lab",
        )  # fmt: skip
        _install(monkeypatch, bound)

        fake = graph_registry.GraphRegistryStore

        class _Recording(fake):  # type: ignore[misc, valid-type]
            def get_graph(self, graph_id: str) -> KnowledgeGraphMeta | None:
                asked.append(graph_id)
                return super().get_graph(graph_id)

        monkeypatch.setattr(graph_registry, "GraphRegistryStore", _Recording)
        vocab, notes = resolve_graph_vocabulary("proj-a", "")
        assert (vocab.ontology_id, notes) == ("lab", [])
        assert asked == [default_graph_id("proj-a")]

    def test_an_unregistered_default_graph_is_the_default_quietly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, None)
        assert resolve_graph_vocabulary("proj-a", "") == (default_vocabulary(), [])


class TestOntologyForGraphLatestVersion:
    def test_id_without_version_resolves_latest_live(self) -> None:
        """ADR-116 D-5: a graph bound by id only used to raise 'unknown ontology'
        because the registry was consulted only for a pinned version."""
        onto = ontology_for_graph({"ontology_id": "lab"}, registry=_FakeOntologyRegistry(None))
        assert onto is _LAB


class TestOntologyOptionalFields:
    def test_new_fields_round_trip(self) -> None:
        onto = Ontology(
            ontology_id="x",
            version="1",
            entity_types=frozenset({"a", "b"}),
            relation_types=frozenset({"r"}),
            transitive_relations=frozenset(),
            aliases=(("a", ("alpha", "A1")), ("r", ("rel",))),
            descriptions=(("a", "the a"),),
            relation_signatures=(("r", "", "a"), ("r", "a", "b")),  # sorted
            class_buckets=(("a", "entity"), ("b", "concept")),
            abstract_types=frozenset({"b"}),
            prefixes=(("ex", "https://example.org/"),),
            source_format="json",
        )
        assert ontology_from_dict(onto.to_dict()) == onto

    def test_old_dicts_without_new_fields_still_load(self) -> None:
        onto = ontology_from_dict(
            {
                "ontology_id": "x",
                "version": "1",
                "entity_types": ["a"],
                "relation_types": ["r"],
            }
        )
        assert onto.aliases == () and onto.class_buckets == ()
        assert "aliases" not in onto.to_dict()

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            ({"class_buckets": {"a": "gadget"}}, "unknown buckets"),
            ({"class_buckets": {"zz": "entity"}}, "undeclared entity types"),
            ({"relation_signatures": [{"relation": "q", "domain": "a", "range": "a"}]}, "undeclared relation"),
            ({"relation_signatures": [{"relation": "r", "domain": "zz", "range": "a"}]}, "undeclared entity"),
            ({"abstract_types": ["zz"]}, "undeclared entity"),
        ],
    )  # fmt: skip
    def test_new_fields_are_checked(self, extra: dict[str, Any], message: str) -> None:
        raw = {"ontology_id": "x", "version": "1", "entity_types": ["a"], "relation_types": ["r"]}
        with pytest.raises(ValueError, match=message):
            ontology_from_dict({**raw, **extra})


class TestPromptCap:
    """Every window's system prompt carries the block: an ontology of 2000
    terms with long descriptions sent ~256 KB per call."""

    def _big(self) -> Ontology:
        names = [f"class_{i:04d}" for i in range(2000)]
        return Ontology(
            ontology_id="big",
            version="1",
            entity_types=frozenset(names),
            relation_types=frozenset(),
            descriptions=tuple((n, "d" * 500) for n in names),
            aliases=tuple((n, tuple(f"alias {n} {k}" for k in range(30))) for n in names[:3]),
        )

    def test_a_large_vocabulary_is_cut_and_says_so(self) -> None:
        from umrs.knowledge import ontology_vocab

        vocab = vocabulary_for(self._big())
        block = vocab.prompt_block()
        assert len(block) <= ontology_vocab.PROMPT_BLOCK_MAX + 100
        assert block.rstrip().endswith("more lines of the vocabulary not listed)")
        assert "d" * (ontology_vocab.PROMPT_DETAIL_MAX + 1) not in block
        assert "alias class_0000 8" not in block
        assert vocab.prompt_truncated

    def test_the_built_in_vocabulary_is_whole(self) -> None:
        vocab = default_vocabulary()
        assert not vocab.prompt_truncated
        assert "not listed" not in vocab.prompt_block()
