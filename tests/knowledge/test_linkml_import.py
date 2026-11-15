"""ADR-116 D-4: LinkML schema subset → Ontology (PyYAML only, no linkml runtime)."""

from __future__ import annotations

import pytest
from umrs.knowledge.linkml_import import (
    MAX_SOURCE_BYTES,
    MAX_TERMS,
    OntologySourceError,
    linkml_to_ontology,
)
from umrs.knowledge.ontology import ontology_from_dict

GOOD_SCHEMA = """\
id: https://example.org/lab-materials
name: lab-materials
version: "2"
description: Materials lab vocabulary.
prefixes:
  linkml: https://w3id.org/linkml/
  lab:
    prefix_prefix: lab
    prefix_reference: https://example.org/lab/
imports:
  - linkml:types
classes:
  Thing:
    abstract: true
    annotations:
      r4_bucket: concept
  Material:
    is_a: Thing
    description: A physical material.
    aliases: [compound, substance]
    annotations:
      r4_bucket:
        tag: r4_bucket
        value: entity
  CrystalStructure:
    is_a: Material
    aliases: [lattice]
  SynthesisMethod:
    is_a: Thing
    slots: [produces]
    attributes:
      uses_precursor:
        range: Material
        aliases: [uses precursor]
      temperature:
        range: float
slots:
  produces:
    range: Material
    description: The method yields the material.
    aliases: [yields]
  part_of:
    domain: Material
    range: Material
    transitive: true
  label:
    range: string
"""


def test_good_schema_imports_classes_relations_and_metadata() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    assert onto.ontology_id == "lab-materials"
    assert onto.version == "2"
    assert onto.description == "Materials lab vocabulary."
    assert onto.source_format == "linkml"
    assert onto.entity_types == {"thing", "material", "crystal_structure", "synthesis_method"}
    # type/enum ranges (label: string, temperature: float) are skipped
    assert onto.relation_types == {"produces", "part_of", "uses_precursor"}
    assert dict(onto.prefixes) == {
        "lab": "https://example.org/lab/",
        "linkml": "https://w3id.org/linkml/",
    }


def test_is_a_maps_to_subtype_and_buckets_inherit() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    assert dict(onto.subtype_of) == {
        "material": "thing",
        "crystal_structure": "material",
        "synthesis_method": "thing",
    }
    buckets = dict(onto.class_buckets)
    assert buckets["material"] == "entity"
    assert buckets["crystal_structure"] == "entity"  # inherited from Material
    assert buckets["synthesis_method"] == "concept"  # inherited from Thing
    assert onto.abstract_types == {"thing"}


def test_aliases_and_descriptions_are_kept() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    aliases = dict(onto.aliases)
    assert aliases["material"] == ("compound", "substance")
    assert aliases["produces"] == ("yields",)
    assert aliases["uses_precursor"] == ("uses precursor",)
    assert dict(onto.descriptions)["produces"] == "The method yields the material."


def test_relation_domain_and_range() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    assert set(onto.relation_signatures) == {
        # domain from the class that lists the slot under `slots`
        ("produces", "synthesis_method", "material"),
        # explicit slot domain
        ("part_of", "material", "material"),
        # attribute: domain is the owning class
        ("uses_precursor", "synthesis_method", "material"),
    }


def test_slot_without_domain_or_users_has_any_domain() -> None:
    schema = GOOD_SCHEMA.replace("    slots: [produces]\n", "")
    onto = linkml_to_ontology(schema)
    assert ("produces", "", "material") in onto.relation_signatures


def test_transitive_is_explicit_never_a_silent_default() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    assert onto.transitive_relations == {"part_of"}


def test_round_trips_through_the_strict_reader() -> None:
    onto = linkml_to_ontology(GOOD_SCHEMA)
    assert ontology_from_dict(onto.to_dict()) == onto


def test_mixins_are_refused() -> None:
    schema = GOOD_SCHEMA.replace("    is_a: Thing\n    slots", "    mixins: [Thing]\n    slots")
    with pytest.raises(OntologySourceError, match="mixins"):
        linkml_to_ontology(schema)


def test_mixin_class_is_refused() -> None:
    schema = GOOD_SCHEMA.replace("    abstract: true\n", "    mixin: true\n", 1)
    with pytest.raises(OntologySourceError, match="mixins"):
        linkml_to_ontology(schema)


@pytest.mark.parametrize(
    "imported",
    ["https://example.org/other.yaml", "core", "../secrets.yaml"],
)
def test_remote_and_file_imports_are_refused(imported: str) -> None:
    schema = GOOD_SCHEMA.replace("  - linkml:types\n", f"  - linkml:types\n  - {imported}\n")
    with pytest.raises(OntologySourceError, match="only 'linkml:types'"):
        linkml_to_ontology(schema)


@pytest.mark.parametrize("field", ["version", "name", "id"])
def test_missing_required_header_is_named(field: str) -> None:
    lines = [ln for ln in GOOD_SCHEMA.splitlines() if not ln.startswith(f"{field}:")]
    with pytest.raises(OntologySourceError, match=f"missing required field.*'{field}'"):
        linkml_to_ontology("\n".join(lines))


def test_unknown_bucket_is_refused() -> None:
    schema = GOOD_SCHEMA.replace("r4_bucket: concept", "r4_bucket: gadget")
    with pytest.raises(OntologySourceError, match="r4_bucket"):
        linkml_to_ontology(schema)


def test_undeclared_parent_is_refused() -> None:
    schema = GOOD_SCHEMA.replace("is_a: Material", "is_a: Mineral")
    with pytest.raises(OntologySourceError, match="is_a 'mineral'"):
        linkml_to_ontology(schema)


def test_is_a_cycle_is_refused() -> None:
    schema = """\
id: x
name: cyc
version: "1"
classes:
  A:
    is_a: B
  B:
    is_a: A
"""
    with pytest.raises(OntologySourceError, match="cycle"):
        linkml_to_ontology(schema)


def test_camel_case_collision_is_refused() -> None:
    schema = 'id: x\nname: c\nversion: "1"\nclasses:\n  FooBar: {}\n  foo_bar: {}\n'
    with pytest.raises(OntologySourceError, match="collides"):
        linkml_to_ontology(schema)


def test_not_yaml_and_not_a_mapping() -> None:
    with pytest.raises(OntologySourceError, match="not valid YAML"):
        linkml_to_ontology("classes: [unclosed")
    with pytest.raises(OntologySourceError, match="YAML mapping"):
        linkml_to_ontology("- just\n- a list\n")


def test_size_cap() -> None:
    big = GOOD_SCHEMA + "#" * (MAX_SOURCE_BYTES + 1)
    with pytest.raises(OntologySourceError, match="larger than"):
        linkml_to_ontology(big)


def test_term_count_cap() -> None:
    classes = "".join(f"  C{i}: {{}}\n" for i in range(MAX_TERMS + 1))
    schema = f'id: x\nname: many\nversion: "1"\nclasses:\n{classes}'
    with pytest.raises(OntologySourceError, match="terms"):
        linkml_to_ontology(schema)


def test_no_classes_is_refused() -> None:
    with pytest.raises(OntologySourceError, match="no classes"):
        linkml_to_ontology('id: x\nname: empty\nversion: "1"\n')


def test_bad_ontology_id_is_refused() -> None:
    schema = GOOD_SCHEMA.replace("name: lab-materials", "name: lab materials/x")
    with pytest.raises(OntologySourceError, match="ontology id"):
        linkml_to_ontology(schema)


# ── hostile YAML (review finding: billion laughs) ─────────────────────────


def _laughs(depth: int, target: str) -> str:
    """A ~500-byte alias tree that expands 10**depth-fold once str()-ed."""
    lines = ['a0: &a0 ["xxxxxxxxxx"]']
    lines += [f"a{i}: &a{i} [{', '.join([f'*a{i - 1}'] * 10)}]" for i in range(1, depth + 1)]
    return "\n".join([*lines, f"{target}: *a{depth}", "id: x", 'version: "1"', "classes: {A: {}}"])


@pytest.mark.parametrize("target", ["name", "description"])
def test_yaml_aliases_are_refused_before_any_expansion(target: str) -> None:
    """safe_load kept the aliases as shared references; str() of the name or
    description then expanded depth 8 to ~1.6 GB of text."""
    import time

    source = _laughs(8, target)
    assert len(source) < 1024
    started = time.monotonic()
    with pytest.raises(OntologySourceError, match="aliases") as caught:
        linkml_to_ontology(source)
    assert time.monotonic() - started < 1.0
    assert len(str(caught.value)) < 1000


def test_a_yaml_anchor_alone_is_refused() -> None:
    schema = GOOD_SCHEMA.replace("  Thing:\n", "  Thing: &thing\n", 1)
    with pytest.raises(OntologySourceError, match="anchors and aliases"):
        linkml_to_ontology(schema)


def test_deep_nesting_is_a_source_error_not_a_recursion_error() -> None:
    schema = GOOD_SCHEMA + "description: " + "[" * 5000 + "]" * 5000 + "\n"
    with pytest.raises(OntologySourceError, match="nests deeper"):
        linkml_to_ontology(schema)


def test_node_count_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from umrs.knowledge import linkml_import

    assert linkml_to_ontology(GOOD_SCHEMA)  # well under the real cap
    monkeypatch.setattr(linkml_import, "MAX_YAML_NODES", 20)
    with pytest.raises(OntologySourceError, match="more than 20 YAML nodes"):
        linkml_to_ontology(GOOD_SCHEMA)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("name: lab-materials", "name: [lab, materials]"),
        ("description: Materials lab vocabulary.", "description: {a: b}"),
        ("    description: A physical material.", "    description: [a, b]"),
    ],
)
def test_structured_text_fields_are_refused(old: str, new: str) -> None:
    with pytest.raises(OntologySourceError, match="must be text"):
        linkml_to_ontology(GOOD_SCHEMA.replace(old, new))
