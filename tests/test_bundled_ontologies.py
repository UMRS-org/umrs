"""ADR-081 KN-2: ontology documents as package data (``umrs.knowledge.ontologies``).

Task kn-s7-01 implements ``expand_curie``, ``parse_doc``, ``load_bundled``
and ``bundled_by_ref``. The bundled YAML files are data owned by the lane;
these tests pin the parser contract and the shipped vocabulary's shape.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from _kn_s7_pending import pending
from umrs.knowledge.ontologies import (
    OntologyDocError,
    bundled_by_ref,
    expand_curie,
    load_bundled,
    parse_doc,
)
from umrs.knowledge.term_types import BUCKETS, CORE_REF, RELATED_TO, STD_REF, UNCLASSIFIED

pytestmark = pending(expand_curie, parse_doc, load_bundled, bundled_by_ref)

DOC: dict[str, Any] = {
    "ontology": "demo",
    "version": "1",
    "iri_base": "http://x/demo#",
    "prefixes": {"d": "http://x/demo#", "s": "http://x/std/"},
    "imports": ["std@1"],
    "classes": [
        {"id": "d:Thing", "bucket": "entity", "aliases": ["thing"]},
        {"id": "d:Image", "label": "Image", "bucket": "file", "parents": ["d:Thing", "s:Obj"]},
    ],
    "relations": [
        {
            "id": "d:partOf",
            "aliases": ["part_of", "thing"],
            "domain": ["d:Thing"],
            "range": ["d:Thing"],
            "transitive": True,
            "inverse_of": "d:hasPart",
        },
    ],
    "attributes": [
        {
            "id": "d:pixelSize",
            "datatype": "number",
            "unit": "unit:MicroM",
            "domain": ["d:Image"],
            "required_on": ["d:Image"],
        },
    ],
}


def _with(**changes: Any) -> dict[str, Any]:
    doc = copy.deepcopy(DOC)
    doc.update(changes)
    return doc


def _with_term(section: str, index: int, **changes: Any) -> dict[str, Any]:
    doc = copy.deepcopy(DOC)
    doc[section][index].update(changes)
    return doc


# ── CURIEs ───────────────────────────────────────────────────────────────────


def test_expand_curie():
    assert expand_curie("d:Image", {"d": "http://x/demo#"}) == "http://x/demo#Image"


@pytest.mark.parametrize("bad", ["Image", "d:", ":Image", "z:Image"])
def test_expand_curie_refuses_non_curies_and_undeclared_prefixes(bad):
    with pytest.raises(OntologyDocError):
        expand_curie(bad, {"d": "http://x/demo#"})


# ── parse_doc ────────────────────────────────────────────────────────────────


def test_parse_doc_fields_and_term_order():
    doc = parse_doc(DOC)
    assert (doc.ref, doc.iri_base, doc.imports) == ("demo@1", "http://x/demo#", ("std@1",))
    assert [(t.term_id, t.kind) for t in doc.terms] == [
        ("d:Thing", "class"),
        ("d:Image", "class"),
        ("d:partOf", "relation"),
        ("d:pixelSize", "attribute"),
    ]


def test_parse_doc_term_values():
    terms = {t.term_id: t for t in parse_doc(DOC).terms}
    image, part, pixel = terms["d:Image"], terms["d:partOf"], terms["d:pixelSize"]
    assert image.iri == "http://x/demo#Image"
    assert (image.label, image.bucket, image.parents) == ("Image", "file", ("d:Thing", "s:Obj"))
    assert terms["d:Thing"].label == "Thing"  # default: the local part
    assert (part.domain_classes, part.range_classes) == (("d:Thing",), ("d:Thing",))
    assert (part.transitive, part.inverse_of, part.aliases) == (
        True,
        "d:hasPart",
        ("part_of", "thing"),
    )
    assert (pixel.datatype, pixel.unit, pixel.required_on) == (
        "number",
        "unit:MicroM",
        ("d:Image",),
    )
    assert (pixel.origin, pixel.verified, pixel.status, pixel.source) == (
        "minted",
        True,
        "active",
        "demo@1",
    )


def test_version_int_becomes_str_and_doc_origin_is_the_term_default():
    doc = parse_doc(_with(version=2, origin="standard"))
    assert doc.ref == "demo@2"
    assert {t.origin for t in doc.terms} == {"standard"}


def test_term_origin_and_source_override_the_document():
    doc = parse_doc(_with_term("classes", 0, origin="imported", source="obo:go@1", verified=False))
    thing = doc.terms[0]
    assert (thing.origin, thing.source, thing.verified) == ("imported", "obo:go@1", False)


def test_alias_may_repeat_across_kinds():
    parse_doc(DOC)  # "thing" is a class alias and a relation alias


@pytest.mark.parametrize(
    ("doc", "needle"),
    [
        (_with(ontology=""), "ontology"),
        (_with(ontology="Bad Name"), "ontology"),
        ({k: v for k, v in DOC.items() if k != "version"}, "version"),
        ({k: v for k, v in DOC.items() if k != "prefixes"}, "prefixes"),
        (_with(colour="red"), "colour"),
        (_with(imports=["std"]), "std"),
        (_with(origin="borrowed"), "borrowed"),
        (_with_term("classes", 0, bucket=None), "d:Thing"),
        (_with_term("classes", 0, bucket="video"), "d:Thing"),
        (_with_term("classes", 0, datatype="string"), "d:Thing"),
        (_with_term("classes", 0, colour="red"), "colour"),
        (_with_term("classes", 1, parents=["q:Obj"]), "q:Obj"),
        (_with_term("classes", 1, id="d:Thing"), "d:Thing"),
        (_with_term("classes", 1, aliases=["THING"]), "(?i)thing"),
        (_with_term("classes", 0, status="gone"), "gone"),
        (_with_term("relations", 0, bucket="entity"), "d:partOf"),
        (_with_term("relations", 0, datatype="string"), "d:partOf"),
        (_with_term("relations", 0, inverse_of="hasPart"), "hasPart"),
        (_with_term("attributes", 0, datatype="colour"), "colour"),
        (_with_term("attributes", 0, required_on=["z:X"]), "z:X"),
    ],
)
def test_parse_doc_refuses_by_name(doc, needle):
    with pytest.raises(OntologyDocError, match=needle):
        parse_doc(doc)


def test_error_names_the_document_ref():
    with pytest.raises(OntologyDocError, match="demo@1"):
        parse_doc(_with_term("classes", 0, bucket=None))


# ── the bundled package data ─────────────────────────────────────────────────


def test_bundled_refs():
    assert [d.ref for d in load_bundled()] == [CORE_REF, STD_REF]
    assert sorted(bundled_by_ref()) == [CORE_REF, STD_REF]


def test_core_imports_std_and_lives_in_the_tag_namespace():
    core = bundled_by_ref()[CORE_REF]
    assert core.imports == (STD_REF,)
    assert core.iri_base.startswith("tag:")
    terms = {t.term_id: t for t in core.terms}
    assert terms[UNCLASSIFIED].kind == "class"
    assert terms[RELATED_TO].kind == "relation"
    assert terms["r4:Document"].iri == core.iri_base + "Document"


def test_every_bucket_has_a_core_root_class():
    core = bundled_by_ref()[CORE_REF]
    roots = {t.bucket for t in core.terms if t.kind == "class"}
    assert roots >= BUCKETS


def test_every_bundled_parent_and_domain_is_defined_in_the_bundle():
    every = {t.term_id: t for d in load_bundled() for t in d.terms}
    referenced = {
        ref
        for t in every.values()
        for ref in (*t.parents, *t.domain_classes, *t.range_classes, *t.required_on)
    }
    assert referenced <= set(every)


def test_std_terms_are_standard_and_core_terms_minted():
    docs = bundled_by_ref()
    assert {t.origin for t in docs[STD_REF].terms} == {"standard"}
    assert {t.origin for t in docs[CORE_REF].terms} == {"minted"}


def test_legacy_relation_slugs_resolve_in_core():
    core = bundled_by_ref()[CORE_REF]
    aliases = {a for t in core.terms if t.kind == "relation" for a in t.aliases}
    legacy = {
        "part_of",
        "produced_by",
        "derived_from",
        "mentions",
        "cites",
        "depends_on",
        "implements",
        "defines",
        "caused_by",
        "same_as",
        "contains",
    }
    assert legacy <= aliases
