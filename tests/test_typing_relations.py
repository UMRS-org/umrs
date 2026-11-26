"""ADR-081 D-4/D-5 edge typing rules: ``umrs.knowledge.edge_typing``.

Pure tests over a hand-built vocabulary. Task kn-s7-04 implements
``prepare_edge``.
"""

from __future__ import annotations

from _kn_s7_pending import pending
from umrs.knowledge.edge_typing import prepare_edge
from umrs.knowledge.term_types import (
    EdgeDraft,
    RegistrySnapshot,
    TermSpec,
    TypingContext,
)

pytestmark = pending(prepare_edge)


def _term(term_id: str, kind: str, **kw: object) -> TermSpec:
    bucket = "entity" if kind == "class" else None
    return TermSpec(term_id, f"http://x/{term_id}", kind, term_id, bucket=bucket, **kw)  # type: ignore[arg-type]


TERMS = (
    _term("ex:Asset", "class"),
    _term("ex:Image", "class", parents=("ex:Asset",)),
    _term("ex:Run", "class"),
    _term("r4:Unclassified", "class"),
    _term("r4:relatedTo", "relation", aliases=("related_to",)),
    _term(
        "ex:partOf",
        "relation",
        aliases=("part_of",),
        domain_classes=("ex:Asset",),
        range_classes=("ex:Asset",),
    ),
    _term("ex:usedBy", "relation"),
    _term("ex:gone", "relation", status="deprecated", aliases=("gone",)),
)
CLOSURE = (
    ("ex:Asset", "ex:Asset", 0),
    ("ex:Asset", "ex:Image", 1),
    ("ex:Image", "ex:Image", 0),
    ("ex:Run", "ex:Run", 0),
    ("r4:Unclassified", "r4:Unclassified", 0),
    ("r4:relatedTo", "r4:relatedTo", 0),
    ("ex:partOf", "ex:partOf", 0),
    ("ex:usedBy", "ex:usedBy", 0),
)
CTX = TypingContext.from_snapshot(
    RegistrySnapshot(ontology_ref="ex@1", mode="warn", terms=TERMS, closure=CLOSURE)
)


def test_alias_resolves_to_the_curie():
    out = prepare_edge(
        CTX, EdgeDraft("e-1", "part_of"), source_class="ex:Image", target_class="ex:Asset"
    )
    assert out.relation_class == "ex:partOf"
    assert out.gaps == () and out.violations == ()
    assert "rel_raw" not in out.content


def test_curie_resolves_exactly():
    assert prepare_edge(CTX, EdgeDraft("e-1", "ex:usedBy")).relation_class == "ex:usedBy"


def test_alias_lookup_is_case_insensitive():
    assert prepare_edge(CTX, EdgeDraft("e-1", "PART_OF")).relation_class == "ex:partOf"


def test_unknown_relation_falls_back_to_related_to_with_a_gap():
    out = prepare_edge(CTX, EdgeDraft("e-1", "frobs", content={"w": 1}))
    assert out.relation_class == "r4:relatedTo"
    assert out.content == {"w": 1, "rel_raw": "frobs"}
    assert [(g.kind, g.value, g.sample_id) for g in out.gaps] == [("relation", "frobs", "e-1")]
    assert [(v.code, v.detail, v.subject_id) for v in out.violations] == [
        ("relation_unknown", "frobs", "e-1")
    ]


def test_empty_relation_falls_back_without_a_gap():
    out = prepare_edge(CTX, EdgeDraft("e-1", ""))
    assert out.relation_class == "r4:relatedTo"
    assert out.content == {"rel_raw": ""}
    assert out.gaps == ()
    assert [v.code for v in out.violations] == ["relation_unknown"]


def test_deprecated_relation_is_unknown():
    out = prepare_edge(CTX, EdgeDraft("e-1", "gone"))
    assert out.relation_class == "r4:relatedTo"
    assert [g.value for g in out.gaps] == ["gone"]


def test_relation_class_overrides_relation_type():
    out = prepare_edge(CTX, EdgeDraft("e-1", "frobs", relation_class="ex:usedBy"))
    assert out.relation_class == "ex:usedBy"
    assert out.gaps == () and "rel_raw" not in out.content


def test_unknown_relation_class_reports_the_class_not_the_type():
    out = prepare_edge(CTX, EdgeDraft("e-1", "part_of", relation_class="ex:nope"))
    assert out.relation_class == "r4:relatedTo"
    assert out.content["rel_raw"] == "ex:nope"


def test_domain_violation_keeps_the_relation():
    out = prepare_edge(
        CTX, EdgeDraft("e-1", "part_of"), source_class="ex:Run", target_class="ex:Image"
    )
    assert out.relation_class == "ex:partOf"
    assert [(v.code, v.detail) for v in out.violations] == [
        ("relation_domain", "ex:partOf: ex:Run")
    ]
    assert out.gaps == ()


def test_range_violation():
    out = prepare_edge(
        CTX, EdgeDraft("e-1", "part_of"), source_class="ex:Image", target_class="ex:Run"
    )
    assert [(v.code, v.detail) for v in out.violations] == [("relation_range", "ex:partOf: ex:Run")]


def test_both_ends_wrong_reports_domain_then_range():
    out = prepare_edge(
        CTX, EdgeDraft("e-1", "part_of"), source_class="ex:Run", target_class="ex:Run"
    )
    assert [v.code for v in out.violations] == ["relation_domain", "relation_range"]


def test_unknown_or_unclassified_endpoints_are_not_checked():
    for source, target in ((None, None), ("r4:Unclassified", "r4:Unclassified")):
        out = prepare_edge(
            CTX, EdgeDraft("e-1", "part_of"), source_class=source, target_class=target
        )
        assert out.violations == ()


def test_relation_without_domain_admits_any_endpoint():
    out = prepare_edge(
        CTX, EdgeDraft("e-1", "ex:usedBy"), source_class="ex:Run", target_class="ex:Image"
    )
    assert out.violations == ()


def test_draft_content_is_not_mutated():
    content = {"w": 1}
    prepare_edge(CTX, EdgeDraft("e-1", "frobs", content=content))
    assert content == {"w": 1}
