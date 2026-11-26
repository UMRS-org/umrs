"""ADR-081 D-4/D-5 node typing rules: ``umrs.knowledge.typing`` (KN-S7).

Pure tests over a hand-built vocabulary (no database, no package data).
Task kn-s7-03 implements ``check_datatype`` and ``prepare_row``.
"""

from __future__ import annotations

import math

import pytest
from _kn_s7_pending import pending
from umrs.knowledge.term_types import (
    NodeDraft,
    RegistrySnapshot,
    TermSpec,
    TypingContext,
)
from umrs.knowledge.typing import check_datatype, prepare_row

pytestmark = pending(check_datatype, prepare_row)


def _cls(term_id: str, bucket: str, parents: tuple[str, ...] = (), **kw: object) -> TermSpec:
    return TermSpec(term_id, f"http://x/{term_id}", "class", term_id, parents, bucket, **kw)  # type: ignore[arg-type]


def _attr(term_id: str, datatype: str | None, **kw: object) -> TermSpec:
    return TermSpec(term_id, f"http://x/{term_id}", "attribute", term_id, datatype=datatype, **kw)  # type: ignore[arg-type]


TERMS = (
    _cls("ex:Asset", "file"),
    _cls("ex:Image", "file", ("ex:Asset",), aliases=("image",)),
    _cls("ex:Run", "entity"),
    _cls("r4:Unclassified", "entity"),
    _attr(
        "ex:pixelSize",
        "number",
        domain_classes=("ex:Image",),
        required_on=("ex:Image",),
        aliases=("pixel_size",),
    ),
    _attr("ex:title", "string"),
    _attr("ex:runKey", "string", domain_classes=("ex:Run",)),
    _attr("ex:assetNote", "string", domain_classes=("ex:Asset",)),
    _attr("ex:old", "string", status="deprecated"),
)
CLOSURE = (
    ("ex:Asset", "ex:Asset", 0),
    ("ex:Asset", "ex:Image", 1),
    ("ex:Image", "ex:Image", 0),
    ("ex:Run", "ex:Run", 0),
    ("r4:Unclassified", "r4:Unclassified", 0),
)


def _ctx(mode: str = "warn") -> TypingContext:
    return TypingContext.from_snapshot(
        RegistrySnapshot(ontology_ref="ex@1", mode=mode, terms=TERMS, closure=CLOSURE)
    )


def _row(node_class: str | None, node_type: str = "entity", **kw: object):
    return prepare_row(_ctx(), NodeDraft("n-1", node_type, node_class, **kw))  # type: ignore[arg-type]


# ── classes ──────────────────────────────────────────────────────────────────


def test_admitted_class_sets_its_bucket_as_the_facet():
    out = _row("ex:Image", node_type="entity", attrs={"ex:pixelSize": 0.5})
    assert (out.node_class, out.node_type) == ("ex:Image", "file")
    assert out.gaps == () and out.violations == () and out.missing_required == ()


def test_class_resolves_by_alias_case_insensitively():
    assert _row("IMAGE", attrs={"pixel_size": 1}).node_class == "ex:Image"


def test_missing_class_is_unclassified_with_the_writer_facet():
    out = _row(None, node_type="concept")
    assert (out.node_class, out.node_type) == ("r4:Unclassified", "concept")
    assert [v.code for v in out.violations] == ["class_missing"]
    assert out.violations[0].subject_id == "n-1"
    assert out.gaps == ()


def test_unknown_class_falls_back_records_a_gap_and_keeps_the_raw_class():
    out = _row("ex:Nope", node_type="file")
    assert (out.node_class, out.node_type) == ("r4:Unclassified", "file")
    assert out.content["class_raw"] == "ex:Nope"
    assert [(g.kind, g.value, g.sample_id) for g in out.gaps] == [("class", "ex:Nope", "n-1")]
    assert [(v.code, v.detail) for v in out.violations] == [("class_unknown", "ex:Nope")]


def test_unclassified_with_an_invalid_facet_becomes_entity():
    out = _row("ex:Nope", node_type="video")
    assert out.node_type == "entity"
    assert [v.code for v in out.violations] == ["class_unknown", "bucket_invalid"]
    assert out.violations[1].detail == "video"


def test_explicit_unclassified_is_admitted():
    out = _row("r4:Unclassified", node_type="module")
    assert (out.node_class, out.node_type) == ("r4:Unclassified", "module")
    assert out.violations == () and out.gaps == ()
    assert "class_raw" not in out.content


def test_enforce_mode_reports_the_same_violations():
    out = prepare_row(_ctx("enforce"), NodeDraft("n-1", "entity", "ex:Nope"))
    assert [v.code for v in out.violations] == ["class_unknown"]


# ── attributes ───────────────────────────────────────────────────────────────


def test_undeclared_attr_moves_to_content_undeclared_with_a_gap():
    out = _row("ex:Image", attrs={"pixel_size": 2.0, "colour": "red"})
    assert out.attrs == {"ex:pixelSize": 2.0}
    assert out.content["undeclared"] == {"colour": "red"}
    assert [(g.kind, g.value) for g in out.gaps] == [("attribute", "colour")]
    assert [(v.code, v.detail) for v in out.violations] == [("attr_undeclared", "colour")]


def test_attr_outside_its_domain_moves_without_a_gap():
    out = _row("ex:Image", attrs={"ex:pixelSize": 1, "ex:runKey": "k"})
    assert out.attrs == {"ex:pixelSize": 1}
    assert out.content["undeclared"] == {"ex:runKey": "k"}
    assert [v.code for v in out.violations] == ["attr_domain"]
    assert out.gaps == ()


def test_domain_admits_subclasses_through_the_closure():
    out = _row("ex:Image", attrs={"ex:pixelSize": 1, "ex:assetNote": "n"})
    assert out.attrs == {"ex:pixelSize": 1, "ex:assetNote": "n"}


def test_unclassified_never_satisfies_a_domain_but_keeps_free_attrs():
    out = _row(None, attrs={"ex:assetNote": "n", "ex:title": "t"})
    assert out.attrs == {"ex:title": "t"}
    assert out.content["undeclared"] == {"ex:assetNote": "n"}
    assert [v.code for v in out.violations] == ["class_missing", "attr_domain"]


def test_datatype_violation_moves_the_value():
    out = _row("ex:Image", attrs={"pixel_size": "big"})
    assert out.attrs == {}
    assert out.content["undeclared"] == {"pixel_size": "big"}
    assert [v.code for v in out.violations] == ["attr_datatype"]


def test_second_key_for_one_term_is_a_duplicate():
    out = _row("ex:Image", attrs={"pixel_size": 2, "ex:pixelSize": 1})
    assert out.attrs == {"ex:pixelSize": 1}  # "ex:pixelSize" sorts first
    assert out.content["undeclared"] == {"pixel_size": 2}
    assert [v.code for v in out.violations] == ["attr_duplicate"]


def test_deprecated_attribute_is_undeclared():
    out = _row("ex:Run", attrs={"ex:old": "x"})
    assert out.content["undeclared"] == {"ex:old": "x"}
    assert [g.value for g in out.gaps] == ["ex:old"]


def test_violations_follow_class_then_sorted_attr_keys():
    out = _row("ex:Nope", attrs={"zz": 1, "aa": 2})
    assert [(v.code, v.detail) for v in out.violations] == [
        ("class_unknown", "ex:Nope"),
        ("attr_undeclared", "aa"),
        ("attr_undeclared", "zz"),
    ]


def test_existing_undeclared_content_is_merged_and_the_draft_is_untouched():
    content = {"kind": "image", "undeclared": {"old": 1}}
    attrs = {"colour": "red"}
    out = _row("ex:Image", attrs=attrs, content=content)
    assert out.content == {"kind": "image", "undeclared": {"old": 1, "colour": "red"}}
    assert content == {"kind": "image", "undeclared": {"old": 1}}
    assert attrs == {"colour": "red"}


def test_no_undeclared_key_when_everything_is_kept():
    out = _row("ex:Image", attrs={"pixel_size": 1}, content={"kind": "image"})
    assert out.content == {"kind": "image"}


def test_missing_required_lists_absent_required_attrs():
    assert _row("ex:Image").missing_required == ("ex:pixelSize",)
    assert _row("ex:Image", attrs={"pixel_size": 3}).missing_required == ()
    assert _row("ex:Run").missing_required == ()
    assert _row(None).missing_required == ()


# ── datatypes ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("datatype", "value", "ok"),
    [
        (None, object(), True),
        ("number", 1, True),
        ("number", 1.5, True),
        ("number", True, False),
        ("number", math.nan, False),
        ("number", math.inf, False),
        ("number", "1", False),
        ("integer", 3, True),
        ("integer", 3.0, False),
        ("integer", False, False),
        ("string", "", True),
        ("string", 1, False),
        ("boolean", False, True),
        ("boolean", 0, False),
        ("datetime", "2026-11-27T10:00:00", True),
        ("datetime", "2026-11-27T10:00:00Z", True),
        ("datetime", "2026-11-27T10:00:00+09:00", True),
        ("datetime", "2026-11-27", False),
        ("datetime", "not a time", False),
        ("datetime", 5, False),
        ("date", "2026-11-27", True),
        ("date", "2026-11-27T10:00:00", False),
        ("date", "2026-13-01", False),
        ("interval", [1, 2.5], True),
        ("interval", (2, 2), True),
        ("interval", [3, 1], False),
        ("interval", [1, 2, 3], False),
        ("interval", [True, 2], False),
        ("interval", [1, math.inf], False),
        ("interval", "1-2", False),
        ("json", {"a": [1, None, "x"]}, True),
        ("json", None, True),
        ("json", {"a": math.nan}, False),
        ("json", object(), False),
        ("node", "n-1", True),
        ("node", "", False),
        ("node", 1, False),
        ("colour", "red", False),
    ],
)
def test_check_datatype(datatype, value, ok):
    assert check_datatype(datatype, value) is ok
