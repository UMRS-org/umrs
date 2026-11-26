"""ADR-081 D-2 closure and admitted sets: ``umrs.knowledge.term_closure``.

Pure tests. Task kn-s7-02 implements ``build_closure``, ``closure_rows``
and ``admitted_set``.
"""

from __future__ import annotations

import pytest
from _kn_s7_pending import pending
from umrs.knowledge.term_closure import (
    ClosureError,
    admitted_set,
    build_closure,
    closure_rows,
)
from umrs.knowledge.term_types import OntologyDoc, TermSpec

pytestmark = pending(build_closure, closure_rows, admitted_set)


def _t(term_id: str, kind: str = "class", parents: tuple[str, ...] = ()) -> TermSpec:
    bucket = "entity" if kind == "class" else None
    return TermSpec(term_id, f"http://x/{term_id}", kind, term_id, parents, bucket)


def _doc(oid: str, terms: tuple[TermSpec, ...], imports: tuple[str, ...] = ()) -> OntologyDoc:
    return OntologyDoc(oid, "1", "", {"x": "http://x/"}, imports, "", terms)


# ── closure ──────────────────────────────────────────────────────────────────


def test_closure_is_reflexive_with_depth_zero():
    assert build_closure([_t("x:A")]) == {("x:A", "x:A"): 0}


def test_closure_follows_parents_transitively():
    closure = build_closure([_t("x:A"), _t("x:B", parents=("x:A",)), _t("x:C", parents=("x:B",))])
    assert closure[("x:A", "x:C")] == 2
    assert closure[("x:B", "x:C")] == 1
    assert ("x:C", "x:A") not in closure
    assert len(closure) == 6


def test_closure_keeps_the_minimum_depth_in_a_diamond():
    terms = [
        _t("x:Top"),
        _t("x:Mid", parents=("x:Top",)),
        _t("x:Low", parents=("x:Mid", "x:Top")),
    ]
    assert build_closure(terms)[("x:Top", "x:Low")] == 1


def test_relations_close_separately_and_attributes_take_no_part():
    terms = [
        _t("x:rel", "relation"),
        _t("x:sub", "relation", parents=("x:rel",)),
        _t("x:attr", "attribute"),
    ]
    closure = build_closure(terms)
    assert closure[("x:rel", "x:sub")] == 1
    assert all("x:attr" not in key for key in closure)


def test_unknown_parent_is_refused_by_name():
    with pytest.raises(ClosureError, match="x:Ghost"):
        build_closure([_t("x:A", parents=("x:Ghost",))])


def test_parent_of_another_kind_is_refused():
    with pytest.raises(ClosureError, match="x:r"):
        build_closure([_t("x:r", "relation"), _t("x:A", parents=("x:r",))])


def test_cycle_is_refused():
    with pytest.raises(ClosureError):
        build_closure([_t("x:A", parents=("x:B",)), _t("x:B", parents=("x:A",))])


def test_closure_rows_are_sorted_tuples():
    closure = build_closure([_t("x:B", parents=("x:A",)), _t("x:A")])
    assert closure_rows(closure) == (
        ("x:A", "x:A", 0),
        ("x:A", "x:B", 1),
        ("x:B", "x:B", 0),
    )


# ── admitted sets ────────────────────────────────────────────────────────────


def test_admitted_set_holds_own_terms():
    docs = {"a@1": _doc("a", (_t("x:A"), _t("x:r", "relation")))}
    assert admitted_set("a@1", docs) == {"x:A": "own", "x:r": "own"}


def test_imports_are_transitive_and_name_the_defining_doc():
    docs = {
        "core@1": _doc("core", (_t("x:Core"),), ("std@1",)),
        "std@1": _doc("std", (_t("x:Std"),)),
        "proj@1": _doc("proj", (_t("x:Proj"),), ("core@1",)),
    }
    assert admitted_set("proj@1", docs) == {
        "x:Proj": "own",
        "x:Core": "import:core@1",
        "x:Std": "import:std@1",
    }


def test_own_definition_beats_an_import_and_earlier_import_beats_later():
    docs = {
        "a@1": _doc("a", (_t("x:Shared"),), ("b@1", "c@1")),
        "b@1": _doc("b", (_t("x:Shared"), _t("x:Both"))),
        "c@1": _doc("c", (_t("x:Both"),)),
    }
    assert admitted_set("a@1", docs) == {"x:Shared": "own", "x:Both": "import:b@1"}


def test_import_cycles_are_tolerated():
    docs = {
        "a@1": _doc("a", (_t("x:A"),), ("b@1",)),
        "b@1": _doc("b", (_t("x:B"),), ("a@1",)),
    }
    assert admitted_set("a@1", docs) == {"x:A": "own", "x:B": "import:b@1"}


def test_unknown_ref_or_import_is_refused_by_name():
    docs = {"a@1": _doc("a", (_t("x:A"),), ("missing@9",))}
    with pytest.raises(ClosureError, match="missing@9"):
        admitted_set("a@1", docs)
    with pytest.raises(ClosureError, match="nope@1"):
        admitted_set("nope@1", docs)
