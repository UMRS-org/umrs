"""Subsumption closure and admitted term sets (ADR-081 D-2, slice KN-S7).

Pure functions over :class:`TermSpec` / :class:`OntologyDoc`; the seed
migration and the W2b registration route both build their
``knowledge_term_closure`` and ``knowledge_ontology_terms`` rows here.

- :func:`build_closure` — reflexive closure over the ``parents`` DAG of the
  class and relation terms (attributes take no part). Key
  ``(ancestor, descendant)``, value the minimum number of ``parents`` hops
  (0 for the term itself). Raises :class:`ClosureError` when a parent is not
  among the given terms, when a parent is of another kind than its child,
  or when the hierarchy has a cycle (two distinct terms each above the
  other).
- :func:`closure_rows` — the closure as sorted ``(ancestor, descendant,
  depth)`` tuples (sorted by ancestor, then descendant).
- :func:`admitted_set` — ``term_id -> via`` for one ontology ref: the doc's
  own terms (``"own"``), then each import in declared order, depth first,
  transitively; an imported term's ``via`` is ``"import:<ref of the doc
  that defines it>"``. The first assignment wins (own beats imported; an
  earlier import beats a later one). Import cycles are tolerated (each doc
  is visited once). An unknown ref (the root or any import) raises
  :class:`ClosureError` naming it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from .term_types import CLOSURE_KINDS, OntologyDoc, TermSpec

__all__ = ["ClosureError", "admitted_set", "build_closure", "closure_rows"]


class ClosureError(ValueError):
    """The term hierarchy or the import graph cannot be closed."""


def build_closure(terms: Iterable[TermSpec]) -> dict[tuple[str, str], int]:
    """Reflexive minimum-depth closure over class and relation parents."""
    by_id = {term.term_id: term for term in terms if term.kind in CLOSURE_KINDS}
    closure: dict[tuple[str, str], int] = {}
    for term in by_id.values():
        for ancestor, depth in _ancestors_of(term, by_id).items():
            closure[(ancestor, term.term_id)] = depth
    return closure


def _ancestors_of(term: TermSpec, by_id: Mapping[str, TermSpec]) -> dict[str, int]:
    """Breadth-first walk up ``term``'s parents: ancestor -> minimum hops."""
    depths = {term.term_id: 0}
    queue = deque([term.term_id])
    while queue:
        current = queue.popleft()
        for parent in _checked_parents(by_id[current], by_id):
            if parent == term.term_id:
                raise ClosureError(f"term {term.term_id}: cycle through {current}")
            if parent not in depths:
                depths[parent] = depths[current] + 1
                queue.append(parent)
    return depths


def _checked_parents(term: TermSpec, by_id: Mapping[str, TermSpec]) -> tuple[str, ...]:
    """``term``'s parents, each present in ``by_id`` and of the same kind."""
    for parent_id in term.parents:
        parent = by_id.get(parent_id)
        if parent is None:
            raise ClosureError(f"term {term.term_id}: unknown parent {parent_id}")
        if parent.kind != term.kind:
            raise ClosureError(
                f"term {term.term_id} ({term.kind}): parent {parent_id} "
                f"is of kind {parent.kind}"
            )
    return term.parents


def closure_rows(
    closure: Mapping[tuple[str, str], int],
) -> tuple[tuple[str, str, int], ...]:
    """``(ancestor, descendant, depth)`` rows, sorted."""
    return tuple(
        (ancestor, descendant, depth)
        for (ancestor, descendant), depth in sorted(closure.items())
    )


def admitted_set(ref: str, docs: Mapping[str, OntologyDoc]) -> dict[str, str]:
    """``term_id -> via`` admitted by the ontology ``ref`` (imports included)."""
    admitted: dict[str, str] = {}
    _admit_doc(ref, docs, admitted, set(), own=True)
    return admitted


def _admit_doc(
    ref: str,
    docs: Mapping[str, OntologyDoc],
    admitted: dict[str, str],
    visited: set[str],
    *,
    own: bool,
) -> None:
    """Admit one doc's terms, then its imports depth first, each doc once."""
    if ref in visited:
        return
    visited.add(ref)
    doc = docs.get(ref)
    if doc is None:
        raise ClosureError(f"unknown ontology ref {ref}")
    via = "own" if own else f"import:{ref}"
    for term in doc.terms:
        admitted.setdefault(term.term_id, via)
    for imported in doc.imports:
        _admit_doc(imported, docs, admitted, visited, own=False)
