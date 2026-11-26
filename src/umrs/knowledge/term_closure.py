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

from collections.abc import Iterable, Mapping

from .term_types import OntologyDoc, TermSpec

__all__ = ["ClosureError", "admitted_set", "build_closure", "closure_rows"]


class ClosureError(ValueError):
    """The term hierarchy or the import graph cannot be closed."""


def build_closure(terms: Iterable[TermSpec]) -> dict[tuple[str, str], int]:
    """Reflexive minimum-depth closure over class and relation parents."""
    raise NotImplementedError("kn-s7-02")


def closure_rows(
    closure: Mapping[tuple[str, str], int],
) -> tuple[tuple[str, str, int], ...]:
    """``(ancestor, descendant, depth)`` rows, sorted."""
    raise NotImplementedError("kn-s7-02")


def admitted_set(ref: str, docs: Mapping[str, OntologyDoc]) -> dict[str, str]:
    """``term_id -> via`` admitted by the ontology ``ref`` (imports included)."""
    raise NotImplementedError("kn-s7-02")
