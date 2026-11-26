"""Node typing rules of the knowledge spine (ADR-081 D-4/D-5, slice KN-S7).

``prepare_row`` is the one pure choke point every node write goes through:
``KnowledgeStore.upsert_typed`` / ``upsert_node`` (typed rows) and the
raw-SQL minting helpers call it with a :class:`TypingContext` resolved once
per graph. It never raises on bad input: it rewrites the row the warn-mode
way and reports what it found; the caller raises
:class:`OntologyViolationError` for an enforce-mode graph.

Rules (in this order; gaps and violations are reported in the same order):

1. Class. ``draft.node_class`` empty → class ``r4:Unclassified`` and
   violation ``class_missing``. Not admitted (``ctx.resolve(raw, "class")``
   is None) → ``r4:Unclassified``, ``content["class_raw"] = raw``, gap
   ``("class", raw)`` and violation ``class_unknown`` (detail ``raw``).
   Admitted → its CURIE. An explicit ``r4:Unclassified`` is admitted.
2. Facet. For ``r4:Unclassified`` the node_type is the writer's
   ``draft.node_type`` when it is one of ``BUCKETS``, else ``entity`` plus
   violation ``bucket_invalid`` (detail: the writer's value). Any other
   class: node_type is the class's ``bucket``.
3. Attributes, in sorted key order. A key resolves through
   ``ctx.resolve(key, "attribute")``; the kept attrs are keyed by the CURIE.
   A key that does not resolve, whose term's ``domain_classes`` is non-empty
   and does not admit the class (``r4:Unclassified`` never satisfies a
   domain), whose value fails :func:`check_datatype`, or whose CURIE is
   already kept, moves to ``content["undeclared"][<raw key>]`` with
   violation ``attr_undeclared`` (+ gap ``("attribute", key)``),
   ``attr_domain``, ``attr_datatype`` or ``attr_duplicate`` respectively.
4. ``content`` is a new dict: the draft's content, plus ``undeclared``
   merged over any dict the draft already had under that key.
5. ``missing_required``: for an admitted class other than
   ``r4:Unclassified``, ``ctx.required_attributes(class)`` minus the kept
   attrs.

Gaps and violations carry ``draft.node_id`` as sample/subject id.
"""

from __future__ import annotations

from typing import Any

from .term_types import (
    NodeDraft,
    OntologyViolationError,
    PreparedNode,
    TypingContext,
)

__all__ = [
    "OntologyViolationError",
    "TypingContext",
    "check_datatype",
    "prepare_row",
]


def check_datatype(datatype: str | None, value: Any) -> bool:
    """Does ``value`` satisfy the attribute ``datatype`` (``DATATYPES``)?

    ``None`` datatype admits anything. ``bool`` is never a number or an
    integer. ``number``: finite int/float. ``integer``: int. ``string``: str.
    ``boolean``: bool. ``datetime``: an ISO-8601 string with a time part
    (``len > 10``; a trailing ``Z`` reads as ``+00:00``). ``date``: a
    ``YYYY-MM-DD`` string. ``interval``: a list or tuple of two finite
    numbers, first <= second. ``json``: any value ``json.dumps`` encodes with
    ``allow_nan=False``. ``node``: a non-empty str. Unknown datatype: False.
    """
    raise NotImplementedError("kn-s7-03")


def prepare_row(ctx: TypingContext, draft: NodeDraft) -> PreparedNode:
    """Apply the module rules to one node; never raises on bad input."""
    raise NotImplementedError("kn-s7-03")
