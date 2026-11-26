"""Node typing rules of the knowledge spine (ADR-081 D-4/D-5, slice KN-S7).

(Named ``node_typing``, not ``typing`` as decisions-2026-11-24 §5 wrote: a
``typing.py`` in this directory shadows the standard library for every
script run from here, e.g. ``ingest_models_stage.py``, which
``test_knowledge_ingest_models`` caught on 2026-11-27.)

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

import json
import math
from datetime import date, datetime
from typing import Any

from .term_types import (
    BUCKETS,
    CLASS_RAW_KEY,
    UNCLASSIFIED,
    UNDECLARED_KEY,
    Gap,
    NodeDraft,
    OntologyViolationError,
    PreparedNode,
    TermSpec,
    TypingContext,
    Violation,
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
    if datatype is None:
        return True
    simple = {
        "number": _is_finite_number,
        "integer": _is_integer,
        "string": _is_string,
        "boolean": _is_boolean,
        "datetime": _is_datetime,
        "date": _is_date,
        "interval": _is_interval,
        "json": _is_json,
        "node": _is_node,
    }
    check = simple.get(datatype)
    return check(value) if check is not None else False


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_boolean(value: Any) -> bool:
    return isinstance(value, bool)


def _is_node(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_finite_number(value: Any) -> bool:
    """A finite int/float; ``bool`` is not a number here."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_datetime(value: Any) -> bool:
    """An ISO-8601 string with a time part; a trailing ``Z`` is UTC."""
    if not isinstance(value, str) or len(value) <= 10:
        return False
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _is_date(value: Any) -> bool:
    """A ``YYYY-MM-DD`` string ``date.fromisoformat`` parses."""
    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_interval(value: Any) -> bool:
    """Two finite non-bool numbers, first <= second."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    low, high = value
    return _is_finite_number(low) and _is_finite_number(high) and low <= high


def _is_json(value: Any) -> bool:
    """Anything ``json.dumps`` encodes with ``allow_nan=False``."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _resolve_class(
    ctx: TypingContext,
    draft: NodeDraft,
    gaps: list[Gap],
    violations: list[Violation],
) -> tuple[str, bool]:
    """Rule 1: the admitted class CURIE, and whether it was admitted."""
    raw = draft.node_class
    if not raw:
        violations.append(Violation("class_missing", draft.node_id, ""))
        return UNCLASSIFIED, False
    term = ctx.resolve(raw, "class")
    if term is None:
        gaps.append(Gap("class", raw, draft.node_id))
        violations.append(Violation("class_unknown", draft.node_id, raw))
        return UNCLASSIFIED, False
    return term.term_id, True


def _facet(
    ctx: TypingContext,
    draft: NodeDraft,
    class_id: str,
    violations: list[Violation],
) -> str:
    """Rule 2: the writer's facet for ``r4:Unclassified``, else the bucket."""
    if class_id != UNCLASSIFIED:
        term = ctx.terms[class_id]
        return term.bucket or "entity"
    if draft.node_type in BUCKETS:
        return draft.node_type
    violations.append(Violation("bucket_invalid", draft.node_id, draft.node_type))
    return "entity"


def _attr_problem(
    ctx: TypingContext,
    term: TermSpec | None,
    class_id: str,
    value: Any,
    kept: dict[str, Any],
) -> str | None:
    """Rule 3: the violation code that moves this attr, or None to keep it."""
    if term is None:
        return "attr_undeclared"
    if term.domain_classes and not ctx.under_any(class_id, term.domain_classes):
        return "attr_domain"
    if not check_datatype(term.datatype, value):
        return "attr_datatype"
    if term.term_id in kept:
        return "attr_duplicate"
    return None


def _prepare_attrs(
    ctx: TypingContext,
    draft: NodeDraft,
    class_id: str,
    gaps: list[Gap],
    violations: list[Violation],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rule 3: kept attrs keyed by CURIE, and the moved raw-keyed attrs."""
    kept: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for key in sorted(draft.attrs):
        value = draft.attrs[key]
        term = ctx.resolve(key, "attribute")
        problem = _attr_problem(ctx, term, class_id, value, kept)
        if problem is None and term is not None:
            kept[term.term_id] = value
            continue
        moved[key] = value
        if problem == "attr_undeclared":
            gaps.append(Gap("attribute", key, draft.node_id))
        violations.append(Violation(problem or "attr_undeclared", draft.node_id, key))
    return kept, moved


def prepare_row(ctx: TypingContext, draft: NodeDraft) -> PreparedNode:
    """Apply the module rules to one node; never raises on bad input."""
    gaps: list[Gap] = []
    violations: list[Violation] = []
    class_id, admitted = _resolve_class(ctx, draft, gaps, violations)
    node_type = _facet(ctx, draft, class_id, violations)
    attrs, moved = _prepare_attrs(ctx, draft, class_id, gaps, violations)
    content = _prepare_content(draft, class_id, admitted, moved)
    missing: tuple[str, ...] = ()
    if admitted and class_id != UNCLASSIFIED:
        missing = tuple(
            key for key in ctx.required_attributes(class_id) if key not in attrs
        )
    return PreparedNode(
        node_class=class_id,
        node_type=node_type,
        attrs=attrs,
        content=content,
        gaps=tuple(gaps),
        violations=tuple(violations),
        missing_required=missing,
    )


def _prepare_content(
    draft: NodeDraft,
    class_id: str,
    admitted: bool,
    moved: dict[str, Any],
) -> dict[str, Any]:
    """Rule 4: a new content dict; the draft's own mappings stay untouched."""
    content = dict(draft.content)
    if not admitted and class_id == UNCLASSIFIED and draft.node_class:
        content[CLASS_RAW_KEY] = draft.node_class
    if moved:
        existing = content.get(UNDECLARED_KEY)
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(moved)
        content[UNDECLARED_KEY] = merged
    return content
