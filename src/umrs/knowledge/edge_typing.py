"""Edge typing rules of the knowledge spine (ADR-081 D-4/D-5, slice KN-S7).

``prepare_edge`` is the edge half of the choke point (``prepare_row`` in
``umrs.knowledge.node_typing`` is the node half). Pure; never raises on bad
input; the caller raises for an enforce-mode graph.

Rules:

1. ``raw = draft.relation_class or draft.relation_type``.
2. ``ctx.resolve(raw, "relation")`` is None → relation ``r4:relatedTo``,
   ``content["rel_raw"] = raw`` and violation ``relation_unknown`` (detail
   ``raw``); a non-empty ``raw`` also gives gap ``("relation", raw)``. No
   domain/range check on the fallback.
3. Resolved → its CURIE. Then, when the term's ``domain_classes`` is
   non-empty and ``source_class`` is known (not None, not
   ``r4:Unclassified``) and ``ctx.under_any(source_class, domain)`` is
   False → violation ``relation_domain`` (detail
   ``"<relation>: <source_class>"``). The same for ``range_classes`` /
   ``target_class`` → ``relation_range``. The relation is kept either way.
4. ``content`` is a new dict (the draft's content, plus ``rel_raw``).

Gaps and violations carry ``draft.edge_id`` as sample/subject id. The
legacy ``relation_type`` column keeps the writer's raw text; this function
only decides ``relation_class``.
"""

from __future__ import annotations

from .term_types import (
    REL_RAW_KEY,
    RELATED_TO,
    UNCLASSIFIED,
    EdgeDraft,
    Gap,
    PreparedEdge,
    TermSpec,
    TypingContext,
    Violation,
)

__all__ = ["prepare_edge"]


def _endpoint_violation(
    ctx: TypingContext,
    term: TermSpec,
    code: str,
    endpoint_class: str | None,
    admitted: tuple[str, ...],
    edge_id: str,
) -> Violation | None:
    """The domain/range violation of one endpoint, or None when it is admitted.

    A check is skipped when the term declares no classes or the endpoint
    class is unknown (None or ``r4:Unclassified``).
    """
    if not admitted or endpoint_class is None or endpoint_class == UNCLASSIFIED:
        return None
    if ctx.under_any(endpoint_class, admitted):
        return None
    return Violation(code, edge_id, f"{term.term_id}: {endpoint_class}")


def prepare_edge(
    ctx: TypingContext,
    draft: EdgeDraft,
    *,
    source_class: str | None = None,
    target_class: str | None = None,
) -> PreparedEdge:
    """Apply the module rules to one edge; never raises on bad input."""
    raw = draft.relation_class or draft.relation_type
    content = dict(draft.content)
    term = ctx.resolve(raw, "relation")
    if term is None:
        content[REL_RAW_KEY] = raw
        gaps = (Gap("relation", raw, draft.edge_id),) if raw else ()
        return PreparedEdge(
            relation_class=RELATED_TO,
            content=content,
            gaps=gaps,
            violations=(Violation("relation_unknown", draft.edge_id, raw),),
        )
    violations = tuple(
        found
        for found in (
            _endpoint_violation(
                ctx,
                term,
                "relation_domain",
                source_class,
                term.domain_classes,
                draft.edge_id,
            ),
            _endpoint_violation(
                ctx,
                term,
                "relation_range",
                target_class,
                term.range_classes,
                draft.edge_id,
            ),
        )
        if found is not None
    )
    return PreparedEdge(
        relation_class=term.term_id, content=content, violations=violations
    )
