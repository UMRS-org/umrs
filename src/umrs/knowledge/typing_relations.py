"""Edge typing rules of the knowledge spine (ADR-081 D-4/D-5, slice KN-S7).

``prepare_edge`` is the edge half of the choke point (``prepare_row`` in
``umrs.knowledge.typing`` is the node half). Pure; never raises on bad
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

from .term_types import EdgeDraft, PreparedEdge, TypingContext

__all__ = ["prepare_edge"]


def prepare_edge(
    ctx: TypingContext,
    draft: EdgeDraft,
    *,
    source_class: str | None = None,
    target_class: str | None = None,
) -> PreparedEdge:
    """Apply the module rules to one edge; never raises on bad input."""
    raise NotImplementedError("kn-s7-04")
