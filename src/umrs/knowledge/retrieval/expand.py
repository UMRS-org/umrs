"""Read-time small-to-big: tail folding and neighbour merge (ADR-093 D-5, C5).

Two moves, both AFTER relevance is decided, both by deterministic node ids
(``node-chunk-{scope}{artifact}-{i}``, the writer's scheme) — no re-embed:

- **Tail folding** (with ``merge`` on; the pipeline gates it like the
  neighbour merge, D-5). Fixed 256-token windows leave a short last chunk
  (15-140 characters on SciFact) that e5 ranks high on its own while it
  reads as a fragment. A tail whose predecessor chunk is also a candidate is
  folded into it before the cut: one candidate, one ``k`` slot, the better
  position and scores of the two; after hydration its text is appended to
  the predecessor's by span union. A tail alone keeps its slot.
- **Neighbour merge** (``merge=on``; ``auto`` reads ``UMRS_RAG_MERGE``, off
  until the S2 gate passes). Each kept chunk grows by i+1, then i-1, while
  the passage stays within ``max_passage_tokens`` (400; ``num_tokens`` when
  the node records it, else characters / 4) and the pack's growth stays
  within ``pack_tokens`` (``k_max x 450``; the best-ranked passages grow
  first). A neighbour joins one passage only, and only when it shares its
  passage's project, graph, visibility tier and artifact (a neighbour id is
  string arithmetic, and a node can be imported under a chosen id: review
  security-3, 2026-10-10). Passages of one artifact that then touch merge
  into one when the union stays within the passage budget.
  Text comes from ``search_text`` or the md5-verified artifact slice; an
  excerpt is never stitched (its length is not its span). Merged passages
  report ``part i-j/n``.

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .dense import query_rows
from .hydrate import hydrate
from .types import Candidate, FetchBytes

TAIL_CHARS_ENV = "UMRS_RAG_TAIL_CHARS"
MERGE_ENV = "UMRS_RAG_MERGE"
_TAIL_CHARS = 200  # ~48 tokens: C9's merge-back rule for tails
_FULL_SOURCES = ("search_text", "artifact")
_RANGE = "_range"  # content key: the chunk indices a stitched passage covers
# A neighbour joins its host only under the host's own scope: the retrieve
# verb's predicate the host passed, nothing wider (the fetch handle's rule,
# ``api/rag_retrieve._NEIGHBOURS_SQL``).
_SCOPE_SQL = (
    "SELECT n.node_id, n.project_id, n.graph_id, n.visibility, "
    "n.content->>'artifact_id' FROM knowledge_nodes n "
    "WHERE n.node_id = ANY(%s) AND n.content->>'kind' = 'chunk'"
)


def merge_enabled(mode: str) -> bool:
    """``on``/``off`` as asked; ``auto`` follows ``UMRS_RAG_MERGE`` (off)."""
    if mode in ("on", "off"):
        return mode == "on"
    return os.environ.get(MERGE_ENV, "off").strip().lower() == "on"


def tail_chars() -> int:
    try:
        return int(os.environ.get(TAIL_CHARS_ENV, _TAIL_CHARS))
    except ValueError:
        return _TAIL_CHARS


# ── ids, spans, sizes ───────────────────────────────────────────────────────


def id_prefix(candidate: Candidate) -> str | None:
    """``node-chunk-{scope}{artifact}`` — ``None`` for an id off the scheme."""
    suffix = f"-{candidate.chunk_index}"
    if not candidate.node_id.endswith(suffix):
        return None
    return candidate.node_id[: -len(suffix)]


def neighbour_id(candidate: Candidate, index: int) -> str | None:
    """The id of chunk ``index`` of the same artifact and scope."""
    prefix = id_prefix(candidate)
    return f"{prefix}-{index}" if prefix is not None and index >= 0 else None


def chunk_range(candidate: Candidate) -> tuple[int, int]:
    covered = candidate.content.get(_RANGE)
    if isinstance(covered, (list, tuple)) and len(covered) == 2:
        return int(covered[0]), int(covered[1])
    return candidate.chunk_index, candidate.chunk_index


def _span(candidate: Candidate) -> tuple[int, int] | None:
    if candidate.span is not None:
        return candidate.span
    start, end = (
        candidate.content.get("chunk_start"),
        candidate.content.get("chunk_end"),
    )
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None


def is_tail(candidate: Candidate, limit: int | None = None) -> bool:
    """A short non-first chunk: the remainder a fixed window leaves."""
    span = _span(candidate)
    floor = tail_chars() if limit is None else limit
    return candidate.chunk_index > 0 and span is not None and span[1] - span[0] < floor


def is_full(candidate: Candidate) -> bool:
    """Verified text whose length is its span, so offsets can stitch it."""
    span = _span(candidate)
    ok = candidate.text_source in _FULL_SOURCES and span is not None
    return ok and len(candidate.text) == span[1] - span[0]


def tokens(candidate: Candidate) -> int:
    """``num_tokens`` the node records, else characters / 4."""
    counted = candidate.content.get("num_tokens")
    if isinstance(counted, int) and counted > 0 and _RANGE not in candidate.content:
        return counted
    return (len(candidate.text) + 3) // 4


def union_text(pieces: Sequence[Candidate]) -> tuple[tuple[int, int], str]:
    """Span union by offset. A gap between chunks (whitespace the window
    skipped) becomes as many spaces, so the text length stays the span."""
    ordered = sorted(pieces, key=lambda c: _span(c) or (0, 0))
    start, end = _span(ordered[0]) or (0, len(ordered[0].text))
    text = ordered[0].text
    for piece in ordered[1:]:
        p_start, p_end = _span(piece) or (end, end + len(piece.text))
        if p_start >= end:
            text += " " * (p_start - end) + piece.text
        elif p_end > end:
            text += piece.text[end - p_start :]
        end = max(end, p_end)
    return (start, end), text


def _part_label(host: Candidate, first: int, last: int) -> None:
    if first == last:
        return
    count = f"/{host.chunk_count}" if host.chunk_count else ""
    part = f"part {first + 1}-{last + 1}{count}"
    base = str(host.content.get("section") or "")
    host.section = f"{base} · {part}" if base else part


def _stitch(host: Candidate, pieces: Sequence[Candidate]) -> None:
    """Make ``host`` the passage over ``pieces`` (verified, one artifact)."""
    span, text = union_text(pieces)
    first = min(chunk_range(c)[0] for c in pieces)
    last = max(chunk_range(c)[1] for c in pieces)
    host.span, host.text = span, text
    host.content = {**host.content, _RANGE: [first, last]}
    _part_label(host, first, last)


def _absorb_scores(host: Candidate, other: Candidate) -> None:
    """The host keeps the better of each score of the two."""
    host.cosine = max(x for x in (host.cosine, other.cosine, -1.0) if x is not None)
    host.fused = max(host.fused, other.fused)
    for lane, score in other.lane_scores.items():
        host.lane_scores[lane] = max(score, host.lane_scores.get(lane, score))
    for lane, rank in other.lane_ranks.items():
        host.lane_ranks[lane] = min(rank, host.lane_ranks.get(lane, rank))
    if other.rerank_logit is not None:
        mine = (
            host.rerank_logit if host.rerank_logit is not None else other.rerank_logit
        )
        host.rerank_logit = max(other.rerank_logit, mine)


# ── tail folding (before the cut) ───────────────────────────────────────────


def _hosts(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    """tail node id -> the predecessor candidate that hosts it."""
    by_id = {c.node_id: c for c in candidates}
    hosts: dict[str, Candidate] = {}
    for candidate in candidates:
        pred = by_id.get(neighbour_id(candidate, candidate.chunk_index - 1) or "")
        if pred is not None and is_tail(candidate) and not is_tail(pred):
            hosts[candidate.node_id] = pred
    return hosts


def fold_tails(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """``(candidates with tails folded in, {host id: [folded tails]})``.

    The host takes the better position and the better scores of the pair.
    """
    hosts = _hosts(candidates)
    folded: dict[str, list[Candidate]] = {}
    out: list[Candidate] = []
    placed: set[str] = set()
    for candidate in candidates:
        host = hosts.get(candidate.node_id, candidate)
        if host is not candidate:
            _absorb_scores(host, candidate)
            folded.setdefault(host.node_id, []).append(candidate)
        if host.node_id not in placed:
            placed.add(host.node_id)
            out.append(host)
    return out, folded


def folded_members(
    kept: Iterable[Candidate], folded: dict[str, list[Candidate]]
) -> list[Candidate]:
    """The tails folded into ``kept`` (to hydrate with them)."""
    return [tail for host in kept for tail in folded.get(host.node_id, ())]


def attach_folded(
    kept: Sequence[Candidate], folded: dict[str, list[Candidate]]
) -> None:
    """After hydration: append each kept host's folded tails by span union."""
    for host in kept:
        tails = [t for t in folded.get(host.node_id, ()) if is_full(t)]
        if tails and is_full(host):
            _stitch(host, [host, *tails])


# ── neighbour merge (after the cut) ─────────────────────────────────────────


def _wanted(growable: Sequence[Candidate], have: set[str]) -> dict[str, Candidate]:
    """Neighbour candidates to fetch: one past each end of every passage."""
    wanted: dict[str, Candidate] = {}
    for host in growable:
        first, last = chunk_range(host)
        for index in (last + 1, first - 1):
            nid = neighbour_id(host, index)
            if nid and nid not in have:
                wanted.setdefault(nid, Candidate(node_id=nid, chunk_index=index))
    return wanted


@dataclass
class _Growth:
    """What every passage's growth shares: the neighbours already taken and
    the scope (project, graph, tier, artifact) of each host and neighbour."""

    scopes: dict[str, tuple]
    claimed: set[str] = field(default_factory=set)

    def same(self, a: Candidate, b: Candidate) -> bool:
        """``a`` and ``b`` share project, graph, tier and artifact (two
        passages whose scopes were not read count as the same)."""
        return self.scopes.get(a.node_id) == self.scopes.get(b.node_id)

    def joins(self, host: Candidate, neighbour: Candidate | None) -> bool:
        """A fetched, verified neighbour of the host's own scope that no
        other passage took yet."""
        if neighbour is None or neighbour.node_id in self.claimed:
            return False
        known = neighbour.node_id in self.scopes
        return known and self.same(host, neighbour) and is_full(neighbour)


def _scopes(pool: Any, node_ids: Iterable[str]) -> dict[str, tuple]:
    """``node id -> (project, graph, visibility, artifact)`` in one read
    (none for a single passage: nothing to join it to)."""
    ids = sorted(set(node_ids))
    if len(ids) < 2:
        return {}
    rows = query_rows(pool, _SCOPE_SQL, (ids,))
    return {str(row[0]): tuple(row[1:5]) for row in rows}


def _in_scope(
    wanted: dict[str, Candidate],
    hosts: Sequence[Candidate],
    scopes: dict[str, tuple],
) -> dict[str, Candidate]:
    """The neighbours sharing some host's scope: no other is ever hydrated."""
    allowed = {scopes.get(h.node_id) for h in hosts} - {None}
    return {nid: c for nid, c in wanted.items() if scopes.get(nid) in allowed}


def _grow(
    host: Candidate,
    fetched: dict[str, Candidate],
    budget: int,
    growth: _Growth,
    room: float,
) -> int:
    """Grow ``host`` within the passage ``budget`` and the pack's ``room``;
    returns the tokens it added."""
    pieces, size, added = [host], tokens(host), 0
    first, last = chunk_range(host)
    for index in (last + 1, first - 1):
        neighbour = fetched.get(neighbour_id(host, index) or "")
        if not growth.joins(host, neighbour):
            continue
        extra = tokens(neighbour)
        if size + extra <= budget and added + extra <= room:
            pieces.append(neighbour)
            growth.claimed.add(neighbour.node_id)
            size, added = size + extra, added + extra
    if len(pieces) > 1:
        _stitch(host, pieces)
    return added


def _touches(a: Candidate, b: Candidate) -> bool:
    """Same artifact and scope, chunk ranges overlapping or adjacent."""
    if id_prefix(a) is None or id_prefix(a) != id_prefix(b):
        return False
    (a_first, a_last), (b_first, b_last) = chunk_range(a), chunk_range(b)
    return a_first <= b_last + 1 and b_first <= a_last + 1


def _joinable(host: Candidate, candidate: Candidate, budget: int) -> bool:
    """Both verified, and their union within the passage budget."""
    if not (is_full(host) and is_full(candidate)):
        return False
    _span, text = union_text([host, candidate])
    return (len(text) + 3) // 4 <= budget


def _dedupe(kept: list[Candidate], budget: int, growth: _Growth) -> list[Candidate]:
    """Touching verified passages of one artifact and scope become one, at
    the better rank, when the union stays within ``budget`` (else both
    stay); the pack's total is never raised by it."""
    out: list[Candidate] = []
    for candidate in kept:
        host = next(
            (h for h in out if _touches(h, candidate) and growth.same(h, candidate)),
            None,
        )
        if host is None or not _joinable(host, candidate, budget):
            out.append(candidate)
            continue
        _absorb_scores(host, candidate)
        _stitch(host, [host, candidate])
    return out


def _pack_room(kept: Sequence[Candidate], pack_tokens: int | None) -> float:
    """Tokens the pack may still grow by (unbounded without a cap)."""
    if pack_tokens is None:
        return math.inf
    return max(0, pack_tokens - sum(tokens(c) for c in kept))


def merge_neighbours(
    pool: Any,
    kept: list[Candidate],
    *,
    max_passage_tokens: int = 400,
    pack_tokens: int | None = None,
    fetch_bytes: FetchBytes | None = None,
    documents: dict[str, str | None] | None = None,
) -> list[Candidate]:
    """Grow each kept, verified passage by its neighbours (one read for all
    of them), best rank first, then merge touching passages of one artifact.

    ``pack_tokens`` caps the pack (D-5: ``k_max x 450``): growth stops when
    it is reached, so the lowest-ranked passages lose theirs first. A
    neighbour outside its host's project, graph, tier or artifact is never
    hydrated or stitched (one scope read for hosts and neighbours).
    """
    growable = [c for c in kept if is_full(c)]
    wanted = _wanted(growable, {c.node_id for c in kept})
    growth = _Growth(_scopes(pool, [*wanted, *(c.node_id for c in growable)]))
    wanted = _in_scope(wanted, growable, growth.scopes)
    if wanted:
        hydrate(pool, list(wanted.values()), fetch_bytes, documents=documents)
    fetched = {nid: c for nid, c in wanted.items() if c.text}
    room = _pack_room(kept, pack_tokens)
    for host in growable:
        room -= _grow(host, fetched, max_passage_tokens, growth, room)
    return _dedupe(list(kept), max_passage_tokens, growth)
