"""Retrieval DTOs — what one retrieve call returns (ADR-093 D-2, D-3).

Stdlib only (ADR-005). UMRS builds these from its retrieval core and serves
them as the JSON of ``GET /umrs/rag/retrieve``; R4, the context-pack renderer
and the eval read them back with ``from_dict``. The shapes are the wire
contract, so a field is added, never renamed.

Relevance and sufficiency are decided by UMRS (ADR-093 D-3; CONSTITUTION
§1.3): a consumer reads ``sufficiency.status``, never re-cuts ``passages``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

#: Sufficiency verdicts, best first. ``unknown`` is the offline reader's: it
#: holds no calibrated null, so it never claims either way (D-12).
SUFFICIENCY_STATUSES: tuple[str, ...] = ("sufficient", "partial", "insufficient", "unknown")

#: Where a passage's text came from: the stored chunk body, the md5-verified
#: slice of the artifact, or the node's 500-character excerpt.
TEXT_SOURCES: tuple[str, ...] = ("search_text", "artifact", "node", "excerpt")

#: The lane of a graph bridge (ADR-093 D-14): a passage UMRS appends AFTER the
#: cut on a site whose extraction model is large, never one of the cut's own.
GRAPH_BRIDGE_LANE = "graph"


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage the cut kept, best first by ``rank`` (1-based)."""

    rank: int
    source_id: str  # the chunk node id; the fetch handle's key
    artifact_id: str
    source_name: str  # trusted artifact metadata, never passage text
    text: str
    section: str = ""
    chunk_index: int = 0
    chunk_count: int = 0
    span: tuple[int, int] | None = None  # character offsets in the artifact
    text_source: str = "artifact"
    scores: dict[str, float] = field(default_factory=dict)  # dense, lexical, fused, rerank_*
    lanes: tuple[str, ...] = ()  # the lanes that found it

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["span"] = list(self.span) if self.span else None
        out["lanes"] = list(self.lanes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedPassage:
        return cls(
            rank=_int(data, "rank"),
            source_id=_str(data, "source_id"),
            artifact_id=_str(data, "artifact_id"),
            source_name=_str(data, "source_name"),
            text=_str(data, "text"),
            section=_str(data, "section"),
            chunk_index=_int(data, "chunk_index"),
            chunk_count=_int(data, "chunk_count"),
            span=_span(data.get("span")),
            text_source=_str(data, "text_source", "artifact"),
            scores=_floats(data.get("scores")),
            lanes=_strings(data.get("lanes")),
        )


@dataclass(frozen=True)
class RetrievalSufficiency:
    """Whether the passages answer anything, and on what basis.

    ``basis`` names the rule (``z_null``, ``lexical``, ``rerank_p``,
    ``threshold``, ``no_vectors``, ``offline``); ``value`` is what it measured
    and ``threshold`` what it was held to, so a reader can see the margin.
    """

    status: str = "unknown"
    basis: str = ""
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RetrievalSufficiency:
        data = data or {}
        status = str(data.get("status") or "unknown")
        return cls(
            status=status if status in SUFFICIENCY_STATUSES else "unknown",
            basis=str(data.get("basis") or ""),
            value=_float_or_none(data.get("value")),
            threshold=_float_or_none(data.get("threshold")),
            detail=str(data.get("detail") or ""),
        )


@dataclass(frozen=True)
class RetrieveResult:
    """One retrieve call: the kept passages, the verdict and how it was reached."""

    query: str
    passages: tuple[RetrievedPassage, ...] = ()
    sufficiency: RetrievalSufficiency = field(default_factory=RetrievalSufficiency)
    searched: int = 0  # candidate rows in scope (0 = nothing to search)
    space: str = ""  # the embedding space the dense lane used
    lanes: dict[str, dict[str, Any]] = field(default_factory=dict)  # lane -> {status, reason, n}
    timing_ms: dict[str, float] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)  # only when asked (trace=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "passages": [p.to_dict() for p in self.passages],
            "sufficiency": self.sufficiency.to_dict(),
            "searched": self.searched,
            "space": self.space,
            "lanes": dict(self.lanes),
            "timing_ms": dict(self.timing_ms),
            "trace": dict(self.trace),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrieveResult:
        return cls(
            query=_str(data, "query"),
            passages=_passages(data.get("passages")),
            sufficiency=RetrievalSufficiency.from_dict(data.get("sufficiency")),
            searched=_int(data, "searched"),
            space=_str(data, "space"),
            lanes=_lanes(data.get("lanes")),
            timing_ms=_floats(data.get("timing_ms")),
            trace=dict(data.get("trace") or {}),
        )


# ── graph bridges (ADR-093 D-14) ─────────────────────────────────────────────


def is_graph_bridge(passage: RetrievedPassage) -> bool:
    """A passage the graph lane alone found, appended after the cut (D-14)."""
    return tuple(passage.lanes) == (GRAPH_BRIDGE_LANE,)


def bounded_passages(passages: Sequence[RetrievedPassage], limit: int) -> list[RetrievedPassage]:
    """The cut's first ``limit`` passages, then every graph bridge.

    A reader's passage bound (a pack's ``k_max``, the partial cap) bounds the
    cut. The bridges ride after it, as UMRS appends them, so a bound never
    trades a bridge for a cut passage or the reverse. A token cap still
    bounds the whole pack.
    """
    cut = [p for p in passages if not is_graph_bridge(p)]
    return [*cut[: max(0, limit)], *(p for p in passages if is_graph_bridge(p))]


# ── from_dict helpers: a missing or null field reads as its empty value ──────


def _str(data: dict[str, Any], key: str, default: str = "") -> str:
    return str(data.get(key) or default)


def _int(data: dict[str, Any], key: str) -> int:
    return int(data.get(key) or 0)


def _span(value: Any) -> tuple[int, int] | None:
    return (int(value[0]), int(value[1])) if value else None


def _floats(value: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in (value or {}).items()}


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in value or ())


def _passages(value: Any) -> tuple[RetrievedPassage, ...]:
    return tuple(RetrievedPassage.from_dict(p) for p in value or ())


def _lanes(value: Any) -> dict[str, dict[str, Any]]:
    return {str(k): dict(v) for k, v in (value or {}).items()}


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
