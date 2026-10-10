"""Shared types of the retrieval core (ADR-093 D-2).

Composition-layer code: nothing here imports ``umrs.api``. The api layer
authenticates, resolves graph access and hands the core resolved graph ids
plus an encoder (and, when staged, a reranker) as callables.

Every lane returns a :class:`LaneResult`; candidates are merged by node id,
so the same chunk found by two lanes is one :class:`Candidate` with two
entries in ``lane_scores``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: ``auto`` follows the site default (env), ``on``/``off`` override it.
STAGE_MODES: tuple[str, ...] = ("auto", "on", "off")
CUTOFF_MODES: tuple[str, ...] = ("adaptive", "fixed")

#: (space/model name, one vector per text or None, reason when None). Texts
#: are queries; the encoder applies the model's query prefix.
BatchEncoder = Callable[[Sequence[str]], "tuple[str, list[list[float]] | None, str]"]

#: (raw logits, one per passage, or None; info: model, revision, reason, ms).
Reranker = Callable[[str, Sequence[str]], "tuple[list[float] | None, dict[str, Any]]"]

#: Object-store read used to hydrate full chunk text: source_ref -> bytes.
FetchBytes = Callable[[str], "bytes | None"]


@dataclass
class Candidate:
    """One chunk node on its way through the pipeline."""

    node_id: str
    artifact_id: str = ""
    chunk_index: int = 0
    content: dict[str, Any] = field(default_factory=dict)  # knowledge_nodes.content
    lane_scores: dict[str, float] = field(default_factory=dict)  # lane -> raw score
    lane_ranks: dict[str, int] = field(default_factory=dict)  # lane -> 1-based rank
    cosine: float | None = None  # dense similarity, also for lexical-only finds
    fused: float = 0.0  # weighted RRF over the lanes
    rerank_logit: float | None = None
    # Filled by hydration:
    text: str = ""
    text_source: str = ""  # search_text | artifact | node | excerpt
    source_name: str = ""
    source_ref: str = ""
    section: str = ""
    chunk_count: int = 0
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class RetrieveSpec:
    """One retrieve request, already authenticated and scoped by the api layer.

    ``graph_ids`` is ``None`` for the whole project (the legacy rule, where
    rows with ``graph_id ''`` stay visible), or the resolved, readable graph
    ids the caller selected. ``threshold`` set means legacy absolute mode.

    Sized by the reading model (ADR-093 D-14): ``partial_k`` is how many
    passages a ``partial`` verdict keeps; ``profile`` and ``reader_model``
    name the profile the api layer resolved (reported as ``lanes.profile``;
    ``""`` = not resolved, no report). ``graph_lane`` is ``on`` or ``off``,
    already resolved by the api layer (the knowledge layer never reads the
    registry); ``graph_gate`` says how it was resolved (``""`` = not
    resolved: the bridge stage reports nothing while off).
    """

    queries: tuple[str, ...]
    project_id: str
    graph_ids: tuple[str, ...] | None = None
    exclude_graph_ids: tuple[str, ...] = ()
    include_ambient: bool = False
    k_max: int = 4
    pool: int = 50
    per_artifact: int = 2
    lexical: str = "auto"
    rerank: str = "auto"
    merge: str = "auto"
    cutoff: str = "adaptive"
    threshold: float | None = None
    trace: bool = False
    partial_k: int = 2
    profile: str = ""
    reader_model: str = ""
    graph_lane: str = "off"
    graph_gate: str = ""

    @property
    def query(self) -> str:
        return self.queries[0] if self.queries else ""


@dataclass
class LaneResult:
    """What one lane found, or why it did not run (ADR-083 D-4: reported)."""

    name: str
    candidates: list[Candidate] = field(default_factory=list)
    status: str = "ok"  # ok | off | unavailable
    reason: str = ""
    ms: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "n": len(self.candidates)}
        if self.reason:
            out["reason"] = self.reason
        out.update(self.info)
        return out
