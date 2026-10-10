"""One scored retrieve path for every UMRS consumer (ADR-093 D-2).

``retrieve(pool, spec, *, encode, rerank, fetch_bytes)`` runs, in order:

1. encode — every query variant in one batch;
2. dense — exact cosine, a pool of ``spec.pool`` per variant (``dense.py``);
3. lexical + ident — the FTS and identifier lanes (``lexical.py``), once
   per query variant (``lex#i``/``ident#i`` when there are several);
   lexical-only finds get their cosines in ONE statement;
4. fuse — weighted RRF across variants and lanes (``fusion.py``);
5. fold tails (only with ``merge`` on: read-time small-to-big, D-5, stays
   behind the S2 knob), then at most ``per_artifact`` chunks per artifact
   within the top 30;
6. hydrate — full text and trusted labels (``hydrate.py``);
7. rerank — the cross-encoder, when on and staged (``rerank.py``);
8. cut + sufficiency (``cutoff.py``, null from ``calibrate.py``); a
   ``partial`` verdict keeps ``spec.partial_k`` passages (the profile's);
9. neighbour merge, when on (``expand.py``);
10. graph bridges, when ``spec.graph_lane`` is on (``graph_lane.py``): at
    most two passages from documents the kept ones did not come from,
    appended after them, never displacing one (D-14, large sites only).

The profile that sized the spec (ADR-093 D-14) is reported as
``lanes.profile = {name, reader_model}`` when the api layer resolved one.

A lane that fails reports ``unavailable`` with the exception's TYPE only; the
message (hosts, users, SQL) goes to the server log, never to the caller.

The lanes written beside this module (lexical, fusion, rerank) are imported
lazily: a missing module makes its stage ``unavailable`` with the import
error as the reason (ADR-083 D-4), never a silent fallback. Legacy mode
(``spec.threshold`` set) runs the pre-ADR-093 path exactly: dense top ``k``,
the absolute filter, nothing else.

**No chat-model call happens anywhere on this path** (acceptance receipt 3):
relevance and sufficiency are scored here, never generated.

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from r4_contracts.retrieval import (
    RetrievalSufficiency,
    RetrievedPassage,
    RetrieveResult,
)

from . import calibrate, cutoff, dense, expand, graph_lane, hydrate, profiles
from .types import (
    BatchEncoder,
    Candidate,
    FetchBytes,
    LaneResult,
    Reranker,
    RetrieveSpec,
)

logger = logging.getLogger(__name__)

RERANK_POOL = 30
RERANK_PER_ARTIFACT = 3
LEXICAL_LIMIT = 50
IDENT_LIMIT = 20
MAX_PASSAGE_TOKENS = 400
#: D-5: the merged pack stays within ``k_max x PACK_TOKENS_PER_PASSAGE``.
PACK_TOKENS_PER_PASSAGE = 450
#: A ``partial`` verdict shows at most this many passages to a small reader
#: (ADR-093 D-3): weak evidence comes with few distractors. Decided here, so
#: the R4 router, the loop store and /rag/query apply one rule; the spec's
#: ``partial_k`` carries the reading model's profile value (D-14).
PARTIAL_K = 2

#: stage -> (module under umrs.knowledge.retrieval, attribute)
_OPTIONAL: dict[str, tuple[str, str]] = {
    "lex": ("lexical", "lexical_lane"),
    "ident": ("lexical", "ident_lane"),
    "lex_strength": ("lexical", "lexical_hit_strength"),
    "fuse": ("fusion", "fuse"),
    "rerank": ("rerank", "rerank_lane"),
    "rerank_enabled": ("rerank", "rerank_enabled"),
}


class Unavailable:
    """A stage whose module could not be imported; it carries the reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __bool__(self) -> bool:
        return False


def _optional(module: str, attr: str) -> Callable[..., Any] | Unavailable:
    try:
        found = importlib.import_module(f"umrs.knowledge.retrieval.{module}")
        return getattr(found, attr)
    except (ImportError, AttributeError) as exc:
        return Unavailable(f"{module}.{attr} unavailable ({type(exc).__name__}: {exc})")


def default_lanes() -> dict[str, Any]:
    table: dict[str, Any] = {
        "dense": dense.dense_lane,
        "cosines": dense.cosines_for,
        "hydrate": hydrate.hydrate,
        "null": calibrate.null_for,
        "merge": expand.merge_neighbours,
        "graph": graph_lane.bridge_lane,
    }
    for stage, (module, attr) in _OPTIONAL.items():
        table[stage] = _optional(module, attr)
    return table


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _stage_report(stage: Any, *, off: bool, off_reason: str) -> dict[str, Any] | None:
    """The report of a stage that does not run, else ``None``."""
    if off:
        return {"status": "off", "n": 0, "reason": off_reason}
    if isinstance(stage, Unavailable):
        return {"status": "unavailable", "n": 0, "reason": stage.reason}
    return None


class _Run:
    """One retrieve call: the stages, their timings and their reports."""

    def __init__(
        self,
        pool: Any,
        spec: RetrieveSpec,
        *,
        encode: BatchEncoder,
        reranker: Reranker | None,
        fetch_bytes: FetchBytes | None,
        lanes: dict[str, Any],
    ) -> None:
        self.pool, self.spec, self.encode = pool, spec, encode
        self.reranker, self.fetch_bytes, self.lanes = reranker, fetch_bytes, lanes
        self.timing: dict[str, float] = {}
        self.report: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, str | None] = {}
        self.model, self.vectors, self.searched = "", None, 0
        self.stamp: str | None = None  # the scope's last re-embed (null cache key)
        self.trace: dict[str, Any] = {}

    # ── 1-2: encode and dense ───────────────────────────────────────────

    def encode_queries(self) -> None:
        started = time.perf_counter()
        self.model, self.vectors, reason = self.encode(list(self.spec.queries))
        self.timing["encode"] = _ms(started)
        if not self.vectors:
            self.report["dense"] = {
                "status": "unavailable",
                "n": 0,
                "reason": reason or "no query vector",
            }

    def dense(self) -> list[Candidate]:
        if not self.vectors:
            return []
        started = time.perf_counter()
        limit = self.spec.k_max if self.legacy else self.spec.pool
        lane: LaneResult = self.lanes["dense"](
            self.pool, self.vectors, self.model, self.spec, limit=limit
        )
        self.timing["dense"] = _ms(started)
        stamp = lane.info.pop("stamp", None)
        self.stamp = None if stamp is None else str(stamp)
        self.report["dense"] = lane.report()
        in_scope = lane.info.get("in_scope")
        self.searched = int(in_scope) if in_scope is not None else len(lane.candidates)
        return lane.candidates

    @property
    def legacy(self) -> bool:
        return self.spec.threshold is not None

    # ── 3: lexical lanes ────────────────────────────────────────────────

    def lexical(self) -> list[LaneResult]:
        off = self.legacy or self.spec.lexical == "off"
        reason = "legacy threshold mode" if self.legacy else "lexical=off"
        started = time.perf_counter()
        found = [
            lane
            for name, limit in (("lex", LEXICAL_LIMIT), ("ident", IDENT_LIMIT))
            for lane in self._lexical_lane(name, limit, off, reason)
        ]
        self.timing["lexical"] = _ms(started)
        return found

    def _lexical_lane(
        self, name: str, limit: int, off: bool, reason: str
    ) -> list[LaneResult]:
        """The lane once per query variant (C1: RRF across variants and lanes),
        so the terms a history-aware retry adds reach it too (D-8)."""
        stage = self.lanes.get(name)
        skipped = _stage_report(stage, off=off, off_reason=reason)
        if skipped is not None:
            self.report[name] = skipped
            return []
        queries = self.spec.queries or ("",)
        results = [
            _call_lane(stage, name, self.pool, q, self.spec, limit) for q in queries
        ]
        self.report[name] = _variant_report(results)
        return _usable(results, name)

    def fill_cosines(
        self, lanes: list[LaneResult], known: dict[str, Candidate]
    ) -> None:
        """Lexical finds take the dense lane's cosine and identity when it
        found them too; the rest get their cosines in ONE statement."""
        missing = sorted(_adopt_dense(lanes, known))
        if not missing or not self.vectors:
            return
        cosines = self.lanes["cosines"](self.pool, missing, self.vectors, self.model)
        for lane in lanes:
            for candidate in lane.candidates:
                candidate.cosine = cosines.get(candidate.node_id, candidate.cosine)

    # ── 4: fusion ───────────────────────────────────────────────────────

    def fuse(
        self, dense_found: list[Candidate], lexical: list[LaneResult]
    ) -> list[Candidate]:
        started = time.perf_counter()
        rankings = _rankings(dense_found, lexical, len(self.vectors or ()))
        fused = self._fuse_rankings(rankings)
        self.timing["fuse"] = _ms(started)
        return fused

    def _fuse_rankings(self, rankings: dict[str, list[Candidate]]) -> list[Candidate]:
        if len(rankings) <= 1:
            single = next(iter(rankings.values()), [])
            for rank, candidate in enumerate(single, start=1):
                candidate.fused = 1.0 / (60 + rank)
            self.report["fusion"] = {
                "status": "off",
                "n": len(single),
                "reason": "one ranking",
            }
            return list(single)
        fuse = self.lanes.get("fuse")
        if isinstance(fuse, Unavailable) or fuse is None:
            reason = fuse.reason if isinstance(fuse, Unavailable) else "no fusion stage"
            self.report["fusion"] = {"status": "unavailable", "n": 0, "reason": reason}
            return _concatenate(rankings)
        fused = list(fuse(rankings))
        self.report["fusion"] = {
            "status": "ok",
            "n": len(fused),
            "rankings": sorted(rankings),
        }
        return fused

    # ── 5-7: fold, cap, hydrate, rerank ─────────────────────────────────

    def hydrate(self, candidates: list[Candidate]) -> None:
        if not candidates:
            return
        started = time.perf_counter()
        self.lanes["hydrate"](
            self.pool, candidates, self.fetch_bytes, documents=self.documents
        )
        self.timing["hydrate"] = round(
            self.timing.get("hydrate", 0.0) + _ms(started), 2
        )

    def rerank(self, pool30: list[Candidate]) -> tuple[list[Candidate], bool]:
        stage, enabled = self.lanes.get("rerank"), self.lanes.get("rerank_enabled")
        on = (
            bool(enabled(self.spec.rerank))
            if callable(enabled)
            else self.spec.rerank == "on"
        )
        skipped = _stage_report(
            stage, off=not on, off_reason=f"rerank={self.spec.rerank}"
        )
        if skipped is not None:
            self.report["rerank"] = skipped
            return pool30, False
        self.hydrate(pool30)
        started = time.perf_counter()
        lane: LaneResult = stage(
            self.spec.query,
            pool30,
            self.reranker,
            limit=RERANK_POOL,
            per_artifact=RERANK_PER_ARTIFACT,
        )
        self.timing["rerank"] = _ms(started)
        self.report["rerank"] = lane.report()
        if lane.status != "ok" or not lane.candidates:
            return pool30, False
        return lane.candidates, True

    # ── 8: cut and sufficiency ──────────────────────────────────────────

    def null(self, mode: str) -> Any:
        skipped = _null_skipped(mode, self.searched, bool(self.vectors))
        if skipped:
            self.report["null"] = {"status": "off", "n": 0, "reason": skipped}
            return None
        started = time.perf_counter()
        try:
            stats = self.lanes["null"](
                self.pool,
                self.spec,
                self.encode,
                self.model,
                vectors=self.searched,
                stamp=self.stamp,
            )
        except Exception as exc:  # noqa: BLE001 — no null means only the lexical rule
            logger.warning("null calibration failed: %s", exc)
            self.report["null"] = {
                "status": "unavailable",
                "n": 0,
                "reason": f"null calibration failed ({type(exc).__name__})",
            }
            return None
        self.timing["null"] = _ms(started)
        self.report["null"] = (
            {"status": "ok", **stats.to_dict()}
            if stats is not None
            else {"status": "unavailable", "n": 0, "reason": "probes not encoded"}
        )
        return stats

    def lexical_evidence(
        self, kept: list[Candidate], mode: str
    ) -> cutoff.LexicalEvidence:
        """The lexical rule's inputs over the KEPT passages: coverage of the
        top one by the best-covered variant, and (fused) the largest single
        matched IDF among them. What each may decide is ``cutoff``'s rule."""
        measure = self.lanes.get("lex_strength")
        strength = (
            float(measure(kept, per_term=True))
            if mode == "fused" and callable(measure)
            else None
        )
        return cutoff.lexical_evidence(self.spec.queries, kept, strength=strength)

    # ── 9: merge ────────────────────────────────────────────────────────

    def merge(self, kept: list[Candidate]) -> list[Candidate]:
        if not expand.merge_enabled(self.spec.merge):
            self.report["merge"] = {
                "status": "off",
                "n": 0,
                "reason": f"merge={self.spec.merge}",
            }
            return kept
        started = time.perf_counter()
        merged = self.lanes["merge"](
            self.pool,
            kept,
            max_passage_tokens=MAX_PASSAGE_TOKENS,
            pack_tokens=max(1, self.spec.k_max) * PACK_TOKENS_PER_PASSAGE,
            fetch_bytes=self.fetch_bytes,
            documents=self.documents,
        )
        self.timing["merge"] = _ms(started)
        self.report["merge"] = {"status": "ok", "n": len(merged)}
        return merged

    # ── 10: graph bridges (D-14) ────────────────────────────────────────

    def bridge(
        self, kept: list[Candidate], verdict: RetrievalSufficiency
    ) -> list[Candidate]:
        """``kept`` plus at most two bridge passages, appended, when the
        graph lane is on and the scorer found evidence to bridge from."""
        skipped = _bridge_skipped(self, verdict)
        if skipped is not None:
            if skipped:
                self.report["graph"] = skipped
            return kept
        started = time.perf_counter()
        vector = self.vectors[0] if self.vectors else None
        lane: LaneResult = self.lanes["graph"](
            self.pool, self.spec, vector=vector, model=self.model, kept=kept
        )
        self.hydrate(lane.candidates)
        found = _with_text(lane.candidates)
        self.timing["graph"] = _ms(started)
        self.report["graph"] = _bridge_report(lane, found)
        return [*kept, *found]


# ── helpers ─────────────────────────────────────────────────────────────────


def _rankings(
    dense_found: list[Candidate], lexical: list[LaneResult], variants: int
) -> dict[str, list[Candidate]]:
    """The rankings fusion merges: one per dense variant, one per lexical lane."""
    rankings: dict[str, list[Candidate]] = {}
    if variants > 1:
        for i in range(variants):
            lane = f"dense#{i}"
            ranked = sorted(
                (c for c in dense_found if lane in c.lane_ranks),
                key=lambda c: c.lane_ranks[lane],
            )
            rankings[lane] = ranked
    elif dense_found:
        rankings["dense"] = list(dense_found)
    for lane in lexical:
        rankings[lane.name] = list(lane.candidates)
    return {name: ranked for name, ranked in rankings.items() if ranked}


def _call_lane(
    stage: Callable[..., LaneResult],
    name: str,
    pool: Any,
    query: str,
    spec: RetrieveSpec,
    limit: int,
) -> LaneResult:
    """One lane run; a failure is reported by its type, logged in full."""
    try:
        return stage(pool, query, spec, limit=limit)
    except Exception as exc:  # noqa: BLE001 — a lane reports, it never raises
        logger.warning("%s lane failed: %s", name, exc)
        reason = f"{name} lane failed ({type(exc).__name__})"
        return LaneResult(name=name, status="unavailable", reason=reason)


def _variant_report(results: list[LaneResult]) -> dict[str, Any]:
    """The first variant's report; with several, the total and each one's."""
    report = results[0].report()
    if len(results) > 1:
        report["n"] = sum(len(r.candidates) for r in results)
        report["variants"] = [
            {"status": r.status, "n": len(r.candidates)} for r in results
        ]
    return report


def _usable(results: list[LaneResult], name: str) -> list[LaneResult]:
    """The rankings that found something; ``name#i`` per variant when there
    are several (fusion gives them their base lane's weight)."""
    if len(results) > 1:
        for index, lane in enumerate(results):
            lane.name = f"{name}#{index}"
    return [lane for lane in results if lane.status == "ok" and lane.candidates]


def _adopt_dense(lanes: list[LaneResult], known: dict[str, Candidate]) -> set[str]:
    """Copy what the dense lane knows onto lexical candidates of the same
    node; return the ids it never saw."""
    missing: set[str] = set()
    for candidate in (c for lane in lanes for c in lane.candidates):
        twin = known.get(candidate.node_id)
        if twin is None:
            if candidate.cosine is None:
                missing.add(candidate.node_id)
            continue
        candidate.cosine = twin.cosine
        candidate.content = candidate.content or twin.content
        candidate.artifact_id = candidate.artifact_id or twin.artifact_id
        candidate.chunk_index = candidate.chunk_index or twin.chunk_index
    return missing


def _concatenate(rankings: dict[str, list[Candidate]]) -> list[Candidate]:
    """Without fusion: each ranking in turn, a node once (its first place)."""
    seen: dict[str, Candidate] = {}
    for ranked in rankings.values():
        for candidate in ranked:
            seen.setdefault(candidate.node_id, candidate)
    return list(seen.values())


def cap_per_artifact(
    candidates: Sequence[Candidate], per_artifact: int, *, top: int = RERANK_POOL
) -> list[Candidate]:
    """The first ``top`` candidates with at most ``per_artifact`` per artifact."""
    taken: dict[str, int] = {}
    out: list[Candidate] = []
    for candidate in candidates:
        key = candidate.artifact_id or candidate.node_id
        if taken.get(key, 0) >= per_artifact:
            continue
        taken[key] = taken.get(key, 0) + 1
        out.append(candidate)
        if len(out) >= top:
            break
    return out


def _cut_mode(run: _Run, reranked: bool, lexical: list[LaneResult]) -> str:
    if run.legacy:
        return "legacy"
    if reranked:
        return "rerank"
    return "fused" if lexical else "dense"


def _scores(candidate: Candidate) -> dict[str, float]:
    values = {
        "dense": candidate.cosine,
        "lexical": candidate.lane_scores.get("lex"),
        "ident": candidate.lane_scores.get("ident"),
        "fused": candidate.fused or None,
        "rerank_logit": candidate.rerank_logit,
        "graph": candidate.lane_scores.get("graph"),
    }
    if candidate.rerank_logit is not None:
        values["rerank_p"] = cutoff.sigmoid(candidate.rerank_logit)
    return {name: round(float(v), 6) for name, v in values.items() if v is not None}


def _lanes_of(candidate: Candidate) -> tuple[str, ...]:
    names = {lane.split("#", 1)[0] for lane in candidate.lane_ranks}
    return tuple(sorted(names))


def to_passage(rank: int, candidate: Candidate) -> RetrievedPassage:
    return RetrievedPassage(
        rank=rank,
        source_id=candidate.node_id,
        artifact_id=candidate.artifact_id,
        source_name=candidate.source_name or candidate.artifact_id or candidate.node_id,
        text=candidate.text,
        section=candidate.section,
        chunk_index=candidate.chunk_index,
        chunk_count=candidate.chunk_count,
        span=candidate.span,
        text_source=candidate.text_source or "excerpt",
        scores=_scores(candidate),
        lanes=_lanes_of(candidate),
    )


def _trace_rows(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": c.node_id,
            "artifact_id": c.artifact_id,
            "chunk_index": c.chunk_index,
            "scores": _scores(c),
            "lanes": list(_lanes_of(c)),
        }
        for c in candidates[:RERANK_POOL]
    ]


def _null_skipped(mode: str, searched: int, encoded: bool) -> str:
    """Why the dense null is not consulted, or ``""`` when it is."""
    if mode == "rerank":
        return "rerank mode judges by p_1"
    if not encoded:
        return "no query vector"
    if searched < calibrate.MIN_VECTORS:
        return f"fewer than {calibrate.MIN_VECTORS} vectors in scope"
    return ""


def _unscored(run: _Run) -> RetrievalSufficiency:
    reason = run.report.get("dense", {}).get("reason") or "no query vector"
    return RetrievalSufficiency(
        status="insufficient", basis="unavailable", detail=str(reason)
    )


def _with_text(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.text]


def _off(reason: str) -> dict[str, Any]:
    return {"status": "off", "n": 0, "reason": reason}


def _bridge_skipped(run: _Run, verdict: RetrievalSufficiency) -> dict[str, Any] | None:
    """Why the graph bridge stage does not run (``{}``: off and unresolved,
    so nothing is reported), or ``None`` when it runs."""
    spec = run.spec
    if spec.graph_lane != "on":
        return _off(spec.graph_gate) if spec.graph_gate else {}
    if run.legacy:
        return _off("legacy threshold mode")
    if verdict.status == "insufficient":
        reason = "insufficient verdict: no evidence to bridge from"
        return {"status": "ok", "n": 0, "fired": False, "reason": reason}
    return None


def _bridge_report(lane: LaneResult, found: list[Candidate]) -> dict[str, Any]:
    """The lane's report over the bridges that kept their text."""
    ids = {c.node_id for c in found}
    report = lane.report()
    report["n"] = len(found)
    if "bridges" in report:
        report["bridges"] = [b for b in report["bridges"] if b.get("source_id") in ids]
    return report


def _legacy(
    run: _Run, found: list[Candidate]
) -> tuple[list[Candidate], RetrievalSufficiency]:
    kept = cutoff.cut(found, run.spec, mode="legacy")
    run.hydrate(kept)
    kept = _with_text(kept)
    verdict = cutoff.sufficiency(
        kept,
        mode="legacy",
        null=None,
        lexical_hit=False,
        searched=run.searched,
        threshold=run.spec.threshold,
    )
    if not found and run.report.get("dense", {}).get("status") != "ok":
        verdict = _unscored(run)
    return kept, verdict


def _fold(
    run: _Run, fused: list[Candidate]
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """Tail folding is read-time small-to-big (D-5): it runs only with
    ``merge`` on, like the neighbour merge, so the S1 arm ships none of it.
    A folded host takes the better scores of the pair: the passage holds
    the tail's text, so its best-matching part is at least the tail's."""
    if not expand.merge_enabled(run.spec.merge):
        return fused, {}
    return expand.fold_tails(fused)


def _cut(
    run: _Run, found: list[Candidate], lexical: list[LaneResult]
) -> tuple[list[Candidate], str]:
    """Steps 4-8a: fuse, fold, cap, rerank, cut, hydrate the kept."""
    folded_list, folded = _fold(run, run.fuse(found, lexical))
    pool30 = cap_per_artifact(folded_list, run.spec.per_artifact)
    ordered, reranked = run.rerank(pool30)
    mode = _cut_mode(run, reranked, lexical)
    started = time.perf_counter()
    kept = cutoff.cut(ordered, run.spec, mode=mode)
    run.timing["cut"] = _ms(started)
    run.hydrate([*kept, *expand.folded_members(kept, folded)])
    kept = _with_text(kept)
    expand.attach_folded(kept, folded)
    if run.spec.trace:
        run.trace.update(
            mode=mode,
            candidates=_trace_rows(ordered),
            folded={k: [t.node_id for t in v] for k, v in folded.items()},
        )
    return kept, mode


def _adaptive(
    run: _Run, found: list[Candidate], lexical: list[LaneResult]
) -> tuple[list[Candidate], RetrievalSufficiency, str]:
    kept, mode = _cut(run, found, lexical)
    null = run.null(mode)
    verdict = cutoff.sufficiency(
        kept,
        mode=mode,
        null=null,
        lexical_hit=run.lexical_evidence(kept, mode),
        searched=run.searched,
    )
    if run.report.get("dense", {}).get("status") != "ok":
        # No query vector (or a failed scan): nothing is calibrated, and
        # lexical rows alone are no evidence (review core-4/core-5).
        verdict = _unscored(run)
    if verdict.status == "partial":
        kept = kept[: max(1, run.spec.partial_k)]
    return run.merge(kept), verdict, mode


def retrieve(
    pool: Any,
    spec: RetrieveSpec,
    *,
    encode: BatchEncoder,
    rerank: Reranker | None = None,
    fetch_bytes: FetchBytes | None = None,
    lanes: dict[str, Callable[..., Any]] | None = None,
) -> RetrieveResult:
    """Retrieve, cut and judge sufficiency for ``spec`` — no model generation.

    ``lanes`` replaces stages by name (``dense``, ``cosines``, ``lex``,
    ``ident``, ``lex_strength``, ``fuse``, ``rerank``, ``rerank_enabled``,
    ``hydrate``, ``null``, ``merge``, ``graph``); tests inject fakes
    through it.
    """
    started = time.perf_counter()
    run = _Run(
        pool,
        spec,
        encode=encode,
        reranker=rerank,
        fetch_bytes=fetch_bytes,
        lanes={**default_lanes(), **(lanes or {})},
    )
    run.encode_queries()
    found = run.dense()
    if run.legacy:
        kept, verdict = _legacy(run, found)
        mode = "legacy"
        run.lexical()
        for stage in ("fusion", "rerank", "merge"):
            run.report[stage] = {
                "status": "off",
                "n": 0,
                "reason": "legacy threshold mode",
            }
    else:
        lexical = run.lexical()
        run.fill_cosines(lexical, {c.node_id: c for c in found})
        kept, verdict, mode = _adaptive(run, found, lexical)
    kept = run.bridge(kept, verdict)
    if spec.profile:
        run.report["profile"] = profiles.report(spec.profile, spec.reader_model)
    run.timing["total"] = _ms(started)
    if spec.trace:
        run.trace.setdefault("mode", mode)
    return RetrieveResult(
        query=spec.query,
        passages=tuple(to_passage(i, c) for i, c in enumerate(kept, start=1)),
        sufficiency=verdict,
        searched=run.searched,
        space=run.report.get("dense", {}).get("space", ""),
        lanes=run.report,
        timing_ms=run.timing,
        trace=run.trace,
    )
