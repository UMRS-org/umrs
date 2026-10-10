"""Read-time small-to-big: tail folding and neighbour merge (ADR-093 D-5, C5).

SciFact abstracts embed as about 1.84 fixed 256-token chunks, so a kept first
chunk often lacks the results half, and the remainder is a 15-140 character
tail that e5 ranks high on its own. Both are repaired at read time by
deterministic node ids — no re-embed.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from umrs.knowledge.retrieval import expand
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.types import Candidate

_DOC = ("Alpha beta gamma. " * 30).strip()  # 539 characters, the artifact's text
_END = len(_DOC)
_PREFIX = "node-chunk-g-art-1"


def _chunk(index, start, end, *, text=None, source="artifact", cosine=0.8, **content):
    body = _DOC[start:end] if text is None else text
    return Candidate(
        node_id=f"{_PREFIX}-{index}",
        artifact_id="art-1",
        chunk_index=index,
        content={
            "kind": "chunk",
            "artifact_id": "art-1",
            "chunk_index": index,
            "chunk_start": start,
            "chunk_end": end,
            **content,
        },
        cosine=cosine,
        text=body,
        text_source=source,
        span=(start, end),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_MERGE", raising=False)
    monkeypatch.delenv("UMRS_RAG_TAIL_CHARS", raising=False)
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)


# ── ids, spans ──────────────────────────────────────────────────────────────


def test_neighbour_ids_follow_the_writers_scheme():
    chunk = _chunk(3, 0, 10)
    assert expand.neighbour_id(chunk, 4) == f"{_PREFIX}-4"
    assert expand.neighbour_id(chunk, 2) == f"{_PREFIX}-2"
    assert expand.neighbour_id(_chunk(0, 0, 10), -1) is None
    assert expand.neighbour_id(Candidate(node_id="ragdoc-x-chunk2", chunk_index=2), 3) is None


def test_union_text_stitches_by_offset_and_keeps_the_span_length():
    left, right = _chunk(0, 0, 100), _chunk(1, 102, 200)
    span, text = expand.union_text([right, left])
    assert span == (0, 200) and len(text) == 200 and text == _DOC[:100] + "  " + _DOC[102:200]
    overlap_span, overlap = expand.union_text([_chunk(0, 0, 120), _chunk(1, 100, 200)])
    assert overlap_span == (0, 200) and overlap == _DOC[:200]


def test_only_verified_text_whose_length_is_its_span_can_be_stitched():
    assert expand.is_full(_chunk(0, 0, 100))
    assert not expand.is_full(_chunk(0, 0, 100, source="excerpt"))
    assert not expand.is_full(_chunk(0, 0, 100, text="short"))


def test_merge_auto_reads_the_site_switch(monkeypatch):
    assert not expand.merge_enabled("auto") and expand.merge_enabled("on")
    monkeypatch.setenv("UMRS_RAG_MERGE", "on")
    assert expand.merge_enabled("auto") and not expand.merge_enabled("off")


# ── tail folding ────────────────────────────────────────────────────────────


def test_a_tail_folds_into_its_predecessor_at_the_better_position():
    tail = _chunk(1, 500, _END, cosine=0.93)
    head = _chunk(0, 0, 500, cosine=0.85)
    other = Candidate(node_id="node-chunk-g-art-2-0", chunk_index=0, cosine=0.90)
    tail.lane_ranks, head.lane_ranks = {"dense": 1}, {"dense": 3}
    folded_list, folded = expand.fold_tails([tail, other, head])
    assert [c.node_id for c in folded_list] == [head.node_id, other.node_id]
    assert folded == {head.node_id: [tail]}
    assert head.cosine == 0.93 and head.lane_ranks == {"dense": 1}


def test_a_tail_without_its_predecessor_keeps_its_own_slot():
    tail = _chunk(1, 500, _END)
    assert expand.fold_tails([tail]) == ([tail], {})


def test_a_long_last_chunk_is_not_a_tail(monkeypatch):
    last = _chunk(1, 300, _END)  # 239 characters
    assert not expand.is_tail(last)
    monkeypatch.setenv("UMRS_RAG_TAIL_CHARS", "300")
    assert expand.is_tail(last)


def test_attach_folded_appends_the_tail_text():
    head, tail = _chunk(0, 0, 500), _chunk(1, 500, _END)
    expand.attach_folded([head], {head.node_id: [tail]})
    assert head.text == _DOC[:_END] and head.span == (0, _END) and head.section == "part 1-2"


def test_an_unverified_tail_is_not_stitched():
    head, tail = _chunk(0, 0, 500), _chunk(1, 500, _END, source="excerpt")
    expand.attach_folded([head], {head.node_id: [tail]})
    assert head.text == _DOC[:500]


# ── neighbour merge ─────────────────────────────────────────────────────────


_SCOPE = ("p1", "g", "project")  # (project, graph, tier) of every fixture chunk


def _scope_of(node_id):
    """``node-chunk-g-{artifact}-{i}``: the fixture's own scope and artifact."""
    artifact = node_id.removeprefix("node-chunk-g-").rsplit("-", 1)[0]
    return (*_SCOPE, artifact)


class _Pool:
    """Answers hydrate's node read for the neighbours it is asked about, and
    the merge's scope read (``foreign`` overrides a node's scope)."""

    def __init__(self, chunks, foreign=None):
        self.chunks = {c.node_id: c for c in chunks}
        self.foreign = dict(foreign or {})
        self.reads = []

    def getconn(self):
        return self

    def putconn(self, _conn):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.rows = []
        if "LEFT JOIN" in sql:
            self.reads.append(sorted(params[0]))
            self.rows = [self._row(nid) for nid in params[0] if nid in self.chunks]
        elif sql == expand._SCOPE_SQL:
            self.rows = [(nid, *self.foreign.get(nid, _scope_of(nid))) for nid in params[0]]

    def _row(self, node_id):
        """The artifact path: offsets plus the md5 of the slice."""
        chunk = self.chunks[node_id]
        start, end = chunk.span
        md5 = hashlib.md5(_DOC[start:end].encode()).hexdigest()
        content = {**chunk.content, "chunk_md5": md5}
        return (node_id, "g", json.dumps(content), "artifacts/doc", "note.md", None, None)

    def fetchall(self):
        return self.rows


def _fetch(_ref):
    return _DOC.encode()


def test_a_kept_chunk_grows_by_its_neighbours_within_the_token_budget():
    kept = _chunk(1, 200, 400)
    pool = _Pool([_chunk(0, 0, 200), _chunk(2, 400, _END)])
    [merged] = expand.merge_neighbours(pool, [kept], max_passage_tokens=400, fetch_bytes=_fetch)
    assert merged.text == _DOC[0:_END] and merged.span == (0, _END)
    assert merged.section == "part 1-3"
    assert pool.reads == [[f"{_PREFIX}-0", f"{_PREFIX}-2"]]  # one read for all


def test_a_neighbour_that_would_overflow_the_budget_is_left_out():
    kept = _chunk(1, 200, 400)
    pool = _Pool([_chunk(0, 0, 200), _chunk(2, 400, _END)])
    [merged] = expand.merge_neighbours(pool, [kept], max_passage_tokens=90, fetch_bytes=_fetch)
    assert merged.span == (200, _END)  # 50 + 35 tokens fit; the left neighbour does not


def test_num_tokens_recorded_on_the_node_is_preferred():
    assert expand.tokens(_chunk(0, 0, 400, num_tokens=17)) == 17
    assert expand.tokens(_chunk(0, 0, 400)) == 100


def test_touching_passages_of_one_artifact_become_one():
    first, second = _chunk(0, 0, 270, cosine=0.9), _chunk(1, 270, _END, cosine=0.8)
    other = Candidate(
        node_id="node-chunk-g-art-2-0",
        chunk_index=0,
        cosine=0.85,
        text="x",
        text_source="artifact",
        span=(0, 1),
    )
    merged = expand.merge_neighbours(_Pool([]), [first, other, second], max_passage_tokens=400)
    assert [c.node_id for c in merged] == [first.node_id, other.node_id]
    assert first.text == _DOC and first.section == "part 1-2"


def test_excerpt_passages_are_never_merged():
    kept = _chunk(1, 200, 400, source="excerpt", text=_DOC[200:300])
    pool = _Pool([_chunk(0, 0, 200)])
    [merged] = expand.merge_neighbours(pool, [kept], fetch_bytes=_fetch)
    assert merged.text == _DOC[200:300] and pool.reads == []


def test_a_known_chunk_count_shows_in_the_part_label():
    kept = _chunk(0, 0, 200)
    kept.chunk_count = 3
    expand.merge_neighbours(_Pool([_chunk(1, 200, 400)]), [kept], fetch_bytes=_fetch)
    assert kept.section == "part 1-2/3"


# ── review fixes (2026-10-10): the 400-token passage and the pack cap (core-8) ─

_LONG = "word " * 900  # 4,500 characters: room for three 1,200-character chunks


def _long_chunk(index, *, cosine=0.8):
    start, end = 1200 * index, 1200 * (index + 1)
    return Candidate(
        node_id=f"node-chunk-g-long-{index}",
        artifact_id="long",
        chunk_index=index,
        content={"kind": "chunk", "artifact_id": "long", "chunk_index": index},
        cosine=cosine,
        text=_LONG[start:end],
        text_source="search_text",
        span=(start, end),
    )


def test_touching_kept_passages_merge_only_within_the_passage_budget():
    """Two adjacent kept 300-token chunks stay two passages under a 400 cap."""
    out = expand.merge_neighbours(
        _Pool([]), [_long_chunk(1), _long_chunk(2)], max_passage_tokens=400
    )
    assert [expand.tokens(c) for c in out] == [300, 300]
    assert [c.node_id for c in out] == ["node-chunk-g-long-1", "node-chunk-g-long-2"]


def _art2(index, start, end):
    chunk = _chunk(index, start, end)
    chunk.node_id, chunk.artifact_id = f"node-chunk-g-art-2-{index}", "art-2"
    chunk.content = {**chunk.content, "artifact_id": "art-2"}
    return chunk


def test_growth_stops_at_the_pack_cap_lowest_rank_first():
    """k_max x 450 bounds the pack: the better passage grows, the worse not."""
    first, second = _chunk(1, 200, 300), _art2(1, 200, 300)
    pool = _Pool([_chunk(2, 300, 400), _art2(2, 300, 400)])
    uncapped = expand.merge_neighbours(
        pool, [_chunk(1, 200, 300), _art2(1, 200, 300)], fetch_bytes=_fetch
    )
    assert [c.span for c in uncapped] == [(200, 400), (200, 400)]
    out = expand.merge_neighbours(pool, [first, second], pack_tokens=90, fetch_bytes=_fetch)
    assert [c.span for c in out] == [(200, 400), (200, 300)]
    assert sum(expand.tokens(c) for c in out) <= 90


def test_a_neighbour_joins_one_passage_only():
    """Chunks 1 and 3 kept: chunk 2 grows one of them, never both."""
    left, right = _chunk(1, 100, 200), _chunk(3, 300, 400)
    pool = _Pool([_chunk(2, 200, 300)])
    out = expand.merge_neighbours(pool, [left, right], max_passage_tokens=60, fetch_bytes=_fetch)
    covered = [c.span for c in out]
    assert covered == [(100, 300), (300, 400)]


# ── review fix (2026-10-10): a neighbour is stitched only under its host's scope
# (security-3, the merge=on leg) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "scope",
    [
        ("p-victim", "g", "project", "art-1"),
        ("p1", "g-other", "project", "art-1"),
        ("p1", "g", "ambient", "art-1"),
        ("p1", "g", "project", "art-9"),
    ],
    ids=["project", "graph", "tier", "artifact"],
)
def test_a_neighbour_of_another_scope_is_never_hydrated_or_stitched(scope):
    """A node imported under the arithmetic id of chunk 2 (another project,
    graph, tier or artifact) is neither read nor joined; chunk 0 still is."""
    planted = f"{_PREFIX}-2"
    pool = _Pool([_chunk(0, 0, 200), _chunk(2, 400, _END)], foreign={planted: scope})
    [merged] = expand.merge_neighbours(
        pool, [_chunk(1, 200, 400)], max_passage_tokens=400, fetch_bytes=_fetch
    )
    assert merged.span == (0, 400) and merged.text == _DOC[0:400]
    assert pool.reads == [[f"{_PREFIX}-0"]]  # the planted node is never hydrated


def test_a_neighbour_joins_only_the_host_whose_scope_it_shares():
    """Two kept hosts of one id prefix but different projects: the shared
    neighbour joins the one of its own project, never the better-ranked one."""
    left, right = _chunk(1, 100, 200), _chunk(3, 300, 400)
    other = ("p2", "g", "project", "art-1")
    pool = _Pool([_chunk(2, 200, 300)], foreign={right.node_id: other, f"{_PREFIX}-2": other})
    out = expand.merge_neighbours(pool, [left, right], max_passage_tokens=400, fetch_bytes=_fetch)
    assert [c.span for c in out] == [(100, 200), (200, 400)]
