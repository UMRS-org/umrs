"""Tests for catalog scoring and structured search filters."""

import pytest
from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.data_catalog.scoring import score_meta, token_matches, tokenize


def test_score_meta_boosts_tag_and_ready_hits():
    meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/batch-3-yields.csv",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        summary="csv data",
        detected_modality="structured",
        tags=["yield_pct", "batch_id"],
        filename_signals=["batch-3-yields.csv", "batch", "3", "yields"],
    )
    score = score_meta(meta, ["yield"], query="csv yield columns")
    assert score >= 2.0


def test_fake_store_filter_by_modality_and_tag():
    store = FakeCatalogMetaStore()
    csv_meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/batch.csv",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="structured",
        tags=["batch_id"],
    )
    pdf_meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/paper.pdf",
        object_etag="def",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="document",
        tags=["paper"],
    )
    store.upsert(csv_meta, "proj-1")
    store.upsert(pdf_meta, "proj-1")

    hits = store.search("proj-1", "", modality="structured", tag="batch_id")
    assert len(hits) == 1
    assert hits[0].meta.s3_key.endswith("batch.csv")


# ── a match is a content word, not a substring (LIVE-06, ADR-093 C7) ────────


def _ready(**fields) -> CatalogObjectMeta:
    base = {
        "s3_key": "proj-1/uploads/training_results.csv",
        "object_etag": "e",
        "state": CatalogState.READY,
        "owner": "u",
        "acl_scope": "proj-1",
        "updated_at": "2026-10-01T00:00:00",
        "summary": "yields obtained in each batch",
        "detected_modality": "structured",
        "tags": ["yield_pct"],
        "filename_signals": ["training_results.csv"],
    }
    base.update(fields)
    return CatalogObjectMeta(**base)


def _fallback_search(query: str, metas: list[CatalogObjectMeta]):
    """The recency-window fallback: the strict FTS AND-match misses first."""
    from fake_pg_pool import FakePool
    from umrs.data_catalog.pg_store import CatalogMetaStore

    rows = [
        {
            "s3_key": m.s3_key,
            "object_etag": m.object_etag,
            "state": m.state.value,
            "owner": m.owner,
            "acl_scope": m.acl_scope,
            "updated_at": m.updated_at,
            "summary": m.summary,
            "deterministic_summary": "",
            "tags": list(m.tags),
            "filename_signals": list(m.filename_signals),
            "detected_modality": m.detected_modality,
            "visibility": "project",
        }
        for m in metas
    ]
    return CatalogMetaStore(FakePool([[], rows])).search("proj-1", query, limit=20)


NONSENSE = [
    "best pizza in naples",
    "a pocket of thunder rents the attic to polite turnips",
    "what is the meaning of it",
]


@pytest.mark.parametrize("query", NONSENSE)
def test_nonsense_with_short_common_words_matches_no_row(query):
    """Review of ADR-093 (2026-10-10): 'in' is a substring of 'training' and
    of most summaries, so 'best pizza in Naples' still returned READY rows."""
    tokens = tokenize(query)
    assert token_matches(_ready(), tokens) == 0
    assert score_meta(_ready(), tokens, query=query) == 0.0
    assert _fallback_search(query, [_ready()]) == []


def test_a_content_word_still_matches_the_start_of_a_word():
    meta = _ready()
    assert token_matches(meta, ["yield", "batch", "zebra"]) == 2  # 'yield' begins 'yields'
    assert token_matches(meta, ["pct"]) == 1  # a tag's part after '_'
    assert token_matches(meta, ["rain"]) == 0  # inside 'training', not its start
    assert [h.meta.s3_key for h in _fallback_search("batch yields", [meta])] == [meta.s3_key]


def test_a_structured_hint_alone_scores_nothing():
    """'data' in the query used to add 0.5 to every structured row."""
    assert score_meta(_ready(), tokenize("zebra data"), query="zebra data") == 0.0


# ── scripts written without spaces (review of ADR-093, 2026-10-10) ──────────

JA_SUMMARY = "このデータはバッチ3の収率の測定結果です"


def _ja() -> CatalogObjectMeta:
    return _ready(
        s3_key="proj-1/run.csv",
        summary=JA_SUMMARY,
        tags=[],
        filename_signals=["run.csv"],
    )


@pytest.mark.parametrize("query", ["収率", "測定結果", "バッチ3の収率", "収率 測定", "収率の測定"])
def test_a_japanese_term_matches_inside_an_unspaced_summary(query):
    """Word-start matching (LIVE-06) zeroed every Japanese catalogue query:
    an unspaced CJK run is one 'word', so '収率' never began one."""
    tokens = tokenize(query)
    assert token_matches(_ja(), tokens) >= 1
    assert score_meta(_ja(), tokens, query=query) > 0
    assert [h.meta.s3_key for h in _fallback_search(query, [_ja()])] == ["proj-1/run.csv"]


@pytest.mark.parametrize("query", ["の", "ナポリのピザ", "東京タワー", "pizza"])
def test_a_japanese_query_the_row_does_not_hold_still_matches_nothing(query):
    tokens = tokenize(query)
    assert token_matches(_ja(), tokens) == 0
    assert score_meta(_ja(), tokens, query=query) == 0.0
    assert _fallback_search(query, [_ja()]) == []


def test_a_two_character_kanji_term_is_not_dropped_beside_a_longer_one():
    """Two kanji make a word: '収率' must count next to '測定結果'."""
    meta = _ready(summary="収率の一覧", tags=[], filename_signals=["a.csv"])
    assert token_matches(meta, tokenize("収率 測定結果")) == 1


def test_word_starts_still_rule_latin_text_beside_japanese():
    meta = _ready(summary=f"training yields; {JA_SUMMARY}")
    assert token_matches(meta, ["rain", "収率"]) == 1  # 'rain' is inside 'training'
