"""KN-S6 task kn-s6-01: the per-space index signature, plan and coverage.

Pure functions in ``umrs.knowledge.embedding_spaces``: what comment a current
``hnsw_<space_id>`` carries, which statements bring the database to it, and
which stored model names a space's index predicate misses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from umrs.knowledge import embedding_spaces as es
from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    EmbeddingSpace,
    EmbeddingSpaceError,
    IndexState,
)


SPACE = EmbeddingSpace(
    space_id="me5b",
    model_name="/models/me5b",
    dimension=768,
    legacy_names=("intfloat/multilingual-e5-base", "me5b"),
)
SIGNATURE = (
    "umrs-space:v1 space=me5b dim=768 "
    "names=/models/me5b,intfloat/multilingual-e5-base,me5b"
)


# ── index_signature / comment_index_sql ─────────────────────────────────────


def test_the_signature_names_space_dimension_and_every_name_in_order():
    assert es.index_signature(SPACE) == SIGNATURE


def test_the_bundled_signature():
    assert es.index_signature(BUNDLED_SPACE) == (
        "umrs-space:v1 space=e5_base_v2 dim=768 names=/opt/umrs/models/e5-base-v2,"
        "intfloat/e5-base-v2,/models/e5-base-v2,e5-base-v2"
    )


def test_the_signature_uses_normalized_names_without_duplicates():
    space = replace(SPACE, legacy_names=("/models/me5b/", "me5b", "me5b"))
    assert es.index_signature(space) == "umrs-space:v1 space=me5b dim=768 names=/models/me5b,me5b"


def test_the_signature_moves_when_a_legacy_name_is_added():
    grown = replace(SPACE, legacy_names=(*SPACE.legacy_names, "/data/me5b"))
    assert es.index_signature(grown) != es.index_signature(SPACE)
    assert es.index_signature(grown).endswith(",/data/me5b")


def test_the_signature_moves_with_the_dimension():
    assert "dim=1024" in es.index_signature(replace(SPACE, dimension=1024))


def test_the_signature_validates_the_space():
    with pytest.raises(EmbeddingSpaceError):
        es.index_signature(replace(SPACE, space_id="Bad-Id"))


def test_the_comment_statement():
    assert es.comment_index_sql(SPACE) == f"COMMENT ON INDEX hnsw_me5b IS '{SIGNATURE}'"


# ── index_actions ───────────────────────────────────────────────────────────

CREATE = es.hnsw_index_ddl(SPACE)
DROP = es.drop_index_ddl(SPACE)


def _comment() -> str:
    return es.comment_index_sql(SPACE)


def test_a_missing_index_is_created_then_commented():
    assert es.index_actions(SPACE, IndexState()) == (CREATE, _comment())


def test_a_current_index_needs_nothing():
    state = IndexState(exists=True, valid=True, comment=SIGNATURE)
    assert es.index_actions(SPACE, state) == ()


def test_an_invalid_build_is_dropped_and_rebuilt():
    state = IndexState(exists=True, valid=False, comment=SIGNATURE)
    assert es.index_actions(SPACE, state) == (DROP, CREATE, _comment())


@pytest.mark.parametrize("comment", [None, "", "umrs-space:v1 space=me5b dim=768 names=/models/me5b"])
def test_a_stale_or_missing_comment_is_rebuilt(comment):
    state = IndexState(exists=True, valid=True, comment=comment)
    assert es.index_actions(SPACE, state) == (DROP, CREATE, _comment())


def test_a_retired_space_loses_its_index():
    retired = replace(SPACE, status="retired")
    state = IndexState(exists=True, valid=True, comment=SIGNATURE)
    assert es.index_actions(retired, state) == (es.drop_index_ddl(retired),)


def test_a_retired_space_without_an_index_needs_nothing():
    assert es.index_actions(replace(SPACE, status="retired"), IndexState()) == ()


def test_every_action_is_concurrent():
    for statement in es.index_actions(SPACE, IndexState(exists=True)):
        if not statement.startswith("COMMENT"):
            assert " CONCURRENTLY " in statement


# ── uncovered_names ─────────────────────────────────────────────────────────


def test_a_basename_match_outside_the_names_is_uncovered():
    stored = ["/data/models/e5-base-v2", "/opt/umrs/models/e5-base-v2", "other/model"]
    assert es.uncovered_names(BUNDLED_SPACE, stored) == ("/data/models/e5-base-v2",)


def test_registered_names_are_covered_after_normalizing():
    stored = ["/models/e5-base-v2/", " intfloat/e5-base-v2 ", "e5-base-v2"]
    assert es.uncovered_names(BUNDLED_SPACE, stored) == ()


def test_uncovered_names_are_normalized_deduplicated_and_ordered():
    stored = ["/b/e5-base-v2/", "/a/e5-base-v2", "/b/e5-base-v2", ""]
    assert es.uncovered_names(BUNDLED_SPACE, stored) == ("/b/e5-base-v2", "/a/e5-base-v2")


def test_a_space_without_a_bare_name_has_no_basename_matches():
    space = EmbeddingSpace(space_id="clip", model_name="/models/clip-b16", dimension=512)
    assert es.uncovered_names(space, ["/other/clip-b16", "/models/clip-b16"]) == ()


def test_uncovered_names_ignore_status():
    retired = replace(BUNDLED_SPACE, status="retired")
    assert es.uncovered_names(retired, ["/x/e5-base-v2"]) == ("/x/e5-base-v2",)
