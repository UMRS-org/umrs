"""Site-file embedding spaces (slice KN-S5, task kn-s5-01)."""

from __future__ import annotations

import pytest
from tests.kn_s5_pending import until_implemented
from umrs.knowledge import embedding_space_site as site
from umrs.knowledge.embedding_spaces import (
    CODE_INVALID,
    EmbeddingSpace,
    EmbeddingSpaceError,
)

pytestmark = until_implemented("kn-s5-01", lambda: site.spaces_from_site(None))

ME5B = {
    "space_id": "me5b",
    "model_name": "/models/me5b",
    "dimension": 768,
    "model_ref": "intfloat/multilingual-e5-base",
    "revision": "abc123",
    "modalities": ["text"],
    "metric": "cosine",
    "query_prefix": "query: ",
    "document_prefix": "passage: ",
    "locality": "site",
    "legacy_names": ["me5b", "intfloat/multilingual-e5-base"],
}
LIST_PREFIX = "site.yaml knowledge.embedding_spaces: "


def _entry_prefix(i: int) -> str:
    return f"site.yaml knowledge.embedding_spaces[{i}]: "


def _refused(knowledge) -> EmbeddingSpaceError:
    with pytest.raises(EmbeddingSpaceError) as err:
        site.spaces_from_site(knowledge)
    assert err.value.code == CODE_INVALID
    return err.value


def _message(err: EmbeddingSpaceError) -> str:
    return str(err).split(": ", 1)[1]


@pytest.mark.parametrize(
    "knowledge", [None, {}, {"embedding_spaces": None}, {"embedding_spaces": []}, {"other": 1}]
)
def test_nothing_declared_is_an_empty_list(knowledge):
    assert site.spaces_from_site(knowledge) == []


def test_a_full_entry_becomes_an_active_site_space():
    (space,) = site.spaces_from_site({"embedding_spaces": [ME5B]})
    assert space == EmbeddingSpace(
        space_id="me5b",
        model_name="/models/me5b",
        dimension=768,
        model_ref="intfloat/multilingual-e5-base",
        revision="abc123",
        modalities=("text",),
        metric="cosine",
        query_prefix="query: ",
        document_prefix="passage: ",
        locality="site",
        legacy_names=("me5b", "intfloat/multilingual-e5-base"),
        status="active",
        source="site",
    )


def test_a_minimal_entry_takes_the_defaults():
    (space,) = site.spaces_from_site(
        {"embedding_spaces": [{"space_id": "x", "model_name": "/m/x", "dimension": 8}]}
    )
    assert space.modalities == ("text",) and space.legacy_names == ()
    assert space.model_ref == "" and space.query_prefix == ""
    assert space.status == "active" and space.source == "site"


def test_entries_keep_file_order():
    second = {"space_id": "a", "model_name": "/m/a", "dimension": 4}
    spaces = site.spaces_from_site({"embedding_spaces": [ME5B, second]})
    assert [s.space_id for s in spaces] == ["me5b", "a"]


@pytest.mark.parametrize("knowledge", [[1], "x", 3])
def test_knowledge_that_is_not_a_mapping_is_refused(knowledge):
    assert str(_refused(knowledge)).split(": ", 1)[1].startswith(LIST_PREFIX)


@pytest.mark.parametrize("value", [{"a": 1}, "me5b", 3])
def test_a_spaces_value_that_is_not_a_list_is_refused(value):
    assert _message(_refused({"embedding_spaces": value})).startswith(LIST_PREFIX)


def test_an_entry_that_is_not_a_mapping_names_its_index():
    err = _refused({"embedding_spaces": [ME5B, "me5b"]})
    assert _message(err).startswith(_entry_prefix(1))


@pytest.mark.parametrize("key", ["space_id", "model_name", "dimension"])
def test_a_missing_required_key_is_named(key):
    entry = {k: v for k, v in ME5B.items() if k != key}
    msg = _message(_refused({"embedding_spaces": [entry]}))
    assert msg.startswith(_entry_prefix(0)) and key in msg


@pytest.mark.parametrize("key", ["status", "source", "index_name", "typo"])
def test_an_unknown_key_is_named(key):
    msg = _message(_refused({"embedding_spaces": [{**ME5B, key: "x"}]}))
    assert msg.startswith(_entry_prefix(0)) and key in msg


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dimension", "768"),
        ("dimension", True),
        ("modalities", "text"),
        ("modalities", [1]),
        ("legacy_names", "me5b"),
        ("legacy_names", [None]),
        ("model_ref", 5),
        ("query_prefix", ["q"]),
    ],
)
def test_wrong_types_are_refused(key, value):
    msg = _message(_refused({"embedding_spaces": [{**ME5B, key: value}]}))
    assert msg.startswith(_entry_prefix(0))


@pytest.mark.parametrize(
    ("key", "value"),
    [("space_id", "Bad-Id"), ("model_name", "x y"), ("dimension", 0), ("metric", "l2")],
)
def test_validation_errors_carry_the_entry_prefix(key, value):
    msg = _message(_refused({"embedding_spaces": [{**ME5B, key: value}]}))
    assert msg.startswith(_entry_prefix(0))


def test_a_duplicate_space_id_names_the_later_entry():
    twin = {**ME5B, "model_name": "/models/other"}
    msg = _message(_refused({"embedding_spaces": [ME5B, twin]}))
    assert msg.startswith(_entry_prefix(1)) and "me5b" in msg


def test_a_duplicate_model_name_names_the_later_entry():
    twin = {**ME5B, "space_id": "other"}
    msg = _message(_refused({"embedding_spaces": [ME5B, twin]}))
    assert msg.startswith(_entry_prefix(1)) and "/models/me5b" in msg


def test_site_embedding_spaces_reads_the_site_config(monkeypatch):
    from umrs.orchestrator import site_config

    cfg = site_config.default_site_config()
    cfg.knowledge = {"embedding_spaces": [ME5B]}
    monkeypatch.setattr(site_config, "get_site_config", lambda *_a, **_k: cfg)
    assert [s.space_id for s in site.site_embedding_spaces()] == ["me5b"]


def test_site_embedding_spaces_is_empty_without_a_declaration(monkeypatch):
    from umrs.orchestrator import site_config

    cfg = site_config.default_site_config()
    monkeypatch.setattr(site_config, "get_site_config", lambda *_a, **_k: cfg)
    assert site.site_embedding_spaces() == []
