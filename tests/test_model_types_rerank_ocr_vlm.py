"""Model types ``reranker``, ``ocr`` and ``vlm`` (2026-11-02, ADR-106 service tier).

A reranker served by vLLM's pooling runner has no chat route, so typed ``llm``
it never passed the chat readiness probe; typed ``llm`` it was also a chat
candidate for background knowledge jobs. These tests pin the behaviour each
type carries.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from umrs.client.control_commands import MODEL_TYPE_CHOICES
from umrs.model_registry import schema
from umrs.model_registry.models import (
    CHAT_MODEL_TYPES,
    MODEL_TYPE_VALUES,
    ModelType,
    normalized_model_type,
)


def _check_values(sql: str) -> set[str]:
    match = re.search(r"CHECK \(model_type IN \(([^)]*)\)\)", sql)
    assert match, "model_type CHECK not found"
    return {v.strip().strip("'") for v in match.group(1).split(",")}


def test_new_types_exist():
    assert {t.value for t in ModelType} >= {"reranker", "ocr", "vlm"}


def test_db_check_constraint_lists_every_type():
    assert _check_values(schema.CREATE_MODEL_CATALOG_TABLE) == set(MODEL_TYPE_VALUES)
    assert schema.MODEL_REGISTRY_SCHEMA_VERSION >= 9


def test_v9_migration_widens_the_constraint():
    statements: list[str] = []
    cur = SimpleNamespace(execute=statements.append)
    schema._migrate_to_v9(cur)
    assert "DROP CONSTRAINT IF EXISTS model_catalog_entries_model_type_check" in statements[0]
    assert _check_values(statements[1]) == set(MODEL_TYPE_VALUES)


def test_client_choices_match_the_enum():
    assert set(MODEL_TYPE_CHOICES) == set(MODEL_TYPE_VALUES)


@pytest.mark.parametrize("value", [None, "", "LLM", " llm "])
def test_unset_or_cased_type_reads_as_llm(value):
    assert normalized_model_type(value) == "llm"


@pytest.mark.parametrize(
    ("model_type", "probe"),
    [
        ("llm", "chat"),
        ("vlm", "chat"),
        ("embedding", "models"),
        ("reranker", "models"),
        ("ocr", "models"),
    ],
)
def test_readiness_probe_follows_the_type(model_type, probe):
    from umrs.api.endpoints import _resolve_readiness_probe_mode

    entry = {"model_type": model_type, "serving_profile": {}}
    assert _resolve_readiness_probe_mode("m", entry, serving_backend="vllm") == probe


def _endpoint(model_type: str | None, **meta):
    metadata = {"model_type": model_type, **meta}
    return SimpleNamespace(metadata=metadata, model="m")


@pytest.mark.parametrize(
    ("model_type", "chat"),
    [
        ("llm", True),
        ("vlm", True),
        (None, True),
        ("reranker", False),
        ("ocr", False),
        ("embedding", False),
        ("vision", False),
    ],
)
def test_knowledge_chat_candidates(model_type, chat):
    from umrs.jobs.knowledge_adapters import _is_chat_endpoint

    assert _is_chat_endpoint(_endpoint(model_type)) is chat
    assert (normalized_model_type(model_type) in CHAT_MODEL_TYPES) is chat


def test_knowledge_still_skips_an_embedding_service_kind():
    from umrs.jobs.knowledge_adapters import _is_chat_endpoint

    assert _is_chat_endpoint(_endpoint("llm", service_kind="openai_embeddings")) is False


@pytest.mark.parametrize(
    ("model_type", "supports_vision", "capable"),
    [
        ("vlm", False, True),
        ("vision", False, True),
        ("llm", True, True),
        ("llm", False, False),
        ("ocr", True, False),  # OCR extracts text; it does not caption
        ("reranker", True, False),
    ],
)
def test_perception_vision_capability(model_type, supports_vision, capable):
    from umrs.data_catalog.perception import _is_vision_capable

    ep = _endpoint(model_type, capabilities={"supports_vision": supports_vision})
    assert _is_vision_capable(ep) is capable


@pytest.mark.parametrize(
    ("model_type", "chat"), [("vlm", True), ("ocr", False), ("reranker", False)]
)
def test_perception_summary_model_is_a_chat_model(model_type, chat):
    from umrs.data_catalog.perception import _is_chat_model

    assert _is_chat_model(_endpoint(model_type)) is chat
