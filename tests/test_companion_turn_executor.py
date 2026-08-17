"""Companion turn executor tests (wave T3c).

Roundtrip semantics against the durable attempt registry: executes only
in-flight attempts, commits response content (ADR-028 replay plane),
fails terminal on upstream errors; env resolution precedence and the
0074 content-plane registry shape pinned alongside.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fake_pg_pool import FakePool  # noqa: E402
from umrs.jobs.companion_turn_executor import (  # noqa: E402
    DEFAULT_LITELLM_URL,
    DEFAULT_MODEL,
    execute_turn,
    resolve_base_url,
    resolve_model,
)
from umrs.jobs.companion_worker import CompanionTurnAttemptRegistry  # noqa: E402


def _row(**overrides):
    row = {
        "attempt_id": "at-1",
        "job_id": "job-w1",
        "registry_key": "alice@default",
        "session_id": "s-1",
        "request_token": "rt-1",
        "user_message_hash": "h",
        "state": "in_flight",
        "committed_response_ref": None,
        "parent_attempt_id": None,
        "research_session_id": None,
        "user_message": "what is the audit surface?",
        "response_content": None,
        "mode": "ask",
    }
    row.update(overrides)
    return row


def _registry(script):
    return CompanionTurnAttemptRegistry(FakePool(list(script)))


# ── roundtrip ─────────────────────────────────────────────────────────────


def test_execute_turn_commits_content_with_attempt_ref():
    registry = _registry([_row()])
    captured = {}

    def chat(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "The audit surface is /umrs/governance."

    asyncio.run(execute_turn(registry, "at-1", chat=chat, model="m", base_url="http://llm"))
    updates = [s for s, _ in registry._pool.record if s.startswith("UPDATE companion_turns")]
    assert len(updates) == 1
    params = registry._pool.record[-1][1]
    assert params == ("attempt:at-1", "The audit surface is /umrs/governance.", "at-1")
    assert captured["messages"][-1] == {
        "role": "user",
        "content": "what is the audit surface?",
    }
    assert captured["kwargs"]["model"] == "m"
    assert captured["kwargs"]["base_url"] == "http://llm"


def test_execute_turn_fails_terminal_on_upstream_error():
    registry = _registry([_row()])

    def chat(_messages, **_kwargs):
        raise RuntimeError("litellm unreachable")

    asyncio.run(execute_turn(registry, "at-1", chat=chat, model="m", base_url="http://llm"))
    updates = [s for s, _ in registry._pool.record if s.startswith("UPDATE companion_turns")]
    assert len(updates) == 1
    assert "state = 'failed'" in updates[0]


def test_execute_turn_skips_non_in_flight_attempt():
    registry = _registry([_row(state="committed", committed_response_ref="attempt:at-1")])
    called = []

    asyncio.run(
        execute_turn(
            registry,
            "at-1",
            chat=lambda *_a, **_k: called.append(1) or "noop",
            model="m",
            base_url="http://llm",
        )
    )
    assert called == []
    assert not any(s.startswith("UPDATE") for s, _ in registry._pool.record)


def test_execute_turn_missing_attempt_is_noop():
    registry = _registry([None])
    asyncio.run(
        execute_turn(
            registry,
            "at-ghost",
            chat=lambda *_a, **_k: "noop",
            model="m",
            base_url="http://llm",
        )
    )
    assert not any(s.startswith("UPDATE") for s, _ in registry._pool.record)


# ── env resolution ────────────────────────────────────────────────────────


def test_model_resolution_precedence(monkeypatch: pytest.MonkeyPatch):
    assert resolve_model() == DEFAULT_MODEL
    monkeypatch.setenv("UMRS_AGENT_MODEL", "agent-model")
    assert resolve_model() == "agent-model"
    monkeypatch.setenv("UMRS_COMPANION_MODEL", "companion-model")
    assert resolve_model() == "companion-model"


def test_base_url_resolution_precedence(monkeypatch: pytest.MonkeyPatch):
    assert resolve_base_url() == DEFAULT_LITELLM_URL
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://agent:11434")
    assert resolve_base_url() == "http://agent:11434"
    monkeypatch.setenv("UMRS_COMPANION_LITELLM_URL", "http://companion:1")
    assert resolve_base_url() == "http://companion:1"


# ── registry content plane (0074) ─────────────────────────────────────────


def test_begin_persists_message_and_mode():
    registry = _registry([None, _row(mode="auto")])
    registry.begin(
        job_id="job-w1",
        registry_key="alice@default",
        session_id="s-1",
        request_token="rt-1",
        user_message="persist me",
        mode="auto",
    )
    insert_sql, insert_params = registry._pool.record[0]
    assert "user_message" in insert_sql and "mode" in insert_sql
    assert "persist me" in insert_params
    assert "auto" in insert_params
    select_sql, _ = registry._pool.record[1]
    for col in ("user_message", "response_content", "mode"):
        assert col in select_sql


def test_complete_writes_response_content():
    registry = _registry([])
    registry.complete("at-1", committed_response_ref="attempt:at-1", response_content="body")
    sql, params = registry._pool.record[0]
    assert "COALESCE" in sql and "response_content" in sql
    assert params == ("attempt:at-1", "body", "at-1")


def test_migration_0074_content_plane_columns():
    sql = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "umrs"
        / "migrations"
        / "0074_companion_turn_content.sql"
    ).read_text()
    for fragment in (
        "ADD COLUMN IF NOT EXISTS user_message",
        "ADD COLUMN IF NOT EXISTS response_content",
        "ADD COLUMN IF NOT EXISTS mode",
    ):
        assert fragment in sql
