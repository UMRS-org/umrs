"""Companion turn executor tests (wave T3c).

Roundtrip semantics against the durable attempt registry: executes only
in-flight attempts, commits response content (ADR-028 replay plane),
fails terminal on upstream errors; env resolution precedence and the
0074 content-plane registry shape pinned alongside.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# splice removed (ADR-053): pytest auto-inserts tests/ for sibling helpers; umrs editable
from fake_pg_pool import FakePool  # noqa: E402
from umrs.jobs.companion_turn_executor import (  # noqa: E402
    DEFAULT_LITELLM_URL,
    DEFAULT_MODEL,
    ChatTarget,
    execute_turn,
    kick,
    project_from_registry_key,
    resolve_base_url,
    resolve_chat_target,
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


# ── stale-attempt reaper (T3c follow-up) ─────────────────────────────────


class _RcCursor:
    def __init__(self, rowcount, record):
        self.rowcount = rowcount
        self._record = record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._record.append((" ".join(str(sql).split()), params))


class _RcPool:
    def __init__(self, rowcount):
        self.rowcount = rowcount
        self.record: list = []

    def getconn(self):
        return self

    def putconn(self, _conn):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, cursor_factory=None):
        del cursor_factory
        return _RcCursor(self.rowcount, self.record)


def test_fail_if_stale_reaps_old_attempts_and_is_guarded():
    pool = _RcPool(rowcount=1)
    registry = CompanionTurnAttemptRegistry(pool)
    assert registry.fail_if_stale("at-1", 300) is True
    sql, params = pool.record[0]
    assert "state = 'in_flight'" in sql and "make_interval" in sql
    assert params == ("at-1", 300)

    pool2 = _RcPool(rowcount=0)
    registry2 = CompanionTurnAttemptRegistry(pool2)
    assert registry2.fail_if_stale("at-1", 300) is False


def test_stale_ttl_env_defaults_and_override(monkeypatch: pytest.MonkeyPatch):
    from umrs.api.companion import _stale_ttl_seconds

    assert _stale_ttl_seconds() == 300
    monkeypatch.setenv("UMRS_COMPANION_TURN_STALE_TTL_SEC", "90")
    assert _stale_ttl_seconds() == 90
    monkeypatch.setenv("UMRS_COMPANION_TURN_STALE_TTL_SEC", "10")
    assert _stale_ttl_seconds() == 30  # floor


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


# ── live-endpoint routing (T3c follow-up) ─────────────────────────────────


class _FakeEndpoint:
    def __init__(self, url="", metadata=None):
        self.litellm_url = url
        self.metadata = metadata or {}


class _FakeRegistry:
    def __init__(self, endpoint=None, error=None):
        self._endpoint = endpoint
        self._error = error
        self.calls: list[tuple] = []

    def select_endpoint(self, project_id, model, strategy=None):
        self.calls.append((project_id, model, strategy))
        if self._error is not None:
            raise self._error
        return self._endpoint


def _patch_registry(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    import umrs.orchestrator.endpoint_registry as endpoint_registry

    monkeypatch.setattr(endpoint_registry, "get_registry", lambda: registry)


def test_resolve_chat_target_prefers_live_endpoint(monkeypatch):
    reg = _FakeRegistry(
        _FakeEndpoint("http://node-3:11434", {"served_model_name": "rivault-gpt-oss-static"})
    )
    _patch_registry(monkeypatch, reg)
    target = resolve_chat_target("rivault-gpt-oss-static", project_id="default")
    assert target == ChatTarget(base_url="http://node-3:11434", model="rivault-gpt-oss-static")
    assert reg.calls == [("default", "rivault-gpt-oss-static", "least_connections")]


def test_resolve_chat_target_none_when_no_endpoint_or_error(monkeypatch):
    _patch_registry(monkeypatch, _FakeRegistry(None))
    assert resolve_chat_target("rivault-gpt-oss-static") is None
    _patch_registry(monkeypatch, _FakeRegistry(error=RuntimeError("registry down")))
    assert resolve_chat_target("rivault-gpt-oss-static") is None


def test_resolve_chat_target_none_when_url_missing(monkeypatch):
    _patch_registry(monkeypatch, _FakeRegistry(_FakeEndpoint("")))
    assert resolve_chat_target("rivault-gpt-oss-static") is None


def test_project_from_registry_key():
    assert project_from_registry_key("alice@proj-a") == "proj-a"
    assert project_from_registry_key("alice@") == "default"
    assert project_from_registry_key("alice") == "default"


def test_kick_uses_registry_target_for_project(monkeypatch):
    reg = _FakeRegistry(_FakeEndpoint("http://node-9:11434", {"served_model_name": "other-model"}))
    _patch_registry(monkeypatch, reg)
    captured = {}

    def chat(messages, **kwargs):
        captured.update(kwargs)
        return "ok"

    registry = _registry([_row()])

    async def _main():
        task = kick(registry, "at-1", chat=chat, registry_key="alice@proj-z")
        await task

    asyncio.run(_main())
    assert captured["base_url"] == "http://node-9:11434"
    assert captured["model"] == "other-model"
    assert reg.calls == [("proj-z", DEFAULT_MODEL, "least_connections")]


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
