"""U3 S3d: session-archive participants through ``add_session_archive_participant``.

Core's archive runs every registered participant inside its own transaction
and no longer knows the companion trust store; the in-tree r4 part registers
the trust revoke + tombstone (ADR-035 item 8).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.extensions import load_extensions, loader, reset_extension_context
from umrs.research_sessions.store import ResearchSession, SessionStore

STORE_SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "research_sessions" / "store.py"


@pytest.fixture(autouse=True)
def _fresh_context(monkeypatch):
    monkeypatch.delenv(loader.ENV, raising=False)
    reset_extension_context()
    yield
    reset_extension_context()


def _store_with_session() -> SessionStore:
    store = SessionStore()
    store.create(ResearchSession(session_id="rs-1", project_id="p1", research_goal="g"))
    return store


def test_core_archive_no_longer_knows_the_trust_ledger():
    # Fails on the pre-S3d store, which wrote the tombstone and revoked trust.
    text = STORE_SRC.read_text(encoding="utf-8")
    assert "companion_session_archives" not in text
    assert "revoke_session_scope_conn" not in text


def test_participants_run_on_the_archive_cursor_with_the_archiver():
    store = _store_with_session()
    seen: list[tuple[str, str, bool]] = []

    def participant(cur, session_id, archived_by):
        cur.execute("SELECT status FROM research_sessions WHERE session_id = ?", (session_id,))
        seen.append((session_id, archived_by, cur.fetchone()[0] == "archived"))

    archived = store.archive_with_cascade("rs-1", participants=[participant])
    assert archived.status == "archived"
    # The participant saw the archive's own uncommitted update: same transaction.
    assert seen == [("rs-1", "research-sessions.patch", True)]


def test_a_failing_participant_rolls_the_archive_back():
    store = _store_with_session()

    def boom(_cur, _sid, _by):
        raise RuntimeError("participant failed")

    with pytest.raises(RuntimeError):
        store.archive_with_cascade("rs-1", participants=[boom])
    assert store.get("rs-1").status != "archived"


def test_r4_registers_the_trust_participant():
    from umrs.companion_trust.archive_participant import revoke_trust_on_archive

    assert load_extensions(("r4",)).session_archive_participants == [revoke_trust_on_archive]
    assert load_extensions(()).session_archive_participants == []


def test_default_participants_come_from_the_enabled_extensions(monkeypatch):
    monkeypatch.setenv(loader.ENV, "none")
    reset_extension_context()
    store = _store_with_session()
    assert store.archive_with_cascade("rs-1").status == "archived"


def test_trust_participant_is_a_no_op_when_the_ledger_is_unwired(monkeypatch):
    import umrs.shared.trust_store_access as trust_access
    from umrs.companion_trust.archive_participant import revoke_trust_on_archive

    monkeypatch.setattr(trust_access, "_trust_store", None)
    store = _store_with_session()
    assert store.archive_with_cascade("rs-1", participants=[revoke_trust_on_archive])
