"""ADR-099 D-8 stores: durable questions (0127) and agent settings (0128).

The questions store runs on its sqlite fallback; the settings store in
memory and against the SQL shape the Postgres path issues. With a reachable
database (``DATABASE_URL``) both run against the migrations themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from tests.fake_pg_pool import FakePool
from umrs.companion_trust.chat_turn_questions import (
    ChatTurnQuestionsStore,
    QuestionClosed,
    cancel_session_questions,
)
from umrs.venues.agent_settings import (
    SettingsInvalid,
    VenueAgentSettingsStore,
    validate_agent_settings,
)

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src/umrs/migrations"


def _ask(store, **over):
    args = {
        "session_id": "s1",
        "turn_id": "t1",
        "user_id": "alice",
        "question": "Which one?",
        "options": ["a", "", "b"],
    }
    args.update(over)
    return store.ask(**args)


class TestQuestionsStore:
    def test_ask_is_pending(self):
        store = ChatTurnQuestionsStore()
        q = _ask(store)
        assert (q["status"], q["options"], q["answered_at"]) == ("pending", ["a", "b"], "")
        assert q["created_at"]
        assert [r["question_id"] for r in store.pending(session_id="s1")] == [q["question_id"]]

    def test_answer_closes_it(self):
        store = ChatTurnQuestionsStore()
        q = _ask(store)
        done = store.answer(q["question_id"], "b", answered_by="alice")
        assert (done["status"], done["answer"], done["answered_by"]) == ("answered", "b", "alice")
        assert done["answered_at"]
        assert store.pending(session_id="s1") == []

    def test_terminal_transitions_only(self):
        store = ChatTurnQuestionsStore()
        qid = _ask(store)["question_id"]
        store.answer(qid, "x", answered_by="alice")
        with pytest.raises(QuestionClosed):
            store.answer(qid, "y", answered_by="alice")
        # expire on a closed question changes nothing
        assert store.expire(qid)["status"] == "answered"
        other = _ask(store)["question_id"]
        expired = store.expire(other)
        assert (expired["status"], expired["answered_at"]) == ("expired", "")
        with pytest.raises(QuestionClosed):
            store.answer(other, "late", answered_by="alice")

    def test_unknown_question(self):
        store = ChatTurnQuestionsStore()
        assert store.get("nope") is None
        with pytest.raises(KeyError):
            store.answer("nope", "x", answered_by="alice")
        with pytest.raises(KeyError):
            store.expire("nope")

    def test_idempotent_question_id(self):
        store = ChatTurnQuestionsStore()
        first = _ask(store, question_id="q-1")
        again = _ask(store, question_id="q-1", question="changed")
        assert again["question"] == first["question"] == "Which one?"

    def test_by_status_and_bounds(self):
        store = ChatTurnQuestionsStore()
        a = _ask(store, question="x" * 5000)["question_id"]
        _ask(store, session_id="s2")
        store.answer(a, "y" * 9000, answered_by="alice")
        assert len(store.get(a)["question"]) == 4000
        assert len(store.get(a)["answer"]) == 8000
        assert [r["status"] for r in store.by_status(session_id="s1", status="all")] == ["answered"]
        with pytest.raises(ValueError):
            store.by_status(session_id="s1", status="bogus")

    def test_cancel_pending_closes_one_owners_pending_questions(self):
        # A3: a new worker, a stop or the lease end leaves no stale question.
        store = ChatTurnQuestionsStore()
        stale = _ask(store)["question_id"]
        answered = _ask(store)["question_id"]
        store.answer(answered, "x", answered_by="alice")
        other_session = _ask(store, session_id="s2")["question_id"]
        other_owner = _ask(store, user_id="bob")["question_id"]
        assert store.cancel_pending(session_id="s1", user_id="alice") == 1
        assert store.get(stale)["status"] == "cancelled"
        assert store.get(answered)["status"] == "answered"
        assert store.get(other_session)["status"] == "pending"
        assert store.get(other_owner)["status"] == "pending"
        with pytest.raises(QuestionClosed):
            store.answer(stale, "late", answered_by="alice")


class _ConnPool:
    def __init__(self, conn=None, *, fail: bool = False):
        self.conn = conn
        self.fail = fail
        self.returned = 0
        self.rolled_back = False

    def getconn(self):
        if self.fail:
            return self
        return self.conn

    def putconn(self, _conn):
        self.returned += 1

    # a broken connection: every cursor fails
    def cursor(self):
        raise RuntimeError("connection lost")

    def rollback(self):
        self.rolled_back = True


class TestCancelSessionQuestions:
    def test_cancels_through_the_factory_and_returns_the_connection(self):
        store = ChatTurnQuestionsStore()
        qid = _ask(store)["question_id"]
        pool = _ConnPool(object())
        n = cancel_session_questions(
            pool, session_id="s1", user_id="alice", factory=lambda _conn: store
        )
        assert n == 1 and store.get(qid)["status"] == "cancelled"
        assert pool.returned == 1

    def test_no_pool_and_errors_cancel_nothing_and_never_raise(self):
        assert cancel_session_questions(None, session_id="s1", user_id="alice") == 0
        pool = _ConnPool(fail=True)
        assert cancel_session_questions(pool, session_id="s1", user_id="alice") == 0
        assert pool.rolled_back is True and pool.returned == 1

    def test_postgres_without_the_table_skips(self):
        pool = FakePool([[(False,)]])
        assert cancel_session_questions(pool, session_id="s1", user_id="alice") == 0
        assert [sql for sql, _ in pool.record] == [
            "SELECT to_regclass('chat_turn_questions') IS NOT NULL"
        ]

    def test_postgres_shape(self):
        pool = FakePool([[(True,)], None])
        pool.conn.commit = lambda: None
        cancel_session_questions(pool, session_id="s1", user_id="alice")
        sql, params = pool.record[1]
        assert sql.startswith("UPDATE chat_turn_questions SET status = 'cancelled'")
        assert "status = 'pending'" in sql and params == ("s1", "alice")


class TestSettingsValidation:
    def test_known_keys_are_checked_unknown_kept(self):
        doc = validate_agent_settings(
            {"permission_mode": "trust", "session_approved_tools": ["b", "a", "a"], "x": 1}
        )
        assert doc == {
            "permission_mode": "trust",
            "session_approved_tools": ["a", "b"],
            "x": 1,
            "schema": 1,
        }

    @pytest.mark.parametrize(
        ("doc", "field"),
        [
            ({"permission_mode": "yolo"}, "permission_mode"),
            ({"interaction_mode": "turbo"}, "interaction_mode"),
            ({"ask_user_mode": "never"}, "ask_user_mode"),
            ({"workflow_armed": "yes"}, "workflow_armed"),
            ({"approval_ttl_s": -1}, "approval_ttl_s"),
            ({"approval_ttl_s": True}, "approval_ttl_s"),
            ({"session_approved_tools": ["rm -rf"]}, "session_approved_tools"),
            ({"blob": "x" * 20000}, "settings"),
            ([], "settings"),
        ],
    )
    def test_refusals_name_the_field(self, doc, field):
        with pytest.raises(SettingsInvalid) as err:
            validate_agent_settings(doc)
        assert err.value.field == field


class TestSettingsStore:
    def test_memory_rev_grows_per_user_and_session(self):
        # A7: a row is (user_id, session_id): another user's write for the
        # same session id is their own row and never claims or blocks this one.
        store = VenueAgentSettingsStore()
        assert store.get("s1", user_id="alice") == (0, {})
        assert store.put(session_id="s1", user_id="bob", settings={"b": 1}) == 1
        assert store.put(session_id="s1", user_id="alice", settings={"a": 1}) == 1
        assert store.put(session_id="s1", user_id="alice", settings={"a": 2}) == 2
        assert store.get("s1", user_id="alice") == (2, {"a": 2})
        assert store.get("s1", user_id="bob") == (1, {"b": 1})
        assert store.get("s1", user_id="carol") == (0, {})

    def test_pg_shape(self):
        pool = FakePool([[(3,)], [(3, {"ask_user_mode": "auto"})]])
        store = VenueAgentSettingsStore(pool)
        assert store.put(session_id="s1", user_id="alice", settings={"ask_user_mode": "auto"}) == 3
        sql, params = pool.record[0]
        assert "ON CONFLICT (user_id, session_id) DO UPDATE" in sql
        assert "WHERE" not in sql
        assert params[:2] == ("s1", "alice") and json.loads(params[2]) == {"ask_user_mode": "auto"}
        assert store.get("s1", user_id="alice") == (3, {"ask_user_mode": "auto"})
        assert pool.record[1][1] == ("s1", "alice")
        assert "AND user_id = %s" in pool.record[1][0]

    def test_migration_keys_the_row_by_user_and_session(self):
        ddl = (_MIGRATIONS / "0128_companion_venue_agent_settings.sql").read_text()
        assert "PRIMARY KEY (user_id, session_id)" in ddl
        assert "session_id TEXT        PRIMARY KEY" not in ddl


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            for name in ("0127_chat_turn_questions.sql", "0128_companion_venue_agent_settings.sql"):
                cur.execute((_MIGRATIONS / name).read_text())
    finally:
        pool.putconn(conn)
    return pool


class TestPostgres:
    def test_migrations_and_stores(self):
        pool = _pg_pool_or_skip()
        sid = f"parity-{uuid4().hex[:8]}"
        try:
            _pg_settings_leg(pool, sid)
            stale = _pg_questions_leg(pool, sid)
            assert cancel_session_questions(pool, session_id=sid, user_id="alice") == 1
            assert _on_conn(pool, lambda store: store.get(stale))["status"] == "cancelled"
        finally:
            _cleanup(pool, sid)
            pool.closeall()


def _on_conn(pool, fn):
    conn = pool.getconn()
    try:
        return fn(ChatTurnQuestionsStore(conn))
    finally:
        pool.putconn(conn)


def _pg_settings_leg(pool, sid: str) -> None:
    """0128 keyed (user_id, session_id): bob's row never blocks alice's."""
    settings = VenueAgentSettingsStore(pool)
    assert settings.put(session_id=sid, user_id="alice", settings={"a": 1}) == 1
    assert settings.put(session_id=sid, user_id="alice", settings={"a": 2}) == 2
    assert settings.put(session_id=sid, user_id="bob", settings={"b": 1}) == 1
    assert settings.get(sid, user_id="alice") == (2, {"a": 2})
    assert settings.get(sid, user_id="bob") == (1, {"b": 1})


def _pg_questions_leg(pool, sid: str) -> str:
    """Terminal transitions on 0127; returns a still-pending question id."""

    def leg(store: ChatTurnQuestionsStore) -> str:
        assert store.table_present() is True
        qid = _ask(store, session_id=sid)["question_id"]
        assert store.answer(qid, "b", answered_by="alice")["status"] == "answered"
        with pytest.raises(QuestionClosed):
            store.answer(qid, "c", answered_by="alice")
        assert store.expire(qid)["status"] == "answered"  # expiry keeps an answer
        return _ask(store, session_id=sid)["question_id"]

    return _on_conn(pool, leg)


def _cleanup(pool, sid: str) -> None:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chat_turn_questions WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM companion_venue_agent_settings WHERE session_id = %s", (sid,))
    finally:
        pool.putconn(conn)
