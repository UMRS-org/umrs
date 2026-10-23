"""ADR-099 D-8 (RS-1 lane A): remote ask_user questions and agent settings.

The worker raises a durable question through its rail and polls it; the
owner answers from any client. The owner's client sends the agent settings
its local bridge applies; the worker reads them through the rail.

Same fakes as the venue route tests (in-memory broker + registry,
FakePodman provisioner, snapshot pool, fake job store); the questions store
is the sqlite fallback, the settings store the in-memory one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.test_companion_venue_routes import (
    READY_ENV,
    FakeJobStore,
    FakePodman,
    FakeSealed,
    _meta,
    _Pool,
    _run_lane,
)
from umrs.api.companion_venue import router as venue_router
from umrs.api.companion_venue_parity import router as parity_router
from umrs.companion_trust.chat_turn_questions import ChatTurnQuestionsStore
from umrs.jobs.secret_broker import SecretBroker
from umrs.venues.agent_settings import VenueAgentSettingsStore
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

H = {"x-user-id": "alice"}
BASE = "/umrs/companion/sessions/sess-1"


class _TokenPodman(FakePodman):
    """Captures the session token the provisioner hands to podman."""

    def __init__(self):
        super().__init__()
        self.token = ""

    def __call__(self, *args, input_text=None, **kw):
        if args[:2] == ("secret", "create"):
            self.token = input_text or ""
        return super().__call__(*args, input_text=input_text, **kw)


@pytest.fixture
def h():
    pool = _Pool()
    broker = SecretBroker()
    podman = _TokenPodman()
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="venue-parity-"))
    )
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=provisioner,
        pool=pool,
        sealed_store=FakeSealed({("alice", "openai-main"): _meta("openai-main")}),
        job_store=FakeJobStore(),
        site_name="rccs",
    )
    questions = ChatTurnQuestionsStore()
    settings = VenueAgentSettingsStore()
    app = FastAPI()
    app.include_router(venue_router)
    app.include_router(parity_router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    app.state.companion_venue_questions_factory = lambda _conn: questions
    app.state.companion_venue_settings_store = settings
    harness = SimpleNamespace(
        client=TestClient(app),
        pool=pool,
        broker=broker,
        podman=podman,
        service=service,
        jobs=service.job_store,
        questions=questions,
        settings=settings,
    )
    pool.seed("alice", "sess-1")
    started = harness.client.post(
        f"{BASE}/venue/start",
        json={"source_refs": ["openai-main"], "web_policy": "none"},
        headers=H,
    )
    assert started.status_code == 200, started.text
    assert _run_lane(harness) == [("running", "job-1")]
    harness.rail = {"authorization": f"Bearer {podman.token}"}
    return harness


def _ask(h, question="Which dataset?", options=("a", "b"), **extra):
    body = {"turn_id": "t1", "question": question, "options": list(options), **extra}
    return h.client.post(f"{BASE}/rail/questions", json=body, headers=h.rail)


class TestQuestions:
    def test_worker_question_is_listed_for_the_owner(self, h):
        r = _ask(h)
        assert r.status_code == 200, r.text
        q = r.json()["question"]
        assert (q["status"], q["user_id"], q["options"]) == ("pending", "alice", ["a", "b"])
        listed = h.client.get(f"{BASE}/questions", headers=H)
        assert listed.status_code == 200
        assert [row["question_id"] for row in listed.json()["questions"]] == [q["question_id"]]

    def test_owner_answer_reaches_the_worker(self, h):
        qid = _ask(h).json()["question"]["question_id"]
        answered = h.client.post(
            f"{BASE}/questions/{qid}/answer", json={"answer": "use b"}, headers=H
        )
        assert answered.status_code == 200, answered.text
        assert answered.json()["question"]["answered_by"] == "alice"

        polled = h.client.get(f"{BASE}/rail/questions/{qid}", headers=h.rail).json()
        assert polled["question"]["status"] == "answered"
        assert polled["question"]["answer"] == "use b"
        assert h.client.get(f"{BASE}/questions", headers=H).json()["questions"] == []

    def test_answer_to_a_closed_question_is_409(self, h):
        qid = _ask(h).json()["question"]["question_id"]
        first = h.client.post(f"{BASE}/questions/{qid}/answer", json={"answer": "x"}, headers=H)
        assert first.status_code == 200
        again = h.client.post(f"{BASE}/questions/{qid}/answer", json={"answer": "y"}, headers=H)
        assert again.status_code == 409
        assert again.headers["x-platform-error-code"] == "companionvenue.question_closed"

    def test_expired_question_refuses_a_late_answer(self, h):
        qid = _ask(h).json()["question"]["question_id"]
        expired = h.client.post(f"{BASE}/rail/questions/{qid}/expire", headers=h.rail)
        assert expired.status_code == 200
        assert expired.json()["question"]["status"] == "expired"
        late = h.client.post(f"{BASE}/questions/{qid}/answer", json={"answer": "x"}, headers=H)
        assert late.status_code == 409
        assert late.headers["x-platform-error-code"] == "companionvenue.question_closed"

    def test_non_owner_gets_404(self, h):
        qid = _ask(h).json()["question"]["question_id"]
        mallory = {"x-user-id": "mallory"}
        listed = h.client.get(f"{BASE}/questions", headers=mallory)
        assert listed.status_code == 404
        answer = h.client.post(
            f"{BASE}/questions/{qid}/answer", json={"answer": "x"}, headers=mallory
        )
        assert answer.status_code == 404
        assert answer.headers["x-platform-error-code"] == "companionvenue.not_found"
        assert h.questions.get(qid)["status"] == "pending"

    def test_question_of_another_session_is_404(self, h):
        other = h.questions.ask(
            session_id="sess-2", turn_id="", user_id="alice", question="?", options=[]
        )
        qid = other["question_id"]
        assert h.client.get(f"{BASE}/rail/questions/{qid}", headers=h.rail).status_code == 404
        r = h.client.post(f"{BASE}/questions/{qid}/answer", json={"answer": "x"}, headers=H)
        assert r.status_code == 404

    def test_rail_needs_the_lease_token(self, h):
        assert _ask(h).status_code == 200
        r = h.client.post(
            f"{BASE}/rail/questions", json={"question": "?", "options": []}, headers=H
        )
        assert r.status_code == 401

    def test_question_id_of_another_session_is_refused(self, h):
        h.questions.ask(
            session_id="sess-2",
            turn_id="",
            user_id="bob",
            question="private",
            options=[],
            question_id="q-fixed",
        )
        r = _ask(h, question_id="q-fixed")
        assert r.status_code == 409
        assert "private" not in r.text

    def test_bounds(self, h):
        assert _ask(h, question="x" * 4001).status_code == 422
        qid = _ask(h).json()["question"]["question_id"]
        r = h.client.post(f"{BASE}/questions/{qid}/answer", json={"answer": "y" * 8001}, headers=H)
        assert r.status_code == 422
        bad = h.client.get(f"{BASE}/questions?status=bogus", headers=H)
        assert bad.status_code == 400


class TestSettings:
    DOC = {
        "schema": 1,
        "permission_mode": "auto_safe",
        "interaction_mode": "autopilot",
        "write_requires_plan": False,
        "workflow_armed": True,
        "ask_user_mode": "ask",
        "approval_ttl_s": 1800,
        "session_approved_tools": ["file_write"],
        "future_key": {"kept": True},
    }

    def test_rail_reads_nothing_before_the_first_push(self, h):
        r = h.client.get(f"{BASE}/rail/settings", headers=h.rail)
        assert r.status_code == 200
        assert r.json() == {"session_id": "sess-1", "rev": 0, "settings": {}}

    def test_owner_push_reaches_the_rail_with_a_growing_rev(self, h):
        first = h.client.put(f"{BASE}/venue/settings", json=self.DOC, headers=H)
        assert first.status_code == 200, first.text
        assert first.json()["rev"] == 1
        seen = h.client.get(f"{BASE}/rail/settings", headers=h.rail).json()
        assert seen["rev"] == 1
        assert seen["settings"]["interaction_mode"] == "autopilot"
        assert seen["settings"]["future_key"] == {"kept": True}
        second = h.client.put(
            f"{BASE}/venue/settings", json={**self.DOC, "ask_user_mode": "auto"}, headers=H
        )
        assert second.json()["rev"] == 2
        assert (
            h.client.get(f"{BASE}/rail/settings", headers=h.rail).json()["settings"][
                "ask_user_mode"
            ]
            == "auto"
        )

    def test_invalid_values_are_400_with_the_field(self, h):
        r = h.client.put(f"{BASE}/venue/settings", json={"permission_mode": "yolo"}, headers=H)
        assert r.status_code == 400
        assert r.headers["x-platform-error-code"] == "companionvenue.bad_request"
        assert "permission_mode" in r.text

    def test_non_owner_cannot_push(self, h):
        r = h.client.put(f"{BASE}/venue/settings", json=self.DOC, headers={"x-user-id": "mallory"})
        assert r.status_code == 404
        assert h.settings.get("sess-1") == (0, {})

    def test_rail_never_reads_another_users_row(self, h):
        h.settings.put(session_id="sess-1", user_id="bob", settings={"ask_user_mode": "auto"})
        seen = h.client.get(f"{BASE}/rail/settings", headers=h.rail).json()
        assert (seen["rev"], seen["settings"]) == (0, {})


def test_owner_routes_answer_a_typed_503_without_a_pool(h):
    h.service.pool = None
    r = h.client.get(f"{BASE}/questions", headers=H)
    assert r.status_code == 503
    assert r.headers["x-platform-error-code"] == "companionvenue.unavailable"
