"""K1.6 (knowledge-sharing plan, 2026-10-24): ``/save`` keeps chat turns
verbatim, attributed, in the right tier.

Before K1.6 there was no ``session_turns`` source kind (``/k2/ingest``
refused it by name), so a chat answer reached knowledge only when the model
re-typed it (``knowledge_build_from_context``): no secret scan, no session
back-link, and the tier followed the request actor alone, so a kept
assistant answer landed established as the person's.

K-1 (ADR-084 D-2, decided 2026-10-24): the person's own text lands
established; an assistant answer lands proposed (origin
``agent_authored_user_kept``) unless the person vouches, which lands it
established with them as establisher; an agent-initiated save is proposed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

GH_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
ANSWER = "  MOF-5 binds CO2 at 0.8 mmol/g (S1).\n\nSecond paragraph, kept as is.\n"


def _source(role: str = "assistant", text: str = ANSWER, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "session_turns",
        "session_id": "sess-42",
        "turns": [{"turn_id": "t2", "role": role, "text": text}],
        "title": "MOF-5 uptake",
        "tags": ["mof", "co2"],
        **extra,
    }


# ── the plan: tier, verbatim text, back-link ──────────────────────────────


def test_a_kept_assistant_answer_is_proposed_and_verbatim() -> None:
    from umrs.knowledge.session_turns import KEPT_ORIGIN, plan_session_save

    save = plan_session_save(_source(), actor="user", user_id="alice")
    assert save.origin == KEPT_ORIGIN
    assert save.spec_extra == {"kept_origin": KEPT_ORIGIN}
    assert save.source["content"] == ANSWER  # exactly turn 2's text
    meta = save.source["metadata"]
    assert meta["attested_by_client"] is True
    assert (meta["session_id"], meta["turn_ids"], meta["roles"]) == (
        "sess-42",
        ["t2"],
        ["assistant"],
    )
    assert (meta["title"], meta["tags"]) == ("MOF-5 uptake", ["mof", "co2"])


def test_a_vouched_answer_is_established_by_the_person() -> None:
    from umrs.knowledge.session_turns import VOUCHED_ORIGIN, plan_session_save

    save = plan_session_save(_source(vouch=True), actor="user", user_id="alice")
    assert save.origin == VOUCHED_ORIGIN
    # The establisher is the job's owner at write time, never a spec field
    # (review of K1 part B: a raw spec named any user).
    assert save.spec_extra == {"kept_origin": VOUCHED_ORIGIN}
    assert save.established_by == "alice"


def test_the_persons_own_text_is_established() -> None:
    from umrs.knowledge.session_turns import plan_session_save

    save = plan_session_save(_source(role="user"), actor="", user_id="alice")
    assert (save.origin, save.spec_extra, save.established_by) == ("user", {}, "alice")


def test_an_agent_initiated_save_is_proposed_and_cannot_vouch() -> None:
    from umrs.knowledge.session_turns import plan_session_save

    save = plan_session_save(_source(vouch=True), actor="agent", user_id="alice")
    assert (save.origin, save.spec_extra) == ("agent", {})


@pytest.mark.parametrize(
    "text",
    [
        f"use this: {GH_TOKEN}",
        "header eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1LTEifQ.c2lnbmF0dXJlLWhlcmU",
        "password: hunter2",
    ],
)
def test_token_shaped_text_is_refused_by_name(text: str) -> None:
    from umrs.knowledge.session_turns import SessionSaveRefused, plan_session_save

    with pytest.raises(SessionSaveRefused) as refused:
        plan_session_save(_source(text=text), actor="user", user_id="alice")
    assert refused.value.code == "secret_detected"
    assert "credential-like" in str(refused.value)
    assert GH_TOKEN not in str(refused.value)  # a masked span, never the secret


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "session_turns", "turns": []},
        {"kind": "session_turns", "turns": [{"turn_id": "t2", "role": "tool", "text": "x"}]},
        {"kind": "session_turns", "turns": [{"turn_id": "t2", "role": "user", "text": "  "}]},
    ],
)
def test_malformed_turns_are_refused(source: dict[str, Any]) -> None:
    from umrs.knowledge.session_turns import SessionSaveRefused, plan_session_save

    with pytest.raises(SessionSaveRefused):
        plan_session_save(source, actor="user", user_id="alice")


# ── the route: scanned before anything is queued ──────────────────────────


@pytest.fixture
def ingest(monkeypatch) -> tuple[TestClient, list[dict[str, Any]]]:
    from umrs.api import knowledge_v2
    from umrs.api.files import get_project_id

    specs: list[dict[str, Any]] = []

    class _Runtime:
        async def submit_job_sync(self, **kwargs: Any) -> str:
            specs.append(kwargs["spec"])
            return "job-k16"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    monkeypatch.setattr(knowledge_v2, "_resolve_ingest_target", lambda *_a, **_k: "g-k16")
    monkeypatch.setattr("umrs.api.governance.guard_ingest", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.enforce_graph_mutation_authorization",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.mark_proposal_executed", lambda *_a, **_k: None
    )
    app = FastAPI()
    app.include_router(knowledge_v2.router)
    app.dependency_overrides[get_project_id] = lambda: "proj-k16"
    return TestClient(app, raise_server_exceptions=False), specs


def test_the_route_queues_the_verbatim_turn_with_its_tier(ingest) -> None:
    client, specs = ingest
    resp = client.post(
        "/umrs/k2/ingest", json={"sources": [_source()]}, headers={"x-user-id": "alice"}
    )
    assert resp.status_code == 200, resp.text
    (spec,) = specs
    assert (spec["origin_actor"], spec["kept_origin"]) == ("user", "agent_authored_user_kept")
    (source,) = spec["sources"]
    assert (source["kind"], source["content"]) == ("session_turns", ANSWER)
    assert "turns" not in source  # only the server-built source travels


def test_the_route_refuses_a_token_before_queueing(ingest) -> None:
    client, specs = ingest
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [_source(text=f"key {GH_TOKEN}")]},
        headers={"x-user-id": "alice"},
    )
    assert resp.status_code == 400
    assert "github_token" in resp.text and GH_TOKEN not in resp.text
    assert specs == []


def test_a_session_save_travels_alone(ingest) -> None:
    client, specs = ingest
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [_source(), {"kind": "inline_text", "uri": "a.md", "content": "x"}]},
        headers={"x-user-id": "alice"},
    )
    assert resp.status_code == 400 and specs == []


# ── the job: stored metadata, contribution origin, row tier ───────────────


class _Storage:
    def put_artifact(self, _project: str, content: bytes) -> tuple[str, int, None]:
        import hashlib

        return hashlib.sha256(content).hexdigest(), len(content), None

    def key(self, project: str, content_hash: str) -> str:
        return f"projects/{project}/knowledge/artifacts/{content_hash}"


def test_the_stored_artifact_keeps_the_back_link(monkeypatch) -> None:
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge.session_turns import plan_session_save

    monkeypatch.setattr(ka, "_bridge_a2l_ref", lambda *_a: None)
    source = plan_session_save(_source(), actor="user", user_id="alice").source
    result = ka.KnowledgeIngestResult()
    ids: list[str] = []
    descs: list[dict[str, Any]] = []
    ka._put_inline_text_sync("j", "a", _Storage(), "proj-k16", source, "art-s", result, ids, descs)
    (desc,) = descs
    meta = desc["metadata"]
    assert (meta["session_id"], meta["turn_ids"], meta["attested_by_client"]) == (
        "sess-42",
        ["t2"],
        True,
    )
    assert meta["source_name"] == "MOF-5 uptake"


@pytest.mark.parametrize(
    ("spec", "origin"),
    [
        (
            {"origin_actor": "user", "kept_origin": "agent_authored_user_kept"},
            "agent_authored_user_kept",
        ),
        (
            {
                "origin_actor": "user",
                "kept_origin": "agent_authored_user_vouched",
                "established_by": "alice",
            },
            "agent_authored_user_vouched",
        ),
        ({"origin_actor": "agent", "kept_origin": "agent_authored_user_vouched"}, "agent"),
        ({"origin_actor": "user"}, "user"),
    ],
)
def test_the_contribution_records_the_save_origin(spec, origin) -> None:
    from umrs.jobs import knowledge_adapters as ka

    job = SimpleNamespace(user_id="alice", spec=spec)
    assert ka._ingest_contributor(job, "proj-k16").origin_actor == origin


def _rows(monkeypatch, spec: dict[str, Any]) -> list[Any]:
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_llm_extract_document", lambda *_a, **_k: None)
    nodes, edges = ka._nodes_edges_for_artifact(
        "art-s", "ir-s", "document", ANSWER.encode(), {}, "proj-k16", "g-k16"
    )
    job = SimpleNamespace(spec=spec, job_id="job-k16", user_id="alice")
    nodes, _edges = ka._tiered_rows(nodes, edges, job, spec.get("origin_actor", "user"))
    return nodes


def test_a_kept_answer_lands_proposed(monkeypatch) -> None:
    nodes = _rows(monkeypatch, {"origin_actor": "user", "kept_origin": "agent_authored_user_kept"})
    assert nodes and {n.status for n in nodes} == {"proposed"}
    assert {n.content["origin_actor"] for n in nodes} == {"agent_authored_user_kept"}


def test_a_vouched_answer_lands_established_with_its_establisher(monkeypatch) -> None:
    spec = {"origin_actor": "user", "kept_origin": "agent_authored_user_vouched"}
    nodes = _rows(monkeypatch, spec)
    assert nodes and {n.status for n in nodes} == {"established"}
    assert {n.content["review"]["established_by"] for n in nodes} == {"alice"}


def test_an_agent_save_lands_proposed(monkeypatch) -> None:
    nodes = _rows(monkeypatch, {"origin_actor": "agent"})
    assert {n.status for n in nodes} == {"proposed"}


def test_the_understand_job_inherits_the_tier_fields() -> None:
    from umrs.knowledge.session_turns import tier_fields

    spec = {
        "origin_actor": "user",
        "kept_origin": "agent_authored_user_vouched",
        "established_by": "mallory",  # a stale or forged field never travels
    }
    assert tier_fields(spec) == {"kept_origin": "agent_authored_user_vouched"}
    assert tier_fields({"origin_actor": "user"}) == {}
