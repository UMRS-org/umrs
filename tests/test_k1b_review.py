"""Review of K1 part B (knowledge-sharing plan, 2026-10-24): the /save
findings of the security, correctness and product reviews.

* H1: a raw ``knowledge_ingest`` / ``knowledge_understand`` spec submitted
  through ``POST /umrs/jobs`` set ``kept_origin``, ``established_by`` and a
  ``session_turns`` source: unscanned text landed established "vouched by"
  any user it named. Only the ingest route builds a save now, and the
  establisher is the job's owner.
* M1 / correctness 1: the save scan reused the egress key-name patterns, so
  ``pwd = os.getcwd()`` refused an ordinary answer while ``{"password":
  "…"}``, URL credentials, ``Bearer …``, Google/GitLab/Stripe keys, text past
  500 000 characters and the title and tags passed.
* M2 / product 1: ``/save tN --vouch`` after a keep changed nothing (the
  bytes deduplicate, the contribution keeps its first origin, the rows are
  not re-extracted) while the receipt said "established": refused by name.
* Product 2: the route returns the origin and tier it decided, so the
  client prints the server's answer, not its own guess.
* L2: a member who added identical bytes saw another member's save
  metadata (session id, title, tags).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from test_k1_session_turns import _source, ingest  # noqa: F401 — ingest: fixture
from test_memfix_spec_project import _post


@pytest.fixture
def route(request: Any) -> Any:
    """The K1.6 ingest route with its runtime stubbed (``ingest``)."""
    return request.getfixturevalue("ingest")


# ── M1 / correctness 1: the save scan ─────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Set max_token: 512 in the config",
        "token = tok[0]",
        "pwd = os.getcwd()",
        "the password: field is not needed",
        "export API_KEY=${API_KEY}",
        "password: ********",
        "access_token = response.json()['access_token']",
    ],
)
def test_ordinary_answers_with_code_are_saved(text: str) -> None:
    from umrs.knowledge.session_turns import plan_session_save

    save = plan_session_save(_source(text=text), actor="user", user_id="alice")
    assert save.source["content"] == text


_LONG_PAD = "lorem ipsum " * 50_000  # 600 000 characters


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ('{"password": "S3cretPw!"}', "password"),
        ("{'api_key': 'a1b2c3d4e5f6g7h8'}", "api_key"),
        ("postgresql://umrs:S3cretPw@db.example:5432/umrs", "url_credentials"),
        ("Authorization: Bearer 9f8e7d6c5b4a39281706f5e4d3c2b1a0", "bearer_token"),
        ("key AIza" + "B" * 35, "google_api_key"),
        ("glpat-" + "x1Y2z3" * 4, "gitlab_token"),
        ("sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc", "stripe_key"),
        (_LONG_PAD + " password: hunter2", "password"),
    ],
)
def test_credentials_the_egress_patterns_missed_are_refused(text: str, name: str) -> None:
    from umrs.knowledge.session_turns import SessionSaveRefused, plan_session_save

    with pytest.raises(SessionSaveRefused) as refused:
        plan_session_save(_source(text=text), actor="user", user_id="alice")
    assert refused.value.code == "secret_detected"
    assert name in str(refused.value)
    assert "S3cretPw" not in str(refused.value)


@pytest.mark.parametrize(
    "labels",
    [
        {"title": "notes ghp_" + "A1b2C3d4E5f6G7h8I9j0K1"},
        {"tags": ["ok", "hf_" + "a1B2" * 9]},
    ],
)
def test_a_token_in_the_title_or_tags_is_refused(labels: dict[str, Any]) -> None:
    from umrs.knowledge.session_turns import SessionSaveRefused, plan_session_save

    with pytest.raises(SessionSaveRefused):
        plan_session_save(_source(**labels), actor="user", user_id="alice")


# ── H1: only the ingest route builds a save ───────────────────────────────


def _forged_spec(**over: Any) -> dict[str, Any]:
    return {
        "origin_actor": "user",
        "kept_origin": "agent_authored_user_vouched",
        "established_by": "alice",
        "sources": [{"kind": "session_turns", "uri": "x.md", "content": "model text"}],
        **over,
    }


@pytest.mark.parametrize("job_type", ["knowledge_ingest", "knowledge_understand"])
@pytest.mark.parametrize(
    "spec",
    [
        _forged_spec(),
        _forged_spec(kept_origin=None, established_by=None),
        {"sources": [{"kind": "inline_text", "content": "x"}], "established_by": "alice"},
    ],
)
def test_post_jobs_refuses_a_client_built_save(monkeypatch, job_type, spec) -> None:
    spec = {k: v for k, v in spec.items() if v is not None}
    resp, store = _post(monkeypatch, {"job_type": job_type, "spec": spec})
    assert resp.status_code == 400, resp.text
    assert "/umrs/k2/ingest" in resp.text
    assert store.created == []


def test_post_jobs_refuses_a_forged_save_in_a_fan_out_child(monkeypatch) -> None:
    child = {"job_type": "knowledge_ingest", "spec": _forged_spec()}
    resp, store = _post(
        monkeypatch, {"job_type": "knowledge_verify", "spec": {}, "fan_out_jobs": [child]}
    )
    assert resp.status_code == 400, resp.text
    assert store.created == []


def test_post_jobs_still_takes_a_plain_knowledge_ingest(monkeypatch) -> None:
    spec = {"sources": [{"kind": "inline_text", "uri": "a.md", "content": "x"}]}
    resp, store = _post(monkeypatch, {"job_type": "knowledge_ingest", "spec": spec})
    assert resp.status_code == 201, resp.text
    assert len(store.created) == 1


def test_the_shared_refusal_covers_both_seams() -> None:
    from umrs.jobs.models import JobType
    from umrs.server.job_tracking_handler import spec_save_refusal

    assert spec_save_refusal(JobType.KNOWLEDGE_INGEST, _forged_spec())
    assert spec_save_refusal(JobType.KNOWLEDGE_UNDERSTAND, {"kept_origin": "x"})
    assert spec_save_refusal(JobType.KNOWLEDGE_INGEST, {"sources": []}) is None
    assert spec_save_refusal(JobType.DATA_INGESTION, _forged_spec()) is None


def test_the_establisher_is_the_jobs_owner_not_a_spec_field(monkeypatch) -> None:
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_llm_extract_document", lambda *_a, **_k: None)
    nodes, edges = ka._nodes_edges_for_artifact(
        "art-s", "ir-s", "document", b"text", {}, "proj-k16", "g-k16"
    )
    spec = {"kept_origin": "agent_authored_user_vouched", "established_by": "alice"}
    job = SimpleNamespace(spec=spec, job_id="job-m", user_id="mallory")
    nodes, _edges = ka._tiered_rows(nodes, edges, job, "user")
    assert {n.content["review"]["established_by"] for n in nodes} == {"mallory"}


# ── product 2: the route says what it decided ─────────────────────────────


@pytest.mark.parametrize(
    ("extra", "headers", "decided"),
    [
        ({}, {}, {"origin": "agent_authored_user_kept", "tier": "proposed"}),
        ({"role": "user"}, {}, {"origin": "user", "tier": "established"}),
        ({}, {"x-r4-actor": "agent"}, {"origin": "agent", "tier": "proposed"}),
    ],
)
def test_the_route_returns_the_origin_and_tier_it_decided(route, extra, headers, decided):
    client, _specs = route
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [_source(**extra)]},
        headers={"x-user-id": "alice", **headers},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["save"] == decided


def test_an_upload_carries_no_save_receipt(route) -> None:
    client, _specs = route
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [{"kind": "inline_text", "uri": "a.md", "content": "x"}]},
        headers={"x-user-id": "alice"},
    )
    assert resp.status_code == 200 and "save" not in resp.json()


# ── M2: a vouch over a stored proposed save is refused by name ────────────


def _stored(monkeypatch, origins: set[str]) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []

    async def _pool(_request: Any) -> object:
        return object()

    async def _origins(_pool: Any, project: str, _hash: str, graph: str) -> set[str]:
        seen.append((project, graph))
        return origins

    monkeypatch.setattr("umrs.api._asyncpg_pool.knowledge_asyncpg_pool", _pool)
    monkeypatch.setattr("umrs.knowledge.contributions.stored_save_origins", _origins)
    return seen


@pytest.mark.parametrize("stored", [{"agent_authored_user_kept"}, {"agent", "user"}])
def test_a_vouch_over_a_stored_proposed_save_is_refused(route, monkeypatch, stored):
    client, specs = route
    seen = _stored(monkeypatch, stored)
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [_source(vouch=True)]},
        headers={"x-user-id": "alice"},
    )
    assert resp.status_code == 400, resp.text
    assert "already in the graph as proposed" in resp.text
    assert "r4 knowledge review" in resp.text
    assert seen == [("proj-k16", "g-k16")] and specs == []


@pytest.mark.parametrize("stored", [set(), {"user"}, {"agent_authored_user_vouched"}])
def test_a_first_vouch_is_queued(route, monkeypatch, stored):
    client, specs = route
    _stored(monkeypatch, stored)
    resp = client.post(
        "/umrs/k2/ingest",
        json={"sources": [_source(vouch=True)]},
        headers={"x-user-id": "alice"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["save"]["tier"] == "established"
    assert specs[0]["kept_origin"] == "agent_authored_user_vouched"
    assert "established_by" not in specs[0]


# ── L2: another member's save metadata ────────────────────────────────────


def _row(saved_by: str, contributors: list[str]) -> dict[str, Any]:
    meta = {
        "filename": "t2.md",
        "session_id": "sess-alice",
        "title": "Alice's notes",
        "tags": ["private"],
        "saved_by": saved_by,
        "attested_by_client": True,
    }
    return {"metadata": meta, "contributions": [{"user_id": u} for u in contributors]}


def test_save_metadata_is_hidden_from_a_member_who_cannot_see_the_save() -> None:
    from umrs.knowledge.contributions import public_metadata

    meta = public_metadata(_row("alice", ["bob"]), "bob")
    assert meta == {"filename": "t2.md"}


@pytest.mark.parametrize(
    ("viewer", "contributors"), [("alice", ["bob"]), ("bob", ["alice", "bob"])]
)
def test_save_metadata_shows_to_the_saver_or_beside_a_visible_save(viewer, contributors) -> None:
    from umrs.knowledge.contributions import public_metadata

    assert public_metadata(_row("alice", contributors), viewer)["session_id"] == "sess-alice"


def test_metadata_without_a_save_is_unchanged() -> None:
    from umrs.knowledge.contributions import public_metadata

    row = {"metadata": {"filename": "a.md", "title": "A"}, "contributions": []}
    assert public_metadata(row, "bob") == {"filename": "a.md", "title": "A"}
