"""K1.6 on Postgres: a ``/save`` lands as an artifact the project lists with
the person's name and the save's origin (knowledge plan 2026-10-24).

The route builds the source (``plan_session_save``), the ingest job stores
it (``_put_inline_text_sync``, ``_store_artifacts_in_db_sync``) and the
contribution records who saved it and whether they kept or vouched for an
assistant answer. Skipped without the CI-shape Postgres.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from test_k1_contributions_pg import (  # noqa: F401 — sync_pool/world: fixtures
    _client,
    _db_url,
    sync_pool,
    world,
)
from test_k1_session_turns import _source, _Storage

pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def env(request: Any) -> tuple[Any, Any]:
    return request.getfixturevalue("world"), request.getfixturevalue("sync_pool")


def _save(w: Any, *, user: str, graph: str | None = None, **extra: Any) -> str:
    """One /save through the ingest job's own insert path; the stored id."""
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge.session_turns import plan_session_save

    plan = plan_session_save(_source(**extra), actor="user", user_id=user)
    spec = {"graph_id": graph or w.default, "origin_actor": "user", **plan.spec_extra}
    job = SimpleNamespace(user_id=user, spec=spec)
    result = ka.KnowledgeIngestResult()
    descs: list[dict[str, Any]] = []
    ka._put_inline_text_sync(
        "j", "a", _Storage(), w.project, plan.source, "art-s0", result, [], descs
    )
    stored = ka._store_artifacts_in_db_sync(
        w.project, descs, contributor=ka._ingest_contributor(job, w.project)
    )
    assert not stored.failed, stored.failed
    return stored.deduped.get("art-s0", "art-s0")


def test_a_kept_answer_is_listed_with_who_saved_it(env, monkeypatch) -> None:
    w, pool = env
    monkeypatch.setattr("umrs.jobs.knowledge_adapters._bridge_a2l_ref", lambda *_a: None)
    aid = _save(w, user="alice")
    resp = _client("bob", w.project, pool, monkeypatch).get("/umrs/k2/artifacts")
    assert resp.status_code == 200, resp.text
    (row,) = [a for a in resp.json()["artifacts"] if a["artifact_id"] == aid]
    (contribution,) = row["contributions"]
    assert (contribution["user_id"], contribution["origin_actor"]) == (
        "alice",
        "agent_authored_user_kept",
    )
    assert contribution["title"] == "MOF-5 uptake"
    assert row["metadata"]["attested_by_client"] is True
    assert row["metadata"]["session_id"] == "sess-42"


def test_a_vouched_answer_records_the_vouch(env, monkeypatch) -> None:
    w, pool = env
    monkeypatch.setattr("umrs.jobs.knowledge_adapters._bridge_a2l_ref", lambda *_a: None)
    aid = _save(w, user="alice", vouch=True, text="A vouched answer.")
    resp = _client("alice", w.project, pool, monkeypatch).get("/umrs/k2/artifacts")
    (row,) = [a for a in resp.json()["artifacts"] if a["artifact_id"] == aid]
    assert row["contributions"][0]["origin_actor"] == "agent_authored_user_vouched"


# ── review of K1 part B (2026-10-24) ──────────────────────────────────────


def _ingest_client(user: str, w: Any, pool: Any, monkeypatch: Any) -> tuple[Any, list[Any]]:
    """``POST /umrs/k2/ingest`` on the K1.2 test app, the runtime stubbed."""
    specs: list[Any] = []

    class _Runtime:
        async def submit_job_sync(self, **kwargs: Any) -> str:
            specs.append(kwargs["spec"])
            return "job-pg"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    monkeypatch.setattr("umrs.api.governance.guard_ingest", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.enforce_graph_mutation_authorization",
        lambda *_a, **_k: None,
    )
    return _client(user, w.project, pool, monkeypatch), specs


def test_a_vouch_after_a_keep_is_refused_by_name(env, monkeypatch) -> None:
    """``s`` keeps an answer, then ``/save t2 --vouch`` (the receipt's own
    hint): the contribution stayed kept and the rows proposed, while the
    client printed "established"."""
    w, pool = env
    monkeypatch.setattr("umrs.jobs.knowledge_adapters._bridge_a2l_ref", lambda *_a: None)
    _save(w, user="alice")
    client, specs = _ingest_client("alice", w, pool, monkeypatch)
    resp = client.post("/umrs/k2/ingest", json={"sources": [_source(vouch=True)]})
    assert resp.status_code == 400, resp.text
    assert "already in the graph as proposed" in resp.text
    assert specs == []


def test_a_first_vouch_reaches_the_queue_with_its_tier(env, monkeypatch) -> None:
    w, pool = env
    client, specs = _ingest_client("alice", w, pool, monkeypatch)
    resp = client.post(
        "/umrs/k2/ingest", json={"sources": [_source(vouch=True, text="Fresh answer.")]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["save"] == {"origin": "agent_authored_user_vouched", "tier": "established"}
    assert specs[0]["kept_origin"] == "agent_authored_user_vouched"


def test_identical_bytes_do_not_show_another_members_save_metadata(env, monkeypatch) -> None:
    """Alice saves into her personal graph; bob adds the same bytes to the
    default graph: his listing showed her session id, title and tags."""
    from test_k1_contributions_pg import _ingest

    w, pool = env
    monkeypatch.setattr("umrs.jobs.knowledge_adapters._bridge_a2l_ref", lambda *_a: None)
    aid = _save(w, user="alice", graph=w.personal, text="shared bytes")
    assert _ingest(w, user="bob", graph=w.default, content="shared bytes") == aid
    bob = _client("bob", w.project, pool, monkeypatch).get("/umrs/k2/artifacts")
    (row,) = [a for a in bob.json()["artifacts"] if a["artifact_id"] == aid]
    assert "session_id" not in row["metadata"] and "title" not in row["metadata"]
    alice = _client("alice", w.project, pool, monkeypatch).get("/umrs/k2/artifacts")
    (mine,) = [a for a in alice.json()["artifacts"] if a["artifact_id"] == aid]
    assert mine["metadata"]["session_id"] == "sess-42"
