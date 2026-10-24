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


def _save(w: Any, *, user: str, **extra: Any) -> str:
    """One /save through the ingest job's own insert path; the stored id."""
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge.session_turns import plan_session_save

    plan = plan_session_save(_source(**extra), actor="user", user_id=user)
    job = SimpleNamespace(
        user_id=user, spec={"graph_id": w.default, "origin_actor": "user", **plan.spec_extra}
    )
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
