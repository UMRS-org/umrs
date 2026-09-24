from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.skills import router, set_skill_registry
from umrs.skills import SkillRegistry, SkillScope

from r4_contracts.skills import SkillManifest


def _client() -> TestClient:
    set_skill_registry(SkillRegistry())
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers(
    user: str = "alice", project: str = "project-a", admin: bool = False
) -> dict[str, str]:
    return {
        "x-user-id": user,
        "x-project-id": project,
        "x-verified-project-id": project,
        "x-verified-is-admin": str(admin).lower(),
    }


def test_register_and_search_project_skill() -> None:
    client = _client()

    response = client.post(
        "/umrs/skills",
        headers=_headers(),
        json={
            "scope": "project",
            "manifest": {
                "name": "rag-literature",
                "version": "1.0.0",
                "description": "Search project literature with RAG",
                "tags": ["rag", "research"],
            },
            "skill_md": "Use project RAG before answering literature questions.",
        },
    )

    assert response.status_code == 201
    skill = response.json()
    assert skill["scope"] == "project"
    assert skill["project_id"] == "project-a"
    assert skill["owner_id"] == "alice"

    search = client.get("/umrs/skills/search?q=literature", headers=_headers())

    assert search.status_code == 200
    assert search.json()[0]["manifest"]["name"] == "rag-literature"


def test_project_skill_hidden_from_other_project() -> None:
    client = _client()
    response = client.post(
        "/umrs/skills",
        headers=_headers(project="project-a"),
        json={
            "scope": "project",
            "manifest": {"name": "project-only", "description": "private"},
            "skill_md": "Project A only.",
        },
    )
    assert response.status_code == 201

    hidden = client.get(
        "/umrs/skills/search?q=project-only",
        headers=_headers(project="project-b"),
    )

    assert hidden.status_code == 200
    assert hidden.json() == []


def test_system_skill_registration_requires_admin() -> None:
    client = _client()

    denied = client.post(
        "/umrs/skills",
        headers=_headers(admin=False),
        json={
            "scope": "system",
            "manifest": {"name": "system-skill", "description": "global"},
            "skill_md": "System instructions.",
        },
    )

    assert denied.status_code == 403

    allowed = client.post(
        "/umrs/skills",
        headers=_headers(user="root", admin=True),
        json={
            "scope": "system",
            "manifest": {"name": "system-skill", "description": "global"},
            "skill_md": "System instructions.",
        },
    )

    assert allowed.status_code == 201
    assert allowed.json()["scope"] == "system"


# ── D-27 skill-edit plane: pending proposals + merge-on-green review ────────


def _registry_with_proposal(name: str = "compiled-method", project: str = "project-a"):
    registry = SkillRegistry()
    proposal = registry.register(
        SkillManifest(name=name, version="1.0.0", description="loop-proposed"),
        "# compiled\n\n## Objective\nrun the standard analysis\n",
        scope=SkillScope.PROJECT,
        owner_id="compile-knowledge-skills",
        project_id=project,
        enabled=False,
        review_state={
            "status": "pending_review",
            "source_ir_ids": ["ir-1", "ir-2"],
            "proposed_at": "2026-09-24T00:00:00+00:00",
        },
    )
    return registry, proposal


def _green_receipt() -> dict:
    return {
        "runner": "r4 skill test",
        "exit_code": 0,
        "tests_present": True,
        "tested_at": "2026-09-24T00:01:00+00:00",
        "output_tail": "ok",
    }


def test_pending_lists_only_loop_proposals() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)
    registry.register(
        SkillManifest(name="normal-skill", version="1.0.0"),
        "enabled, never proposed",
        scope=SkillScope.PROJECT,
        owner_id="alice",
        project_id="project-a",
    )
    registry.register(  # decided earlier — must not surface as pending
        SkillManifest(name="rejected-earlier", version="1.0.0"),
        "decided",
        scope=SkillScope.PROJECT,
        owner_id="compile-knowledge-skills",
        project_id="project-a",
        enabled=False,
        review_state={"status": "rejected", "decision": "rejected"},
    )

    response = client.get("/umrs/skills/pending", headers=_headers())

    assert response.status_code == 200
    items = response.json()
    assert [item["skill_id"] for item in items] == [proposal.skill_id]
    assert items[0]["enabled"] is False
    assert items[0]["review_state"]["status"] == "pending_review"
    assert items[0]["review_state"]["source_ir_ids"] == ["ir-1", "ir-2"]


def test_pending_scoped_to_caller_project() -> None:
    client = _client()
    registry, _ = _registry_with_proposal()
    set_skill_registry(registry)

    other = client.get("/umrs/skills/pending", headers=_headers(project="project-b"))

    assert other.status_code == 200
    assert other.json() == []

    admin = client.get("/umrs/skills/pending", headers=_headers(user="root", admin=True))

    assert admin.status_code == 200
    assert len(admin.json()) == 1


def test_approve_without_receipt_refused_by_name() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "approved"},
    )

    assert response.status_code == 403
    assert response.headers.get("x-platform-error-code") == "skills.merge_not_green"


def test_approve_with_no_tests_receipt_refused() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "approved", "test_receipt": {**_green_receipt(), "tests_present": False}},
    )

    assert response.status_code == 403
    assert response.headers.get("x-platform-error-code") == "skills.merge_not_green"


def test_approve_with_red_receipt_refused() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "approved", "test_receipt": {**_green_receipt(), "exit_code": 3}},
    )

    assert response.status_code == 403
    assert response.headers.get("x-platform-error-code") == "skills.merge_not_green"


def test_approve_with_green_receipt_merges() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(user="alice"),
        json={"decision": "approved", "note": "verified", "test_receipt": _green_receipt()},
    )

    assert response.status_code == 200
    merged = response.json()
    assert merged["enabled"] is True
    state = merged["review_state"]
    assert state["status"] == "approved"
    assert state["reviewer"] == "alice"
    assert state["note"] == "verified"
    assert state["test_receipt"]["exit_code"] == 0
    assert state["source_ir_ids"] == ["ir-1", "ir-2"]  # proposal block survives

    listed = client.get("/umrs/skills/pending", headers=_headers())
    assert listed.json() == []


def test_agent_actor_cannot_review() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers={**_headers(), "x-r4-actor": "agent"},
        json={"decision": "rejected"},
    )

    assert response.status_code == 403
    assert response.headers.get("x-platform-error-code") == "skills.forbidden"


def test_reject_keeps_disabled_then_conflict_on_re_review() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    rejected = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "rejected", "note": "not reusable yet"},
    )

    assert rejected.status_code == 200
    assert rejected.json()["enabled"] is False
    assert rejected.json()["review_state"]["status"] == "rejected"

    redecided = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "approved", "test_receipt": _green_receipt()},
    )

    assert redecided.status_code == 409
    assert redecided.headers.get("x-platform-error-code") == "skills.conflict"


def test_review_of_non_proposal_is_conflict() -> None:
    client = _client()
    registry, _ = _registry_with_proposal()
    set_skill_registry(registry)
    normal = registry.register(
        SkillManifest(name="normal-skill", version="1.0.0"),
        "enabled, never proposed",
        scope=SkillScope.PROJECT,
        owner_id="alice",
        project_id="project-a",
    )

    response = client.post(
        f"/umrs/skills/{normal.skill_id}/review",
        headers=_headers(),
        json={"decision": "approved", "test_receipt": _green_receipt()},
    )

    assert response.status_code == 409
    assert response.headers.get("x-platform-error-code") == "skills.conflict"


def test_review_unknown_skill_not_found() -> None:
    client = _client()
    registry, _ = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        "/umrs/skills/skl-missing/review",
        headers=_headers(),
        json={"decision": "rejected"},
    )

    assert response.status_code == 404
    assert response.headers.get("x-platform-error-code") == "skills.not_found"


def test_review_bad_decision_is_bad_request() -> None:
    client = _client()
    registry, proposal = _registry_with_proposal()
    set_skill_registry(registry)

    response = client.post(
        f"/umrs/skills/{proposal.skill_id}/review",
        headers=_headers(),
        json={"decision": "maybe"},
    )

    assert response.status_code == 400
    assert response.headers.get("x-platform-error-code") == "skills.bad_request"
