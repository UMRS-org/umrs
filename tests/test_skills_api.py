from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.skills import router, set_skill_registry
from umrs.skills import SkillRegistry


def _client() -> TestClient:
    set_skill_registry(SkillRegistry())
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers(user: str = "alice", project: str = "project-a", admin: bool = False) -> dict[str, str]:
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
