from __future__ import annotations

import json

from umrs.core.TaskCmdStatus import TaskCmd
from umrs.model_registry.models import ModelScope
from umrs.server import model_registry_handler as handler


def test_handle_register_rejects_system_scope_for_non_admin(tmp_path, monkeypatch):
    model_dir = tmp_path / "system-model"
    model_dir.mkdir()

    monkeypatch.setenv("MODEL_ALLOWED_PATHS", f"{tmp_path}/*")
    monkeypatch.setattr(handler, "_catalog_path", lambda: str(tmp_path / "catalog.json"))

    payload = json.dumps(
        {
            "name": "system-model",
            "path": str(model_dir),
            "scope": ModelScope.SYSTEM.value,
        }
    )

    result = json.loads(handler._handle_register("alice", payload))

    assert result["status"] == "error"
    assert "system" in result["message"].lower()


def test_handle_register_rejects_unapproved_project_path(tmp_path, monkeypatch):
    model_dir = tmp_path / "outside-project-root"
    model_dir.mkdir()

    monkeypatch.setenv("MODEL_ALLOWED_PATHS", f"{tmp_path}/*")
    monkeypatch.setenv("UMRS_MODEL_PROJECT_MEMBERSHIPS", json.dumps({"alice": ["proj-123"]}))
    monkeypatch.setattr(handler, "_catalog_path", lambda: str(tmp_path / "catalog.json"))

    payload = json.dumps(
        {
            "name": "project-model",
            "path": str(model_dir),
            "scope": ModelScope.PROJECT.value,
            "project_id": "proj-123",
        }
    )

    result = json.loads(handler._handle_register("alice", payload))

    assert result["status"] == "error"
    assert "allowed" in result["message"].lower()


def test_handle_download_rejects_project_scope_without_membership(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MODEL_ALLOWED_PATHS",
        "/shared/projects/{project_id}/models/*:/shared/users/{owner_id}/models/*",
    )
    monkeypatch.setenv("UMRS_MODEL_PROJECT_MEMBERSHIPS", json.dumps({"alice": ["proj-123"]}))
    monkeypatch.setattr(handler, "_catalog_path", lambda: str(tmp_path / "catalog.json"))

    payload = json.dumps(
        {
            "repo_id": "org/demo-model",
            "name": "demo-model",
            "scope": ModelScope.PROJECT.value,
            "project_id": "proj-999",
            "staging_dir": "/shared/projects/proj-999/models/experimental",
        }
    )

    result = json.loads(handler._handle_download("alice", payload))

    assert result["status"] == "error"
    assert "project" in result["message"].lower()


def test_handle_list_filters_models_by_visibility(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "_catalog_path", lambda: str(tmp_path / "catalog.json"))
    monkeypatch.setenv("UMRS_MODEL_PROJECT_MEMBERSHIPS", json.dumps({"alice": ["proj-123"]}))

    handler._save_catalog(
        {
            "models": [
                {"name": "sys", "scope": "system", "storage_ref": "/models/sys"},
                {
                    "name": "mine",
                    "scope": "user",
                    "owner_id": "alice",
                    "storage_ref": "/shared/users/alice/models/mine",
                },
                {
                    "name": "other",
                    "scope": "user",
                    "owner_id": "bob",
                    "storage_ref": "/shared/users/bob/models/other",
                },
                {
                    "name": "team",
                    "scope": "project",
                    "project_id": "proj-123",
                    "storage_ref": "/shared/projects/proj-123/models/team",
                },
                {
                    "name": "secret-team",
                    "scope": "project",
                    "project_id": "proj-999",
                    "storage_ref": "/shared/projects/proj-999/models/secret",
                },
            ]
        }
    )

    result = json.loads(handler._handle_list("alice", "{}"))

    assert result["status"] == "ok"
    assert [model["name"] for model in result["models"]] == ["sys", "mine", "team"]


def test_handle_model_command_prefers_backend_for_list():
    class DummyBackend:
        def __init__(self):
            self.calls = []

        def list_models(self, client_id, payload_str):
            self.calls.append((client_id, payload_str))
            return json.dumps({"status": "ok", "models": [{"name": "shared-model"}]})

    backend = DummyBackend()

    result = json.loads(
        handler.handle_model_command(TaskCmd.MODEL_LIST, "alice", "{}", backend=backend)
    )

    assert result["status"] == "ok"
    assert result["models"] == [{"name": "shared-model"}]
    assert backend.calls == [("alice", "{}")]
