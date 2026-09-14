"""Route-level tests for UMRS model job submission paths."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.jobs import get_job_store
from umrs.api.models import get_model_store, router
from umrs.jobs.models import JobType
from umrs.model_registry.models import (
    ExecutionAuthority,
    ModelEntry,
    ModelScope,
    ModelStatus,
    ModelType,
    ServingProfile,
)


def _create_test_client(store, job_store) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    return TestClient(app)


def test_stage_model_enqueues_durable_job_with_caller_project_id():
    store = MagicMock()
    job_store = MagicMock()

    staged_model = ModelEntry(
        model_id="mdl-stage-123",
        name="qwen-7b-user",
        scope=ModelScope.USER,
        owner_id="alice",
        storage_ref="/shared/users/alice/models/qwen-7b-user",
        model_type=ModelType.LLM,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        description="User-scoped Qwen",
        serving_profile=ServingProfile(),
        execution_authority_default=ExecutionAuthority.COMPOSE,
        status=ModelStatus.STAGING,
    )
    store.create_model.return_value = staged_model
    job_store.create_job.return_value = "job-123"

    with _create_test_client(store, job_store) as client:
        response = client.post(
            "/umrs/models/stage",
            params={
                "user_id": "alice",
                "project_memberships": "project-123",
                "caller_project_id": "project-123",
            },
            json={
                "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "qwen-7b-user",
                "scope": "user",
                "model_type": "llm",
            },
        )

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-123"
    kwargs = job_store.create_job.call_args.kwargs
    assert kwargs["project_id"] == "project-123"
    assert kwargs["user_id"] == "alice"
    assert kwargs["job_type"] == JobType.MODEL_DOWNLOAD
    assert kwargs["spec"]["source"] == "huggingface"
    assert kwargs["spec"]["source_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert kwargs["spec"]["target_path"] == "/shared/users/alice/models/qwen-7b-user"
    assert kwargs["spec"]["model_id"] == "mdl-stage-123"
    assert kwargs["spec"]["metadata"]["owner_id"] == "alice"
    job_store.enqueue_job.assert_called_once_with("job-123")


def test_admin_can_stage_system_model_to_configured_root(monkeypatch):
    monkeypatch.setenv("UMRS_MODEL_STAGING_ROOT", "/home/operator/models")
    store = MagicMock()
    job_store = MagicMock()
    store.create_model.return_value = ModelEntry(
        model_id="mdl-system-stage",
        name="glm-planes",
        scope=ModelScope.SYSTEM,
        storage_ref="/home/operator/models/glm-planes",
        model_type=ModelType.LLM,
        huggingface_id="sapidlabs/glm-planes",
        serving_profile=ServingProfile(),
        execution_authority_default=ExecutionAuthority.SLURM,
        status=ModelStatus.STAGING,
    )
    job_store.create_job.return_value = "job-system-stage"

    with _create_test_client(store, job_store) as client:
        response = client.post(
            "/umrs/models/stage",
            params={"caller_project_id": "default"},
            headers={
                "X-User-ID": "operator",
                "X-Verified-Is-Admin": "true",
                "X-Verified-Project-ID": "default",
            },
            json={
                "huggingface_id": "sapidlabs/glm-planes",
                "name": "glm-planes",
                "scope": "system",
                "model_type": "llm",
            },
        )

    assert response.status_code == 202
    assert job_store.create_job.call_args.kwargs["spec"]["target_path"] == (
        "/home/operator/models/glm-planes"
    )


def test_non_admin_cannot_stage_system_model():
    store = MagicMock()
    job_store = MagicMock()

    with _create_test_client(store, job_store) as client:
        response = client.post(
            "/umrs/models/stage",
            params={"user_id": "alice", "caller_project_id": "default"},
            json={
                "huggingface_id": "sapidlabs/glm-planes",
                "name": "glm-planes",
                "scope": "system",
                "model_type": "llm",
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin role required to stage system models"
    store.create_model.assert_not_called()
    job_store.create_job.assert_not_called()


def test_download_model_creates_staging_copy_and_local_copy_job():
    store = MagicMock()
    job_store = MagicMock()

    source_model = ModelEntry(
        model_id="mdl-system-001",
        name="qwen-system",
        scope=ModelScope.SYSTEM,
        storage_ref="/shared/models/system/qwen-system",
        model_type=ModelType.LLM,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        description="Shared model",
        serving_profile=ServingProfile(),
        execution_authority_default=ExecutionAuthority.SLURM,
        status=ModelStatus.AVAILABLE,
    )
    target_model = ModelEntry(
        model_id="mdl-copy-001",
        name="qwen-team-copy",
        scope=ModelScope.PROJECT,
        project_id="project-123",
        storage_ref="/shared/projects/project-123/models/qwen-team-copy",
        model_type=ModelType.LLM,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        description="Shared model",
        serving_profile=ServingProfile(),
        execution_authority_default=ExecutionAuthority.SLURM,
        status=ModelStatus.STAGING,
    )
    store.get_model_by_name.return_value = source_model
    store.check_policy_access.return_value = (True, None)
    store.create_model.return_value = target_model
    job_store.create_job.return_value = "job-456"

    with _create_test_client(store, job_store) as client:
        response = client.post(
            "/umrs/models/qwen-system/download",
            params={
                "user_id": "alice",
                "project_memberships": "project-123",
                "caller_project_id": "project-123",
            },
            json={
                "target_scope": "project",
                "project_id": "project-123",
                "custom_name": "qwen-team-copy",
            },
        )

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-456"

    create_kwargs = store.create_model.call_args.kwargs
    assert create_kwargs["name"] == "qwen-team-copy"
    assert create_kwargs["scope"] == ModelScope.PROJECT
    assert create_kwargs["project_id"] == "project-123"
    assert create_kwargs["status"] == ModelStatus.STAGING
    assert create_kwargs["execution_authority_default"] == ExecutionAuthority.SLURM

    job_kwargs = job_store.create_job.call_args.kwargs
    assert job_kwargs["project_id"] == "project-123"
    assert job_kwargs["job_type"] == JobType.MODEL_DOWNLOAD
    assert job_kwargs["spec"]["source"] == "local"
    assert job_kwargs["spec"]["source_id"] == "/shared/models/system/qwen-system"
    assert job_kwargs["spec"]["target_path"] == "/shared/projects/project-123/models/qwen-team-copy"
    assert job_kwargs["spec"]["model_id"] == "mdl-copy-001"
    assert job_kwargs["spec"]["metadata"]["source_model_id"] == "mdl-system-001"
    job_store.enqueue_job.assert_called_once_with("job-456")


def test_create_model_keeps_system_scope_owner_null():
    store = MagicMock()
    job_store = MagicMock()

    store.create_model.return_value = ModelEntry(
        model_id="mdl-system-123",
        name="qwen3-32b",
        scope=ModelScope.SYSTEM,
        storage_ref="/lvs0/dne1/rccs-dmp/masaru.nagaso/umrs_models/Qwen3-32B",
        model_type=ModelType.LLM,
        huggingface_id="Qwen/Qwen3-32B",
        description="System-scoped Qwen3 32B",
        serving_profile=ServingProfile(),
        execution_authority_default=ExecutionAuthority.COMPOSE,
        status=ModelStatus.AVAILABLE,
    )

    with _create_test_client(store, job_store) as client:
        response = client.post(
            "/umrs/models",
            params={
                "user_id": "alice",
                "is_admin": "true",
            },
            json={
                "name": "qwen3-32b",
                "scope": "system",
                "storage_ref": "/lvs0/dne1/rccs-dmp/masaru.nagaso/umrs_models/Qwen3-32B",
                "model_type": "llm",
                "huggingface_id": "Qwen/Qwen3-32B",
                "description": "System-scoped Qwen3 32B",
            },
        )

    assert response.status_code == 201
    create_kwargs = store.create_model.call_args.kwargs
    assert create_kwargs["scope"] == ModelScope.SYSTEM
    assert create_kwargs["owner_id"] is None
