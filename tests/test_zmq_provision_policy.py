"""ZMQ endpoint_provision submit policy (H-1, 2026-10-19).

The durable ZMQ submit now enforces the same policy as ``POST /umrs/jobs``
endpoint_provision (shared ``jobs/endpoint_policy.py``): catalog-model ACL,
admin-only recipe fields (env_vars, container_image, tier=system), and the
mount sources + model_path under the caller's own roots
(``UMRS_ENDPOINT_CALLER_MOUNT_ROOTS``). On yesterday's tree every refusal
case here answered ``status: ok`` with the foreign spec enqueued.
"""

from __future__ import annotations

import json

import pytest
from test_job_tracking_handler_durable import FakeJobStore
from umrs.server.job_tracking_handler import DurableJobCommandBackend


def _system_entry(name: str) -> dict:
    return {"name": name, "scope": "system", "storage_ref": f"/models/{name}"}


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setenv("UMRS_LOG_LEVEL", "ERROR")
    monkeypatch.setattr(
        "umrs.api.endpoints._resolve_registered_model_entry",
        _system_entry,
    )
    # System-tier recipes are refused by the site's own check — no site
    # config in tests, so the policy seam decides; keep it quiet.


def _submit(spec: dict, *, is_admin: bool = False, fan_out: list | None = None):
    body = {
        "project_id": "proj-a",
        "user_id": "alice",
        "job_type": "endpoint_provision",
        "spec": {"model": "model-x", **spec},
        "priority": 1,
    }
    if fan_out:
        body["fan_out_jobs"] = fan_out
    backend = DurableJobCommandBackend(FakeJobStore())
    return backend, json.loads(backend.submit("alice", json.dumps(body), is_admin=is_admin))


_ALICE_ROOTS = {
    "project_data": {"source": "/projects/proj-a/data"},
    "rag_index": {"source": "/home/alice/index"},
    "mcp_servers": [{"source": "/shared/users/alice/mcp"}],
}

_FOREIGN_MOUNTS = [
    ("project_data", {"source": "/projects/proj-b/data"}),
    ("rag_index", {"source": "/home/bob/index"}),
    ("mcp_servers", [{"source": "/shared/users/bob/mcp"}]),
]


class TestMountSourcePolicy:
    @pytest.mark.parametrize(("kind", "spec"), _FOREIGN_MOUNTS)
    def test_foreign_mount_sources_refused(self, kind, spec):
        _backend, result = _submit({"mounts": {kind: spec}})
        assert result["status"] == "error"
        assert "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS" in json.dumps(result)

    def test_mount_sources_under_own_roots_admitted(self):
        backend, result = _submit({"mounts": _ALICE_ROOTS})
        assert result["status"] == "ok", result
        assert len(backend._job_store.jobs) == 1

    def test_model_path_outside_roots_refused(self):
        _backend, result = _submit({"model_path": "/home/bob/weights"})
        assert result["status"] == "error"
        assert "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS" in json.dumps(result)

    def test_model_path_under_roots_admitted(self):
        _backend, result = _submit({"model_path": "/projects/proj-a/models/m"})
        assert result["status"] == "ok", result


class TestRecipePolicy:
    def test_env_vars_refused(self):
        _backend, result = _submit({"env_vars": {"RIVAULT_SKIP": "1"}})
        assert result["status"] == "error"

    def test_container_image_refused(self):
        _backend, result = _submit({"container_image": "custom:latest"})
        assert result["status"] == "error"

    def test_tier_system_refused(self):
        _backend, result = _submit({"tier": "system"})
        assert result["status"] == "error"

    def test_admin_bypasses_recipe_fields(self):
        _backend, result = _submit(
            {"env_vars": {"RIVAULT_SKIP": "1"}, "mounts": _ALICE_ROOTS},
            is_admin=True,
        )
        assert result["status"] == "ok", result


class TestModelCatalogPolicy:
    def test_unregistered_model_refused(self, monkeypatch):
        monkeypatch.setattr(
            "umrs.api.endpoints._resolve_registered_model_entry",
            lambda _name: None,
        )
        _backend, result = _submit({})
        assert result["status"] == "error"
        assert "model catalog" in json.dumps(result).lower()

    def test_foreign_owned_catalog_storage_allowed_but_checked(self):
        # storage_ref outside the caller's roots while NOT caller-named:
        # allowed only when the caller may provision the entry (system scope here).
        _backend, result = _submit({"model_path": "/models/model-x"})
        assert result["status"] == "ok", result


class TestFanOutPolicy:
    def test_fan_out_child_with_foreign_mount_refused(self):
        child = {
            "name": "c",
            "job_type": "endpoint_provision",
            "spec": {
                "model": "model-x",
                "mounts": {"project_data": {"source": "/home/bob/data"}},
            },
        }
        _backend, result = _submit({}, fan_out=[child])
        assert result["status"] == "error"
        assert "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS" in json.dumps(result)

    def test_fan_out_children_under_own_roots_admitted(self):
        child = {
            "name": "c",
            "job_type": "endpoint_provision",
            "spec": {"model": "model-x", "mounts": _ALICE_ROOTS},
        }
        backend, result = _submit({}, fan_out=[child])
        assert result["status"] == "ok", result
        assert len(backend._job_store.jobs) == 1
