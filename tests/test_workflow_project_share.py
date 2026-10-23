"""ADR-080 project shares on every path that resolves a workflow for a caller.

Live 2026-10-23: a member of project ``opossum`` (not in the owner's circle)
saw the owner's workflow shared with ``project:opossum`` in /work, but the
definition card was a 404: ``GET /workflows/{name}/package`` resolved without
the caller's verified project, so a ``project:`` scope never admitted. The same
omission sat in ``/triggers``, in the research loop's port (``trigger_as``), in
loop start's workflow listing and in the durable re-dispatch fallback.

Each path admits a ``project:`` share only for a VERIFIED project: the signed
gateway header (HTTP), the loop's start-time stamp of that header (loops), the
spec record the server writes for a verified trigger (durable jobs).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.server import workflow_registry_handler as wrh

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    catalog_file = str(tmp_path / "workflow_catalog.json")
    monkeypatch.setattr(wrh, "_catalog_path", lambda: catalog_file)
    wrh.set_workflow_circles_resolver(lambda _user: [])
    yield catalog_file
    wrh.set_workflow_circles_resolver(lambda _user: [])


def _register(tmp_path, owner: str, access: list[str], name: str = "cnn-track") -> str:
    source = tmp_path / f"{owner}-{name}"
    source.mkdir()
    workflow = {
        "name": name,
        "version": "0.2.9",
        "steps": [{"name": "s1", "entrypoint": "mod:Step"}],
        "triggers": [{"event": "object_created", "bucket": "videos"}],
    }
    (source / "workflow.json").write_text(json.dumps(workflow))
    (source / "step.py").write_text("print('hi')\n")
    wrh.register_workflow(
        workflow_name=name,
        version="0.2.9",
        user_id=owner,
        source_path=str(source),
        workflow_json=workflow,
    )
    wrh.set_workflow_access(name, owner, access)
    return str(source)


@pytest.fixture
def shared(tmp_path) -> str:
    """mnagaso's workflow shared like the live one: a circle and a project."""
    return _register(tmp_path, "mnagaso", ["circle:bdr/opossum", "project:opossum"])


# ── HTTP: /package (definition card) and /triggers ──────────────────────────


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


_ROUTES = ("/package", "/triggers")


@pytest.mark.parametrize("suffix", _ROUTES)
def test_a_project_member_opens_a_project_shared_workflow(client, shared, suffix):
    response = client.get(
        f"/umrs/workflows/cnn-track{suffix}", headers=_headers("ryoshimi2", "opossum")
    )
    assert response.status_code == 200, response.text
    body = response.json()
    if suffix == "/package":
        assert body["source_path"] == shared
        assert [f["path"] for f in body["files"]] == ["step.py", "workflow.json"]
    else:
        assert body["triggers"] == [{"event": "object_created", "bucket": "videos"}]


@pytest.mark.parametrize("suffix", _ROUTES)
def test_another_verified_project_still_gets_404(client, shared, suffix):  # noqa: ARG001 — registers the share
    response = client.get(
        f"/umrs/workflows/cnn-track{suffix}", headers=_headers("ryoshimi2", "other-proj")
    )
    assert response.status_code == 404


@pytest.mark.parametrize("suffix", _ROUTES)
def test_owner_and_admin_unchanged(client, shared, suffix):  # noqa: ARG001 — registers the share
    owner = client.get(f"/umrs/workflows/cnn-track{suffix}", headers=_headers("mnagaso", "bdr"))
    admin = client.get(
        f"/umrs/workflows/cnn-track{suffix}", headers=_headers("root", "ops", admin=True)
    )
    assert owner.status_code == 200 and admin.status_code == 200


@pytest.mark.parametrize("suffix", _ROUTES)
def test_a_circle_share_still_opens(client, shared, suffix):  # noqa: ARG001 — registers the share
    wrh.set_workflow_circles_resolver(lambda user: ["bdr/opossum"] if user == "circlemate" else [])
    response = client.get(
        f"/umrs/workflows/cnn-track{suffix}", headers=_headers("circlemate", "bdr")
    )
    assert response.status_code == 200


# ── research loop: trigger_as + loop start listing ──────────────────────────


def _port():
    from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort

    store = MagicMock()
    store.create_job.return_value = "job-1"
    return UmrsWorkflowExecutionPort(store), store


def test_the_loop_port_runs_a_project_share_in_the_verified_project(shared):
    port, store = _port()
    port.trigger_as(
        "cnn-track", {"project_id": "opossum"}, owner="ryoshimi2", owner_project="opossum"
    )
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["spec"]["source_path"] == shared
    assert kwargs["project_id"] == "opossum"
    # the durable job records its project as verified for a later re-resolve
    assert kwargs["spec"]["verified_project_id"] == "opossum"


@pytest.mark.parametrize(
    ("owner_project", "run_project"),
    [(None, "opossum"), ("opossum", "elsewhere"), ("other-proj", "other-proj")],
)
def test_the_loop_port_refuses_without_a_matching_verified_project(
    shared,  # noqa: ARG001 — registers the share
    owner_project,
    run_project,
):
    port, store = _port()
    with pytest.raises(RuntimeError, match="not one the loop's owner"):
        port.trigger_as(
            "cnn-track",
            {"project_id": run_project},
            owner="ryoshimi2",
            owner_project=owner_project,
        )
    store.create_job.assert_not_called()


def test_the_engine_passes_the_loops_stamped_project(shared):
    from umrs.knowledge.loop import AutonomousResearchLoop

    port, store = _port()
    engine = AutonomousResearchLoop(workflow_port=port)
    stamped = SimpleNamespace(
        project_id="opossum",
        metadata={"owner_user_id": "ryoshimi2", "owner_verified_project": "opossum"},
    )
    engine._trigger_workflow(stamped, "cnn-track", {"project_id": "opossum"})
    assert store.create_job.call_args.kwargs["spec"]["source_path"] == shared

    unstamped = SimpleNamespace(project_id="opossum", metadata={"owner_user_id": "ryoshimi2"})
    with pytest.raises(RuntimeError, match="not one the loop's owner"):
        engine._trigger_workflow(unstamped, "cnn-track", {"project_id": "opossum"})


def test_verified_loop_project_needs_the_stamp_to_match_the_loop():
    from umrs.knowledge.loop import verified_loop_project

    def loop(project, **meta):
        return SimpleNamespace(project_id=project, metadata=meta)

    owner = {"owner_user_id": "u"}
    assert verified_loop_project(loop("p", **owner, owner_verified_project="p")) == "p"
    assert verified_loop_project(loop("p", **owner, owner_verified_project="q")) is None
    assert verified_loop_project(loop("p", **owner)) is None
    assert verified_loop_project(loop("p", owner_verified_project="p")) is None


def test_loop_start_lists_workflows_shared_with_the_owner(tmp_path, shared):  # noqa: ARG001 — registers the share
    from umrs.api.loop import _registered_workflow_names

    _register(tmp_path, "carol", ["user:ryoshimi2"], name="user-shared")
    _register(tmp_path, "carol", [], name="private")
    assert _registered_workflow_names("ryoshimi2", "opossum") == ["cnn-track", "user-shared"]
    assert _registered_workflow_names("ryoshimi2", "other-proj") == ["user-shared"]
    assert _registered_workflow_names("ryoshimi2", None) == ["user-shared"]
    assert "private" in _registered_workflow_names("", None)  # ownerless (mvp): all


class TestLoopStartStamp:
    @pytest.fixture(autouse=True)
    def _mvp_loop_api(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        from umrs.api import loop as loop_api
        from umrs.knowledge import engine_loop_runner
        from umrs.knowledge.loop import AutonomousResearchLoop

        loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
        monkeypatch.setattr(engine_loop_runner, "register", lambda *_a, **_k: None)
        loop_api._configure_hosting()
        yield
        loop_api.clear_loop_store()

    def _start(self) -> tuple[TestClient, str]:
        from umrs.api.loop import router as loop_router
        from umrs.knowledge.engine_loop_runner import _RUNNERS

        app = FastAPI()
        app.include_router(loop_router)
        client = TestClient(
            app, headers={"X-User-ID": "ryoshimi2", "X-Verified-Project-ID": "opossum"}
        )
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "opossum", "research_goal": "g", "max_iterations": 2},
        )
        assert resp.status_code == 201, resp.text
        loop_id = resp.json()["loop_id"]
        _RUNNERS.pop(loop_id, None)
        return client, loop_id

    def test_start_stamps_the_verified_project(self):
        from umrs.api.loop import get_loop
        from umrs.knowledge.loop import verified_loop_project

        _client, loop_id = self._start()
        loop = get_loop(loop_id)
        assert loop.metadata["owner_verified_project"] == "opossum"
        assert verified_loop_project(loop) == "opossum"

    def test_a_state_write_never_replaces_the_stamped_identity(self):
        from umrs.api.loop import get_loop

        client, loop_id = self._start()
        resp = client.put(
            f"/umrs/loops/{loop_id}/state",
            json={
                "metadata": {
                    "owner_user_id": "mallory",
                    "owner_verified_project": "other-proj",
                    "note": "kept",
                }
            },
        )
        assert resp.status_code == 200, resp.text
        meta = get_loop(loop_id).metadata
        assert meta["owner_user_id"] == "ryoshimi2"
        assert meta["owner_verified_project"] == "opossum"
        assert meta["note"] == "kept"


# ── durable re-dispatch without a usable source_path ────────────────────────


def _job(spec_extra: dict, project: str = "opossum", user: str = "ryoshimi2"):
    from umrs.jobs.models import Job, JobAttempt, JobType

    spec = {"workflow_name": "cnn-track", "execution_id": "wf-0123456789ab", "params": {}}
    spec.update(spec_extra)
    job = Job(job_id="j", project_id=project, user_id=user, job_type=JobType.WORKFLOW_RUN,
              spec=spec, idempotency_key="k")  # fmt: skip
    attempt = JobAttempt(attempt_id="a", job_id="j", attempt_no=1, backend="workflow-run")
    return job, attempt


def _dispatch(monkeypatch, job, attempt) -> dict:
    from umrs.jobs.workflow_run import WorkflowRunBackend

    launched: dict = {}

    def fake_create(**kw):
        launched.update(kw)
        return SimpleNamespace(execution_id=kw.get("execution_id"), durable_job_id="j")

    registry = SimpleNamespace(
        running_here=lambda _eid: None,
        get=lambda _eid: None,
        create=fake_create,
        launch=lambda _record, _runner: None,
    )
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    monkeypatch.setattr(wrh, "_take_over_prior_run", lambda *_a: None)
    monkeypatch.setattr("umrs.server.step_router.build_registry_backed_router", lambda **_k: None)
    asyncio.run(WorkflowRunBackend().submit_job(job, attempt))
    return launched


def test_a_verified_durable_job_re_resolves_a_project_share(monkeypatch, shared):  # noqa: ARG001 — registers the share
    job, attempt = _job({"verified_project_id": "opossum"})
    launched = _dispatch(monkeypatch, job, attempt)
    assert launched["workflow_name"] == "cnn-track"
    assert launched["user_id"] == "ryoshimi2"


@pytest.mark.parametrize(
    ("extra", "project"),
    [({}, "opossum"), ({"verified_project_id": "opossum"}, "other-proj")],
)
def test_an_unverified_durable_job_never_admits_a_project_share(
    monkeypatch,
    shared,  # noqa: ARG001 — registers the share
    extra,
    project,
):
    job, attempt = _job(extra, project=project)
    with pytest.raises(ValueError, match="not found in registry"):
        _dispatch(monkeypatch, job, attempt)


def test_the_durable_fallback_keeps_owner_and_circle_resolution(monkeypatch, shared):  # noqa: ARG001 — registers the share
    owner_job, attempt = _job({}, project="bdr", user="mnagaso")
    assert _dispatch(monkeypatch, owner_job, attempt)["user_id"] == "mnagaso"
    wrh.set_workflow_circles_resolver(lambda user: ["bdr/opossum"] if user == "mate" else [])
    circle_job, attempt = _job({}, project="bdr", user="mate")
    assert _dispatch(monkeypatch, circle_job, attempt)["user_id"] == "mate"


class _Jobs:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def submit_workflow_run(self, _client_id, payload):
        self.payloads.append(json.loads(payload))
        return json.dumps({"status": "ok", "job": {"job_id": "j1"}})


@pytest.mark.parametrize("verified", [True, False])
def test_a_trigger_records_the_project_as_verified_only_when_it_was(tmp_path, verified):
    _register(tmp_path, "ryoshimi2", [])  # own workflow: resolves either way
    jobs = _Jobs()
    raw = wrh.handle_workflow_command(
        "RUN_WORKFLOW",
        "ryoshimi2",
        json.dumps(
            {
                "workflow_name": "cnn-track",
                "params": {"project_id": "opossum"},
                "_identity_verified": verified,
            }
        ),
        job_backend=jobs,
        trusted_identity=verified,
    )
    assert json.loads(raw)["status"] == "ok", raw
    spec = jobs.payloads[0]["spec"]
    assert jobs.payloads[0]["project_id"] == "opossum"
    assert spec.get("verified_project_id") == ("opossum" if verified else None)
