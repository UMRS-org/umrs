"""Which step lane a ``user_account`` run may take (ADR-112 B2; I-1).

Fixer findings 2026-11-08:

- only the SSH-submitting staged lane has a gated path; a ``user_account``
  run (signed or not) on a local ``sbatch`` lane refuses
  ``hpcaccess.lane_not_gated`` at the trigger and in the runner, and the
  runner's step binding fails by name if one still reaches that executor
  (it would submit as the server's own login);
- a site with the gated path but no helper lane (plan §5 B2.0 §10.5) runs
  uploaded code for a ``user_account`` run; a service run there has no
  HPC lane.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.hpc_accounts.store import InMemorySiteAccountStore
from umrs.orchestrator.hpc_identity import HpcAccessRefused
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)
from umrs.shared import hpc_user_work_access as access

BUILTIN = {
    "execution_mode": "slurm",
    "steps": [{"name": "s1", "entrypoint": "umrs.pipeline_sdk.templates:Noop"}],
}
LANE_ENV = (
    "UMRS_SLURM_STEP_SUBMIT",
    "UMRS_STEP_EXECUTOR",
    "UMRS_STEP_REMOTE_ENABLED",
    "UMRS_STEP_IMAGE",
    "UMRS_STEP_HELPER",
    "UMRS_SLURM_WORKDIR_ROOT",
)


def _verified(store: InMemorySiteAccountStore) -> None:
    store.issue_key("alice", "tk-000000000001", "SHA256:x")
    store.record_verified(
        "alice",
        "tk-000000000001",
        {
            "site_login": "alice_hpc",
            "verified_login": "alice_hpc",
            "verified_assocs": [{"account": "grp-a"}],
            "gate_version": "2",
            "gate_sha256": "0" * 64,
            "signer_fingerprints": [],
            "uv_required": False,
        },
    )
    store.upsert_link("alice", "proj-a", "grp-a")


@pytest.fixture
def enrolled(monkeypatch):
    """alice enrolled at a site that allows both modes; bob is not."""
    for name in LANE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("umrs.shared.uploaded_code.load_site_env", lambda: None)
    monkeypatch.setattr("umrs.server.step_trust._load_site_env", lambda: None)
    site = SiteConfig(name="site-x", extra={"ssh_user": "op"})
    site.scheduler.hpc_access = HpcAccessConfig(
        modes_allowed=["service_account", "user_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(login="op", personal=True, allowed_projects=[]),
        user_account=UserAccountAccess(methods=["gated_key"], signing_domain="ex.org"),
    )
    tables: dict = {"accounts": {}, "links": {}}
    _verified(InMemorySiteAccountStore("site-x", tables))
    monkeypatch.setattr(access, "current_site", lambda: site)
    monkeypatch.setattr(access, "user_is_admin", lambda _u: False)
    monkeypatch.setattr(access, "user_in_project", lambda _u, _p: True)
    monkeypatch.setattr(access, "_store_factory", lambda v: InMemorySiteAccountStore(v, tables))
    return monkeypatch


def test_a_user_account_run_on_a_local_sbatch_lane_refuses_at_the_trigger(enrolled):
    from umrs.server.hpc_user_work import workflow_user_work_identity

    with pytest.raises(HpcAccessRefused) as exc:
        workflow_user_work_identity(BUILTIN, None, user_id="alice", project_id="proj-a")
    assert exc.value.reason == "lane_not_gated"


def test_a_user_account_run_on_a_local_sbatch_lane_refuses_in_the_runner(enrolled):
    from umrs.server import workflow_runner as wr

    name, refusal = wr._workflow_executor_name(
        BUILTIN, BUILTIN["steps"], None, injected=False, run_identity=("alice", "proj-a", True)
    )
    assert name is None and "lane_not_gated" in refusal


def test_the_ssh_lane_keeps_its_gated_path(enrolled):
    from umrs.server import workflow_runner as wr
    from umrs.server.hpc_user_work import workflow_user_work_identity

    enrolled.setenv("UMRS_SLURM_STEP_SUBMIT", "remote")
    identity = workflow_user_work_identity(BUILTIN, None, user_id="alice", project_id="proj-a")
    assert identity.mode == "user_account"
    name, refusal = wr._workflow_executor_name(
        BUILTIN, BUILTIN["steps"], None, injected=False, run_identity=("alice", "proj-a", True)
    )
    assert (name, refusal) == ("slurm", "")


def test_the_runner_binding_refuses_a_slurm_executor_without_a_gated_path(enrolled):
    from umrs.server import workflow_runner as wr
    from umrs.server.slurm_step_executor import SlurmStepExecutor
    from umrs.server.step_executor import InProcessExecutor, StepExecutionError

    record = SimpleNamespace(intent_id="i" * 64, execution_id="wf-0123456789ab")
    local = SlurmStepExecutor.__new__(SlurmStepExecutor)
    with pytest.raises(StepExecutionError) as exc:
        wr._require_gated_lane(local, record)
    assert "lane_not_gated" in exc.value.message
    wr._require_gated_lane(InProcessExecutor(), record)  # a template step's lane
    wr._require_gated_lane(local, SimpleNamespace(intent_id=None))


def _gated_only(monkeypatch) -> None:
    monkeypatch.setenv("UMRS_SLURM_STEP_SUBMIT", "remote")
    monkeypatch.setenv("UMRS_STEP_REMOTE_ENABLED", "1")
    monkeypatch.setenv("UMRS_STEP_IMAGE", "localhost/umrs-step:x")


def test_a_gated_only_site_runs_uploaded_code_for_a_signed_run(enrolled, tmp_path):
    from umrs.server import workflow_runner as wr
    from umrs.server.step_trust import gated_lane_available, uploaded_code_lane

    _gated_only(enrolled)
    assert gated_lane_available() and uploaded_code_lane(None) == "slurm"
    (tmp_path / "pkg").mkdir()
    steps = [{"name": "s1", "entrypoint": "pkg.step:Step"}]
    wf = {"steps": steps}
    signed = wr._workflow_executor_name(
        wf, steps, str(tmp_path), injected=False, run_identity=("alice", "proj-a", True)
    )
    assert signed == ("slurm", "")
    access.current_site().scheduler.hpc_access.service_account.allowed_projects = ["proj-b"]
    service = wr._workflow_executor_name(
        wf, steps, str(tmp_path), injected=False, run_identity=("bob", "proj-b", False)
    )
    assert service == (None, wr._NO_HPC_LANE)


def test_a_site_without_user_accounts_has_no_gated_path(enrolled):
    from umrs.server.step_trust import gated_lane_available

    _gated_only(enrolled)
    access.current_site().scheduler.hpc_access.modes_allowed = ["service_account"]
    assert gated_lane_available() is False
