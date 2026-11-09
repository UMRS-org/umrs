"""Enrolment precedence in the HPC identity resolver (ADR-112 wave B2).

Plan §5 B2.0 §10.1: where the site allows ``user_account`` and the person
has a link (any status) for the **verified** project, their own account
decides: active account + active link run as that login and Slurm
account; an inactive account is ``not_enrolled``; an inactive link is
``allocation_not_linked``. Otherwise ``default_mode`` decides: under
``user_account`` no account is ``not_enrolled`` and no link is
``allocation_not_linked``; under ``service_account`` the wave A rules hold.
The enrolment is never a fallback (I-1), and only staged steps have a gated
path: any other user-work lane that resolves to ``user_account`` refuses
``lane_not_gated`` before anything reaches the cluster.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.hpc_accounts.store import InMemorySiteAccountStore
from umrs.orchestrator.hpc_identity import (
    Enrolment,
    HpcAccessRefused,
    Lane,
    resolve_hpc_identity,
)
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)
from umrs.shared import hpc_user_work_access as access

ACTIVE_ACCOUNT = {"status": "active", "verified_login": "alice_hpc", "username": "alice"}
ACTIVE_LINK = {"project_id": "proj-a", "status": "active", "slurm_account": "grp-a"}


def _site(*, default: str = "service_account", allows_user: bool = True) -> SiteConfig:
    site = SiteConfig(name="site-x", extra={"ssh_user": "op1"})
    modes = ["service_account", "user_account"] if allows_user else ["service_account"]
    site.scheduler.hpc_access = HpcAccessConfig(
        modes_allowed=modes,
        default_mode=default,
        service_account=ServiceAccountAccess(
            login="op1", personal=True, allowed_projects=["proj-a"]
        ),
        user_account=UserAccountAccess(methods=["gated_key"], signing_domain="ex.org"),
    )
    return site


def _resolve(site, enrolment, *, project="proj-a", admin=False):
    return resolve_hpc_identity(
        site,
        lane=Lane.USER_WORK,
        user_id="alice",
        project_id=project,
        is_admin=admin,
        enrolment=enrolment,
    )


def _reason(site, enrolment, **kw) -> str:
    with pytest.raises(HpcAccessRefused) as info:
        _resolve(site, enrolment, **kw)
    return info.value.reason


def _with(account=None, link=None, *, account_set=None, link_set=None) -> Enrolment:
    acc = None if account is None else {**account, **(account_set or {})}
    lnk = None if link is None else {**link, **(link_set or {})}
    return Enrolment(account=acc, link=lnk)


# ── the matrix: enrolment x link status x default_mode ───────────────────────


@pytest.mark.parametrize("default", ["service_account", "user_account"])
def test_an_active_account_and_link_run_as_the_person_whatever_the_default(default):
    ident = _resolve(_site(default=default), _with(ACTIVE_ACCOUNT, ACTIVE_LINK))
    assert (ident.mode, ident.login, ident.slurm_account) == (
        "user_account",
        "alice_hpc",
        "grp-a",
    )
    assert (ident.personal, ident.declared, ident.principal) == (False, True, "alice")
    assert ident.to_metrics()["access_mode"] == "user_account"


@pytest.mark.parametrize("default", ["service_account", "user_account"])
@pytest.mark.parametrize("status", ["pending", "suspended", "revoked"])
def test_a_link_with_an_inactive_account_is_not_enrolled(default, status):
    enrolment = _with(ACTIVE_ACCOUNT, ACTIVE_LINK, account_set={"status": status})
    assert _reason(_site(default=default), enrolment) == "not_enrolled"


@pytest.mark.parametrize("default", ["service_account", "user_account"])
def test_a_suspended_link_is_allocation_not_linked_never_the_platform_login(default):
    enrolment = _with(ACTIVE_ACCOUNT, ACTIVE_LINK, link_set={"status": "suspended"})
    # An admin with a listed project would pass wave A's rule: the link wins.
    assert _reason(_site(default=default), enrolment, admin=True) == "allocation_not_linked"


@pytest.mark.parametrize(
    ("enrolment", "reason"),
    [
        (None, "not_enrolled"),
        (Enrolment(), "not_enrolled"),
        (_with({**ACTIVE_ACCOUNT, "status": "pending"}), "not_enrolled"),
        (_with(ACTIVE_ACCOUNT), "allocation_not_linked"),
    ],
)
def test_default_user_account_without_a_link_refuses_by_name(enrolment, reason):
    assert _reason(_site(default="user_account"), enrolment, admin=True) == reason


@pytest.mark.parametrize("enrolment", [None, Enrolment(), _with(ACTIVE_ACCOUNT)])
def test_default_service_account_without_a_link_keeps_the_wave_a_rules(enrolment):
    site = _site(default="service_account")
    ident = _resolve(site, enrolment)  # proj-a is listed
    assert (ident.mode, ident.login) == ("service_account", "op1")
    assert _reason(site, enrolment, project="proj-z") == "project_not_allowed"


def test_a_link_is_ignored_where_the_site_does_not_allow_user_accounts():
    site = _site(default="service_account", allows_user=False)
    ident = _resolve(site, _with(ACTIVE_ACCOUNT, ACTIVE_LINK))
    assert ident.mode == "service_account"


def test_platform_lanes_never_resolve_to_the_person():
    ident = resolve_hpc_identity(
        _site(default="user_account"),
        lane=Lane.PLATFORM,
        user_id="alice",
        project_id="proj-a",
        is_admin=False,
        enrolment=_with(ACTIVE_ACCOUNT, ACTIVE_LINK),
    )
    assert (ident.mode, ident.login) == ("service_account", "op1")


# ── the lookup (this venue's tables, the verified project's link) ────────────


@pytest.fixture
def tables(monkeypatch):
    shared = {"accounts": {}, "links": {}}
    monkeypatch.setattr(
        access, "_store_factory", lambda venue: InMemorySiteAccountStore(venue, shared)
    )
    monkeypatch.setattr(access, "user_is_admin", lambda _u: False)
    return shared


def _enrol(tables, venue="site-x", user="alice", project="proj-a", status="active"):
    store = InMemorySiteAccountStore(venue, tables)
    store.issue_key(user, "tk-000000000001", "SHA256:x")
    store.record_verified(
        user,
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
    store.upsert_link(user, project, "grp-a")
    if status != "active":
        store.suspend_links(user, "test", project)
    return store


def test_read_enrolment_reads_this_venue_and_the_verified_projects_link(tables):
    _enrol(tables)
    _enrol(tables, venue="site-y", project="proj-b")  # another front's row
    enrolment = access.read_enrolment(_site(), "alice", "proj-a")
    assert enrolment.account["verified_login"] == "alice_hpc"
    assert enrolment.link["slurm_account"] == "grp-a"
    assert access.read_enrolment(_site(), "alice", "proj-b").link is None
    assert access.read_enrolment(_site(), "alice", None).link is None


def test_read_enrolment_is_skipped_where_user_accounts_are_not_allowed(tables):
    assert access.read_enrolment(_site(allows_user=False), "alice", "proj-a") is None


def test_unreadable_site_accounts_refuse_not_enrolled(monkeypatch):
    def broken(_venue):
        raise RuntimeError("database down")

    monkeypatch.setattr(access, "_store_factory", broken)
    with pytest.raises(HpcAccessRefused) as info:
        access.read_enrolment(_site(), "alice", "proj-a")
    assert info.value.reason == "not_enrolled"


def test_resolve_user_work_returns_the_person_for_the_gated_lane(tables, monkeypatch):
    _enrol(tables)
    monkeypatch.setattr(access, "current_site", lambda: _site())
    ident = access.resolve_user_work(user_id="alice", project_id="proj-a", gated=True)
    assert (ident.mode, ident.login, ident.slurm_account) == (
        "user_account",
        "alice_hpc",
        "grp-a",
    )


def test_a_suspended_link_refuses_through_the_lookup(tables, monkeypatch):
    _enrol(tables, status="suspended")
    monkeypatch.setattr(access, "current_site", lambda: _site())
    refused = access.user_work_refusal(user_id="alice", project_id="proj-a", gated=True)
    assert refused is not None and refused.code == "hpcaccess.allocation_not_linked"


# ── lanes without a gated path: lane_not_gated, before anything runs ─────────


def test_a_user_work_lane_without_a_gated_path_refuses_lane_not_gated(tables, monkeypatch):
    _enrol(tables)
    monkeypatch.setattr(access, "current_site", lambda: _site())
    refused = access.user_work_refusal(user_id="alice", project_id="proj-a")
    assert refused is not None and refused.code == "hpcaccess.lane_not_gated"


def test_hpc_script_submit_refuses_lane_not_gated_before_any_ssh(tables, monkeypatch):
    from umrs.jobs.hpc_script import HpcScriptBackend

    _enrol(tables)
    monkeypatch.setattr(access, "current_site", lambda: _site())
    commands: list[str] = []

    async def factory():
        async def run_remote(command, **_kw):
            commands.append(command)

        return SimpleNamespace(run_remote=run_remote)

    job = SimpleNamespace(
        job_id="j-1", attempt_id=None, user_id="alice", project_id="proj-a",
        spec={"script_path": "/x"},
    )
    attempt = SimpleNamespace(attempt_id="a-1", metrics={}, backend_ref="")
    with pytest.raises(HpcAccessRefused) as info:
        asyncio.run(HpcScriptBackend(ssh=factory).submit_job(job, attempt))
    assert info.value.code == "hpcaccess.lane_not_gated"
    assert commands == [] and "hpc_identity" not in attempt.metrics


def test_cluster_loop_submit_refuses_lane_not_gated_before_submitting(tables, monkeypatch):
    from umrs.jobs.cluster_venue import ClusterSlurmSubmitter
    from umrs.jobs.models import Job, JobAttempt, JobType

    _enrol(tables)
    monkeypatch.setattr(access, "current_site", lambda: _site())
    submitted: list[dict] = []

    async def submit_wrapped(**kwargs):
        submitted.append(kwargs)
        return "s-1"

    spec = {"loop_id": "loop-1", "driver": "engine", "venue": "cluster:site-x"}
    job = Job(
        job_id="job-1", project_id="proj-a", user_id="alice",
        job_type=JobType.RESEARCH_LOOP, spec=spec, idempotency_key="k1",
    )
    attempt = JobAttempt(attempt_id="att-1", job_id="job-1", attempt_no=1, backend="slurm")
    with pytest.raises(HpcAccessRefused) as info:
        asyncio.run(
            ClusterSlurmSubmitter(submit_wrapped=submit_wrapped).submit(
                job=job, attempt=attempt, spec=spec
            )
        )
    assert info.value.code == "hpcaccess.lane_not_gated"
    assert submitted == [] and "hpc_identity" not in attempt.metrics
