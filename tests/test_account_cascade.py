"""RS-0 S-2: the account and project cascade.

Disabling a user, removing them from a project, and (ADR-097 ``enforce``)
losing a membership revoke the user's leases on this site
(``SecretBroker.revoke_by_owner``), cancel their other open jobs, revoke
their CURVE sessions (whole account only) and write ONE ``audit_events``
row. Remote sessions end through the revoked lease on their status
checker's next cycle. The admin CLI runs the cascade at once; the server's
standing sweep re-derives it from live state within a minute.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from umrs.admin import account_cascade as ac
from umrs.admin.user_store import UserRecord, UserStore
from umrs.jobs.models import JobType
from umrs.jobs.secret_broker import SecretBroker

SITE = "rccs"


# ── fakes ─────────────────────────────────────────────────────────────────


class _Cur:
    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []

    def execute(self, sql, params=()):
        self._conn.executed.append((sql, params))
        if "FROM projects" in sql:
            self._rows = list(self._conn.projects)

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Conn:
    """Captures SQL; answers the projects lookup from ``projects`` rows
    ``(project_id, owner_id, status)``."""

    def __init__(self, projects=()):
        self.projects = list(projects)
        self.executed: list = []
        self.commits = 0

    def cursor(self, **_kw):
        return _Cur(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def audit_rows(self):
        return [p for s, p in self.executed if "INSERT INTO audit_events" in s]


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def getconn(self):
        return self.conn

    def putconn(self, _c):
        pass


class _Jobs:
    """JobStore slice: ``list_jobs`` + ``cancel_job`` over in-memory jobs."""

    site_name = SITE

    def __init__(self, jobs):
        self.jobs = jobs
        self.cancelled: list[str] = []

    def list_jobs(self, *, user_id, project_id, status, limit):
        del limit
        return [
            j
            for j in self.jobs
            if j.user_id == user_id
            and j.status == status.value
            and (not project_id or j.project_id == project_id)
        ]

    def cancel_job(self, job_id, reason=""):
        del reason
        self.cancelled.append(job_id)


class _Curve:
    def __init__(self):
        self.revoked: list[str] = []

    def revoke_user(self, user):
        self.revoked.append(user)
        return 2


def _job(job_id, *, user="alice", project="p1", status="running", venue=SITE, kind=None):
    return SimpleNamespace(
        job_id=job_id,
        user_id=user,
        project_id=project,
        status=status,
        venue=venue,
        job_type=kind or JobType.WORKFLOW_RUN,
        spec={},
    )


def _rec(name="alice", *, role="user", disabled=False, projects=("p1", "p2")):
    return UserRecord(username=name, role=role, disabled=disabled, projects=list(projects))


def _broker():
    return SecretBroker(venue=SITE)


def _lease(broker, owner="alice", *, project="p1", session="", venue=None):
    lease, _ = broker.issue_lease(
        f"job-{owner}-{project}",
        "inference-token",
        "inference",
        owner_id=owner,
        project=project,
        session_id=session,
    )
    if venue is not None:
        lease.venue = venue
    return lease


# ── revoke_by_owner ───────────────────────────────────────────────────────


class TestRevokeByOwner:
    def test_revokes_every_live_lease_the_owner_holds_here(self):
        broker = _broker()
        a1 = _lease(broker, project="p1", session="s1")
        a2 = _lease(broker, project="p2")
        bob = _lease(broker, "bob")
        revoked = broker.revoke_by_owner("alice", actor="t")
        assert {lease.lease_id for lease in revoked} == {a1.lease_id, a2.lease_id}
        assert not broker.get_lease(a1.lease_id).is_valid
        assert broker.get_lease(bob.lease_id).is_valid
        ops = [e.operation for e in broker.get_audit_trail(lease_id=a1.lease_id)]
        assert "revoke" in ops

    def test_project_scope_leaves_other_projects(self):
        broker = _broker()
        p1 = _lease(broker, project="p1")
        p2 = _lease(broker, project="p2")
        revoked = broker.revoke_by_owner("alice", project="p1")
        assert [lease.lease_id for lease in revoked] == [p1.lease_id]
        assert broker.get_lease(p2.lease_id).is_valid

    def test_other_sites_leases_stay_and_pre_0124_leases_count_as_ours(self):
        broker = _broker()
        other = _lease(broker, venue="ai4s")
        legacy = _lease(broker, project="p2", venue="")
        revoked = broker.revoke_by_owner("alice")
        assert [lease.lease_id for lease in revoked] == [legacy.lease_id]
        assert broker.get_lease(other.lease_id).is_valid

    def test_issued_leases_carry_the_site(self):
        broker = _broker()
        assert _lease(broker).venue == SITE
        assert ("alice", "p1") in broker.active_owner_projects()

    def test_empty_owner_revokes_nothing(self):
        broker = _broker()
        _lease(broker, owner="")
        assert broker.revoke_by_owner("") == []


# ── the cascade ───────────────────────────────────────────────────────────


class TestRunCascade:
    def _run(self, project=None, **kw):
        broker = _broker()
        _lease(broker, project="p1", session="s1")
        _lease(broker, project="p2", session="s2")
        jobs = _Jobs(
            [
                _job("j-p1", project="p1"),
                _job("j-p2", project="p2", status="queued"),
                _job("j-sess", kind=JobType.FRONTEND_SESSION),
                _job("j-other-site", venue="ai4s"),
                _job("j-done", status="completed"),
                _job("j-bob", user="bob"),
            ]
        )
        curve, conn = _Curve(), _Conn()
        result = ac.run_cascade(
            "alice",
            project=project,
            reason="account_disabled",
            actor="admin-cli:root",
            trigger="admin-cli",
            broker=broker,
            job_store=jobs,
            curve_provider=curve,
            audit_pool=_Pool(conn),
            **kw,
        )
        return result, broker, jobs, curve, conn

    def test_disable_ends_everything_as_one_audited_event(self):
        result, broker, jobs, curve, conn = self._run()
        counts = (result.leases_revoked, result.sessions_ended, result.curve_sessions_revoked)
        assert counts == (2, 2, 2)
        assert sorted(jobs.cancelled) == ["j-p1", "j-p2"]  # never the session job
        assert (curve.revoked, broker.active_owner_projects()) == (["alice"], set())
        (row,) = conn.audit_rows()  # exactly one event
        assert (row[3], row[4], result.audited) == (ac.CASCADE_EVENT, "admin-cli:root", True)
        details = json.loads(row[9])
        assert {k: details[k] for k in ("user", "leases_revoked", "jobs_cancelled", "venue")} == {
            "user": "alice",
            "leases_revoked": 2,
            "jobs_cancelled": 2,
            "venue": SITE,
        }

    def test_project_removal_is_scoped_and_keeps_curve(self):
        result, broker, jobs, curve, conn = self._run(project="p1")
        assert result.leases_revoked == 1 and result.project == "p1"
        assert jobs.cancelled == ["j-p1"]
        assert curve.revoked == []
        assert broker.active_owner_projects() == {("alice", "p2")}
        assert conn.audit_rows()[0][5] == "p1"

    def test_in_process_cascade_tells_the_cancel_bus(self, monkeypatch):
        told: list[str] = []
        monkeypatch.setattr("umrs.jobs.download_safety.broadcast_cancel", told.append)
        monkeypatch.setattr(
            "umrs.jobs.workflow_run.cancel_workflow_execution_for_job", lambda _job: None
        )
        self._run(in_process=True)
        assert sorted(told) == ["j-p1", "j-p2"]

    def test_audit_failure_is_reported_not_raised(self):
        class _Boom(_Conn):
            def cursor(self, **_kw):
                raise RuntimeError("no audit table")

        result = ac.run_cascade(
            "alice",
            project=None,
            reason="r",
            actor="a",
            trigger="sweep",
            broker=_broker(),
            audit_pool=_Pool(_Boom()),
        )
        assert result.audited is False

    def test_a_failing_job_store_still_revokes_and_audits(self):
        class _BrokenJobs(_Jobs):
            def list_jobs(self, **_kw):
                raise RuntimeError("jobs table gone")

        broker, conn = _broker(), _Conn()
        _lease(broker)
        result = ac.run_cascade(
            "alice",
            project=None,
            reason="r",
            actor="a",
            trigger="sweep",
            broker=broker,
            job_store=_BrokenJobs([]),
            audit_pool=_Pool(conn),
        )
        assert (result.leases_revoked, result.jobs_cancelled, result.audited) == (1, 0, True)


class TestSidecarWiring:
    def test_sweep_off_by_flag_or_without_a_database(self, monkeypatch):
        from umrs.server.http_sidecar import _start_account_sweep

        monkeypatch.setenv("UMRS_ACCOUNT_SWEEP_ENABLED", "false")
        assert _start_account_sweep() is None
        monkeypatch.setenv("UMRS_ACCOUNT_SWEEP_ENABLED", "true")
        monkeypatch.setattr("umrs.api._asyncpg_pool._resolve_knowledge_dsn", lambda: ("", False))
        assert _start_account_sweep() is None

    def test_sweep_starts_and_stops_with_a_database(self, monkeypatch):
        from umrs.server.http_sidecar import _start_account_sweep

        monkeypatch.setenv("UMRS_ACCOUNT_SWEEP_ENABLED", "true")
        monkeypatch.setenv("UMRS_ACCOUNT_SWEEP_SEC", "60")
        monkeypatch.setattr(
            "umrs.api._asyncpg_pool._resolve_knowledge_dsn",
            lambda: ("postgresql://nobody@127.0.0.1:1/none", True),
        )
        sweep = _start_account_sweep()
        assert (type(sweep).__name__, sweep.interval) == ("AccountStandingSweep", 60.0)
        sweep.stop(timeout=1)


# ── standing ──────────────────────────────────────────────────────────────


class TestStanding:
    def _members(self, *ids):
        return lambda: set(ids)

    def test_disabled_always_loses(self):
        rec = _rec(disabled=True, role="admin")
        assert ac.standing_loss(rec, "", member_of=self._members(), enforce=False) == (
            ac.ACCOUNT_DISABLED
        )

    def test_membership_loss_only_under_enforce(self):
        rec = _rec(projects=("p2",))
        assert ac.standing_loss(rec, "p1", member_of=self._members("p2"), enforce=False) is None
        assert (
            ac.standing_loss(rec, "p1", member_of=self._members("p2"), enforce=True)
            == ac.MEMBERSHIP_REVOKED
        )

    @pytest.mark.parametrize("role", ["admin", "service"])
    def test_admin_and_service_never_lose_a_project(self, role):
        rec = _rec(role=role, projects=())
        assert ac.standing_loss(rec, "p1", member_of=self._members(), enforce=True) is None

    def test_unknown_account_is_left_alone(self):
        assert ac.standing_loss(None, "p1", member_of=self._members(), enforce=True) is None

    def test_live_projects_are_listed_minus_retired_plus_owned(self):
        conn = _Conn(
            projects=[
                ("p1", "bob", "active"),
                ("p2", "bob", "retired"),
                ("mine", "alice", "active"),
                ("old", "alice", "retired"),
            ]
        )
        assert ac.live_project_ids(conn, _rec(projects=("p1", "p2", "default"))) == {
            "p1",
            "default",
            "mine",
        }

    def test_membership_mode_reads_the_env(self, monkeypatch):
        monkeypatch.delenv("UMRS_PROJECT_MEMBERSHIP", raising=False)
        assert not ac.membership_enforced()
        monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "enforce")
        assert ac.membership_enforced()

    def test_json_disable_wins_over_a_lagging_row(self, tmp_path, monkeypatch):
        store = UserStore(tmp_path)
        store.create("alice")
        store.set_disabled("alice", True)
        lagging = _rec()

        def _init(self, *_args):
            self._dir = tmp_path

        monkeypatch.setattr(UserStore, "__init__", _init)
        monkeypatch.setattr(UserStore, "get", lambda _self, _name: lagging)
        assert ac.load_record("alice").disabled is True


# ── the server-side sweep ─────────────────────────────────────────────────


def _sweep(broker, records, *, projects=(), jobs=None, curve=None):
    conn = _Conn(projects=projects)
    sweep = ac.AccountStandingSweep(
        pool=_Pool(conn),
        broker=broker,
        job_store=jobs,
        record_lookup=records.get,
        curve_provider=lambda: curve,
        interval=5,
    )
    return sweep, conn


class TestSweep:
    def test_disabled_owner_loses_every_lease_curve_and_jobs(self):
        broker = _broker()
        _lease(broker, project="p1", session="s1")
        _lease(broker, project="p2")
        keep = _lease(broker, "bob")
        jobs, curve = _Jobs([_job("j1")]), _Curve()
        sweep, conn = _sweep(
            broker, {"alice": _rec(disabled=True), "bob": _rec("bob")}, jobs=jobs, curve=curve
        )
        results = sweep.run_once()
        assert [(r.user, r.project, r.reason) for r in results] == [
            ("alice", "", ac.ACCOUNT_DISABLED)
        ]
        assert broker.active_owner_projects() == {("bob", "p1")}
        assert broker.get_lease(keep.lease_id).is_valid
        assert (curve.revoked, jobs.cancelled) == (["alice"], ["j1"])
        assert len(conn.audit_rows()) == 1
        assert sweep.run_once() == []  # nothing left: no repeat event

    def test_ex_member_loses_only_that_project_under_enforce(self, monkeypatch):
        monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "enforce")
        broker = _broker()
        _lease(broker, project="p1")
        _lease(broker, project="p2")
        sweep, _conn = _sweep(broker, {"alice": _rec(projects=("p2",))})
        results = sweep.run_once()
        assert [(r.project, r.reason) for r in results] == [("p1", ac.MEMBERSHIP_REVOKED)]
        assert broker.active_owner_projects() == {("alice", "p2")}

    def test_ex_member_keeps_sessions_below_enforce(self, monkeypatch):
        monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "live")
        broker = _broker()
        _lease(broker, project="p1")
        sweep, _conn = _sweep(broker, {"alice": _rec(projects=())})
        assert sweep.run_once() == []

    def test_admin_outside_the_project_is_not_swept(self, monkeypatch):
        monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "enforce")
        broker = _broker()
        _lease(broker, "root", project="anyproj")
        sweep, _conn = _sweep(broker, {"root": _rec("root", role="admin", projects=())})
        assert sweep.run_once() == []

    def test_owner_of_the_project_is_a_member(self, monkeypatch):
        monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "enforce")
        broker = _broker()
        _lease(broker, project="mine")
        sweep, _conn = _sweep(
            broker, {"alice": _rec(projects=())}, projects=[("mine", "alice", "active")]
        )
        assert sweep.run_once() == []

    def test_unknown_owner_is_left_alone(self):
        broker = _broker()
        _lease(broker, "ghost")
        sweep, _conn = _sweep(broker, {})
        assert sweep.run_once() == []

    def test_one_failing_lookup_never_stops_the_sweep(self):
        broker = _broker()
        _lease(broker, "bad")
        _lease(broker, "alice")

        def lookup(user):
            if user == "bad":
                raise RuntimeError("db hiccup")
            return _rec(disabled=True)

        sweep = ac.AccountStandingSweep(
            pool=_Pool(_Conn()),
            broker=broker,
            record_lookup=lookup,
            curve_provider=lambda: None,
            interval=5,
        )
        assert [r.user for r in sweep.run_once()] == ["alice"]

    @pytest.mark.parametrize(
        ("raw", "want"), [(None, 30.0), ("1", 5.0), ("600", 60.0), ("x", 30.0)]
    )
    def test_cadence_stays_within_a_minute(self, monkeypatch, raw, want):
        if raw is None:
            monkeypatch.delenv("UMRS_ACCOUNT_SWEEP_SEC", raising=False)
        else:
            monkeypatch.setenv("UMRS_ACCOUNT_SWEEP_SEC", raw)
        assert ac.sweep_interval_seconds() == want


# ── sessions end through the revoked lease (sweep → status checker) ──────


class TestSessionsEnd:
    def _lane(self, broker, lease_id):
        from tests.fake_pg_pool import FakePool
        from umrs.jobs.frontend_session_backend import (
            FrontendSessionBackend,
            FrontendSessionStatusChecker,
        )
        from umrs.venues.venue_sessions import VenueSessionStore

        class _Prov:
            torn: list[str] = []

            def provision(self, cfg):
                del cfg
                return SimpleNamespace(
                    state="running", reason="", lease_id=lease_id, egress_ip="172.20.8.2"
                )

            def teardown(self, sid):
                self.torn.append(sid)

        config = SimpleNamespace(
            subnet_pool="172.30.0.0/16",
            idle_seconds=600,
            provision_config=lambda **kw: SimpleNamespace(**kw),
        )
        prov = _Prov()
        backend = FrontendSessionBackend(
            config=config,
            provisioner=prov,
            store=VenueSessionStore(),
            broker=broker,
            pool=FakePool(script=[None] * 20),
        )
        return backend, prov, FrontendSessionStatusChecker(backend)

    def _spec(self):
        return {
            "session_id": "sess-1",
            "owner": "alice",
            "project": "p1",
            "source_ids": ["11111111-1111-4111-8111-111111111111"],
            "source_names": ["src"],
            "default_source_id": "11111111-1111-4111-8111-111111111111",
            "default_source_name": "src",
            "web_policy": "public",
            "venue": SITE,
        }

    def test_disabled_users_session_is_torn_down_on_the_next_check(self, monkeypatch):
        monkeypatch.setattr(ac, "load_record", lambda _name: None)
        broker = _broker()
        lease = _lease(broker, project="p1", session="sess-1")
        backend, prov, checker = self._lane(broker, lease.lease_id)
        job = SimpleNamespace(job_id="job-1", spec=self._spec())
        ref = asyncio.run(backend.submit_job(job, SimpleNamespace(attempt_id="a1")))
        attempt = SimpleNamespace(backend_ref=ref)
        assert asyncio.run(checker.check_status(attempt, job)).status == "running"

        sweep, _conn = _sweep(broker, {"alice": _rec(disabled=True)})
        sweep.run_once()
        res = asyncio.run(checker.check_status(attempt, job))
        assert res.status == "succeeded" and res.metrics["end"] == "lease_ended"
        assert prov.torn == ["sess-1"]
        assert backend.store.get("sess-1").state == "stopped"

    def test_a_queued_session_of_a_disabled_user_never_comes_up(self, monkeypatch):
        monkeypatch.setattr(ac, "load_record", lambda _name: _rec(disabled=True))
        backend, prov, _checker = self._lane(_broker(), "lease-x")
        job = SimpleNamespace(job_id="job-1", spec=self._spec())
        with pytest.raises(RuntimeError, match="account_disabled"):
            asyncio.run(backend.submit_job(job, SimpleNamespace(attempt_id="a1")))
        assert prov.torn == []


# ── the admin CLI ─────────────────────────────────────────────────────────


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Admin CLI over a tmp user dir; the cascade's DB legs captured."""
    from umrs.cli.admin import cascade
    from umrs.cli.admin.main import main

    calls: list[tuple] = []
    conn = _Conn(projects=[("owned", "alice", "active")])
    monkeypatch.setattr(cascade, "cli_db_pool", lambda: _Pool(conn))
    monkeypatch.setattr(
        cascade,
        "_run",
        lambda _pool, user, project, reason: calls.append((user, project, reason)) or 0,
    )
    mirrored: list[tuple] = []
    monkeypatch.setattr(
        "umrs.admin.user_store.mirror_record_to_pg",
        lambda _pool, rec, venue: mirrored.append(
            (rec.username, rec.disabled, list(rec.projects), venue)
        ),
    )
    store = UserStore(tmp_path)
    store.create("alice", projects=["p1", "p2", "owned"])
    store.create("root", role="admin")

    def run(*argv):
        return main(["--user-files-dir", str(tmp_path), "--no-reload", *argv])

    return SimpleNamespace(run=run, calls=calls, mirrored=mirrored, store=store)


class TestAdminCli:
    def test_disable_runs_the_account_cascade_and_mirrors_the_row(self, cli):
        assert cli.run("user", "disable", "alice") == 0
        assert cli.calls == [("alice", None, ac.ACCOUNT_DISABLED)]
        assert ("alice", True, ["p1", "p2", "owned"], "") in cli.mirrored

    def test_soft_delete_cascades_too(self, cli):
        assert cli.run("user", "delete", "alice") == 0
        assert cli.calls == [("alice", None, ac.ACCOUNT_DISABLED)]

    def test_remove_from_project_is_scoped_to_that_project(self, cli):
        assert cli.run("user", "remove-from-project", "alice", "p1") == 0
        assert cli.calls == [("alice", "p1", ac.MEMBERSHIP_REVOKED)]
        assert ("alice", False, ["p2", "owned"], "") in cli.mirrored

    def test_removal_from_an_owned_project_keeps_access(self, cli):
        assert cli.run("user", "remove-from-project", "alice", "owned") == 0
        assert cli.calls == []

    def test_admin_removed_from_a_project_keeps_access(self, cli):
        cli.store.add_project("root", "p1")
        assert cli.run("user", "remove-from-project", "root", "p1") == 0
        assert cli.calls == []

    def test_the_only_enabled_admin_cannot_be_disabled_or_deleted(self, cli):
        assert cli.run("user", "disable", "root") == 1
        assert cli.run("user", "delete", "root") == 1
        assert cli.store.get("root").disabled is False and cli.calls == []

    def test_a_second_admin_makes_disabling_one_possible(self, cli):
        cli.store.create("ops", role="admin")
        assert cli.run("user", "disable", "root") == 0
        assert cli.calls == [("root", None, ac.ACCOUNT_DISABLED)]

    def test_enable_reaches_the_row_too(self, cli):
        cli.run("user", "disable", "alice")
        assert cli.run("user", "enable", "alice") == 0
        assert cli.mirrored[-1][:2] == ("alice", False)


# ── Postgres end to end (the CLI against a migrated database) ─────────────


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _ = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("no Postgres configured")
    from umrs.cli.admin.cascade import _ConnPerCallPool

    pool = _ConnPerCallPool(dsn)
    try:
        conn = pool.getconn()
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name = "
                "'secret_leases' AND column_name = 'venue'"
            )
            if cur.fetchone() is None:
                pytest.skip("database not migrated to 0124 (python -m umrs.migrations)")
            from umrs.api.audit_store import ensure_audit_schema

            ensure_audit_schema(conn)
    finally:
        pool.putconn(conn)
    return pool


def _pg(pool, sql, params=()):
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
        conn.commit()
        return rows
    finally:
        pool.putconn(conn)


class TestPostgresEndToEnd:
    def test_cli_disable_revokes_cancels_audits_and_mirrors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_ADMIN_CLI_DB", "on")
        pool = _pg_pool_or_skip()
        user = f"rs0-{os.getpid()}"
        monkeypatch.setattr("umrs.jobs.store.configured_site_name", lambda: "")
        broker = SecretBroker(db_pool=pool, venue="")
        sess = _lease(broker, user, project="p1", session="sess-pg")
        other = _lease(broker, "rs0-bystander", project="p1")
        from umrs.jobs.store import JobStore

        jobs = JobStore(pool)
        job_id = jobs.create_job("p1", user, JobType.WORKFLOW_RUN, {"rs0": user})
        from umrs.cli.admin.main import main

        UserStore(tmp_path).create(user, projects=["p1"])
        assert (
            main(["--user-files-dir", str(tmp_path), "--no-reload", "user", "disable", user]) == 0
        )

        live = (broker.get_lease(sess.lease_id).is_valid, broker.get_lease(other.lease_id).is_valid)
        assert live == (False, True)
        assert jobs.get_job(job_id).status.value == "cancelled"
        (row,) = _pg(
            pool,
            "SELECT actor, details::text FROM audit_events"
            " WHERE event = %s AND details->>'user' = %s",
            (ac.CASCADE_EVENT, user),
        )
        details = json.loads(row[1])
        counts = (details["leases_revoked"], details["jobs_cancelled"])
        assert (row[0].split(":")[0], counts) == ("admin-cli", (1, 1))
        assert _pg(
            pool, "SELECT disabled FROM users WHERE venue = '' AND username = %s", (user,)
        ) == [(True,)]
        _pg(pool, "DELETE FROM users WHERE username = %s", (user,))

    def test_sites_sharing_one_database_keep_their_own_leases(self):
        pool = _pg_pool_or_skip()
        user = f"rs0-sites-{os.getpid()}"
        rccs = SecretBroker(db_pool=pool, venue="rccs")
        ai4s = SecretBroker(db_pool=pool, venue="ai4s")
        legacy = SecretBroker(db_pool=pool, venue="")
        mine_p1 = _lease(rccs, user, project="p1", session="s-rccs")
        mine_p2 = _lease(rccs, user, project="p2")
        theirs = _lease(ai4s, user, project="p1", session="s-ai4s")
        old = _lease(legacy, user, project="p1")
        assert (user, "p1") in rccs.active_owner_projects()

        revoked = rccs.revoke_by_owner(user, project="p1", actor="t")
        assert {lease.lease_id for lease in revoked} == {mine_p1.lease_id, old.lease_id}
        assert rccs.get_lease(mine_p2.lease_id).is_valid
        assert ai4s.get_lease(theirs.lease_id).is_valid
        assert (user, "p1") not in rccs.active_owner_projects()
        assert (user, "p1") in ai4s.active_owner_projects()
        rccs.revoke_by_owner(user)
        ai4s.revoke_by_owner(user)
