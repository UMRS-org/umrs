"""A server-hosted loop whose iteration is a venue-hosted team plan run
(ADR-092 slice 8b).

The existing loop (``EngineLoopRunner``, ADR-036) drives the plan executor
as it drives the agent executor; each round starts one slice-8 run through
the venue verbs (faked here: ``FakeHost`` records the starts and answers
the outcome the run's worker would record), applies the shared decision
table, and stages a published round's finding as a pending proposal. The
ask gate is the runner's own, over the durable approvals inbox. No live
venue, no container, no model.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_companion_venue_routes import READY_ENV, FakeSealed, _meta, _Pool
from test_plan_run_venue import TokenPodman, VenueJobStore
from umrs.api import plan_run_venue
from umrs.companion_trust.approval_inbox import ApprovalInboxStore
from umrs.jobs import plan_run_records as prr
from umrs.jobs.secret_broker import SecretBroker
from umrs.knowledge import ask_policy, plan_iteration, tool_approvals
from umrs.knowledge.engine_loop_runner import EngineLoopRunner
from umrs.knowledge.loop import LoopStatus, ResearchLoop
from umrs.session_policy import PolicyRule, PolicyRuleStore
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

from r4_contracts import loop_plan

TEMPLATE = Path("/tmp/w-a92/r4/src/r4/teams/plan/templates/research-round.json")


def _doc() -> dict[str, Any]:
    """The research-round template as the r4 client resolves and sends it."""
    candidates = [
        TEMPLATE,
        Path(__file__).resolve().parents[2] / "r4/src/r4/teams/plan/templates/research-round.json",
    ]
    for path in candidates:
        if path.is_file():
            doc = json.loads(path.read_text())
            # the client resolves instruction references (r4 plan_sources)
            for name, profile in doc.get("agents", {}).items():
                profile["instructions"] = f"You are the {name}."
            return doc
    pytest.skip("the r4 research-round template is not reachable from this checkout")


PICKED = {
    "status": "picked",
    "winner_index": 0,
    "ranking": [{"index": 0, "score": 0.9}],
    "winner": {"hypothesis": "H1", "protocol": "P1", "refuting_outcome": "R1"},
}
UNDECIDED = {"status": "undecided", "ranking": [{"index": 0, "score": 0.3}]}


class FakeHost:
    """The venue verbs and the proposal lane, recording what the loop asks."""

    def __init__(self, outcomes: list[dict[str, Any]], missing: list[str] | None = None):
        self.outcomes = list(outcomes)
        self.missing = list(missing or [])
        self.starts: list[dict[str, Any]] = []
        self.cancels: list[str] = []
        self.proposals: list[dict[str, Any]] = []
        self.views = 0
        self.on_view: Any = None
        self.current: dict[str, Any] | None = None

    def host(self) -> plan_iteration.PlanRunHost:
        return plan_iteration.PlanRunHost(
            start=self.start,
            view=self.view,
            cancel=self.cancel,
            propose=self.propose,
            venue_missing=lambda: list(self.missing),
        )

    def start(self, **kw):
        self.starts.append(kw)
        self.current = self.outcomes.pop(0)
        return {"run_id": kw["run_id"], "job_id": f"job-{kw['run_id']}"}

    def view(self, **_kw):
        self.views += 1
        if self.on_view is not None:
            self.on_view(self.views)
        outcome = self.current or {}
        if outcome.get("pending"):
            return {"status": "running", "attempts": [{"status": "running"}], "venue": {}}
        status = outcome.get("status", "success")
        summary = {"output": {"decision": outcome.get("review", {})}, "error": outcome.get("error")}
        return {
            "status": "completed",
            "attempts": [{"status": "succeeded", "plan_status": status, "summary": summary}],
            "venue": {"state": "stopped"},
        }

    def cancel(self, **kw):
        self.cancels.append(kw["run_id"])
        return {"cancelled": True}

    def propose(self, **kw):
        self.proposals.append(kw)
        return {"status": "proposed", "graph_id": "g1", "proposal_id": "gp-1"}


class Store:
    def __init__(self):
        self.loops: dict[str, Any] = {}
        self.stopped = False

    def save(self, loop):
        self.loops[loop.loop_id] = loop

    def get(self, loop_id):
        if self.stopped:
            return SimpleNamespace(status=LoopStatus.STOPPED)
        return self.loops.get(loop_id)


@pytest.fixture(autouse=True)
def inbox():
    store = ApprovalInboxStore()
    tool_approvals.configure(store)
    yield store
    tool_approvals.configure(None)
    plan_iteration.configure(None)


def _loop(mode="auto", max_iterations=3, **meta) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="loop-abc",
        project_id="proj-x",
        research_goal="find a better solver",
        max_iterations=max_iterations,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(
        mode=mode,
        owner_user_id="alice",
        iteration_plan={
            "template": "research-round",
            "doc": _doc(),
            "args": {},
            "sources": ["site"],
            "default_model": "qwen3-8b@site",
            "web_policy": "none",
        },
        **meta,
    )
    return loop


def _drive(loop, host: FakeHost, store: Store | None = None) -> ResearchLoop:
    plan_iteration.configure(host.host())
    store = store or Store()
    store.save(loop)
    engine = lambda: plan_iteration.PlanIterationEngine(  # noqa: E731
        executor=plan_iteration.PlanRunIteration(poll_seconds=0.0, sleep=lambda _s: None)
    )
    runner = EngineLoopRunner(loop, engine_factory=engine, store=store, poll_seconds=0.01)
    return runner.drive(loop)


def _decisions(loop) -> list[str]:
    return [it.results[loop_plan.LOOP_DECISION_KEY]["decision"] for it in loop.iterations]


class TestTheRound:
    def test_a_pick_publishes_and_only_proposes(self):
        host = FakeHost([{"review": UNDECIDED}, {"review": PICKED}])
        loop = _drive(_loop(), host)
        (first, second) = host.starts
        published = loop.iterations[-1].results
        assert {
            "decisions": _decisions(loop),
            "status": (loop.status, loop.metadata["state_reason"]),
            "run_ids": [s["run_id"] for s in host.starts],
            "venue": (first["owner"], first["project"], first["source_refs"]),
            "objective": first["definition"]["params"]["objective"],
            "context_1": first["definition"]["params"]["context"],
            "context_2": second["definition"]["params"]["context"],
            "default_model": first["definition"]["manifest"]["default_model"],
            "finding": (published["finding"]["status"], host.proposals[0]["operation"]["status"]),
            "confidence": host.proposals[0]["operation"]["confidence"],
            "candidate_written": published["catalog_candidate"]["written"],
        } == {
            "decisions": ["continue", "publish"],
            "status": (LoopStatus.STOPPED, "published:the reviewers picked a design"),
            "run_ids": ["loop-abc-it1", "loop-abc-it2"],
            "venue": ("alice", "proj-x", ["site"]),
            "objective": "find a better solver",
            "context_1": "",
            "context_2": "Round 1: review undecided (design 0: 0.3)",
            "default_model": "qwen3-8b@site",
            "finding": ("proposed", "proposed"),
            "confidence": 0.0,
            "candidate_written": False,
        }

    def test_no_pick_continues_to_max_iterations(self):
        host = FakeHost([{"review": UNDECIDED}] * 2)
        loop = _drive(_loop(max_iterations=2), host)
        assert _decisions(loop) == ["continue", "continue"]
        assert loop.metadata["state_reason"] == "max_iterations" and host.proposals == []

    def test_a_failed_run_stops_the_loop_with_its_reason(self):
        host = FakeHost([{"status": "cancelled", "error": "gate rejected", "review": PICKED}])
        loop = _drive(_loop(), host)
        assert _decisions(loop) == ["stop"]
        assert loop.metadata["state_reason"] == "plan_cancelled:gate rejected"
        assert host.proposals == [] and len(host.starts) == 1

    def test_the_wall_clock_bounds_each_run(self):
        started = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        loop = _loop(max_iterations=1, max_wall_clock_hours=1.0)
        loop.started_at = started
        host = FakeHost([{"review": UNDECIDED}])
        _drive(loop, host)
        (start,) = host.starts
        own = start["definition"]["plan"]["budget"]["max_wallclock_sec"]
        assert 1700 < own <= 1800 < 3600  # min(the template's 3600, what is left)
        assert start["caps"]["max_wall_clock_sec"] == own

    def test_no_run_starts_without_the_venue(self):
        host = FakeHost([], missing=["UMRS_VENUE_ENABLED"])
        loop = _drive(_loop(), host)
        assert host.starts == [] and _decisions(loop) == ["stop"]
        assert loop.metadata["state_reason"].startswith("plan_iteration_needs_server_venue")

    def test_no_run_starts_without_a_wired_host(self):
        loop = _loop()
        plan_iteration.configure(None)
        store = Store()
        store.save(loop)
        engine = lambda: plan_iteration.PlanIterationEngine()  # noqa: E731
        EngineLoopRunner(loop, engine_factory=engine, store=store).drive(loop)
        assert loop.metadata["state_reason"].startswith("plan_host_unavailable")

    def test_a_stopped_loop_cancels_the_run_in_flight(self):
        host = FakeHost([{"pending": True}])
        store = Store()
        host.on_view = lambda n: setattr(store, "stopped", n >= 2)
        loop = _drive(_loop(), host, store)
        assert host.cancels == ["loop-abc-it1"]
        assert loop.iterations[0].results["plan"]["status"] == "cancelled"


class TestTheGate:
    def test_a_plan_run_is_a_cost_action(self):
        assert ask_policy.action_trigger_classes({"kind": plan_iteration.PLAN_RUN}) == {"cost"}

    def test_ask_mode_parks_on_the_inbox_until_a_human_decides(self):
        host = FakeHost([{"review": PICKED}])
        decided: list[str] = []

        def human():
            for _ in range(200):
                pending = tool_approvals.pending(user_id="alice")
                if pending:
                    assert host.starts == []  # nothing ran while it waited
                    tool_approvals.decide(pending[0]["id"], "once", decided_by="alice")
                    decided.append(pending[0]["action"]["kind"])
                    return
                time.sleep(0.02)

        thread = threading.Thread(target=human)
        thread.start()
        loop = _drive(_loop(mode="ask", max_iterations=1), host)
        thread.join(5)
        assert decided == ["plan_run"] and len(host.starts) == 1
        assert _decisions(loop) == ["publish"]

    def test_a_humans_no_stops_the_loop_and_nothing_runs(self):
        host = FakeHost([{"review": PICKED}])

        def human():
            for _ in range(200):
                pending = tool_approvals.pending(user_id="alice")
                if pending:
                    tool_approvals.decide(pending[0]["id"], "no", decided_by="alice")
                    return
                time.sleep(0.02)

        thread = threading.Thread(target=human)
        thread.start()
        loop = _drive(_loop(mode="ask"), host)
        thread.join(5)
        assert host.starts == [] and loop.iterations[0].status == "rejected"
        assert loop.metadata["state_reason"] == "approval_rejected:plan_run"


class TestTheStartRoute:
    @pytest.fixture
    def loops(self, monkeypatch):
        from umrs.api import loop as loop_api
        from umrs.knowledge.loop import AutonomousResearchLoop

        for name, value in READY_ENV.items():
            monkeypatch.setenv(name, value)
        loop_api.clear_loop_store()
        previous = loop_api._loop_engine
        loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
        registered: list[str] = []
        monkeypatch.setattr(
            "umrs.knowledge.engine_loop_runner.register",
            lambda loop, **kw: registered.append((loop.loop_id, kw["engine_factory"])),
        )
        app = FastAPI()
        app.include_router(loop_api.router)
        app.state.companion_venue_service = _real_service(PolicyRuleStore())
        yield TestClient(app), loop_api, registered
        loop_api._loop_engine = previous
        loop_api.clear_loop_store()

    def _body(self, **over):
        return {
            "project_id": "p",
            "research_goal": "g",
            "iteration_plan": "research-round",
            "iteration_plan_doc": _doc(),
            "plan_sources": ["openai-main"],
            "plan_web_policy": "none",
            **over,
        }

    def test_with_the_venue_a_plan_loop_starts_and_asks_by_default(self, loops):
        client, loop_api, registered = loops
        res = client.post("/umrs/loops", json=self._body())
        assert res.status_code == 201, res.text
        (loop,) = loop_api._loop_store.list_all()
        assert loop.metadata["mode"] == "ask"
        assert loop.metadata["iteration_plan"]["sources"] == ["openai-main"]
        ((_loop_id, factory),) = registered
        assert isinstance(factory(), plan_iteration.PlanIterationEngine)

    def test_a_plan_loop_resolves_no_workflow_even_on_a_real_engine(self, loops, monkeypatch):
        client, loop_api, _registered = loops
        monkeypatch.setattr(loop_api._loop_engine, "_simulated", False)
        res = client.post("/umrs/loops", json=self._body())
        assert res.status_code == 201, res.text
        (loop,) = loop_api._loop_store.list_all()
        assert "workflow_resolution" not in loop.metadata

    def test_an_explicit_mode_is_kept(self, loops):
        client, loop_api, _registered = loops
        assert client.post("/umrs/loops", json=self._body(mode="auto")).status_code == 201
        assert loop_api._loop_store.list_all()[0].metadata["mode"] == "auto"

    @pytest.mark.parametrize(
        ("over", "why"),
        [
            ({"plan_sources": []}, "plan_sources"),
            ({"iteration_plan": "code"}, "cannot run as a loop iteration"),
            ({"plan_args": {"objective": "x"}}, "belong to the loop"),
            ({"iteration_plan_doc": {"plan_version": 1}}, "plan_invalid"),
        ],
    )
    def test_an_unrunnable_plan_is_refused_before_anything_exists(self, loops, over, why):
        client, loop_api, registered = loops
        res = client.post("/umrs/loops", json=self._body(**over))
        assert res.status_code == 422 and why in res.text
        assert loop_api._loop_store.list_all() == [] and registered == []


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ({"status": "cancelled"}, "cancelled"),
        ({"attempts": [{"plan_status": "success", "status": "succeeded"}]}, "success"),
        ({"venue": {"state": "running"}, "attempts": [{"status": "running"}]}, None),
        # the session stopped but the worker's attempt still runs: not over
        ({"venue": {"state": "stopped"}, "attempts": [{"status": "running"}]}, None),
        ({"venue": {"state": "failed", "reason": "pull"}, "attempts": []}, "failed"),
        ({"venue": {"state": "stopped"}, "attempts": [{"status": "lost"}]}, "failed"),
    ],
)
def test_a_runs_outcome_is_what_its_record_says(view, expected):
    outcome = plan_iteration.run_outcome({"status": "running", **view})
    assert (outcome or {}).get("status") == expected


def test_the_server_loop_reads_the_shared_table():
    """UMRS's consumer of r4_contracts.loop_plan (the one pin is there)."""
    assert plan_iteration.loop_plan is loop_plan


def test_the_proposal_lane_stages_a_pending_proposal_only(monkeypatch):
    created: list[dict[str, Any]] = []

    class Graphs:
        def __init__(self, _pool):
            pass

        def list_graphs(self, **_kw):
            return [SimpleNamespace(graph_id="g1")]

    class Proposals:
        def __init__(self, _pool):
            pass

        def create_proposal(self, graph_id, **kw):
            created.append({"graph": graph_id, **kw})
            return SimpleNamespace(proposal_id="gp-1", status="pending")

        def decide(self, *_a, **_kw):  # pragma: no cover - must never be called
            raise AssertionError("a loop never decides a proposal")

    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", Graphs)
    monkeypatch.setattr("umrs.knowledge.graph_registry.ProposalStore", Proposals)
    op = loop_plan.finding_operation(loop_id="L", iteration=1, run_id="r", winner={})
    out = plan_run_venue.propose_finding(
        None,
        project="p",
        owner="alice",
        reason="why",
        source_artifacts=["team_run:r"],
        operation=op,
    )
    assert (out["status"], out["proposal_status"], created[0]["proposed_operations"]) == (
        "proposed",
        "pending",
        [op],
    )


def test_a_job_hosted_plan_loop_is_driven_by_the_plan_executor(monkeypatch):
    from umrs.knowledge import loop_hosting

    registered: list[Any] = []
    monkeypatch.setattr(
        "umrs.knowledge.engine_loop_runner.register",
        lambda _loop, **kw: registered.append(kw["engine_factory"]),
    )
    loop_hosting.configure(
        loop_hosting.HostingWiring(
            store=lambda: Store(),
            engine_factory=lambda: lambda: pytest.fail("not the phased engine"),
            agent_executor=lambda _p: pytest.fail("not the agent executor"),
            checkpoint_store=lambda: None,
            finding_sink=lambda: None,
        )
    )
    try:
        kind = loop_hosting._register_driver(_loop(), Store(), job_id="j", attempt_id="a")
    finally:
        loop_hosting.configure(None)
    assert kind == "engine"
    assert isinstance(registered[0](), plan_iteration.PlanIterationEngine)


def test_each_run_gets_the_tighter_of_the_project_and_loop_caps():
    tighter = plan_run_venue._tighter_caps
    assert tighter({"max_wall_clock_sec": 600}, {"max_wall_clock_sec": 900}) == {
        "max_wall_clock_sec": 600
    }
    assert tighter({}, {"max_wall_clock_sec": 900}) == {"max_wall_clock_sec": 900}
    assert tighter({"max_wall_clock_sec": 600}, {}) == {"max_wall_clock_sec": 600}
    assert tighter({}, {}) == {}


# ── ADR-099 D-2: the loop's runs need the user's web choice (fix P1) ───────


def _real_service(store: PolicyRuleStore, jobs: Any = None) -> FrontendVenueService:
    """A real venue service over the route-test fakes (no podman, no DB)."""
    broker = SecretBroker()
    return FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=FrontendPodmanProvisioner(
            broker,
            podman=TokenPodman(),
            session_root=Path(tempfile.mkdtemp(prefix="plan-loop-")),
        ),
        pool=_Pool(),
        sealed_store=FakeSealed({("alice", "openai-main"): _meta("openai-main")}),
        job_store=jobs if jobs is not None else VenueJobStore(),
        site_name="rccs",
        policy_store=store,
        policy_env={},
    )


class TestTheWebChoice:
    """Old code: a loop started without plan_web_policy was accepted (201)
    and every round was refused at its start (plan_refused at iteration 1);
    plan_web_allow_domains did not exist, so allowlist could never run."""

    @pytest.fixture
    def loops(self, monkeypatch):
        from umrs.api import loop as loop_api
        from umrs.knowledge.loop import AutonomousResearchLoop

        for name, value in READY_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("UMRS_SESSION_WEB_CHOICE_REQUIRED", raising=False)
        loop_api.clear_loop_store()
        previous = loop_api._loop_engine
        loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
        monkeypatch.setattr("umrs.knowledge.engine_loop_runner.register", lambda *_a, **_k: None)
        monkeypatch.setattr("umrs.api.loop._project_budget_caps", lambda _p: {})
        store = PolicyRuleStore()
        jobs = VenueJobStore()
        service = _real_service(store, jobs)
        app = FastAPI()
        app.include_router(loop_api.router)
        app.state.companion_venue_service = service
        yield SimpleNamespace(
            client=TestClient(app), api=loop_api, store=store, service=service, jobs=jobs
        )
        loop_api._loop_engine = previous
        loop_api.clear_loop_store()

    def _body(self, **over):
        return {
            "project_id": "proj-x",
            "research_goal": "find a better solver",
            "iteration_plan": "research-round",
            "iteration_plan_doc": _doc(),
            "plan_sources": ["openai-main"],
            "mode": "auto",
            **over,
        }

    def test_no_choice_is_refused_at_loop_start(self, loops):
        res = loops.client.post("/umrs/loops", json=self._body())
        assert res.status_code == 400, res.text
        assert res.headers["x-platform-error-code"] == "sessionpolicy.web_choice_required"
        assert "plan_web_policy" in res.text and "none | allowlist | public" in res.text
        assert loops.api._loop_store.list_all() == []

    def test_a_choice_wider_than_the_cap_is_refused_at_loop_start(self, loops):
        loops.store.put(
            PolicyRule("rccs", "project", "proj-x", "admin", {"web": {"allow": ["none"]}})
        )
        res = loops.client.post("/umrs/loops", json=self._body(plan_web_policy="public"))
        assert res.status_code == 403, res.text
        assert res.headers["x-platform-error-code"] == "sessionpolicy.widening_refused"
        assert loops.api._loop_store.list_all() == []

    def test_the_choice_reaches_the_run_over_the_real_venue(self, loops):
        """Route start, then round 1 through build_loop_plan_host over a real
        FrontendVenueService: the run's session carries the user's allowlist."""
        body = self._body(plan_web_policy="allowlist", plan_web_allow_domains=[".arxiv.org"])
        res = loops.client.post("/umrs/loops", json=body)
        assert res.status_code == 201, res.text
        (loop,) = loops.api._loop_store.list_all()
        loop.metadata["owner_user_id"] = "alice"
        host = plan_run_venue.build_loop_plan_host(
            None, service=loops.service, job_store=loops.jobs
        )
        executor = plan_iteration.PlanRunIteration(
            host=host, poll_seconds=0.0, sleep=lambda _s: None
        )
        result = executor.run(loop, 1, None, cancel_check=lambda: True)
        assert result["plan"]["status"] == "cancelled", result  # it ran; the test stopped it
        run_id = plan_iteration.run_id_for(loop, 1)
        row = loops.service.store.get(prr.venue_session_id("alice", run_id))
        assert row is not None
        assert row.policy["web"] == {"mode": "allowlist", "domains": [".arxiv.org"]}
        assert row.policy["origin"]["web.mode"] == "choice"


class TestAStoredLoopWithoutAChoice:
    def test_it_stops_before_its_round_naming_the_way_out(self, monkeypatch):
        monkeypatch.delenv("UMRS_SESSION_WEB_CHOICE_REQUIRED", raising=False)
        host = FakeHost([{"review": PICKED}])
        loop = _loop()
        loop.metadata["iteration_plan"]["web_policy"] = ""
        loop = _drive(loop, host)
        assert host.starts == []
        reason = loop.metadata["state_reason"]
        assert reason.startswith("sessionpolicy.web_choice_required")
        assert "plan_web_policy" in reason and "--web" in reason

    def test_with_the_choice_not_required_it_runs_on_the_site_default(self, monkeypatch):
        monkeypatch.setenv("UMRS_SESSION_WEB_CHOICE_REQUIRED", "false")
        host = FakeHost([{"review": PICKED}])
        loop = _loop()
        loop.metadata["iteration_plan"]["web_policy"] = ""
        _drive(loop, host)
        assert len(host.starts) == 1 and host.starts[0]["web_policy"] == ""
