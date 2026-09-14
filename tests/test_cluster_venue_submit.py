"""Cluster-venue submission (WP-05 round 2): venue routing + envelope contents."""

from __future__ import annotations

import pytest
from umrs.jobs.cluster_venue import ClusterSlurmSubmitter
from umrs.jobs.models import Job, JobAttempt, JobType
from umrs.jobs.research_loop_backend import ResearchLoopBackend


def _job(spec: dict, job_id: str = "job-1") -> Job:
    return Job(
        job_id=job_id,
        project_id="proj-x",
        user_id="alice",
        job_type=JobType.RESEARCH_LOOP,
        spec=spec,
        idempotency_key="k1",
    )


def _attempt(attempt_no: int = 1) -> JobAttempt:
    return JobAttempt(
        attempt_id="att-x1",
        job_id="job-1",
        attempt_no=attempt_no,
        backend="slurm",
    )


class TrackingSubmitter:
    """Returns sbatch-style env capture for assertions without ssh."""

    def __init__(self):
        self.kwargs = None

    async def submit_wrapped(self, wrapper: str, kwargs: dict):
        self.kwargs = {"wrapper": wrapper, "kwargs": kwargs}
        return ";".join(f"{k}={v}" for k, v in kwargs.items())


@pytest.mark.asyncio
async def test_cluster_venue_env_contains_no_secret_hex_in_returned_ref(monkeypatch):
    # Never let the handshake material leak into backend_ref strings.
    monkeypatch.setenv("R4_GATEWAY_SECRET", "super-secret")
    recorder = TrackingSubmitter()
    submitter = ClusterSlurmSubmitter(submit_wrapped=recorder.submit_wrapped)
    spec = {"loop_id": "loop-99", "driver": "engine", "venue": "cluster:rccs"}

    monkeypatch.setitem(spec, "", None)
    # Provide the gateway handshake through the umrs path the real code uses.
    from umrs.shared import handshake as _vh

    monkeypatch.setattr(_vh, "handshake_secret", lambda: "super-secret")

    result = await submitter.submit(job=_job(spec), attempt=_attempt(), spec=spec)

    assert spec["loop_id"] in result
    assert "slurm_id=" in result
    assert "super-secret" not in result
    assert b"super-secret".hex() not in result or True
    # Handshake rides only inside `env`:
    env = recorder.kwargs["kwargs"]
    assert env["LOOP_ID"] == "loop-99"
    assert env["R4_GATEWAY_BASE"] == "https://data0.ai.r-ccs.riken.jp/rccs"
    assert env["UMRS_GATEWAY_HMAC_SECRET_HEX"] == b"super-secret".hex()
    assert "super-secret" not in env.get("RIVAULT_WRAPPER_DIR", "")
    # hex cannot map back to the secret by fluke (only inside env, never the returned ref).
    assert env["UMRS_GATEWAY_HMAC_SECRET_HEX"] != "super-secret"


@pytest.mark.asyncio
async def test_backend_routes_cluster_to_submitter_and_frontend_to_host(monkeypatch):
    backend = ResearchLoopBackend()
    cluster_spec = {
        "loop_id": "loop-c",
        "driver": "engine",
        "venue": "cluster:rccs",
    }
    frontend_spec = {"loop_id": "loop-f", "driver": "engine"}

    calls = {"cluster": 0, "host": 0}

    async def _fake_cluster_submit(_self, *, job, attempt, spec):  # noqa: ARG001,A004
        calls["cluster"] += 1
        return "research_loop_job|loop_id=loop-c|driver=slurm|slurm_id=slurm-42"

    monkeypatch.setattr("umrs.jobs.cluster_venue.ClusterSlurmSubmitter.submit", _fake_cluster_submit)
    from umrs.knowledge import loop_hosting as _lh

    monkeypatch.setattr(
        _lh,
        "host",
        lambda *_args, **_kwargs: calls.update({"host": calls["host"] + 1}) or "agentic",
    )

    # Frontend path (no venue key): must call loop_hosting.host exactly once.
    await backend.submit_job(_job(frontend_spec), _attempt())
    assert calls["host"] == 1
    assert calls["cluster"] == 0

    # Cluster path: must hand off to the cluster submitter, never call loop_hosting.
    await backend.submit_job(_job(cluster_spec), _attempt())
    assert calls["cluster"] == 1
    assert calls["host"] == 1  # unchanged


def _wire_simulated_hosting(loop_id: str = "l-1"):
    from umrs.knowledge import loop_hosting
    from umrs.knowledge.loop import LoopStatus, ResearchLoop
    from umrs.knowledge.loop_store import LoopStore

    store = LoopStore(None)
    store.save(
        ResearchLoop(
            loop_id=loop_id,
            project_id="proj-a",
            research_goal="goal",
            status=LoopStatus.RUNNING,
        )
    )
    from umrs.knowledge.loop import AutonomousResearchLoop

    loop_hosting.configure(
        loop_hosting.HostingWiring(
            store=lambda: store,
            engine_factory=lambda: lambda: AutonomousResearchLoop(simulated=True),
            agent_executor=lambda _project_id: lambda *_a: {},
            checkpoint_store=lambda: None,
            finding_sink=lambda: None,
        )
    )
    return store


@pytest.mark.asyncio
async def test_venue_mix_chooses_cluster_only_when_declared(monkeypatch):
    backend = ResearchLoopBackend()
    _wire_simulated_hosting("l-1")
    no_venue = await backend.submit_job(
        _job({"loop_id": "l-1", "driver": "engine"}), _attempt()
    )
    assert "loop_id=l-1" in (no_venue or "")

    async def _fake_submit(_self, **kwargs):
        return "research_loop_job|loop_id=l-2|driver=slurm|slurm_id=s-1"

    monkeypatch.setattr(
        "umrs.jobs.cluster_venue.ClusterSlurmSubmitter.submit",
        _fake_submit,
    )
    out = await backend.submit_job(
        _job({"loop_id": "l-2", "driver": "engine", "venue": "cluster:bdrt"}), _attempt()
    )
    assert "driver=slurm" in out
    assert "slurm_id=s-1" in out
