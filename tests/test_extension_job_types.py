"""U3 S3b: extension-owned job types and the D8 claim policy.

``frontend_session`` registers through ``ExtensionContext.add_job_type`` (the
in-tree r4 part). A process with no lane for an extension type does not claim
its rows: they stay QUEUED for a front that has the extension.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from umrs.extensions import JobTypeRegistration, load_extensions
from umrs.extensions.context import ExtensionContext
from umrs.jobs.adapters import create_backend_factory, create_status_checker
from umrs.jobs.extension_job_types import (
    EXTENSION_JOB_TYPES,
    build_extension_lanes,
    unclaimable_here,
)
from umrs.jobs.models import JobType


def _ctx(**builders) -> ExtensionContext:
    ctx = ExtensionContext().bind("x")
    for value, build in builders.items():
        ctx.add_job_type(JobTypeRegistration(value=value, build_lane=build))
    return ctx


def test_every_extension_type_is_a_reserved_job_type_value():
    assert set(EXTENSION_JOB_TYPES) <= {t.value for t in JobType}


def test_r4_registers_the_frontend_session_lane():
    ctx = load_extensions(("r4",))
    reg = ctx.job_types["frontend_session"]
    assert reg.build_lane.__name__ == "build_frontend_session_lane"
    assert load_extensions(()).job_types == {}


def test_lanes_wire_only_when_the_builder_delivers(caplog):
    caplog.set_level(logging.INFO, logger="umrs.jobs.extension_job_types")
    pair = (object(), object())

    def boom(_pool):
        raise RuntimeError("venue misconfigured")

    lanes = build_extension_lanes(_ctx(frontend_session=lambda _p: pair), "pool")
    assert lanes == {"frontend_session": pair}
    assert build_extension_lanes(_ctx(frontend_session=lambda _p: None), "pool") == {}
    assert build_extension_lanes(_ctx(frontend_session=boom), "pool") == {}
    assert "frontend_session lane wiring skipped" in caplog.text
    assert "not claimed on this server: ['frontend_session']" in caplog.text


def test_unclaimable_is_the_unwired_extension_types():
    assert unclaimable_here([]) == frozenset({"frontend_session"})
    assert unclaimable_here(["frontend_session"]) == frozenset()


def _factory(**kw):
    return create_backend_factory(
        orchestrator=MagicMock(), registry=MagicMock(), port_manager=MagicMock(), **kw
    )


def test_factory_serves_registered_lanes_and_refuses_unwired_extension_types():
    lane = object()
    assert _factory(extension_backends={"frontend_session": lane})(JobType.FRONTEND_SESSION) is lane
    with pytest.raises(RuntimeError, match="frontend_session lane not configured"):
        _factory()(JobType.FRONTEND_SESSION)


def test_status_checker_registers_extension_checkers():
    ext_checker = MagicMock()
    composite = create_status_checker(
        readiness_poller=MagicMock(),
        registry=MagicMock(),
        extension_checkers={"frontend_session": ext_checker},
    )
    assert composite._checkers[JobType.FRONTEND_SESSION] is ext_checker


# ── claim policy against the CI-shaped Postgres ─────────────────────────────


def _env_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:  # noqa: BLE001 - no database: skip like the venue suites
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("TRUNCATE jobs CASCADE")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture
def pool(request):
    p = _env_pool_or_skip()
    request.addfinalizer(p.closeall)
    return p


def _store(pool: Any, unclaimable: frozenset[str] = frozenset()):
    from umrs.jobs.store import JobStore

    store = JobStore(pool, "test-instance")
    store.site_name = "rccs-front"
    store.unclaimable_job_types = unclaimable
    return store


def _queued(store, job_type: JobType, spec: dict) -> str:
    job = store.create_job(
        project_id="proj-x", user_id="alice", job_type=job_type, spec=spec, idempotency_key=None
    )
    store.enqueue_job(job)
    return job


def test_claim_skips_extension_rows_without_a_lane(pool):
    plain = _store(pool, unclaimable=frozenset({"frontend_session"}))
    session = _queued(plain, JobType.FRONTEND_SESSION, {"session_id": "s1"})
    loop = _queued(plain, JobType.RESEARCH_LOOP, {"loop_id": "l1"})
    assert plain.claim_next_jobs(limit=10) == [loop]
    # A front that has the lane claims the row the other one left QUEUED.
    assert session in _store(pool).claim_next_jobs(limit=10)
