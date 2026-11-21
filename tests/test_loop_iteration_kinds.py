"""U3 S3c: loop iteration kinds through ``add_loop_iteration_kind``.

The r4 plan-as-iteration executor (ADR-092 slice 8b) registers as kind
``plan`` (in-tree r4). Core resolves it at the start route, the thread host
(start and re-host after a restart) and the job-hosted driver; without the
extension a plan start or a persisted plan loop is refused by name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from umrs.extensions import loader, reset_extension_context
from umrs.knowledge import loop_iteration_kinds
from umrs.knowledge.loop import ResearchLoop
from umrs_r4.knowledge import plan_iteration


@pytest.fixture(autouse=True)
def _fresh_context(monkeypatch):
    monkeypatch.delenv(loader.ENV, raising=False)
    reset_extension_context()
    yield
    reset_extension_context()


def _without_extensions(monkeypatch) -> None:
    monkeypatch.setenv(loader.ENV, "none")
    reset_extension_context()


def _plan_loop(**meta: Any) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="loop-k",
        project_id="proj-x",
        research_goal="g",
        max_iterations=2,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(iteration_plan={"template": "research-round", "doc": {}}, **meta)
    return loop


def _plain_loop() -> ResearchLoop:
    loop = _plan_loop()
    loop.metadata.pop("iteration_plan")
    return loop


def test_r4_provides_the_plan_kind():
    kind = loop_iteration_kinds.kind_for_loop(_plan_loop())
    assert kind.name == "plan"
    assert kind.matches is plan_iteration.is_plan_loop
    assert loop_iteration_kinds.kind_for_loop(_plain_loop()) is None


def test_start_resolution():
    assert loop_iteration_kinds.kind_for_start(SimpleNamespace(iteration_plan="  ")) is None
    kind = loop_iteration_kinds.kind_for_start(SimpleNamespace(iteration_plan="research-round"))
    assert kind.name == "plan"


def test_without_the_extension_a_plan_loop_is_refused_by_name(monkeypatch):
    _without_extensions(monkeypatch)
    with pytest.raises(loop_iteration_kinds.IterationKindUnavailable, match="'r4'"):
        loop_iteration_kinds.kind_for_loop(_plan_loop())
    with pytest.raises(loop_iteration_kinds.IterationKindUnavailable) as caught:
        loop_iteration_kinds.kind_for_start(SimpleNamespace(iteration_plan="research-round"))
    assert caught.value.kind == "plan"
    assert loop_iteration_kinds.kind_for_loop(_plain_loop()) is None


def test_the_start_route_refuses_a_plan_without_the_extension(monkeypatch):
    from umrs.api import loop as loop_api

    _without_extensions(monkeypatch)
    request = SimpleNamespace(iteration_plan="research-round")
    with pytest.raises(HTTPException) as caught:
        loop_api._start_iteration_kind(request, object(), "alice")
    assert caught.value.status_code == 422
    assert caught.value.headers["x-platform-error-code"] == "loops.unprocessable"
    assert "needs an extension that is not enabled" in caught.value.detail


def _capture_engine_register(monkeypatch) -> list[Any]:
    registered: list[Any] = []
    monkeypatch.setattr(
        "umrs.knowledge.engine_loop_runner.register",
        lambda _loop, **kw: registered.append(kw["engine_factory"]),
    )
    return registered


def test_a_thread_rehosted_plan_loop_resumes_on_the_plan_executor(monkeypatch):
    # Fails on the pre-S3c tree: the re-host fell through to the phased engine.
    from umrs.api import loop as loop_api

    registered = _capture_engine_register(monkeypatch)
    monkeypatch.setattr(loop_api._loop_store, "save", lambda _loop: None)
    monkeypatch.setattr(loop_api, "_get_engine", lambda: pytest.fail("not the phased engine"))
    loop_api._rehost_on_thread(_plan_loop(hosting="thread"))
    assert isinstance(registered[0](), plan_iteration.PlanIterationEngine)


def test_rehost_without_the_extension_refuses(monkeypatch):
    from umrs.api import loop as loop_api

    _without_extensions(monkeypatch)
    registered = _capture_engine_register(monkeypatch)
    monkeypatch.setattr(loop_api._loop_store, "save", lambda _loop: None)
    with pytest.raises(loop_iteration_kinds.IterationKindUnavailable):
        loop_api._rehost_on_thread(_plan_loop(hosting="thread"))
    assert registered == []
