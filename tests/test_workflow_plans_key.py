"""D66-KEY (decisions 2026-11-24 §5 wave 1): the workflow-plan signing key
fails loudly instead of silently.

Without ``UMRS_SECRET_KEY`` plans are signed with a per-process random key,
so a plan signed by one worker process fails its commit on another. Now:

- ``UMRS_REQUIRE_PLAN_KEY=1`` without the key refuses server start, naming
  ``UMRS_SECRET_KEY``;
- ``WEB_CONCURRENCY`` > 1 refuses start with or without the key: the
  in-thread HTTP sidecar runs one uvicorn worker, and uvicorn would end that
  thread instead (2026-11-24 refix);
- one worker without the key keeps working (stand-alone), with a warning and
  readiness component ``plan_key`` degraded as ``plan_key_ephemeral``;
- with the key, a plan signed in one process verifies in another.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from umrs import workflow_plans
from umrs.workflow_plans import (
    PLAN_KEY_EPHEMERAL,
    enforce_plan_key_at_startup,
    plan_key_readiness,
    plan_key_refusal,
    sign_plan,
    verify_plan,
)

KEY = "d66-key-shared-secret-value"
SERVER = Path(__file__).resolve().parents[1] / "src" / "server_umrs.py"
FIELDS = {"project_id": "p1", "user_id": "u1", "workflow_name": "wf", "version": "1.0.0"}


def _child_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("UMRS_SECRET_KEY", "WEB_CONCURRENCY", "UMRS_REQUIRE_PLAN_KEY")
    }
    env.update(HOME=str(tmp_path), PYTHONPATH=os.pathsep.join(sys.path))
    env.update(extra)
    return env


def _run(code: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )


# ── refusal rule ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "env",
    [{"WEB_CONCURRENCY": "2"}, {"UMRS_REQUIRE_PLAN_KEY": "1"}, {"WEB_CONCURRENCY": "4", "UMRS_SECRET_KEY": "short"}],
)
def test_missing_key_refuses_under_workers_or_requirement(env):
    message = plan_key_refusal(env)
    assert message and "UMRS_SECRET_KEY" in message


@pytest.mark.parametrize(
    "env",
    [{}, {"WEB_CONCURRENCY": "1"}, {"UMRS_REQUIRE_PLAN_KEY": "1", "UMRS_SECRET_KEY": KEY}],
)
def test_single_worker_or_configured_key_is_allowed(env):
    assert plan_key_refusal(env) is None


@pytest.mark.parametrize("workers", ["2", "junk"])
def test_several_or_unreadable_workers_refuse_even_with_key(workers):
    """The sidecar's uvicorn.run(app object) exits its thread on workers > 1
    (and int() fails on junk): the key cannot make that serve plans."""
    message = plan_key_refusal({"WEB_CONCURRENCY": workers, "UMRS_SECRET_KEY": KEY})
    assert message and f"WEB_CONCURRENCY={workers}" in message
    assert "one uvicorn worker" in message and "UMRS_REQUIRE_PLAN_KEY=1" in message


async def _never_served_app(*_args) -> None:
    """An ASGI app object; uvicorn exits before it would ever be called."""


def test_uvicorn_refuses_workers_with_an_app_object(monkeypatch):
    """The fact the refusal rests on: uvicorn.run over an app object (what
    the sidecar passes) reads WEB_CONCURRENCY and exits instead of serving."""
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(SystemExit):
        uvicorn.run(_never_served_app, host="127.0.0.1", port=0)


def test_startup_refusal_is_a_system_exit_naming_the_key():
    with pytest.raises(SystemExit) as exc:
        enforce_plan_key_at_startup({"WEB_CONCURRENCY": "2"})
    assert "UMRS_SECRET_KEY" in str(exc.value.code)
    assert "WEB_CONCURRENCY=2" in str(exc.value.code)


def test_single_worker_without_key_starts_with_a_named_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="umrs.workflow_plans"):
        enforce_plan_key_at_startup({})
    assert PLAN_KEY_EPHEMERAL in caplog.text


def test_server_main_refuses_two_workers_without_key(tmp_path):
    """The acceptance row: workers=2 without key -> startup error naming
    UMRS_SECRET_KEY, before any configuration is read (old code exits on
    the missing configuration directory instead)."""
    env = _child_env(tmp_path, WEB_CONCURRENCY="2", UMRS_CONFIG=str(tmp_path / "absent"))
    code = "import runpy, sys; path = sys.argv[1]; sys.argv = [path]; runpy.run_path(path, run_name='__main__')"
    proc = _run(code, env, str(SERVER))
    assert proc.returncode != 0
    assert "UMRS_SECRET_KEY" in proc.stderr
    assert "WEB_CONCURRENCY=2" in proc.stderr


# ── readiness ─────────────────────────────────────────────────────────────


def test_readiness_component_degraded_without_key_never_shows_key():
    assert plan_key_readiness({})["reason"] == PLAN_KEY_EPHEMERAL
    assert plan_key_readiness({})["status"] == "degraded"
    ready = plan_key_readiness({"UMRS_SECRET_KEY": KEY})
    assert ready["status"] == "ok" and KEY not in json.dumps(ready)


def _readiness_body(monkeypatch, key: str | None) -> dict:
    from umrs.server import http_sidecar

    if key is None:
        monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("UMRS_SECRET_KEY", key)
    ok = ({"status": "ok"}, False)
    with (
        patch.object(http_sidecar, "_check_postgres", return_value=ok),
        patch.object(http_sidecar, "_check_job_store", return_value=ok),
        patch.object(http_sidecar, "_check_models", return_value={"status": "ok"}),
        patch.object(http_sidecar, "_check_durable_jobs_readiness", return_value=ok),
        patch.object(http_sidecar, "_check_registry_readiness", return_value={"status": "ok"}),
        patch.object(http_sidecar, "_check_backend", return_value={"status": "ok"}),
    ):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        return asyncio.run(http_sidecar._readiness(request))


def test_readiness_names_plan_key_ephemeral_but_stays_ready(monkeypatch):
    body = _readiness_body(monkeypatch, None)
    assert body["ready"] is True  # one worker still signs and verifies its own plans
    assert body["components"]["plan_key"]["status"] == "degraded"
    assert body["degraded"] == [PLAN_KEY_EPHEMERAL]


def test_readiness_with_key_has_no_degraded_reason(monkeypatch):
    body = _readiness_body(monkeypatch, KEY)
    assert body["components"]["plan_key"]["status"] == "ok"
    assert body["degraded"] == []


# ── cross-process verification ────────────────────────────────────────────

_SIGN = (
    "import json, sys; from umrs.workflow_plans import sign_plan; "
    "signed, digest, _ = sign_plan(json.loads(sys.argv[1])); "
    "print(json.dumps({'signed': signed, 'digest': digest}))"
)


def test_with_key_a_plan_signed_in_one_process_verifies_in_another(tmp_path, monkeypatch):
    proc = _run(_SIGN, _child_env(tmp_path, UMRS_SECRET_KEY=KEY), json.dumps(FIELDS))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout.strip().splitlines()[-1])
    monkeypatch.setenv("UMRS_SECRET_KEY", KEY)
    assert verify_plan(plan["signed"], plan["digest"])


def test_without_key_another_process_cannot_verify(tmp_path, monkeypatch):
    """Why the refusal exists: the per-process key does not cross processes."""
    proc = _run(_SIGN, _child_env(tmp_path), json.dumps(FIELDS))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout.strip().splitlines()[-1])
    monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    monkeypatch.setattr(workflow_plans, "_process_key", None)
    assert not verify_plan(plan["signed"], plan["digest"])
    signed, digest, _ = sign_plan(FIELDS)
    assert verify_plan(signed, digest)  # one process still verifies its own plans
