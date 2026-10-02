"""Every JobStore is claim-scoped to its site, not only the scheduler's.

R3 2026-10-02: the four umrs-front servers share ``umrs_db``. Jobs created
through the ZMQ durable backend / HTTP sidecar store (no ``site_name``) were
written with ``venue=''``, so the ai4s and octopus schedulers claimed rccs
workflow runs and refused them (no HPC lane) — and failed the rccs attempt
that did run. The creating store and the claiming store must agree.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from umrs.jobs.store import JobStore, configured_site_name

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


@pytest.fixture(autouse=True)
def _no_schema(monkeypatch):
    monkeypatch.setattr(JobStore, "_ensure_schema", lambda _self: None)


@pytest.fixture
def no_site(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr(
        "umrs.orchestrator.site_config._default_site_config_path", lambda: ""
    )


def test_store_takes_the_site_name(monkeypatch):
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(DEPLOY / "rccs-front" / "site.yaml"))
    assert JobStore(MagicMock(), "i").site_name == "rccs-front"


@pytest.mark.parametrize("front", ["ai4s-front", "octopus-front", "bdrt-front"])
def test_other_fronts_never_share_rccs_name(monkeypatch, front):
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(DEPLOY / front / "site.yaml"))
    assert configured_site_name() not in ("", "rccs-front")


def test_no_site_config_keeps_the_shared_lane(no_site):
    # Never the env-only "env-fallback" name: a scheduler without site.yaml
    # claims venue '' only, so a store must not tag rows it would never claim.
    assert configured_site_name() == ""
    assert JobStore(MagicMock(), "i").site_name == ""


def test_creation_paths_build_plain_stores():
    # The backends that create jobs construct JobStore(pool) and rely on the
    # constructor for scoping — no caller-side site_name assignment.
    root = Path(__file__).resolve().parents[1] / "src" / "umrs"
    for rel in (
        "server/job_tracking_handler.py",
        "server/http_sidecar.py",
        "api/vision_durable.py",
    ):
        assert "JobStore(" in (root / rel).read_text(), rel
