"""``GET /umrs/sites/me`` (multi-HPC plan S2, 2026-10-24).

The verified caller sees this site's non-secret capabilities, its declared
HPC access mode (``scheduler.hpc_access``, ADR-112; ``undeclared`` without
one), whether its service login is personal, and their **own** user row here
— never another user's, never the Slurm account name, never anything
personal in the open ``/health``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.admin.user_store import UserStore
from umrs.api import sites
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    load_site_config,
)

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"

_VERIFIED = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    users = UserStore(tmp_path / "user_files")
    users.create("alice", slurm_account="grp-a")
    users.create("bob", slurm_account="grp-b")
    users.create("carol")
    users.set_disabled("carol", True)
    monkeypatch.setattr(sites, "_user_store", lambda: users)
    return users


def _site(hpc_access: HpcAccessConfig | None = None) -> SiteConfig:
    site = SiteConfig(name="rccs-front")
    site.scheduler.kind = "slurm-remote"
    site.step_executor.remote_enabled = True
    site.step_executor.gpu = True
    site.scheduler.hpc_access = hpc_access
    return site


def _service_site(*, personal: bool, consent_ref: str = "") -> SiteConfig:
    service = ServiceAccountAccess(
        login="svc", personal=personal, consent_ref=consent_ref, allowed_projects=["p1"]
    )
    return _site(
        HpcAccessConfig(
            operator="R-CCS",
            modes_allowed=["service_account"],
            default_mode="service_account",
            service_account=service,
        )
    )


def _client(monkeypatch, site) -> TestClient:
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setattr(sites, "_load_site", lambda: site)
    app = FastAPI()
    app.include_router(sites.router)
    return TestClient(app)


def test_caller_sees_own_row_and_undeclared_mode(monkeypatch, store) -> None:
    body = _client(monkeypatch, _site()).get("/umrs/sites/me", headers=_VERIFIED).json()
    assert body["site"] == "rccs-front"
    assert body["user_id"] == "alice"
    assert body["access"] == {
        "mode": "undeclared",
        "declared": False,
        "modes_allowed": [],
        "operator": None,
        "personal": False,
        "consent": False,
    }
    assert body["account"] == {"status": "present", "slurm_account": True}
    assert body["capabilities"] == {
        "scheduler_kind": "slurm-remote",
        "staged_lane": True,
        "gpu": True,
    }


def test_only_the_callers_own_row_and_no_account_name(monkeypatch, store) -> None:
    text = _client(monkeypatch, _site()).get("/umrs/sites/me", headers=_VERIFIED).text
    assert "grp-a" not in text and "grp-b" not in text and "bob" not in text


def test_declared_mode_is_reported(monkeypatch, store) -> None:
    declared = HpcAccessConfig(
        operator="R-CCS",
        modes_allowed=["user_account"],
        default_mode="user_account",
        service_account=ServiceAccountAccess(personal=True, consent_ref="SECRET-DOC-ID"),
    )
    resp = _client(monkeypatch, _site(declared)).get("/umrs/sites/me", headers=_VERIFIED)
    assert resp.json()["access"] == {
        "mode": "user_account",
        "declared": True,
        "modes_allowed": ["user_account"],
        "operator": "R-CCS",
        "personal": False,
        "consent": False,
    }
    assert "SECRET-DOC-ID" not in resp.text


def test_service_account_site_reports_a_personal_login(monkeypatch, store) -> None:
    # ADR-112: the caller learns the login is a person's, never whose.
    resp = _client(monkeypatch, _service_site(personal=True)).get(
        "/umrs/sites/me", headers=_VERIFIED
    )
    access = resp.json()["access"]
    assert (access["mode"], access["personal"], access["consent"]) == (
        "service_account",
        True,
        False,
    )
    assert "svc" not in resp.text and "p1" not in resp.text


def test_consent_is_a_flag_never_the_reference(monkeypatch, store) -> None:
    site = _service_site(personal=False, consent_ref="LETTER-2026-17")
    resp = _client(monkeypatch, site).get("/umrs/sites/me", headers=_VERIFIED)
    access = resp.json()["access"]
    assert (access["personal"], access["consent"]) == (False, True)
    assert "LETTER-2026-17" not in resp.text


@pytest.mark.parametrize("front", ["rccs-front", "ai4s-front", "bdrt-front", "octopus-front"])
def test_every_front_reports_its_declared_personal_login(front) -> None:
    # The old loader dropped scheduler.hpc_access, so every front read undeclared.
    access = sites.access_of(load_site_config(str(DEPLOY / front / "site.yaml")))
    assert (access.mode, access.declared, access.personal, access.consent) == (
        "service_account",
        True,
        True,
        False,
    )


def test_default_mode_outside_allowed_is_undeclared(monkeypatch, store) -> None:
    bad = HpcAccessConfig(modes_allowed=["user_account"], default_mode="service_account")
    body = _client(monkeypatch, _site(bad)).get("/umrs/sites/me", headers=_VERIFIED).json()
    assert body["access"]["declared"] is False
    assert body["access"]["mode"] == "undeclared"


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("carol", {"status": "disabled", "slurm_account": False}),
        ("dave", {"status": "absent", "slurm_account": False}),
    ],
)
def test_disabled_and_absent_rows(monkeypatch, store, user, expected) -> None:
    headers = {"X-User-ID": user, "X-Verified-Project-ID": "proj-a"}
    body = _client(monkeypatch, _site()).get("/umrs/sites/me", headers=headers).json()
    assert body["account"] == expected


def test_unreadable_store_is_unknown_not_absent(monkeypatch) -> None:
    def _broken():
        raise OSError("disk gone")

    monkeypatch.setattr(sites, "_user_store", _broken)
    body = _client(monkeypatch, _site()).get("/umrs/sites/me", headers=_VERIFIED).json()
    assert body["account"]["status"] == "unknown"


def test_no_site_config_still_answers(monkeypatch, store) -> None:
    body = _client(monkeypatch, None).get("/umrs/sites/me", headers=_VERIFIED).json()
    assert body["site"] is None
    assert body["access"]["mode"] == "undeclared"
    assert body["capabilities"] == {"scheduler_kind": None, "staged_lane": False, "gpu": False}


def test_anonymous_caller_is_refused(monkeypatch, store) -> None:
    resp = _client(monkeypatch, _site()).get("/umrs/sites/me")
    assert resp.status_code == 401


def test_staged_lane_off_means_no_gpu_lane() -> None:
    site = _site()
    site.step_executor.remote_enabled = False
    caps = sites.capabilities_of(site)
    assert (caps.staged_lane, caps.gpu) == (False, False)


def test_access_of_reads_an_object_block() -> None:
    block = SimpleNamespace(modes_allowed=["service_account"], default_mode="service_account")
    access = sites.access_of(_site(block))
    assert (access.mode, access.declared) == ("service_account", True)


def test_route_is_mounted_and_health_carries_nothing_personal(monkeypatch, store) -> None:
    import inspect

    from umrs.server import http_sidecar

    # Static: building the whole sidecar app here starts process singletons
    # that later workflow tests trip over, so read the mount instead.
    mounts = inspect.getsource(http_sidecar._create_app)
    assert "from umrs.api.sites import router as sites_router" in mounts
    assert "app.include_router(sites_router)" in mounts
    caps = http_sidecar._site_capabilities()
    assert not {"account", "access", "consent_ref", "user_id"} & set(caps)
