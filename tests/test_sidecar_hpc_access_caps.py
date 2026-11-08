"""``/health`` ``site_capabilities`` advertise the declared HPC access mode.

ADR-112 A.3.7: the shared-key delegation keys (``slurm_delegation``,
``service_pubkey_path``, ``slurm_proxy_path``) are no longer advertised, so no
client enrols the operator's key in a user's account; ``hpc_access`` carries
``{default_mode, modes_allowed, personal}`` (empty when undeclared) and
never the login, the consent reference or the project list.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.orchestrator.site_config import HpcAccessConfig, SchedulerConfig, ServiceAccountAccess
from umrs.server import http_sidecar

_RETIRED = {"slurm_delegation", "service_pubkey_path", "slurm_proxy_path"}


def _caps(monkeypatch, scheduler: SchedulerConfig) -> dict:
    site = SimpleNamespace(name="cluster-a", scheduler=scheduler)
    monkeypatch.setattr("umrs.orchestrator.site_config.load_site_config", lambda: site)
    return http_sidecar._site_capabilities()


def test_a_declared_site_advertises_its_mode_only(monkeypatch):
    scheduler = SchedulerConfig(
        kind="slurm-remote",
        slurm_delegation="per_user_ssh",  # deprecated key still in a file
        service_pubkey_path="/keys/umrs-service.pub",
        slurm_proxy_path="/bin/umrs-slurm-proxy",
        hpc_access=HpcAccessConfig(
            modes_allowed=["service_account"],
            default_mode="service_account",
            service_account=ServiceAccountAccess(
                login="operator",
                personal=True,
                consent_ref="doc-1",
                allowed_projects=["p"],
            ),
        ),
    )
    caps = _caps(monkeypatch, scheduler)
    assert not _RETIRED & set(caps)
    assert caps["hpc_access"] == {
        "default_mode": "service_account",
        "modes_allowed": ["service_account"],
        "personal": True,
    }
    assert caps["scheduler_kind"] == "slurm-remote" and caps["site_name"] == "cluster-a"
    flat = repr(caps)
    assert "operator" not in flat and "doc-1" not in flat and "'p'" not in flat


@pytest.mark.parametrize("delegation", ["service_account", "per_user_ssh"])
def test_an_undeclared_site_advertises_an_empty_block(monkeypatch, delegation):
    caps = _caps(monkeypatch, SchedulerConfig(kind="slurm-remote", slurm_delegation=delegation))
    assert caps["hpc_access"] == {}
    assert not _RETIRED & set(caps)
