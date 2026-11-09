"""Every shipped site file loads (ADR-112 wave A review, 2026-11-08).

`scheduler.hpc_access` is validated fail-closed at load: a malformed block
raises, and `load_site_config` failing takes the platform lanes of that front
down with it. Catch that in CI, before a converge does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.orchestrator.site_config import HPC_ACCESS_MODES, load_site_config

_DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
_SITE_FILES = sorted(
    p for p in _DEPLOY.rglob("site*.yaml") if "node_modules" not in p.parts
)


def test_the_glob_finds_the_fronts():
    names = {p.parent.name for p in _SITE_FILES}
    assert {"rccs-front", "ai4s-front", "bdrt-front", "octopus-front"} <= names


@pytest.mark.parametrize("path", _SITE_FILES, ids=lambda p: str(p.relative_to(_DEPLOY)))
def test_every_site_file_loads(path):
    site = load_site_config(str(path))
    access = site.scheduler.hpc_access
    if access is not None:
        assert set(access.modes_allowed) <= set(HPC_ACCESS_MODES)
        assert access.default_mode in access.modes_allowed
