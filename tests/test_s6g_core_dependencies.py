"""U3 S6g: core ``umrs`` no longer depends on r4-contracts.

Core imports no ``r4_contracts`` since U3 S5 (the import fence pins it); the
umrs-r4 extension, which still uses ``loop_plan`` and ``team_engine``, carries
the dependency instead (ADR-117 U3a D9). The lock follows the declaration.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 runners
    import tomli as tomllib

UMRS = Path(__file__).resolve().parents[1]


def _deps(pyproject: Path) -> list[str]:
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]


def _names(deps: list[str]) -> set[str]:
    return {d.split(">")[0].split("=")[0].split("<")[0].strip() for d in deps}


def test_core_declares_no_r4_contracts():
    assert "r4-contracts" not in _names(_deps(UMRS / "src" / "pyproject.toml"))


def test_the_extension_declares_it():
    assert "r4-contracts" in _names(_deps(UMRS / "packages" / "umrs-r4" / "pyproject.toml"))


def test_the_umrs_lock_has_no_r4_contracts():
    assert 'name = "r4-contracts"' not in (UMRS / "src" / "uv.lock").read_text(encoding="utf-8")
