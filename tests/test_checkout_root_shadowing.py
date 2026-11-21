"""No top-level directory of the umrs checkout shadows a ``umrs`` subpackage.

CI can import ``umrs`` as a namespace package spanning the checkout root (see
test_approval_inbox: ``umrs.__file__`` is None there), so ``<root>/plugins``
made ``umrs.plugins`` resolve to it and hid core ``src/umrs/plugins`` (U3 S2,
2026-11-21: three suites failed to import). The extension lives in
``packages/`` for that reason.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "umrs"


def _subpackages() -> set[str]:
    return {p.name for p in CORE.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}


def _root_dirs() -> set[str]:
    return {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")}


def test_no_checkout_root_directory_shadows_a_core_subpackage():
    clash = sorted(_root_dirs() & _subpackages())
    assert not clash, f"top-level dirs shadow umrs subpackages: {clash}"
