"""The .importlinter contracts keep r4's UMRS layering rules (U4-g, SI-01).

The `imports` CI job runs the contracts with import-linter; this test only
checks the configuration, so it needs neither import-linter nor a skip. The
lists below copy r4's ``UMRS_INTERNAL_BOUNDARY_RULES``
(tools/import_boundary_validator.py) as of 2026-11-23.
"""

from __future__ import annotations

import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".importlinter"
SRC = REPO_ROOT / "src"

UPPER_LAYERS = {"umrs.api", "umrs.server"}
FOUNDATION = (
    "umrs.model_registry",
    "umrs.mcp",
    "umrs.rag",
    "umrs.pipeline_sdk",
    "umrs.orchestrator",
    "umrs.jobs",
    "umrs.security",
    "umrs.shared",
    "umrs.inference",
    "umrs.skills",
    "umrs.recipes",
)
SATELLITES = (
    "umrs.knowledge_model",
    "umrs.knowledge_graph",
    "umrs.scientific_memory",
    "umrs.evidence",
)


def _config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    assert parser.read(CONFIG), f"missing {CONFIG}"
    return parser


def _lines(value: str) -> set[str]:
    return {line.strip() for line in value.splitlines() if line.strip()}


def _contracts() -> list[configparser.SectionProxy]:
    parser = _config()
    return [parser[name] for name in parser.sections() if name.startswith("importlinter:contract:")]


def _forbidden_for(module: str) -> set[str]:
    """Union of modules forbidden to *module* across the forbidden contracts."""
    forbidden: set[str] = set()
    for contract in _contracts():
        if module in _lines(contract.get("source_modules", "")):
            forbidden |= _lines(contract.get("forbidden_modules", ""))
    return forbidden


def _exists(module: str) -> bool:
    path = SRC.joinpath(*module.split("."))
    return path.is_dir() or path.with_suffix(".py").is_file()


def test_every_foundation_package_is_fenced_from_api_and_server() -> None:
    missing = [m for m in FOUNDATION if not _forbidden_for(m) >= UPPER_LAYERS]
    assert not missing, f"not forbidden umrs.api/umrs.server: {missing}"


def test_every_present_satellite_is_fenced_from_knowledge_too() -> None:
    wanted = UPPER_LAYERS | {"umrs.knowledge"}
    present = [m for m in SATELLITES if _exists(m)]
    missing = [m for m in present if not _forbidden_for(m) >= wanted]
    assert not missing, f"satellites missing a forbidden layer: {missing}"


def test_knowledge_is_fenced_from_api_and_server() -> None:
    assert _forbidden_for("umrs.knowledge") >= UPPER_LAYERS


def test_contracts_check_direct_imports_only_like_r4() -> None:
    # r4's AST validator flags direct imports and never type-only ones.
    assert _config()["importlinter"].getboolean("exclude_type_checking_imports")
    for contract in _contracts():
        assert contract.get("type") == "forbidden", contract.name
        assert contract.getboolean("allow_indirect_imports"), contract.name


def test_jobs_tolerates_no_lazy_upper_layer_import() -> None:
    # r4 has a strict jobs rule next to its allow_lazy one; the strict one wins.
    for contract in _contracts():
        ignored = _lines(contract.get("ignore_imports", ""))
        assert not [line for line in ignored if line.startswith("umrs.jobs")], contract.name


def test_security_namespace_portion_is_a_root() -> None:
    # src/umrs/security has no __init__.py: grimp skips it unless it is a root.
    if (SRC / "umrs" / "security" / "__init__.py").is_file():
        return
    assert "umrs.security" in _lines(_config()["importlinter"]["root_packages"])
