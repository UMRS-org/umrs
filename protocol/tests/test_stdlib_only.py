"""umrs-protocol imports the standard library and itself, nothing else (ADR-117 D-3).

An AST walk of every module, function-local imports included: each absolute
import root must be a stdlib module or ``umrs_protocol``, and a relative
import must stay inside the package. This is the umrs-side twin of the
superproject's import-boundary rule; UMRS-org CI replaces it with
import-linter contracts in U4.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = "umrs_protocol"
SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE
MODULES = sorted(SRC.rglob("*.py"))
ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | {PACKAGE}


def _package_depth(path: Path) -> int:
    """How many parent levels a relative import may climb and stay inside."""
    return len(path.relative_to(SRC).parts)


def _import_roots(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def violations(source: str, path: Path) -> list[str]:
    """Every import in ``source`` that leaves the stdlib or the package."""
    found = []
    depth = _package_depth(path)
    for node in ast.walk(ast.parse(source, filename=str(path))):
        found += [
            f"{node.lineno}: import {root}"
            for root in _import_roots(node)
            if root not in ALLOWED_ROOTS
        ]
        if isinstance(node, ast.ImportFrom) and node.level > depth:
            found.append(f"{node.lineno}: relative import escapes {PACKAGE} (level {node.level})")
    return found


def test_package_has_the_closure() -> None:
    names = {p.stem for p in MODULES}
    assert len(MODULES) == 17, names  # __init__ + the 16 moved modules
    assert {"__init__", "hpc_gate", "handshake_hmac", "protocol", "errors"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_is_stdlib_only(path: Path) -> None:
    assert violations(path.read_text(encoding="utf-8"), path) == []


def test_fence_catches_the_pre_p0_research_import() -> None:
    """Pre-P0 research.py imported r4_contracts.provenance absolutely."""
    path = SRC / "research.py"
    source = path.read_text(encoding="utf-8")
    pre_p0 = source.replace(
        "from .provenance import ArtifactRef", "from r4_contracts.provenance import ArtifactRef", 1
    )
    assert pre_p0 != source
    assert any("import r4_contracts" in v for v in violations(pre_p0, path))


@pytest.mark.parametrize(
    "line",
    [
        "import httpx",
        "from pydantic import BaseModel",
        "def f():\n    import yaml\n",
        "from .. import r4_contracts",
        "import umrs.api",
    ],
)
def test_fence_catches_non_stdlib(line: str) -> None:
    assert violations(line, SRC / "errors.py")


@pytest.mark.parametrize(
    "line", ["import os", "from .provenance import ArtifactRef", "from umrs_protocol import errors"]
)
def test_fence_allows_stdlib_and_self(line: str) -> None:
    assert violations(line, SRC / "errors.py") == []
