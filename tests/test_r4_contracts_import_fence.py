"""Every ``r4_contracts`` import in umrs is listed, and the list only shrinks.

ADR-117 D-2 moves the UMRS-owned contract modules into ``umrs_protocol``
(``protocol/``); U2.1 onward switches umrs to it one module family per PR
(U2.0 spec §10). This fence keeps the migration one-way:

* an AST scan of ``src/`` and ``tests/``, function-local imports included,
  plus string targets of ``importorskip``/``import_module``/``__import__``;
* each hit is ``<relpath>::<target>``: ``r4_contracts.x`` for
  ``import r4_contracts.x``/``from r4_contracts.x import …``, and
  ``r4_contracts.<name>`` for each ``from r4_contracts import <name>``;
* the fence fails on a hit missing from ``r4_contracts_import_allowlist.txt``
  (a new import: use ``umrs_protocol`` once its family moved) and on a listed
  entry no longer found (stale: delete the row in the same commit).

UMRS-org CI replaces this test with import-linter contracts in U4.

2026-11-24 (U2 exit): the allowlist is empty and the fence requires it; the
last consumer (test_plan_loop_server.py) moved to packages/umrs-r4/tests,
where r4_contracts is a declared dependency (D9). The ``# floor: U3-S5`` row
marks (``loop_plan``, ``team_engine``, ``taint``) and their check went with the
last floor row: an empty list has nothing to mark.

Out of scope, on purpose: ``packages/`` (``umrs-r4`` declares r4-contracts)
and ``docker/server-image/smoke.py``, whose ``check_protocol_alias`` imports
``r4_contracts.hpc_gate`` to prove the r4 server image's alias IS the
``umrs_protocol`` module; that image bundles r4-contracts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests")
ALLOWLIST = UMRS_ROOT / "tests" / "r4_contracts_import_allowlist.txt"
PACKAGE = "r4_contracts"
_DYNAMIC_IMPORTERS = frozenset({"importorskip", "import_module", "__import__"})


def _is_target(name: str | None) -> bool:
    return bool(name) and (name == PACKAGE or str(name).startswith(PACKAGE + "."))


def _static_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if _is_target(alias.name)]
    if not isinstance(node, ast.ImportFrom) or node.level or not _is_target(node.module):
        return []
    if node.module == PACKAGE:
        return [f"{PACKAGE}.{alias.name}" for alias in node.names]
    return [str(node.module)]


def _dynamic_targets(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.Call) or not node.args:
        return []
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    first = node.args[0]
    if name not in _DYNAMIC_IMPORTERS or not isinstance(first, ast.Constant):
        return []
    return [first.value] if isinstance(first.value, str) and _is_target(first.value) else []


def targets_in(source: str, filename: str = "<string>") -> set[str]:
    """The ``r4_contracts`` modules or names a source file imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        found.update(_static_targets(node))
        found.update(_dynamic_targets(node))
    return found


def scan_entries(root: Path = UMRS_ROOT) -> set[str]:
    """``<relpath>::<target>`` for every hit under ``src/`` and ``tests/``."""
    entries: set[str] = set()
    for top in SCAN_DIRS:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            entries.update(f"{rel}::{target}" for target in targets_in(text, rel))
    return entries


def read_allowlist(path: Path = ALLOWLIST) -> list[str]:
    """The entry of each non-comment line (trailing ``#`` comment dropped), in file order."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append(line.partition("#")[0].strip())
    return rows


# ── the fence ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scanned() -> set[str]:
    return scan_entries()


@pytest.fixture(scope="module")
def listed() -> list[str]:
    return read_allowlist()


def test_no_unlisted_r4_contracts_import(scanned: set[str], listed: list[str]) -> None:
    new = sorted(scanned - set(listed))
    assert not new, (
        "new r4_contracts imports (import umrs_protocol for a moved module; the "
        f"allowlist only shrinks, ADR-117 D-2): {new}"
    )


def test_no_stale_allowlist_entry(scanned: set[str], listed: list[str]) -> None:
    stale = sorted(set(listed) - scanned)
    assert not stale, f"allowlist rows no longer imported; delete them: {stale}"


def test_allowlist_rows_unique(listed: list[str]) -> None:
    assert len(listed) == len(set(listed))
    assert all("::" in entry for entry in listed)


def test_core_src_imports_no_r4_contracts(scanned: set[str]) -> None:
    """U3 S5: the plan-run cluster (the last floor users) left for umrs_r4."""
    assert not sorted(e for e in scanned if e.startswith("src/"))


def test_protocol_and_handshake_hmac_have_moved(scanned: set[str]) -> None:
    """U2.x's first family uses UMRS-owned modules directly, never the aliases."""
    family = {"r4_contracts.protocol", "r4_contracts.handshake_hmac"}
    leftovers = sorted(entry for entry in scanned if entry.partition("::")[2] in family)
    assert not leftovers, f"protocol/HMAC imports still use r4_contracts: {leftovers}"


def test_u2_exit_the_allowlist_is_empty(scanned: set[str], listed: list[str]) -> None:
    """U2 exit (ADR-117 plan U2): no r4_contracts import is left in core src or
    tests, floors included; the last consumer moved to packages/umrs-r4/tests."""
    assert not listed, f"the allowlist must stay empty after U2: {listed}"
    assert not sorted(scanned), f"r4_contracts imports in core: {sorted(scanned)}"


# ── the scanner ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import r4_contracts", {"r4_contracts"}),
        ("import r4_contracts.hpc_gate as g", {"r4_contracts.hpc_gate"}),
        ("from r4_contracts.errors import PlatformError", {"r4_contracts.errors"}),
        (
            "from r4_contracts import hpc_gate, errors",
            {"r4_contracts.hpc_gate", "r4_contracts.errors"},
        ),
        ("def f():\n    from r4_contracts import loop_plan\n", {"r4_contracts.loop_plan"}),
        ("g = pytest.importorskip('r4_contracts.hpc_gate')", {"r4_contracts.hpc_gate"}),
        ("m = importlib.import_module('r4_contracts.models')", {"r4_contracts.models"}),
        ("from . import r4_contracts", set()),
        ("import r4_contracts_extra", set()),
        ("x = 'r4_contracts.hpc_gate'", set()),
        ("from umrs_protocol import hpc_gate", set()),
    ],
)
def test_targets_in(source: str, expected: set[str]) -> None:
    assert targets_in(source) == expected


def test_new_import_fails_the_fence(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def f():\n    from r4_contracts import errors\n")
    (tmp_path / "tests" / "test_x.py").write_text("import os\n")
    assert scan_entries(tmp_path) == {"src/mod.py::r4_contracts.errors"}


def test_allowlist_parser(tmp_path: Path) -> None:
    path = tmp_path / "allow.txt"
    path.write_text(
        "# header\n\nsrc/a.py::r4_contracts.errors\nsrc/b.py::r4_contracts.loop_plan  # floor: U3-S5\n"
    )
    assert read_allowlist(path) == [
        "src/a.py::r4_contracts.errors",
        "src/b.py::r4_contracts.loop_plan",
    ]
