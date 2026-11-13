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

The modules ADR-117 U3a D9 leaves in r4-contracts until U3 S5 (``loop_plan``,
``team_engine``, ``taint``) carry ``# floor: U3-S5``; nothing else may.
UMRS-org CI replaces this test with import-linter contracts in U4.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests")
ALLOWLIST = UMRS_ROOT / "tests" / "r4_contracts_import_allowlist.txt"
PACKAGE = "r4_contracts"
FLOOR_MARK = "# floor: U3-S5"
FLOOR_MODULES = ("loop_plan", "team_engine", "taint")
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


def read_allowlist(path: Path = ALLOWLIST) -> list[tuple[str, bool]]:
    """``(entry, floor_marked)`` per non-comment line, in file order."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry, _, comment = line.partition("#")
        rows.append((entry.strip(), FLOOR_MARK in "#" + comment))
    return rows


def _is_floor_target(entry: str) -> bool:
    module = entry.partition("::")[2].split(".")
    return len(module) > 1 and module[1] in FLOOR_MODULES


# ── the fence ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scanned() -> set[str]:
    return scan_entries()


@pytest.fixture(scope="module")
def listed() -> list[tuple[str, bool]]:
    return read_allowlist()


def test_no_unlisted_r4_contracts_import(scanned: set[str], listed: list[tuple[str, bool]]) -> None:
    new = sorted(scanned - {entry for entry, _ in listed})
    assert not new, (
        "new r4_contracts imports (import umrs_protocol for a moved module; the "
        f"allowlist only shrinks, ADR-117 D-2): {new}"
    )


def test_no_stale_allowlist_entry(scanned: set[str], listed: list[tuple[str, bool]]) -> None:
    stale = sorted({entry for entry, _ in listed} - scanned)
    assert not stale, f"allowlist rows no longer imported; delete them: {stale}"


def test_allowlist_rows_unique(listed: list[tuple[str, bool]]) -> None:
    entries = [entry for entry, _ in listed]
    assert len(entries) == len(set(entries))
    assert all("::" in entry for entry in entries)


def test_floor_marks_only_on_u3_s5_modules(listed: list[tuple[str, bool]]) -> None:
    wrong = [entry for entry, marked in listed if marked != _is_floor_target(entry)]
    assert not wrong, f"'{FLOOR_MARK}' belongs on loop_plan/team_engine/taint rows only: {wrong}"


def test_scan_finds_the_known_consumers(scanned: set[str]) -> None:
    """Sanity: the scanner sees both eager and function-local imports."""
    assert "src/umrs/knowledge/plan_iteration.py::r4_contracts.loop_plan" in scanned
    assert "src/umrs/api/hpc_accounts.py::r4_contracts.hpc_gate" in scanned


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
        ("src/a.py::r4_contracts.errors", False),
        ("src/b.py::r4_contracts.loop_plan", True),
    ]
    assert _is_floor_target("src/b.py::r4_contracts.team_engine.services")
    assert not _is_floor_target("src/a.py::r4_contracts.errors")
