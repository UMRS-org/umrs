"""Core never imports the r4 extension or r4-mcp (ADR-117 §6 enforcement).

U3 S2 moved ``knowledge/verifiers/mcp_adapter.py``, core's only ``r4_mcp``
user, into the umrs-r4 extension (``plugins/umrs-r4``). This keeps core free of
``r4_mcp`` and ``umrs_r4`` imports, lazy ones included.
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "umrs"
FORBIDDEN = frozenset({"r4_mcp", "umrs_r4"})


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_never_imports_the_r4_extension_or_r4_mcp():
    offenders = []
    for path in sorted(CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _imported_roots(tree) & FORBIDDEN
        if hits:
            offenders.append(f"{path.relative_to(CORE.parent)}: {sorted(hits)}")
    assert not offenders, "core imports the r4 extension:\n" + "\n".join(offenders)


def test_fence_sees_lazy_imports():
    tree = ast.parse("def f():\n    from r4_mcp.security import X\n    import umrs_r4\n")
    assert _imported_roots(tree) >= FORBIDDEN
