"""ADR-081 D-4 CI insert fence (KN-S7): no untyped raw knowledge INSERT.

Every node and edge write goes through one choke point: the store
(``umrs/knowledge/store.py`` and its typed half ``typed_write.py``) or a
raw-SQL minting helper that names the typed column. This fence parses every
Python source under ``src/`` and ``packages/*/src`` and fails on any string
literal (f-string parts included) that inserts into ``knowledge_nodes``
without naming ``node_class``, or into ``knowledge_edges`` without naming
``relation_class``, outside ``store.py``.

``store.py`` is the choke point itself: its untyped statements are the
legacy path typed writes fall back to, so it is the one exemption.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

UMRS_ROOT = Path(__file__).resolve().parents[1]
EXEMPT = {Path("src/umrs/knowledge/store.py")}
_INSERT = re.compile(r"insert\s+into\s+(knowledge_nodes|knowledge_edges)\b", re.IGNORECASE)
_REQUIRED = {"knowledge_nodes": "node_class", "knowledge_edges": "relation_class"}


def _literal_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return None


def _untyped(text: str) -> list[str]:
    return [
        m.group(1)
        for m in _INSERT.finditer(text)
        if _REQUIRED[m.group(1).lower()] not in text
    ]


def untyped_inserts(source: str) -> list[tuple[int, str]]:
    """(line, table) for each untyped knowledge INSERT literal in ``source``."""
    tree = ast.parse(source)
    parts = {id(p) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for p in n.values}
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        text = None if id(node) in parts else _literal_text(node)
        if text:
            found.extend((getattr(node, "lineno", 0), table) for table in _untyped(text))
    return found


def _sources() -> list[Path]:
    roots = [UMRS_ROOT / "src", *sorted((UMRS_ROOT / "packages").glob("*/src"))]
    return sorted(p for root in roots if root.is_dir() for p in root.rglob("*.py"))


def test_fence_flags_an_untyped_node_insert():
    source = 'SQL = """INSERT INTO knowledge_nodes (node_id, label) VALUES ($1, $2)"""\n'
    assert untyped_inserts(source) == [(1, "knowledge_nodes")]


def test_fence_flags_an_untyped_edge_insert_in_an_fstring():
    source = 'x = 1\nSQL = f"insert  into knowledge_edges (edge_id) VALUES ({x})"\n'
    assert untyped_inserts(source) == [(2, "knowledge_edges")]


def test_fence_passes_typed_inserts():
    source = (
        'A = "INSERT INTO knowledge_nodes (node_id, node_class) VALUES ($1, $2)"\n'
        'B = f"INSERT INTO knowledge_edges (edge_id, relation_class) VALUES ({1})"\n'
        'C = "SELECT * FROM knowledge_nodes"\n'
    )
    assert untyped_inserts(source) == []


def test_no_untyped_knowledge_insert_outside_the_store():
    offenders = []
    for path in _sources():
        rel = path.relative_to(UMRS_ROOT)
        if rel in EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        if not _INSERT.search(text):
            continue
        offenders += [f"{rel}:{line} {table}" for line, table in untyped_inserts(text)]
    assert offenders == [], (
        "raw knowledge INSERT without node_class/relation_class (ADR-081 D-4: write "
        "through KnowledgeStore, or name the typed column via typing.prepare_row): "
        + ", ".join(offenders)
    )
