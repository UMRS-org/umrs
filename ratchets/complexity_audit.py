#!/usr/bin/env python3
"""Audit Python cyclomatic complexity without external dependencies.

The CI gate uses this for changed Python files, while full repository audits can
use it to produce a markdown or JSON backlog report. It intentionally treats
nested functions as independent functions instead of charging their branches to
the enclosing function.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "tmp",
    "venv",
}


@dataclass(frozen=True)
class ComplexityFinding:
    path: str
    line: int
    function: str
    complexity: int
    band: str


@dataclass(frozen=True)
class ParseError:
    path: str
    error: str


class ComplexityVisitor(ast.NodeVisitor):
    """Count branch points for one function body."""

    def __init__(self) -> None:
        self.score = 1

    def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.score += len(node.cases)
        self.generic_visit(node)


class FunctionCollector(ast.NodeVisitor):
    """Collect complexity findings from classes, functions, and methods."""

    def __init__(
        self,
        path: str,
        warn_at: int,
        critical_at: int,
        marker_lines: frozenset[int] = frozenset(),
    ) -> None:
        self.path = path
        self.warn_at = warn_at
        self.critical_at = critical_at
        self.marker_lines = marker_lines
        self.stack: list[str] = []
        self.findings: list[ComplexityFinding] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.lineno in self.marker_lines:
            self._visit_nested_definitions(node)
            return
        visitor = ComplexityVisitor()
        for child in node.body:
            visitor.visit(child)
        self.findings.append(
            ComplexityFinding(
                path=self.path,
                line=node.lineno,
                function=".".join([*self.stack, node.name]) if self.stack else node.name,
                complexity=visitor.score,
                band=band_for_score(visitor.score, self.warn_at, self.critical_at),
            )
        )
        self._visit_nested_definitions(node)

    @staticmethod
    def _noqa_marker_lines(source: str) -> set[int]:
        """Lines waiving C901 via `# noqa: C901` (project's complexity-exempt
        contract — marker-bearing functions are governed, not penalised).

        Honored here too: a whole-file changed-file audit must not re-flag
        debt the McCabe gate + the exemption validator already track
        (WP-13 round-2 promotion wave follow-up).
        """
        return {
            lineno for lineno, line in enumerate(source.splitlines(), 1) if "# noqa: C901" in line
        }

    def _visit_nested_definitions(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self.stack.pop()


def band_for_score(score: int, warn_at: int, critical_at: int) -> str:
    if score >= critical_at:
        return "critical"
    if score >= warn_at:
        return "warning"
    return "target"


def iter_python_files(paths: list[Path], root: Path) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        files.extend(expand_python_path(resolved, root))
    return sorted(set(files))


def expand_python_path(path: Path, root: Path) -> list[Path]:
    if should_skip(path, root):
        return []
    if path.is_file() and path.suffix == ".py":
        return [path]
    if path.is_dir():
        return [candidate for candidate in path.rglob("*.py") if not should_skip(candidate, root)]
    return []


def should_skip(path: Path, root: Path) -> bool:
    # Emacs interlock files (.#foo.py) are dead symlinks by design — never
    # source; parsing them only yields E902-flavoured noise (2026-09-30).
    if path.name.startswith(".#"):
        return True
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return bool(set(parts) & EXCLUDED_DIRS or "site-packages" in parts)


def audit_file(
    path: Path, root: Path, warn_at: int, critical_at: int
) -> tuple[list[ComplexityFinding], ParseError | None]:
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001 - report parse/read failures as audit data.
        return [], ParseError(path=rel, error=str(exc))
    markers = FunctionCollector._noqa_marker_lines(source)
    collector = FunctionCollector(rel, warn_at, critical_at, frozenset(markers))
    collector.visit(tree)
    return collector.findings, None


def audit_paths(
    paths: list[Path], root: Path, warn_at: int, critical_at: int
) -> tuple[list[ComplexityFinding], list[ParseError], int]:
    findings: list[ComplexityFinding] = []
    errors: list[ParseError] = []
    files = iter_python_files(paths, root)
    for path in files:
        file_findings, error = audit_file(path, root, warn_at, critical_at)
        findings.extend(file_findings)
        if error:
            errors.append(error)
    return findings, errors, len(files)


def render_markdown(
    findings: list[ComplexityFinding], errors: list[ParseError], scanned: int, max_complexity: int
) -> str:
    offenders = [finding for finding in findings if finding.complexity > max_complexity]
    lines = [
        "# Complexity Audit",
        "",
        f"- Python files scanned: {scanned}",
        f"- Functions over target ({max_complexity}): {len(offenders)}",
        f"- Hard warnings: {sum(1 for item in offenders if item.band == 'warning')}",
        f"- Critical: {sum(1 for item in offenders if item.band == 'critical')}",
        f"- Parse errors: {len(errors)}",
        "",
        "| Band | Complexity | File | Function |",
        "| --- | ---: | --- | --- |",
    ]
    for finding in sorted(offenders, key=lambda item: (-item.complexity, item.path, item.line)):
        lines.append(
            f"| {finding.band} | {finding.complexity} | `{finding.path}:{finding.line}` | `{finding.function}` |"
        )
    append_parse_errors(lines, errors)
    return "\n".join(lines) + "\n"


def append_parse_errors(lines: list[str], errors: list[ParseError]) -> None:
    if not errors:
        return
    lines.extend(["", "## Parse Errors", ""])
    for error in errors:
        lines.append(f"- `{error.path}`: {error.error}")


def render_json(
    findings: list[ComplexityFinding], errors: list[ParseError], scanned: int, max_complexity: int
) -> str:
    offenders = [finding for finding in findings if finding.complexity > max_complexity]
    payload = {
        "python_files_scanned": scanned,
        "max_complexity": max_complexity,
        "offenders": [
            asdict(finding)
            for finding in sorted(
                offenders, key=lambda item: (-item.complexity, item.path, item.line)
            )
        ],
        "parse_errors": [asdict(error) for error in errors],
    }
    return json.dumps(payload, indent=2) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-complexity", type=int, default=9)
    parser.add_argument("--warn-at", type=int, default=15)
    parser.add_argument("--critical-at", type=int, default=25)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    findings, errors, scanned = audit_paths(args.paths, root, args.warn_at, args.critical_at)
    rendered = render_report(args.format, findings, errors, scanned, args.max_complexity)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 1 if any(finding.complexity > args.max_complexity for finding in findings) else 0


def render_report(
    report_format: str,
    findings: list[ComplexityFinding],
    errors: list[ParseError],
    scanned: int,
    max_complexity: int,
) -> str:
    if report_format == "json":
        return render_json(findings, errors, scanned, max_complexity)
    return render_markdown(findings, errors, scanned, max_complexity)


if __name__ == "__main__":
    raise SystemExit(main())
