"""Lint engine — server-side static analysis for knowledge projects.

Run lightweight lint checks without requiring ruff. Produces structured
findings usable by the TUI, CLI, and web UI.

Stdlib-only: uses ``ast`` module for Python parsing.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

logger = __import__("logging").getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_BLOCK_PATTERNS: list[str] = [
    r"(?i)(password|passwd|pwd)\s*[:=]",
    r"(?i)(secret|secret_key)\s*[:=]",
    r"(?i)(api_key|apikey)\s*[:=]",
    r"(?i)(token|auth_token|access_token)\s*[:=]",
    r"(?i)(private_key)\s*[:=]",
    r"(?i)(aws_secret_access_key)\s*[:=]",
]

# Import boundary rules: forbidden module → allowed_parent modules
# Pattern: "forbidden_module" not in any "allowed_parents"
_IMPORT_BOUNDARIES: list[tuple[str, list[str]]] = [
    ("umrs.api", ["umrs.api"]),
    ("umrs.server", ["umrs.server"]),
    ("r4", ["r4", "r4_contracts", "r4.client"]),
]

# Cyclomatic complexity warning threshold (matches ruff C901 max)
COMPLEXITY_THRESHOLD: int = 9


# ── Finding data model ─────────────────────────────────────────────────────────


@dataclass
class LintFinding:
    """Single lint finding."""

    file: str
    line: int
    column: int
    severity: str  # BLOCKING | WARNING | INFO
    rule: str  # rule identifier
    message: str  # human-readable description
    recommendation: str  # suggested fix

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "recommendation": self.recommendation,
        }


# ── Rule: syntax error ────────────────────────────────────────────────────────


def check_syntax(source: str, file_path: str = "") -> list[LintFinding]:
    """Check Python source for syntax errors using ast.parse."""
    findings: list[LintFinding] = []
    try:
        ast.parse(source)
    except SyntaxError as exc:
        findings.append(LintFinding(
            file=file_path,
            line=exc.lineno or 1,
            column=exc.offset or 0,
            severity="BLOCKING",
            rule="E999_syntax_error",
            message=f"Syntax error: {exc.msg}",
            recommendation="Fix the syntax error at the indicated location.",
        ))
    return findings


# ── Rule: cyclomatic complexity ───────────────────────────────────────────────


def _walk_complexity(node: ast.AST) -> int:
    """Compute cyclomatic complexity of a function/method node."""
    complexity = 1  # base
    for child in ast.walk(node):
        if isinstance(child, (
            ast.If, ast.While, ast.For, ast.ExceptHandler,
            ast.Assert, ast.With, ast.comprehension,
            ast.IfExp, ast.BoolOp,
        )):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each additional boolean operator adds one path
            complexity += len(child.values) - 1
    return complexity


def check_complexity(source: str, file_path: str = "") -> list[LintFinding]:
    """Check function cyclomatic complexity (C901).

    Reports findings for any function/method exceeding COMPLEXITY_THRESHOLD.
    """
    findings: list[LintFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings  # already reported by syntax check

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c = _walk_complexity(node)
            if c > COMPLEXITY_THRESHOLD:
                findings.append(LintFinding(
                    file=file_path,
                    line=node.lineno or 0,
                    column=node.col_offset or 0,
                    severity="BLOCKING",
                    rule="C901_complexity",
                    message=f"{node.name}: complexity {c} exceeds threshold {COMPLEXITY_THRESHOLD}",
                    recommendation=f"Refactor {node.name} to reduce complexity below {COMPLEXITY_THRESHOLD}. Extract helper functions, reduce nesting, or simplify conditionals.",
                ))
    return findings


# ── Rule: unused variables ────────────────────────────────────────────────────


# ── Rule: unused variables ────────────────────────────────────────────────────

def _collect_assigned(tree: ast.AST) -> dict[str, int]:
    """Collect all assigned variable names and their line numbers."""
    assigned: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = target.lineno or 0
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                assigned[node.target.id] = node.target.lineno or 0
        elif isinstance(node, ast.AnnAssign):
            if node.target and isinstance(node.target, ast.Name):
                assigned[node.target.id] = node.target.lineno or 0
    return assigned


def _collect_loaded(tree: ast.AST) -> set[str]:
    """Collect all variable names that are read (Load context)."""
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return loaded


def check_unused_variables(source: str, file_path: str = "") -> list[LintFinding]:
    """Detect assigned but never-read local variables (F841 equivalent)."""
    findings: list[LintFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    assigned = _collect_assigned(tree)
    loaded = _collect_loaded(tree)
    for var_name, line in assigned.items():
        if var_name not in loaded and not var_name.startswith("_"):
            findings.append(LintFinding(
                file=file_path, line=line, column=0,
                severity="WARNING", rule="F841_unused_variable",
                message=f"Variable {var_name!r} assigned but never used",
                recommendation=f"Remove unused variable {var_name!r} or prefix with underscore.",
            ))
    return findings


# ── Rule: import boundary ─────────────────────────────────────────────────────


def _check_import_forbidden(
    alias_name: str,
    file_path: str,
    module_name: str,
    findings: list[LintFinding],
) -> None:
    """Check one import name against all boundary rules."""
    for forbidden, allowed in _IMPORT_BOUNDARIES:
        if alias_name == forbidden or alias_name.startswith(forbidden + "."):
            if module_name not in allowed:
                findings.append(LintFinding(
                    file=file_path, line=0, column=0,
                    severity="BLOCKING", rule="IMPORT_BOUNDARY",
                    message=f"Module {module_name or file_path} imports forbidden module {alias_name}",
                    recommendation=f"Remove import of {alias_name}. {forbidden} is not accessible from {module_name or file_path}.",
                ))


def check_import_boundary(source: str, file_path: str = "", module_name: str = "") -> list[LintFinding]:
    """Detect forbidden imports based on architectural boundary rules."""
    findings: list[LintFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    _check_imports_in_tree(tree, file_path, module_name, findings)
    return findings


def _check_imports_in_tree(
    tree: ast.AST, file_path: str, module_name: str, findings: list[LintFinding],
) -> None:
    """Walk the AST and check all imports against boundary rules."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import_forbidden(alias.name, file_path, module_name, findings)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _check_import_forbidden(node.module, file_path, module_name, findings)


# ── Rule: no docstring ────────────────────────────────────────────────────────


def check_docstrings(source: str, file_path: str = "") -> list[LintFinding]:
    """Flag public functions/methods missing docstrings (informational).

    Public = names not starting with underscore.
    """
    findings: list[LintFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                docstring = ast.get_docstring(node)
                if not docstring:
                    findings.append(LintFinding(
                        file=file_path,
                        line=node.lineno or 0,
                        column=node.col_offset or 0,
                        severity="INFO",
                        rule="NO_DOCSTRING",
                        message=f"{node.name} missing docstring",
                        recommendation=f"Add a docstring explaining the purpose of {node.name}.",
                    ))

    return findings


# ── Rule: duplicate imports ──────────────────────────────────────────────────


def check_duplicate_imports(source: str, file_path: str = "") -> list[LintFinding]:
    """Detect same module imported multiple times."""
    findings: list[LintFinding] = []
    imports: dict[str, int] = {}
    import_lines: dict[str, int] = {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = imports.get(name, 0) + 1
                import_lines[name] = node.lineno or 0
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = imports.get(name, 0) + 1
                    import_lines[name] = node.lineno or 0

    for name, count in imports.items():
        if count > 1:
            line = import_lines[name]
            findings.append(LintFinding(
                file=file_path,
                line=line,
                column=0,
                severity="WARNING",
                rule="DUPLICATE_IMPORT",
                message=f"Module imported {count} times",
                recommendation=f"Consolidate duplicate imports of {name} into a single import statement.",
            ))

    return findings


# ── Rule: content scanning ────────────────────────────────────────────────────


def scan_content(source: str, block_patterns: Optional[list[str]] = None) -> list[LintFinding]:
    """Scan source content for PII/credential patterns (informational).

    Same patterns as egress_fetch.py default block patterns.
    """
    findings: list[LintFinding] = []
    patterns = list(block_patterns) if block_patterns else list(_DEFAULT_BLOCK_PATTERNS)

    for pat in patterns:
        try:
            for match in re.finditer(pat, source):
                # Get line number for the match
                line_no = source[:match.start()].count("\n") + 1
                findings.append(LintFinding(
                    file="",
                    line=line_no,
                    column=0,
                    severity="WARNING",
                    rule="SENSITIVE_PATTERN",
                    message="Potential sensitive pattern detected",
                    recommendation="Remove or externalize the sensitive value.",
                ))
                if len(findings) > 20:  # cap findings
                    break
        except re.error:
            continue

    return findings


# ── LintEngine ────────────────────────────────────────────────────────────────


class LintEngine:
    """Run static analysis on code files.

    Produces structured findings usable by the TUI, CLI, and web UI.
    Stdlib-only — no ruff dependency.
    """

    def __init__(self, include_rules: Optional[list[str]] = None) -> None:
        """Initialize lint engine with optional rule filter.

        Available rules:
            C901_complexity      — cyclomatic complexity > 9
            F841_unused_variable — unused local variables
            E999_syntax_error    — Python syntax errors
            IMPORT_BOUNDARY      — forbidden imports
            NO_DOCSTRING         — public functions missing docstrings (INFO)
            DUPLICATE_IMPORT     — same module imported multiple times
            SENSITIVE_PATTERN    — PII/credential patterns (INFO)
        """
        all_rules = [
            "C901_complexity",
            "F841_unused_variable",
            "E999_syntax_error",
            "IMPORT_BOUNDARY",
            "NO_DOCSTRING",
            "DUPLICATE_IMPORT",
            "SENSITIVE_PATTERN",
        ]
        self.checkers: list[tuple[str, callable]] = []

        if include_rules:
            rules_to_run = include_rules
        else:
            rules_to_run = all_rules

        rule_map = {
            "E999_syntax_error": (check_syntax, []),
            "C901_complexity": (check_complexity, []),
            "F841_unused_variable": (check_unused_variables, []),
            "IMPORT_BOUNDARY": (check_import_boundary, []),
            "NO_DOCSTRING": (check_docstrings, []),
            "DUPLICATE_IMPORT": (check_duplicate_imports, []),
            "SENSITIVE_PATTERN": (scan_content, []),
        }

        for rule_name in rules_to_run:
            checker, args = rule_map.get(rule_name, (lambda s, f="": [], []))
            self.checkers.append((rule_name, checker, args))

    def check_file(self, file_path: str, module_name: str = "") -> list[LintFinding]:
        """Check a single file by path. Returns list of findings."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError as exc:
            return [LintFinding(
                file=file_path, line=0, column=0,
                severity="BLOCKING",
                rule="FILE_ERROR",
                message=f"Cannot read file: {exc}",
                recommendation="Ensure file is accessible and has valid encoding.",
            )]

        return self.check_source(source, file_path, module_name)

    def check_source(
        self,
        source: str,
        file_path: str = "",
        module_name: str = "",
    ) -> list[LintFinding]:
        """Check source code string. Returns list of findings."""
        findings: list[LintFinding] = []

        for rule_name, checker, args in self.checkers:
            try:
                rule_findings = checker(source, file_path, *args)
                # Filter by rule name
                rule_findings = [f for f in rule_findings if f.rule == rule_name or not f.rule]
                findings.extend(rule_findings)
            except Exception as exc:
                logger.warning("Checker %s failed on %s: %s", rule_name, file_path, exc)

        # Sort by severity (BLOCKING first), then line number
        severity_order = {"BLOCKING": 0, "WARNING": 1, "INFO": 2}
        findings.sort(key=lambda f: (severity_order.get(f.severity, 3), f.line))

        return findings

    def check_files(self, file_paths: list[str], module_name: str = "") -> list[LintFinding]:
        """Check multiple files. Returns all findings."""
        all_findings: list[LintFinding] = []
        for fp in file_paths:
            all_findings.extend(self.check_file(fp, module_name))
        return all_findings

    def summary(self, findings: list[LintFinding]) -> dict[str, int]:
        """Produce a summary count of findings by severity."""
        counts: Counter = Counter()
        for f in findings:
            counts[f.severity] += 1
        return {
            "blocking": counts.get("BLOCKING", 0),
            "warning": counts.get("WARNING", 0),
            "info": counts.get("INFO", 0),
            "total": len(findings),
        }
