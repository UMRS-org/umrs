#!/usr/bin/env python3
"""Complexity baseline + ratchet (structure-hardening plan §6.3 step 3.3, ADR-052).

The whole-tree complexity backlog (D-1 evidence: hundreds of functions above
threshold 9 across umrs) cannot be refactored in one wave, so the gate works
as a RATCHET: the current over-threshold set is recorded once, and every
subsequent ``check`` fails when

* a NEW over-threshold ``(file, function)`` pair appears (``new``),
* a baselined pair's complexity GREW (``grew``), or
* the total over-threshold pair count exceeded the baseline's (``count``).

Baselined pairs may only stay flat or disappear (improvement), so the backlog
shrinks monotonically from here on. ``--rebaseline`` is allowed only when the
new count is not larger than the recorded one.

Subcommands:

* ``record <paths...>``  → write ratchets/complexity_baseline.json
* ``check <paths...>``   → exit 1 with a printed report on any ratchet break
* ``check --rebaseline <paths...>`` → same checks count-cap only; rewrite file
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Script mode (`python ratchets/complexity_baseline.py`) puts ratchets/ on
    # sys.path[0]; the repo root is needed for the ratchets package import.
    sys.path.insert(0, str(_REPO_ROOT))

from ratchets.complexity_audit import audit_paths  # noqa: E402

THRESHOLD = 9
BASELINE_FILE = Path(
    os.environ.get("COMPLEXITY_BASELINE_FILE", "ratchets/complexity_baseline.json")
)


@dataclass(frozen=True)
class BaselineEntry:
    """One over-threshold function captured by the baseline."""

    file: str
    function: str
    complexity: int
    fingerprint: str


def _function_fingerprint(file_path: Path, function_name: str) -> str:
    """Stable hash of the function's AST shape (no location attributes)."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name.split(".")[-1]
        ):
            dump = ast.dump(node, annotate_fields=False, include_attributes=False)
            return hashlib.sha256(dump.encode("utf-8")).hexdigest()
    return ""  # function vanished between scan and fingerprint (transient)


def _scan(paths: list[Path], repo_root: Path) -> list[BaselineEntry]:
    findings, errors, _scanned = audit_paths(paths, repo_root, THRESHOLD, sys.maxsize)
    if errors:
        for error in errors:
            print(f"parse error: {error.path}: {error.error}", file=sys.stderr)
    entries = [
        BaselineEntry(
            file=finding.path,
            function=finding.function,
            complexity=finding.complexity,
            fingerprint=_function_fingerprint(repo_root / finding.path, finding.function),
        )
        for finding in findings
        if finding.complexity > THRESHOLD
    ]
    entries.sort(key=lambda e: (e.file, e.function))
    return entries


def _load_baseline(baseline_path: Path) -> dict:
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _write_baseline(baseline_path: Path, entries: list[BaselineEntry]) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "entries": [asdict(entry) for entry in entries],
    }
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row(entry: BaselineEntry, kind: str) -> dict:
    return {
        "kind": kind,
        "file": entry.file,
        "function": entry.function,
        "complexity": entry.complexity,
        "fingerprint": entry.fingerprint,
    }


def _diff_entries(current: list[BaselineEntry], recorded: list[BaselineEntry]) -> list[dict]:
    """Breaks against the recorded baseline (strict ratchet semantics)."""
    recorded_by_pair = {(e.file, e.function): e for e in recorded}
    breaks: list[dict] = []
    for entry in current:
        base = recorded_by_pair.get((entry.file, entry.function))
        if base is None:
            breaks.append(_row(entry, "new"))
        elif entry.complexity > base.complexity:
            breaks.append(
                _row(
                    BaselineEntry(entry.file, entry.function, entry.complexity, entry.fingerprint),
                    f"grew:{base.complexity}->{entry.complexity}",
                )
            )
    if len(current) > len(recorded):
        breaks.append(
            {
                "kind": "count",
                "file": "-",
                "function": f"over-threshold pairs {len(current)} > baseline {len(recorded)}",
                "complexity": len(current),
                "fingerprint": "-",
            }
        )
    return breaks


def _fmt_break(row: dict) -> str:
    return f"{row['file']}: {row['function']} [{row['kind']}] complexity={row['complexity']}"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_record(paths: list[Path], repo_root: Path | None = None) -> int:
    repo_root = repo_root or _default_repo_root()
    entries = _scan(paths, repo_root)
    _write_baseline(repo_root / BASELINE_FILE, entries)
    print(f"baseline recorded: {len(entries)} over-threshold functions -> {BASELINE_FILE}")
    return 0


def _check_rebaseline(current, recorded, baseline_path) -> tuple[int, list[dict]]:
    """Rebaseline mode: allow only when the count did not grow."""
    count_breaks = [b for b in _diff_entries(current, recorded) if b["kind"] == "count"]
    if not count_breaks:
        _write_baseline(baseline_path, current)
        print(f"rebaselined: {len(recorded)} -> {len(current)} over-threshold functions")
        return 0, []
    return 1, count_breaks + [
        {
            "kind": "rebaseline-refused",
            "file": "-",
            "function": "new count exceeds old; fix functions instead",
            "complexity": len(current),
            "fingerprint": "-",
        }
    ]


def cmd_check(paths: list[Path], rebaseline: bool = False, repo_root: Path | None = None) -> int:
    repo_root = repo_root or _default_repo_root()
    baseline_path = repo_root / BASELINE_FILE
    if not baseline_path.exists():
        print(
            f"no baseline at {BASELINE_FILE}; run: ratchets/complexity_baseline.py record src tests"
        )
        return 1
    recorded = [BaselineEntry(**raw) for raw in _load_baseline(baseline_path)["entries"]]
    current = _scan(paths, repo_root)
    if rebaseline:
        rc, breaks = _check_rebaseline(current, recorded, baseline_path)
        for row in breaks:
            print(_fmt_break(row))
        return rc
    breaks = _diff_entries(current, recorded)
    for row in breaks:
        print(_fmt_break(row))
    if breaks:
        return 1
    print(
        f"complexity ratchet OK: {len(current)}/{len(recorded)} baselined pairs still over threshold"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_record = sub.add_parser("record")
    p_record.add_argument("paths", nargs="+", type=Path)
    p_check = sub.add_parser("check")
    p_check.add_argument("paths", nargs="+", type=Path)
    p_check.add_argument("--rebaseline", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "record":
        return cmd_record(args.paths)
    return cmd_check(args.paths, rebaseline=args.rebaseline)


if __name__ == "__main__":
    sys.exit(main())
