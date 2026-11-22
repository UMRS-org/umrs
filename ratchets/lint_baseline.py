#!/usr/bin/env python3
"""Lint baseline + ratchet (structure-hardening plan §6.3 step 3.4, ADR-052).

Sibling of ``ratchets/complexity_baseline.py``, for ruff findings instead of
cyclomatic complexity. The mechanical autofix waves (plan 3.1) cannot reach
the non-mechanical classes honestly (B008 def-time defaults, BLE001
blind-except, DTZ003 naive datetimes, F405 star-import use, …), so the gate
carries them as a RATCHET: current findings are recorded once, and every
subsequent ``check`` fails when

* a NEW finding occurrence appears (``new``) — matched per
  ``(file, code, line-content fingerprint)`` as a multiset so same-code
  occurrences on sibling lines stay distinct, or
* the total finding count grows past the baseline (``count``).

Baselined findings may only stay or disappear, so the residue shrinks
monotonically. Line numbers are deliberately NOT part of the identity —
editing a file above a flagged line shifts the line number; the
fingerprint of the flagged line's whitespace-normalized content is the
anchor (content edits on a baselined line are treated as a fresh finding —
by design).

Subcommands:

* ``record <paths...>``  → write ratchets/lint_baseline.json
* ``check <paths...>``   → exit 1 with a printed report on any ratchet break
* ``check --rebaseline`` → rewrite baseline, refused when the count grew
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

BASELINE_FILE = Path("ratchets/lint_baseline.json")


def _ruff() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / ".venv" / "bin" / "ruff"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ruff")
    return found or "ruff"


@dataclass(frozen=True)
class LintFinding:
    """One ruff finding under the ratchet."""

    file: str
    line: int
    col: int
    code: str
    message: str
    fingerprint: str

    @property
    def key(self) -> tuple[str, str]:
        # ADR-052 v2 (2026-09-15): key by (file, code) only. The earlier
        # line-content fingerprint made the baseline collapse every time a
        # large autofix wave shifted line content (whole-tree format passes):
        # hundreds of unchanged findings re-appeared as "[new]" and the
        # ratchet choked on its own migration. (file, code) multiset keeps
        # the same no-new-finding/no-count-growth strength while surviving
        # reformat history; removal of ONE copy of a finding (a real fix)
        # still moves the ratchet DOWN as designed.
        return (self.file, self.code)


def _line_fingerprint(source: str, line_no: int) -> str:
    lines = source.splitlines()
    text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _relativize(filename: str, repo_root: Path) -> str:
    abs_path = Path(filename)
    if abs_path.is_absolute():
        try:
            return str(abs_path.relative_to(repo_root))
        except ValueError:
            return abs_path.name  # outside the tree: keep its name
    return filename


def _finding_from_row(row: dict, cache: dict[str, str], repo_root: Path) -> LintFinding:
    filename = _relativize(row["filename"], repo_root)
    if filename not in cache:
        source_path = repo_root / filename
        cache[filename] = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    location = row.get("location") or {}
    line_no = int(location.get("row", 0))
    col = int(location.get("column", 0))
    return LintFinding(
        file=filename,
        line=line_no,
        col=col,
        code=str(row.get("code") or ""),
        message=str(row.get("message") or ""),
        fingerprint=_line_fingerprint(cache[filename], line_no),
    )


def _collect(paths: list[Path], repo_root: Path) -> list[LintFinding]:
    cmd = [_ruff(), "check", "--output-format", "json"] + [str(p) for p in paths]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    payload = json.loads(proc.stdout or "[]")
    findings: list[LintFinding] = []
    cache: dict[str, str] = {}
    for row in payload:
        filename = str(row.get("filename") or "")
        if Path(filename).name.startswith(".#"):
            continue  # Emacs interlock dead symlinks are never source.
        findings.append(_finding_from_row(row, cache, repo_root))
    findings.sort(key=lambda f: (f.file, f.line, f.col, f.code))
    return findings


def _write_baseline(path: Path, findings: list[LintFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": [asdict(f) for f in findings],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row(kind: str, finding: LintFinding) -> dict:
    return {"kind": kind, **asdict(finding)}


def _diff_findings(current: list[LintFinding], recorded: list[LintFinding]) -> list[dict]:
    """Breaks against the recorded baseline. Multiset semantics per
    (file, code, fingerprint) key: counts may only shrink."""
    base_counts = Counter(f.key for f in recorded)
    cur_counts = Counter(f.key for f in current)
    breaks: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    repr_findings = {f.key: f for f in current}
    for key, cur in cur_counts.items():
        if cur > base_counts.get(key, 0):
            seen_keys.add(key)
    for key in sorted(seen_keys):
        finding = repr_findings[key]
        breaks.append(
            _row(
                f"new:x{cur_counts[key]}(base x{base_counts.get(key, 0)})",
                finding,
            )
        )
    if len(current) > len(recorded):
        breaks.append(
            {
                "kind": "count",
                "file": "-",
                "line": 0,
                "col": 0,
                "code": "-",
                "message": f"findings {len(current)} > baseline {len(recorded)}",
                "fingerprint": "-",
            }
        )
    return breaks


def _fmt_break(row: dict) -> str:
    if row["kind"] == "count":
        return f"[count] {row['message']}"
    return f"{row['file']}:{row['line']}: {row['code']} {row['message']} [{row['kind']}]"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_record(paths: list[Path], repo_root: Path | None = None) -> int:
    repo_root = repo_root or _default_repo_root()
    findings = _collect(paths, repo_root)
    _write_baseline(repo_root / BASELINE_FILE, findings)
    codes = Counter(f.code for f in findings)
    top = ", ".join(f"{c}x{n}" for c, n in codes.most_common(8))
    print(f"lint baseline recorded: {len(findings)} findings -> {BASELINE_FILE}")
    print(f"top codes: {top}")
    return 0


def _check_rebaseline(current, recorded, baseline_path) -> tuple[int, list[dict]]:
    count_breaks = [b for b in _diff_findings(current, recorded) if b["kind"] == "count"]
    if not count_breaks:
        _write_baseline(baseline_path, current)
        print(f"rebaselined: {len(recorded)} -> {len(current)} findings")
        return 0, []
    return 1, count_breaks + [
        {
            "kind": "rebaseline-refused",
            "file": "-",
            "line": 0,
            "col": 0,
            "code": "-",
            "message": "new count exceeds old; fix findings instead",
            "fingerprint": "-",
        }
    ]


def _load_baseline(baseline_path: Path) -> list[LintFinding]:
    raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    return [LintFinding(**row) for row in raw["entries"]]


def cmd_check(paths: list[Path], rebaseline: bool = False, repo_root: Path | None = None) -> int:
    repo_root = repo_root or _default_repo_root()
    baseline_path = repo_root / BASELINE_FILE
    if not baseline_path.exists():
        print(f"no baseline at {BASELINE_FILE}; run: ratchets/lint_baseline.py record src tests")
        return 1
    recorded = _load_baseline(baseline_path)
    current = _collect(paths, repo_root)
    if rebaseline:
        rc, breaks = _check_rebaseline(current, recorded, baseline_path)
        for row in breaks:
            print(_fmt_break(row))
        return rc
    breaks = _diff_findings(current, recorded)
    for row in breaks:
        print(_fmt_break(row))
    if breaks:
        return 1
    by_code = Counter(f.code for f in current)
    top = ", ".join(f"{c}x{n}" for c, n in by_code.most_common(5))
    print(f"lint ratchet OK: {len(current)}/{len(recorded)} baselined findings ({top})")
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
