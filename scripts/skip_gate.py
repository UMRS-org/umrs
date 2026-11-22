#!/usr/bin/env python3
"""Skip gate (U4, plan §7.2): fail when a test skips for an unlisted reason.

Reads a pytest JUnit XML and ``tests/allowed_skips.txt``. Each allowlist row
is ``<max count> | <reason prefix>``; a skip passes when its reason starts
with a listed prefix and that row's count stays within its maximum. The list
may only shrink: a row no skip matched any more, or a maximum above the
observed count, fails too, so the file always states today's skips exactly.
DB- and S3-backed tests therefore cannot pass by skipping.

Usage: python scripts/skip_gate.py JUNIT.xml [ALLOWLIST]
"""

from __future__ import annotations

import ast
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

DEFAULT_ALLOWLIST = Path(__file__).resolve().parents[1] / "tests" / "allowed_skips.txt"
_COLLECTION = "collection skipped"


def skip_reason(message: str, text: str) -> str:
    """The skip reason; a collection skip keeps it in the element text."""
    if message != _COLLECTION or not text:
        return message.strip()
    try:
        reason = str(ast.literal_eval(text.strip())[-1])
    except (ValueError, SyntaxError, IndexError, TypeError):
        reason = text.strip()
    return reason.removeprefix("Skipped: ").strip()


def junit_skips(path: Path) -> list[str]:
    reasons = []
    for case in ET.parse(path).getroot().iter("testcase"):
        for child in case:
            if child.tag == "skipped":
                reasons.append(skip_reason(child.get("message") or "", child.text or ""))
    return reasons


def read_allowlist(path: Path) -> list[tuple[int, str]]:
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count, _, prefix = line.partition("|")
        rows.append((int(count), prefix.strip()))
    return rows


def _match(reason: str, rows: list[tuple[int, str]]) -> str | None:
    for _, prefix in rows:
        if reason.startswith(prefix):
            return prefix
    return None


def evaluate(reasons: list[str], rows: list[tuple[int, str]]) -> list[str]:
    """Problems found; empty when the run's skips are exactly the listed ones."""
    seen: Counter[str] = Counter()
    problems = []
    for reason in reasons:
        prefix = _match(reason, rows)
        if prefix is None:
            problems.append(f"unlisted skip reason: {reason!r}")
        else:
            seen[prefix] += 1
    for limit, prefix in rows:
        got = seen[prefix]
        if got > limit:
            problems.append(f"{got} skips > {limit} allowed: {prefix!r}")
        elif got < limit:
            problems.append(f"allowlist only shrinks: lower {limit} -> {got} for {prefix!r}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    allowlist = Path(argv[2]) if len(argv) == 3 else DEFAULT_ALLOWLIST
    reasons = junit_skips(Path(argv[1]))
    problems = evaluate(reasons, read_allowlist(allowlist))
    for problem in problems:
        print(f"::error::skip gate: {problem}")
    print(f"skip gate: {len(reasons)} skips, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
