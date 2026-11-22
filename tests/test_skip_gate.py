"""U4 skip gate: unlisted, excess and stale skips all fail.

Drives ``scripts/skip_gate.py`` as CI does (a subprocess on a JUnit file),
so no spec-load harness is needed (ADR-053).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "skip_gate.py"

_CASE = '<testcase classname="tests.a" name="t{i}"><skipped message="{reason}"/></testcase>'
_COLLECTION = (
    '<testcase classname="" name="tests.b"><skipped message="collection skipped">'
    "('/x/tests/b.py', 5, \"Skipped: could not import 'r4.x': No module named 'r4'\")"
    "</skipped></testcase>"
)


def _gate(tmp_path: Path, reasons: list[str], rows: str, *, collection: bool = False):
    cases = [_CASE.format(i=i, reason=r) for i, r in enumerate(reasons)]
    if collection:
        cases.append(_COLLECTION)
    junit = tmp_path / "junit.xml"
    junit.write_text(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
    allow = tmp_path / "allowed.txt"
    allow.write_text(rows)
    return subprocess.run(
        [sys.executable, str(GATE), str(junit), str(allow)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_exact_rows_pass_and_collection_reasons_are_read(tmp_path: Path) -> None:
    rows = "# c\n1 | kubernetes\n1 | could not import 'r4.\n"
    done = _gate(tmp_path, ["kubernetes package not installed"], rows, collection=True)
    assert done.returncode == 0, done.stdout
    assert "2 skips, 0 problems" in done.stdout


def test_unlisted_reason_fails(tmp_path: Path) -> None:
    done = _gate(tmp_path, ["Need --db-url or UMRS_TEST_DB_URL"], "")
    assert done.returncode == 1
    assert "unlisted skip reason: 'Need --db-url or UMRS_TEST_DB_URL'" in done.stdout


def test_excess_count_fails(tmp_path: Path) -> None:
    done = _gate(tmp_path, ["k", "k"], "1 | k\n")
    assert done.returncode == 1
    assert "2 skips > 1 allowed: 'k'" in done.stdout


def test_allowlist_only_shrinks(tmp_path: Path) -> None:
    done = _gate(tmp_path, [], "1 | k\n")
    assert done.returncode == 1
    assert "allowlist only shrinks: lower 1 -> 0 for 'k'" in done.stdout


def test_committed_allowlist_lists_no_db_or_s3_skip() -> None:
    rows = [
        line.partition("|")
        for line in (ROOT / "tests" / "allowed_skips.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rows and all(count.strip().isdigit() and prefix.strip() for count, _, prefix in rows)
    assert not any("db-url" in prefix.lower() or "S3" in prefix for _, _, prefix in rows)
