"""U4 skip gate: unlisted, excess and stale skips all fail."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("skip_gate", ROOT / "scripts" / "skip_gate.py")
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

_JUNIT = """<testsuites><testsuite>
<testcase classname="tests.a" name="t1"><skipped message="kubernetes package not installed"/></testcase>
<testcase classname="" name="tests.b"><skipped message="collection skipped">('/x/tests/b.py', 5, "Skipped: could not import 'r4.x': No module named 'r4'")</skipped></testcase>
<testcase classname="tests.c" name="t2"/>
</testsuite></testsuites>"""


def test_collection_skip_reason_comes_from_the_text(tmp_path: Path) -> None:
    junit = tmp_path / "j.xml"
    junit.write_text(_JUNIT)
    assert gate.junit_skips(junit) == [
        "kubernetes package not installed",
        "could not import 'r4.x': No module named 'r4'",
    ]


def test_exact_rows_pass() -> None:
    rows = [(1, "kubernetes"), (1, "could not import 'r4.")]
    assert gate.evaluate(["kubernetes x", "could not import 'r4.y'"], rows) == []


def test_unlisted_reason_fails() -> None:
    problems = gate.evaluate(["Need --db-url or UMRS_TEST_DB_URL"], [])
    assert problems == ["unlisted skip reason: 'Need --db-url or UMRS_TEST_DB_URL'"]


def test_excess_and_stale_counts_fail() -> None:
    assert gate.evaluate(["k", "k"], [(1, "k")]) == ["2 skips > 1 allowed: 'k'"]
    assert gate.evaluate([], [(1, "k")]) == ["allowlist only shrinks: lower 1 -> 0 for 'k'"]


def test_committed_allowlist_parses() -> None:
    rows = gate.read_allowlist(ROOT / "tests" / "allowed_skips.txt")
    assert rows and all(count > 0 and prefix for count, prefix in rows)
    assert not any("db-url" in p.lower() or "S3" in p for _, p in rows)
