"""Pins for ratchets/lint_baseline.py (plan §6.3 step 3.4, ADR-052)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ratchets import lint_baseline as lb

RUFF = shutil.which("ruff") or ".venv/bin/ruff"


BLIND_EXCEPT = """def swallow():
    try:
        int("x")
    except Exception:
        pass
"""

CLEAN = "def ok():\n    return 1\n"


def _write(root: Path, rel: str, text: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def _scan(tmp_path: Path, *rels: str) -> list[lb.LintFinding]:
    return lb._collect([tmp_path / rel for rel in rels], tmp_path)


def test_collect_flags_blind_except(tmp_path):
    _write(tmp_path, "pkg/mod.py", BLIND_EXCEPT)
    findings = _scan(tmp_path, "pkg")
    codes = {f.code for f in findings}
    assert "BLE001" in codes


def test_collect_clean_tree_is_empty(tmp_path):
    _write(tmp_path, "pkg/ok.py", CLEAN)
    assert _scan(tmp_path, "pkg") == []


def test_fingerprint_ignores_whitespace_changes(tmp_path):
    f = _write(tmp_path, "mod.py", "try:\n    int('x')\nexcept Exception:\n    pass\n")
    h1 = lb._line_fingerprint(f.read_text(), 2)
    f.write_text("try:\n      int('x')\nexcept Exception:\n    pass\n")
    h2 = lb._line_fingerprint(f.read_text(), 2)
    assert h1 == h2


def test_record_then_check_is_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "pkg/mod.py", BLIND_EXCEPT)
    assert lb.cmd_record([tmp_path / "pkg"], repo_root=tmp_path) == 0
    assert (tmp_path / lb.BASELINE_FILE).exists()
    assert lb.cmd_check([tmp_path / "pkg"], repo_root=tmp_path) == 0


def test_new_occurrence_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    recorded = [
        lb.LintFinding("pkg/mod.py", 4, 5, "BLE001", "blind", "oldfp"),
    ]
    baseline = tmp_path / lb.BASELINE_FILE
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"generated": "-", "entries": [e.__dict__ for e in recorded]}))
    _write(tmp_path, "pkg/new.py", BLIND_EXCEPT)
    assert lb.cmd_check([tmp_path / "pkg"], repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "[new:" in out and "count" in out


def test_multiset_counts_identical_flagged_lines(tmp_path):
    """Identical flagged lines collapse into one key but the COUNT is the
    identity: a third occurrence of the same pattern must trip the ratchet."""
    src = "try:\n    int('x')\nexcept Exception:\n    pass\n\ntry:\n    int('y')\nexcept Exception:\n    pass\n"
    _write(tmp_path, "pkg/mod.py", src)
    findings = lb._collect([tmp_path / "pkg"], tmp_path)

    ble = [f for f in findings if f.code == "BLE001"]
    assert len(ble) == 2 and len({f.key for f in ble}) == 1
    # Baseline of the two; adding a third identical block must fail.
    baseline = tmp_path / lb.BASELINE_FILE
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {"generated": "-", "entries": [json.loads(json.dumps(f.__dict__)) for f in findings]}
        )
    )
    _write(tmp_path, "pkg/mod.py", src + "\ntry:\n    int('z')\nexcept Exception:\n    pass\n")
    assert lb.cmd_check([tmp_path / "pkg"], repo_root=tmp_path) == 1


def test_disappearing_finding_is_fine(tmp_path):
    current: list[lb.LintFinding] = []
    recorded = [lb.LintFinding("pkg/mod.py", 4, 5, "BLE001", "m", "fp")]
    assert lb._diff_findings(current, recorded) == []


def test_rebaseline_allowed_when_count_not_larger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Recorded 2 entries; current scan finds 2 as well (BLE001 + S110 on the
    # same file) — count did not grow, so rebaseline rewrites the file.
    recorded = [
        lb.LintFinding("pkg/a.py", 4, 5, "BLE001", "m", "fp1"),
        lb.LintFinding("pkg/b.py", 4, 5, "S110", "m", "fp2"),
    ]
    baseline = tmp_path / lb.BASELINE_FILE
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"generated": "-", "entries": [e.__dict__ for e in recorded]}))
    _write(tmp_path, "pkg/mod.py", BLIND_EXCEPT)
    assert lb.cmd_check([tmp_path / "pkg"], rebaseline=True, repo_root=tmp_path) == 0
    data = json.loads(baseline.read_text())
    assert len(data["entries"]) == 2


def test_rebaseline_refused_when_count_grows(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    recorded = [lb.LintFinding("pkg/a.py", 4, 5, "BLE001", "m", "fp1")]
    baseline = tmp_path / lb.BASELINE_FILE
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"generated": "-", "entries": [e.__dict__ for e in recorded]}))
    _write(tmp_path, "pkg/mod.py", BLIND_EXCEPT)  # BLE001 + S110 = 2 > 1
    assert lb.cmd_check([tmp_path / "pkg"], rebaseline=True, repo_root=tmp_path) == 1
    assert "rebaseline-refused" in capsys.readouterr().out


def test_missing_baseline_reports_actionable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert lb.cmd_check([tmp_path], repo_root=tmp_path) == 1
    assert "record src" in capsys.readouterr().out


def test_real_baseline_is_present_and_self_consistent():
    repo_root = Path(__file__).resolve().parent.parent
    data = json.loads((repo_root / lb.BASELINE_FILE).read_text())
    # Contract: non-trivial residue + the ratchet's own check passes. Absolute
    # size moves with every deliberate rebaseline (3782 fresh; 1927 post-3.1).
    assert data["entries"]  # recorded, not emptied; the ratchet only shrinks it
    assert lb.cmd_check([repo_root / "src", repo_root / "tests"], repo_root=repo_root) == 0


def test_line_shift_above_finding_does_not_misfire(tmp_path):
    """Editing code ABOVE a baselined finding changes its line number; the
    fingerprint anchor keeps the ratchet quiet."""
    _write(tmp_path, "pkg/mod.py", BLIND_EXCEPT)
    recorded = lb._collect([tmp_path / "pkg"], tmp_path)
    _write(tmp_path, "pkg/mod.py", "X = 1\nY = 2\n" + BLIND_EXCEPT)
    current = lb._collect([tmp_path / "pkg"], tmp_path)
    assert lb._diff_findings(current, recorded) == []


def test_edited_flagged_line_changes_key_code_membership(tmp_path):
    """(file, code)-key semantics (ADR-052 v2): editing a flagged line only
    counts when the code-family membership changes — a plain re-edit of the
    same rule instance is (deliberately) NOT a new finding, because large
    format waves would otherwise flood the ratchet with self-noise.
    Changing the rule itself (BLE001's sibling B036) still counts as new."""
    _write(tmp_path, "pkg/mod.py", "try:\n    int('x')\nexcept Exception:\n    pass\n")
    recorded = lb._collect([tmp_path / "pkg"], tmp_path)
    _write(tmp_path, "pkg/mod.py", "try:\n    int('zz')\nexcept Exception:\n    pass\n")
    current = lb._collect([tmp_path / "pkg"], tmp_path)
    assert lb._diff_findings(current, recorded) == []
    _write(tmp_path, "pkg/mod2.py", "try:\n    int('x')\nexcept Exception:\n    pass\n")
    current2 = lb._collect([tmp_path / "pkg"], tmp_path)
    breaks = lb._diff_findings(current2, recorded)
    assert breaks and all(b["kind"].startswith("new") or b["kind"] == "count" for b in breaks)


def test_count_break_prints_totals(tmp_path):
    current = [lb.LintFinding("pkg/a.py", i, 1, "BLE001", "m", f"fp{i}") for i in range(3)]
    recorded = current[:2]
    breaks = lb._diff_findings(current, recorded)
    count_rows = [b for b in breaks if b["kind"] == "count"]
    assert count_rows and "3 > baseline 2" in count_rows[0]["message"]


def test_fmt_break_hides_finite_rows_for_count():
    row = {
        "kind": "count",
        "message": "findings 5 > baseline 4",
        "file": "-",
        "line": 0,
        "col": 0,
        "code": "-",
        "fingerprint": "-",
    }
    assert lb._fmt_break(row) == "[count] findings 5 > baseline 4"
    finding_row = {
        "kind": "new:x1(base x0)",
        "file": "a.py",
        "line": 7,
        "col": 3,
        "code": "BLE001",
        "message": "blind",
        "fingerprint": "-",
    }
    text = lb._fmt_break(finding_row)
    assert text.startswith("a.py:7: BLE001") and text.endswith("[new:x1(base x0)]")
