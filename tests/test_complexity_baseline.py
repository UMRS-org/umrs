"""Pins for ratchets/complexity_baseline.py (plan §6.3 step 3.3, ADR-052)."""

from __future__ import annotations

import json
from pathlib import Path

from ratchets import complexity_baseline as cb


def _nested_function(depth: int) -> str:
    """Emit a function with `depth` nested ifs: complexity = depth + 1."""
    lines = ["def hot(x):"]
    for level in range(depth):
        lines.append("    " * (level + 1) + f"if x > {level}:")
    lines.append("    " * (depth + 1) + "return 1")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


# 9 nested ifs -> complexity 10 (over threshold 9); 10 nested -> complexity 11.
HOT_FN = _nested_function(9)
HOTTER_FN = _nested_function(10)


def _write(root: Path, rel: str, text: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def _record(tmp_path: Path, monkeypatch, entries: list[cb.BaselineEntry]) -> Path:
    baseline_file = tmp_path / cb.BASELINE_FILE
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated": "-", "threshold": 9, "entries": [e.__dict__ for e in entries]}
    baseline_file.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    return baseline_file


def test_record_then_check_is_clean(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "pkg/mod.py", HOT_FN)
    entries = cb._scan([tmp_path / "pkg"], tmp_path)
    assert len(entries) == 1 and entries[0].complexity == 10
    baseline_file = tmp_path / cb.BASELINE_FILE
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    assert cb.cmd_record([tmp_path / "pkg"], repo_root=tmp_path) == 0
    assert baseline_file.exists()
    assert cb.cmd_check([tmp_path / "pkg"], repo_root=tmp_path) == 0
    assert "1/1" in capsys.readouterr().out


def test_new_over_threshold_pair_fails(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "pkg/new.py", HOT_FN)
    _record(tmp_path, monkeypatch, [cb.BaselineEntry("pkg/old.py", "hot1", 10, "fp1")])
    assert cb.cmd_check([tmp_path / "pkg"], repo_root=tmp_path) == 1
    assert "[new]" in capsys.readouterr().out


def test_grew_grows_fails(tmp_path):
    _write(tmp_path, "pkg/mod.py", HOTTER_FN)
    current = cb._scan([tmp_path / "pkg"], tmp_path)
    recorded = [cb.BaselineEntry("pkg/mod.py", "hot", 10, "oldfp")]
    breaks = cb._diff_entries(current, recorded)
    grew = [b for b in breaks if b["kind"].startswith("grew")]
    assert [b["kind"] for b in grew] == ["grew:10->11"]


def test_count_increase_fires_when_over_threshold_set_grows():
    def h(name):
        return cb.BaselineEntry(f"pkg/{name[-1]}.py", name, 10, "")

    current = [h(f"h{i}") for i in range(3)]
    recorded = current[:2]
    kinds = {b["kind"] for b in cb._diff_entries(current, recorded)}
    assert "count" in kinds and "new" in kinds
    swap_kinds = {b["kind"] for b in cb._diff_entries(current[:2], [current[0], h("h9")])}
    assert "count" not in swap_kinds  # same size swaps are reported as [new] only


def test_shrinking_backlog_passes():
    current = [cb.BaselineEntry("pkg/a.py", "h1", 10, "")]
    recorded = current + [cb.BaselineEntry("pkg/b.py", "h2", 12, "")]
    assert cb._diff_entries(current, recorded) == []


def test_rebaseline_allowed_only_when_not_larger(tmp_path, monkeypatch):
    recorded = [
        cb.BaselineEntry("pkg/a.py", "h1", 10, ""),
        cb.BaselineEntry("pkg/b.py", "h2", 10, ""),
    ]
    baseline_file = _record(tmp_path, monkeypatch, recorded)
    _write(tmp_path, "pkg/mod.py", HOT_FN)  # single hot -> count 1 <= 2
    assert cb.cmd_check([tmp_path / "pkg"], rebaseline=True, repo_root=tmp_path) == 0
    assert len(json.loads(baseline_file.read_text())["entries"]) == 1


def test_rebaseline_refused_when_larger(tmp_path, monkeypatch, capsys):
    recorded = [cb.BaselineEntry("pkg/a.py", "h1", 10, "")]
    _record(tmp_path, monkeypatch, recorded)
    _write(tmp_path, "pkg/mod.py", HOT_FN.replace("def hot(", "def hot1("))
    _write(tmp_path, "pkg/mod2.py", HOT_FN.replace("def hot(", "def hot2("))
    assert cb.cmd_check([tmp_path / "pkg"], rebaseline=True, repo_root=tmp_path) == 1
    assert "rebaseline-refused" in capsys.readouterr().out


def test_clean_tree_has_zero_entries(tmp_path):
    _write(tmp_path, "pkg/ok.py", "def ok():\n    return 1\n")
    assert cb._scan([tmp_path / "pkg"], tmp_path) == []


def test_missing_baseline_reports_actionable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cb.cmd_check([tmp_path], repo_root=tmp_path) == 1
    assert "record src" in capsys.readouterr().out


def test_real_baseline_is_present_and_self_consistent():
    repo_root = Path(__file__).resolve().parent.parent
    data = json.loads((repo_root / cb.BASELINE_FILE).read_text())
    assert data["threshold"] == 9
    assert data["entries"]  # recorded, not emptied; the ratchet only shrinks it
    assert cb.cmd_check([repo_root / "src", repo_root / "tests"], repo_root=repo_root) == 0


def test_fingerprint_is_stable_for_identical_source(tmp_path):
    f1 = _write(tmp_path, "a.py", HOT_FN)
    h1 = cb._function_fingerprint(f1, "hot")
    h2 = cb._function_fingerprint(f1, "hot")
    assert h1 and h1 == h2
