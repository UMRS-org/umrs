"""Benchmark scripts under src/scripts/benchmarks (plan U1 A8).

Smoke: each Python script imports and its ``--help`` works. Behaviour: the
refactored server-log parser prints exactly what upstream's version printed
for a fixed log, and the populate scripts' argument defaults match the
constants upstream hard-coded (minus the host-specific paths).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "src" / "scripts" / "benchmarks"

# script -> modules it needs at import time (skip when one is missing)
SCRIPTS = {
    "bench_baremetal_ingest.py": ("torch", "torchvision", "PIL"),
    "bench_knn.py": (),
    "bench_latency.py": (),
    "parse_server_log.py": (),
    "populate.py": ("boto3", "tqdm"),
    "populate_buckets.py": (),
}


def _run(script, *args, tmp_path):
    return subprocess.run(
        [sys.executable, str(BENCH_DIR / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_script_help(script, tmp_path):
    missing = [m for m in SCRIPTS[script] if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"{script} needs {missing}")
    result = _run(script, "--help", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []  # --help writes nothing


SERVER_LOG = """\
2026-10-01 10:00:00,000 INFO umrs.auth: Auth session created: user=alice role=user
2026-10-01 10:00:01,000 INFO MessageQueue: recv cmd=PROCESS_OBJECTS client_id=alice fields=5 has_auth=true
2026-10-01 10:00:01,500 INFO umrs.batch: Batch assembled: ('alice', 100)
2026-10-01 10:00:02,000 INFO umrs.batch: Tasks created: ('alice', 7)
2026-10-01 10:00:02,100 INFO umrs.s3: filter_already_indexed: skipping 3 already indexed objects
2026-10-01 10:00:03,000 INFO umrs.s3: S3 events ready ('alice', 97)
2026-10-01 10:00:03,500 INFO umrs.task: [TASK] client=alice task_id=7 → Running
2026-10-01 10:00:09,250 INFO umrs.task: [TASK] client=alice task_id=7 → Completed
2026-10-01 10:00:09,300 INFO umrs.timing: [TIMING] task_id=7,8 status=ok model=resnet path=batch batch_tasks=7,8 total_s=6.2 forward_s=1.25
2026-10-01 10:00:10,000 INFO umrs.auth: Auth session created: user=bob role=user
2026-10-01 10:00:10,200 INFO MessageQueue: recv cmd=PROCESS_OBJECTS client_id=bob fields=5 has_auth=true
2026-10-01 10:00:10,300 INFO umrs.s3: filter_already_indexed: skipping 5 already indexed objects
2026-10-01 10:00:11,000 INFO umrs.s3: S3 events ready ('bob', 12)
2026-10-01 10:00:12,000 INFO umrs.task: [TASK] client=bob task_id=99 → Completed
unrelated line without timestamp
"""

# Output of upstream main's parse_server_log.py on SERVER_LOG (captured
# before the refactor into LogParser).
EXPECTED_TABLE = (
    "task_id   user        n_batch  n_skipped n_processed upload_s3(s)  "
    "s3_check(s) server_exec(s)  end_to_end(s) \n"
    + "-" * 109 + "\n"
    "7         alice       100      3         97          1.0           "
    "1.0         5.75            8.25          \n"
    "?         bob         None     5         12          0.2           "
    "None        None            None          \n"
)
EXPECTED_CSV = (
    "task_id,user,n_batch,n_skipped,n_processed,upload_s3(s),s3_check(s),"
    "server_exec(s),end_to_end(s),status,model,path,batch_tasks,s3_events(s),"
    "index_check_sql(s),index_check_head(s),dispatch(s),lock_wait(s),"
    "fetch_preprocess(s),forward(s),embed(s),store(s),count_update(s),"
    "server_total(s)\n"
    '7,alice,100,3,97,1.0,1.0,5.75,8.25,ok,resnet,batch,"7,8",NA,NA,NA,NA,NA,'
    "NA,1.25,NA,NA,NA,6.2\n"
    "?,bob,,5,12,0.2,,,,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n"
)


@pytest.mark.parametrize(
    ("extra", "expected"), [((), EXPECTED_TABLE), (("--csv",), EXPECTED_CSV)]
)
def test_parse_server_log_matches_upstream_output(extra, expected, tmp_path):
    log = tmp_path / "server.log"
    log.write_text(SERVER_LOG, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BENCH_DIR / "parse_server_log.py"), str(log), *extra],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_populate_defaults_match_upstream_sweep(tmp_path):
    pytest.importorskip("boto3")
    pytest.importorskip("tqdm")
    populate = _load("populate")
    args = populate.parse_args(["--log-dir", str(tmp_path)])
    assert args.users == [{"z": 14, "target": 20_000}]
    assert args.state_file == str(tmp_path / "populate_state.json")
    assert args.verify_tls is False
    assert populate.rounds_needed(20_000) == 4
    assert list(populate.all_prefixes())[:2] == ["5000aa", "5000ab"]
    args = populate.parse_args(["--user", "7:10000", "--user", "9:50000"])
    assert args.users == [{"z": 7, "target": 10_000}, {"z": 9, "target": 50_000}]


def test_populate_buckets_resumes_after_done_rounds(tmp_path, monkeypatch):
    mod = _load("populate_buckets")
    args = mod.parse_args(["--log-dir", str(tmp_path), "--sleep", "0", "--user", "10:21000"])
    assert args.state_file == str(tmp_path / "mass_upload_state.json")
    calls = []
    monkeypatch.setattr(mod, "run_upload", lambda z, prefix, idx, total, a: calls.append((prefix, idx)))
    state = {"10": 1}
    mod.populate_user(args.users[0], state, args)
    assert calls == [("ab", 2), ("ac", 3)]
    assert state == {"10": 3}
