"""Benchmark scripts under src/scripts/benchmarks (plan U1 A8).

Smoke: each Python script imports and its ``--help`` works. Behaviour: the
refactored server-log parser prints what upstream's version printed for a
fixed log (minus the never-written [TIMING] columns; the skip line in this
server's wording), and the populate scripts' argument defaults match the
constants upstream hard-coded (minus the host-specific paths). That the
parser reads the lines the real handler writes is checked in
test_process_objects_prefilter.py.
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
2026-10-01 10:00:02,100 INFO umrs.server.indexed_prefilter: [PROCESS] skipping 3 already-indexed object(s): {'alice.xp.images': 3}
2026-10-01 10:00:03,000 INFO umrs.s3: S3 events ready ('alice', 97)
2026-10-01 10:00:03,500 INFO umrs.task: [TASK] client=alice task_id=7 → Running
2026-10-01 10:00:09,250 INFO umrs.task: [TASK] client=alice task_id=7 → Completed
2026-10-01 10:00:10,000 INFO umrs.auth: Auth session created: user=bob role=user
2026-10-01 10:00:10,200 INFO MessageQueue: recv cmd=PROCESS_OBJECTS client_id=bob fields=5 has_auth=true
2026-10-01 10:00:10,300 INFO umrs.s3: filter_already_indexed: skipping 5 already indexed objects
2026-10-01 10:00:11,000 INFO umrs.s3: S3 events ready ('bob', 12)
2026-10-01 10:00:12,000 INFO umrs.task: [TASK] client=bob task_id=99 → Completed
unrelated line without timestamp
"""

# Output of upstream main's parse_server_log.py on SERVER_LOG (captured
# before the refactor into LogParser), without the [TIMING] columns.
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
    "server_exec(s),end_to_end(s)\n"
    "7,alice,100,3,97,1.0,1.0,5.75,8.25\n"
    "?,bob,,5,12,0.2,,,\n"
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


MULTI_CLIENT_LOG = """\
2026-10-01 10:00:01,500 INFO umrs.core.MessageQueue: [PROCESS] Batch assembled: 3 client request(s) — [('alice', 5), ('bob', 2), ('alice', 4)]
2026-10-01 10:00:02,000 INFO umrs.server.commands.process_objects: [PROCESS] Tasks created: [('alice', 1), ('bob', 2), ('alice', 3)]
2026-10-01 10:00:02,100 INFO umrs.server.indexed_prefilter: [PROCESS] skipping 9 already-indexed object(s): {'a': 9} per client: [('alice', 9)]
2026-10-01 10:00:02,200 INFO umrs.server.commands.process_objects: [PROCESS] S3 events ready — launching threads for 1 client(s): [('bob', 2)]
"""


def test_parse_server_log_matches_batch_counts_by_position():
    """A client sending twice in one batch keeps each request's own count;
    the per-client skip goes to that client's first task (review U1)."""
    tasks = _load("parse_server_log").parse_lines(MULTI_CLIENT_LOG.splitlines())

    rows = {tid: t.summary_row()[1:5] for tid, t in tasks.items()}
    assert rows == {"1": ["alice", 5, 9, 0], "2": ["bob", 2, 0, 2], "3": ["alice", 4, 0, 0]}


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
    assert args.insecure is False
    assert populate.rounds_needed(20_000) == 4
    assert list(populate.all_prefixes())[:2] == ["5000aa", "5000ab"]
    args = populate.parse_args(["--user", "7:10000", "--user", "9:50000"])
    assert args.users == [{"z": 7, "target": 10_000}, {"z": 9, "target": 50_000}]


def test_populate_buckets_resumes_after_done_rounds(tmp_path, monkeypatch):
    mod = _load("populate_buckets")
    args = mod.parse_args(["--log-dir", str(tmp_path), "--sleep", "0", "--user", "10:21000"])
    assert args.state_file == str(tmp_path / "mass_upload_state.json")
    calls = []
    monkeypatch.setattr(mod, "run_upload", lambda _z, prefix, idx, _total, _a: calls.append((prefix, idx)))
    state = {"10": 1}
    mod.populate_user(args.users[0], state, args)
    assert calls == [("ab", 2), ("ac", 3)]
    assert state == {"10": 3}


# --- populate: TLS verification and one retry layer (review U1) -----------------------


def _populate_s3cfg(tmp_path, extra=""):
    path = tmp_path / "s3cfg"
    path.write_text(
        "[default]\nhost_base = store:9000\nuse_https = True\naccess_key = a\n"
        f"secret_key = s\n{extra}",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def populate_clients(monkeypatch):
    pytest.importorskip("boto3")
    pytest.importorskip("tqdm")
    populate = _load("populate")
    built = []
    monkeypatch.setattr(populate.boto3, "client", lambda *_a, **kw: built.append(kw) or kw)
    monkeypatch.delenv("UMRS_S3_MAX_ATTEMPTS", raising=False)
    return populate, built


@pytest.mark.parametrize(
    ("extra", "insecure", "verify"),
    [("", False, True),
     ("check_ssl_certificate = True\n", False, True),
     ("check_ssl_certificate = False\n", False, False),
     ("", True, False)],
)
def test_populate_verifies_tls_unless_told_not_to(populate_clients, tmp_path, extra,
                                                  insecure, verify):
    populate, built = populate_clients
    populate.make_client(_populate_s3cfg(tmp_path, extra), insecure=insecure)
    (kwargs,) = built
    assert kwargs["verify"] is verify
    assert kwargs["endpoint_url"] == "https://store:9000"


def test_populate_retries_in_botocore_only(populate_clients, tmp_path):
    populate, built = populate_clients
    populate.make_client(_populate_s3cfg(tmp_path))
    (kwargs,) = built
    assert kwargs["config"].retries == {"total_max_attempts": 5, "mode": "standard"}
    assert not hasattr(populate, "_with_retry")


def test_populate_insecure_flag():
    pytest.importorskip("boto3")
    pytest.importorskip("tqdm")
    populate = _load("populate")
    assert populate.parse_args(["--insecure"]).insecure is True


# --- bench_knn: only successful replies are timed (review U1) -------------------------


class _KnnMQ:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def request_knn(self, username, k, s3_key, bucket, **_kwargs):
        self.calls.append((username, k, s3_key, bucket))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def close(self):
        self.closed = True


_OK_REPLY = '{"status": "ok", "results": []}'


def test_bench_knn_reopens_its_socket_after_a_timeout(monkeypatch):
    """A DEALER socket does not pair replies with requests: after a timeout
    the late reply answered the next request on the same socket."""
    bench_knn = _load("bench_knn")
    first = _KnnMQ([_OK_REPLY, TimeoutError("slow")])
    second = _KnnMQ([_OK_REPLY, _OK_REPLY])
    made = [first, second]
    monkeypatch.setattr(bench_knn, "_make_mq", lambda *_a: made.pop(0))

    latencies, errors = bench_knn._worker("h", 1, False, 4, 0, "u", 3, "q.png", "b")

    assert (len(latencies), errors) == (3, 1)
    assert len(first.calls) == 2  # nothing more is sent on the timed-out socket
    assert first.closed
    assert len(second.calls) == 2
    assert second.closed


def test_bench_knn_counts_error_replies_instead_of_timing_them(monkeypatch):
    bench_knn = _load("bench_knn")
    replies = ['{"status": "ok", "results": []}', '{"status": "error", "message": "no"}',
               "SERVER_OVERLOADED", "UNAUTHORIZED:KNN", TimeoutError("slow"),
               '{"status": "ok", "results": []}']
    mq = _KnnMQ(replies)
    monkeypatch.setattr(bench_knn, "_make_mq", lambda *_a: mq)

    stats = bench_knn.run_bench(N=1, K=6, server_ip="h", control_port=1, tunnel_mode=False,
                                warmup=0, username="u", k=3, s3_key="q.png", bucket="b")

    assert stats["n"] == 2
    assert stats["errors"] == 4
    assert mq.calls[0] == ("u", 3, "q.png", "b")


def test_bench_knn_warm_up_stops_on_an_error_reply(monkeypatch):
    bench_knn = _load("bench_knn")
    mq = _KnnMQ(['{"status": "error", "message": "No bucket entry found for path: b"}'])
    monkeypatch.setattr(bench_knn, "_make_mq", lambda *_a: mq)
    connection = type("C", (), {"server_host": "h", "control_port": 1, "tunnel_mode": False})
    args = type("A", (), {"username": "u", "k": 3, "bucket": "b"})
    with pytest.raises(SystemExit, match="No bucket entry"):
        bench_knn._warm_cache(connection, args, "q.png")


class _BenchS3:
    def __init__(self):
        self.uploads = []
        self.deleted = []

    def upload_file(self, path, bucket, key, **_kwargs):
        self.uploads.append((path, bucket, key))

    def delete_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.deleted.append((Bucket, Key))


def test_bench_knn_uploads_its_query_under_the_reserved_key_and_removes_it(
    monkeypatch, tmp_path
):
    """Review U1: the bench uploaded its query as <bucket>/<basename>,
    overwriting a corpus object of that name and leaving it to be searched."""
    from types import SimpleNamespace

    bench_knn = _load("bench_knn")
    image = tmp_path / "cat.jpeg"
    image.write_bytes(b"jpeg")
    s3 = _BenchS3()
    seen = []
    monkeypatch.setattr(bench_knn, "_connect_s3", lambda *_a: s3)
    monkeypatch.setattr(bench_knn, "resolve_connection", lambda **_k: SimpleNamespace(
        server_host="h", control_port=1, tunnel_mode=False))
    monkeypatch.setattr(bench_knn, "_benchmark", lambda _a, _c, key: seen.append(key) or [])
    monkeypatch.setattr(sys, "argv", ["bench_knn.py", "--image", str(image), "--bucket", "b",
                                      "--username", "u", "--s3cfg", "c"])

    bench_knn.main()

    (key,) = seen
    assert key.startswith(".umrs-knn/") and key.endswith("-cat.jpeg")
    assert s3.uploads == [(str(image), "b", key)]
    assert s3.deleted == [("b", key)]


def test_bench_knn_has_no_client_id_flag():
    bench_knn = _load("bench_knn")
    with pytest.raises(SystemExit):
        bench_knn._build_parser().parse_args(
            ["--image", "i", "--bucket", "b", "--username", "u", "--s3cfg", "c",
             "--client-id", "x"])


# --- shared helpers --------------------------------------------------------------------


def test_bench_scripts_share_one_stats_module():
    common = _load("_bench_common")
    for name in ("bench_knn", "bench_latency"):
        module = _load(name)
        for helper in ("_stats", "_percentile", "_fmt_ms", "_append_csv"):
            assert not hasattr(module, helper), f"{name}.{helper} is a copy"
    stats = common.stats([10.0, 20.0, 30.0, 40.0])
    assert (stats["n"], stats["min_ms"], stats["max_ms"], stats["p50_ms"]) == (4, 10.0, 40.0, 25.0)
    assert common.fmt_ms({}, "mean_ms") == "nan"


def test_append_csv_creates_the_directory(tmp_path):
    common = _load("_bench_common")
    path = tmp_path / "log" / "out.csv"
    common.append_csv(str(path), [{"a": 1}])
    common.append_csv(str(path), [{"a": 2}])
    assert path.read_text().splitlines() == ["a", "1", "2"]


# --- knn_sweep_cold_warm.sh --------------------------------------------------------------


def test_knn_sweep_stays_within_the_server_k_cap():
    from umrs.server.knn_worker import DEFAULT_MAX_K

    script = (BENCH_DIR / "knn_sweep_cold_warm.sh").read_text(encoding="utf-8")
    (line,) = [ln for ln in script.splitlines() if ln.startswith("K_VALUES=")]
    values = [int(v) for v in line.split("(", 1)[1].rstrip(")").split()]
    assert max(values) <= DEFAULT_MAX_K


def test_knn_sweep_passes_shellcheck():
    import shutil

    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(["shellcheck", str(BENCH_DIR / "knn_sweep_cold_warm.sh")],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout
