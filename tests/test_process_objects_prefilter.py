"""PROCESS_OBJECTS_BATCH with the indexed pre-filter, end to end (plan U1 A5).

Drives ``ProcessObjectsBatchHandler.execute`` through the real UMRSThreads
pool and thread/status stores, with fake S3, DB, event queues and backend.
The STATUS assertions are the regression for upstream 8f9fe001: the thread
status reports the submitted count (plus how many were already indexed),
never the post-filter count.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from diskcache import Cache
from umrs.core.TaskCmdStatus import TaskStatus
from umrs.core.UMRSThreads import ThreadStatus, UMRSThreads
from umrs.server import processing
from umrs.server.commands.process_objects import ProcessObjectsBatchHandler
from umrs.shared.buckets import BucketsCfg, ServerBucketEntry

_KEY = "process_objects"


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _query, params):
        self._keys = set(params[0])

    def fetchall(self):
        return [row for row in self._rows if row[0] in self._keys]


class _DataBase:
    def __init__(self, rows):
        self.rows = rows
        self.checked_out = 0

    def has_column(self, _table, column):
        return column in ("s3_object", "checksum", "model")

    def get_conn(self):
        self.checked_out += 1
        return SimpleNamespace(cursor=lambda: _Cursor(self.rows))

    def put_conn(self, _conn, close=False):  # noqa: ARG002 — pool signature
        self.checked_out -= 1

    def count_s3_objects(self, _table):
        return len(self.rows)


class _S3:
    def head_object(self, Bucket, Key):  # noqa: N803, ARG002 — boto3 signature
        return {"ETag": f'"etag-{Key}"'}


class _Model:
    model_name = "clip"
    type = None
    return_type = "embedding"


class _Backend:
    def __init__(self):
        self.items: list[dict] = []

    def compute_and_store_batch(self, items):
        self.items.extend(items)
        return [True] * len(items)


class _Tasks:
    def __init__(self):
        self.status: dict[int, list[str]] = {}

    def create_task(self, _cmd, _payload):
        task_id = len(self.status) + 1
        self.status[task_id] = []
        return task_id

    def update_status(self, task_id, status):
        self.status[task_id].append(status)


class _DataQueue:
    def __init__(self, objects):
        self.objects = objects

    def wait_for_s3_events(self, nb_objects, timeout_s=60):  # noqa: ARG002
        return list(self.objects[:nb_objects])


def _bucket_entry(user, data_base):
    return ServerBucketEntry(
        bucket_alias=user, s3_username=user, sql_username=user, data_base=data_base,
        db_name=f"db_{user}", table_name=f"tbl_{user}", model=_Model(),
        store_embeddings_flag=True, modality_name="image", task="embedding",
        s3_connection=_S3(), metadata={}, xp_name="xp", num_objects=0,
        return_type="embedding",
    )


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_SERVICE_LIFECYCLE", "manual")
    monkeypatch.delenv("UMRS_PREFILTER_INDEXED", raising=False)
    owners = {"a": "alice", "b": "bob"}
    monkeypatch.setattr(processing, "_client_may_use", lambda b, c: owners.get(b) == c)

    indexed = [(f"a/k{i}", f"etag-k{i}", "clip") for i in range(3)]
    cfg = BucketsCfg()
    cfg.store("a", _bucket_entry("alice", _DataBase(indexed)))
    cfg.store("b", _bucket_entry("bob", _DataBase([])))

    threads = UMRSThreads(Cache(str(tmp_path / "status")), Cache(str(tmp_path / "threads")))
    tasks = _Tasks()
    backend = _Backend()
    queues: dict[str, dict] = {}
    ctx = SimpleNamespace(
        task_manager=SimpleNamespace(tasks_status=tasks, active_threads=threads),
        control_mq=SimpleNamespace(client_dictionary=queues),
        buckets_cfg=cfg, device="cpu", embedding_backend=backend,
    )
    stack = SimpleNamespace(ctx=ctx, threads=threads, tasks=tasks, backend=backend,
                            queues=queues, cfg=cfg)
    yield stack
    threads.close()


def _run(stack, events_by_client):
    for cid, objects in events_by_client.items():
        stack.queues[cid] = {"data_queue": _DataQueue(objects)}
    payload = [(cid, len(objs)) for cid, objs in events_by_client.items()]
    ProcessObjectsBatchHandler().execute(payload, stack.ctx)
    for thread in threading.enumerate():
        if thread.name == "process-objects-batch":
            thread.join(timeout=10)
    stack.threads.waitall()


def _works(stack, bucket):
    store = stack.threads.thread_dict
    infos = [store[key] for key in list(store.iterkeys())]
    return [
        (info["status"], work)
        for info in infos
        for work in info["work"] if work["bucket"] == bucket
    ]


def _status_works(stack, user):
    data = stack.threads.status_store[user][_KEY]["data"]
    return [work for info in data for work in info["work"]]


def _work_counts(works):
    return [(status, w["nb_objects"], w["nb_skipped_indexed"]) for status, w in works]


def test_status_reports_submitted_count_when_some_are_indexed(stack):
    objects = [f"s3://a/k{i}" for i in range(5)]  # k0..k2 already indexed

    _run(stack, {"alice": objects})

    assert [item["s3_keys"] for item in stack.backend.items] == [["k3", "k4"]]
    # nb_objects is the submitted count, not the 2 that ran.
    assert _work_counts(_works(stack, "a")) == [(ThreadStatus.DONE, 5, 3)]
    assert _status_works(stack, "alice")[0]["nb_objects"] == 5
    summary = stack.threads.status_store["alice"][_KEY]["summary"]
    assert "5 (3 already indexed)" in summary
    assert stack.tasks.status[1][-1] == TaskStatus.COMPLETED


def test_all_indexed_finishes_without_a_pool_thread(stack):
    objects = [f"s3://a/k{i}" for i in range(3)]

    _run(stack, {"alice": objects})

    assert stack.backend.items == []
    works = _works(stack, "a")
    assert len(works) == 1
    status, work = works[0]
    assert status == ThreadStatus.DONE
    assert work["nb_objects"] == 3
    assert work["nb_skipped_indexed"] == 3
    assert _status_works(stack, "alice")[0]["nb_skipped_indexed"] == 3
    assert stack.tasks.status[1] == [TaskStatus.RUNNING, TaskStatus.COMPLETED]


def test_other_tenants_are_not_filtered_by_alices_index(stack):
    """bob's bucket holds the same keys but its own (empty) table: all run."""
    _run(stack, {"alice": ["s3://a/k0"], "bob": [f"s3://b/k{i}" for i in range(3)]})

    assert [(i["path_to_bucket"], i["s3_keys"]) for i in stack.backend.items] == [
        ("b", ["k0", "k1", "k2"])
    ]
    (_, bob_work), = _works(stack, "b")
    assert bob_work["nb_objects"] == 3
    assert "nb_skipped_indexed" not in bob_work


def test_refused_client_still_errors(stack):
    """bob naming alice's bucket is refused, not 'completed' by her index."""
    _run(stack, {"bob": ["s3://a/k0"]})

    assert stack.backend.items == []
    assert stack.tasks.status[1][-1] == TaskStatus.ERROR
    assert TaskStatus.COMPLETED not in stack.tasks.status[1]


def test_knob_off_indexes_everything(stack, monkeypatch):
    monkeypatch.setenv("UMRS_PREFILTER_INDEXED", "0")
    objects = [f"s3://a/k{i}" for i in range(5)]

    _run(stack, {"alice": objects})

    assert [item["s3_keys"] for item in stack.backend.items] == [[f"k{i}" for i in range(5)]]
    (_, work), = _works(stack, "a")
    assert work["nb_objects"] == 5
    assert "nb_skipped_indexed" not in work


# --- fail open (review U1) ------------------------------------------------------


def test_a_failing_prefilter_indexes_everything(stack, monkeypatch):
    from umrs.server.commands import process_objects

    def _boom(*_args):
        raise RuntimeError("db down")

    monkeypatch.setattr(process_objects, "filter_already_indexed", _boom)
    objects = [f"s3://a/k{i}" for i in range(5)]

    _run(stack, {"alice": objects})

    assert [item["s3_keys"] for item in stack.backend.items] == [[f"k{i}" for i in range(5)]]
    assert stack.tasks.status[1][-1] == TaskStatus.COMPLETED


def test_a_failing_skip_report_still_launches_the_rest_of_the_batch(stack, monkeypatch):
    """The STATUS bookkeeping for a fully skipped bucket ran outside the
    fail-open guard: one diskcache error there killed the collector thread,
    so bob's new objects were never indexed and both tasks stayed LAUNCHED."""

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(stack.threads, "record_skipped_work", _boom)

    _run(stack, {"alice": [f"s3://a/k{i}" for i in range(3)],
                 "bob": [f"s3://b/k{i}" for i in range(2)]})

    assert [(i["path_to_bucket"], i["s3_keys"]) for i in stack.backend.items] == [
        ("b", ["k0", "k1"])
    ]
    assert stack.tasks.status[1][-1] == TaskStatus.COMPLETED  # alice: all indexed
    assert stack.tasks.status[2][-1] == TaskStatus.COMPLETED  # bob


# --- the benchmark log parser reads this server's log (review U1) ----------------


def _parse_server_log_module():
    path = (Path(__file__).resolve().parents[1] / "src" / "scripts" / "benchmarks"
            / "parse_server_log.py")
    spec = importlib.util.spec_from_file_location("bench_parse_server_log", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_server_log_reads_the_lines_this_handler_writes(stack, caplog):
    """parse_server_log.py matched upstream's 'filter_already_indexed:
    skipping N' wording, which this server never logs: n_skipped was always 0."""
    from umrs.server.structured_logging import TEXT_LOG_FORMAT

    with caplog.at_level(logging.INFO):
        _run(stack, {"alice": [f"s3://a/k{i}" for i in range(5)]})  # k0..k2 indexed
    formatter = logging.Formatter(TEXT_LOG_FORMAT)
    lines = [formatter.format(record) for record in caplog.records]

    tasks = _parse_server_log_module().parse_lines(lines)

    (task,) = tasks.values()
    assert (task.task_id, task.user) == ("1", "alice")
    assert (task.n_skipped, task.n_objects_s3) == (3, 2)
    assert task.t_running is not None
    assert task.t_completed is not None
