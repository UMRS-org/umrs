"""KNN command: request validation, scope, dispatch and wire (plan U1 A6).

Port of upstream 1798ac05 / 487e3efb / e04b3d4e / bd0e6bc1 / 8912c150.
These tests need no database; the search itself is covered against
pgvector in ``test_knn_search_pg.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError
from umrs.server import knn_worker
from umrs.server.commands import project_check
from umrs.server.commands.base import HandlerContext
from umrs.server.commands.knn import KnnHandler
from umrs.server.commands.registry_builder import build_registry
from umrs.server.knn_worker import KnnError, KnnRequest, KnnRequestError
from umrs.server.step_staging import register_bucket_registry
from umrs.shared.auth import USER_ALLOWED_COMMANDS
from umrs.shared.pretty_print import pretty_print_knn
from umrs.shared.storage import user_buckets as ub

ALICE_BUCKET = "alice.xp.images"
BOB_BUCKET = "bob.xp.images"
SENDER = b"sender-1"


def _model(name="clip"):
    return SimpleNamespace(model_name=name, type="local", return_type="embedding")


def _entry(owner, *, store=True, model=None):
    return SimpleNamespace(
        s3_username=owner, data_base=MagicMock(), table_name="t", db_name="d",
        modality_name="image", task="embedding", store_embeddings_flag=store,
        model=model if model is not None else _model(), s3_connection=MagicMock(),
        num_objects=0,
    )


class _Registry:
    """BucketsCfg-shaped: provisioned buckets and their owners."""

    def __init__(self, entries):
        self.entries = entries

    def get(self, name, fields=None, filter=None):
        entry = self.entries.get(name)
        fields = fields or filter
        if entry is None or fields is None:
            return entry
        return tuple(getattr(entry, f) for f in fields)

    def find_by_output_path(self, _output):
        return None, None

    def values(self):
        return list(self.entries.values())


class _NoPersonalBuckets:
    def owner_of(self, _bucket):
        return None

    def can_access(self, _bucket, _user_id):
        return False


@pytest.fixture()
def buckets():
    registry = _Registry({
        ALICE_BUCKET: _entry("alice"),
        BOB_BUCKET: _entry("bob"),
        "alice.xp.scripts": _entry("alice", model=SimpleNamespace(
            model_name="s", type="user_script", return_type="embedding")),
        "alice.xp.raw": _entry("alice", store=False),
    })
    register_bucket_registry(registry)
    ub.set_user_bucket_store(_NoPersonalBuckets())
    yield registry
    register_bucket_registry(None)
    ub.set_user_bucket_store(None)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UMRS_KNN_MAX_K", raising=False)
    monkeypatch.delenv("UMRS_ZMQ_PROJECT_MEMBERSHIP", raising=False)
    project_check.set_membership_resolver(None)
    yield
    project_check.set_membership_resolver(None)


def _args(**overrides):
    body = {"n": 5, "object": "q.png", "bucket": ALICE_BUCKET}
    body.update(overrides)
    return json.dumps({k: v for k, v in body.items() if v is not None})


# --- request validation -------------------------------------------------------


def test_a_valid_request_parses():
    request = knn_worker.parse_knn_request(_args(subdir="pets"))
    assert request == KnnRequest(n=5, s3_object="q.png", bucket=ALICE_BUCKET, subdir="pets")


@pytest.mark.parametrize("k", [0, -1, "abc", 2.5, True, None, [3]])
def test_k_must_be_a_positive_integer(k):
    with pytest.raises(KnnRequestError):
        knn_worker.parse_knn_request(_args(n=k) if k is not None else json.dumps(
            {"object": "q.png", "bucket": ALICE_BUCKET}))


def test_k_is_capped_at_the_default_maximum():
    assert knn_worker.parse_knn_request(_args(n=10_000)).n == knn_worker.DEFAULT_MAX_K == 100


def test_k_cap_follows_the_env_knob(monkeypatch):
    monkeypatch.setenv("UMRS_KNN_MAX_K", "7")
    assert knn_worker.parse_knn_request(_args(n=50)).n == 7
    assert knn_worker.parse_knn_request(_args(n="3")).n == 3


@pytest.mark.parametrize("raw", ["0", "-4", "many"])
def test_an_invalid_knob_keeps_the_default(monkeypatch, raw):
    monkeypatch.setenv("UMRS_KNN_MAX_K", raw)
    assert knn_worker.max_k() == knn_worker.DEFAULT_MAX_K


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("pets", "pets"), ("pets/", "pets"), ("./pets//cats/", "pets/cats"), ("", None),
     (None, None), (".", None), ("./", None)],
)
def test_subdir_is_normalised(raw, expected):
    assert knn_worker.normalise_subdir(raw) == expected


@pytest.mark.parametrize(
    "subdir", ["..", "../bob", "pets/../../x", "/etc", "/", "pets\\..\\x", "a\x00b"]
)
def test_subdir_cannot_escape_the_bucket(subdir):
    with pytest.raises(KnnRequestError):
        knn_worker.parse_knn_request(_args(subdir=subdir))


@pytest.mark.parametrize("obj", ["", "../x.png", "/abs.png", "a/../../b.png", 7])
def test_unsafe_object_keys_are_refused(obj):
    with pytest.raises(KnnRequestError):
        knn_worker.parse_knn_request(_args(object=obj))


@pytest.mark.parametrize("bucket", ["", "a/b", 3])
def test_bucket_must_be_a_name(bucket):
    with pytest.raises(KnnRequestError):
        knn_worker.parse_knn_request(_args(bucket=bucket))


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", None])
def test_malformed_payloads_are_refused(raw):
    with pytest.raises(KnnRequestError):
        knn_worker.parse_knn_request(raw)


# --- bucket scope ---------------------------------------------------------------


def test_own_bucket_is_searchable(buckets):
    assert knn_worker.searchable_entry(buckets, ALICE_BUCKET, "alice") is buckets.entries[ALICE_BUCKET]


@pytest.mark.parametrize("bucket", [BOB_BUCKET, "nobody.xp.images"])
def test_another_users_bucket_reads_like_a_missing_one(buckets, bucket):
    with pytest.raises(KnnError, match="No bucket entry found"):
        knn_worker.searchable_entry(buckets, bucket, "alice")


@pytest.mark.parametrize("bucket", ["alice.xp.scripts", "alice.xp.raw"])
def test_buckets_without_a_plain_embedding_table_are_refused(buckets, bucket):
    with pytest.raises(KnnError, match="no embedding table"):
        knn_worker.searchable_entry(buckets, bucket, "alice")


# --- handler + registry ------------------------------------------------------------


class _MQ:
    """control_mq stand-in: authorization verdict, bound session, sent replies."""

    def __init__(self, *, admin=False, user="alice"):
        self.sent: list[tuple[str, object]] = []
        self._admin = admin
        self._entry = SimpleNamespace(user_id=user, grants=SimpleNamespace(projects=("default",)))

    def send_string(self, message, sender_id=None):
        self.sent.append((message, sender_id))

    def get_auth_context(self, client_id):
        from umrs.shared.auth import AuthContext, UserRole

        role = UserRole.ADMIN if self._admin else UserRole.USER
        return AuthContext(user_id=client_id, username=client_id, role=role)

    def verified_session(self, _zmq_sender_id):
        return self._entry


def _ctx(mq, buckets_cfg):
    task_manager = MagicMock()
    task_manager.tasks_status.create_task.return_value = 41
    return HandlerContext(
        task_manager=task_manager, control_mq=mq,
        server_cfg=SimpleNamespace(buckets_cfg=buckets_cfg), device="cpu",
        embedding_backend=None, configure_file="",
    )


def _dispatch(ctx, body, client_id="alice"):
    payload = (client_id, body if isinstance(body, str) else json.dumps(body), SENDER)
    build_registry().dispatch(TaskCmd.KNN, payload, ctx)


def _replies(mq):
    return [json.loads(message) for message, _sender in mq.sent]


def test_knn_is_a_client_command_behind_authorization():
    registry = build_registry()
    assert isinstance(registry._handlers[TaskCmd.KNN], KnnHandler)
    assert TaskCmd.KNN not in registry._internal_cmds
    assert "KNN" in USER_ALLOWED_COMMANDS


def test_own_bucket_launches_one_search(buckets):
    mq = _MQ()
    ctx = _ctx(mq, buckets)
    _dispatch(ctx, {"n": 3, "object": "pets/q.png", "bucket": ALICE_BUCKET, "subdir": "pets/"})

    launch = ctx.task_manager.active_threads.launch_knn
    launch.assert_called_once()
    args = launch.call_args.args
    assert args[0] == "alice"
    assert args[1] == KnnRequest(n=3, s3_object="pets/q.png", bucket=ALICE_BUCKET, subdir="pets")
    assert args[2] is buckets.entries[ALICE_BUCKET]
    ctx.task_manager.associate_thread_with_task.assert_called_once()
    assert mq.sent == []


def test_another_users_bucket_is_refused_before_launch(buckets):
    mq = _MQ(user="bob")
    ctx = _ctx(mq, buckets)
    _dispatch(ctx, {"n": 3, "object": "q.png", "bucket": ALICE_BUCKET}, client_id="bob")

    ctx.task_manager.active_threads.launch_knn.assert_not_called()
    assert _replies(mq) == [{"status": "error", "message": f"No bucket entry found for path: {ALICE_BUCKET}"}]


def test_a_bad_request_is_refused_before_launch(buckets):
    mq = _MQ()
    ctx = _ctx(mq, buckets)
    _dispatch(ctx, {"n": 0, "object": "q.png", "bucket": ALICE_BUCKET})

    ctx.task_manager.active_threads.launch_knn.assert_not_called()
    (reply,) = _replies(mq)
    assert reply["status"] == "error"
    assert "n must be >= 1" in reply["message"]


def test_a_spoofed_project_never_reaches_the_handler(buckets):
    """ADR-097 D-7: KNN goes through the registry's project check."""
    project_check.set_membership_resolver(lambda _user: ["default"])
    mq = _MQ()
    ctx = _ctx(mq, buckets)
    _dispatch(ctx, {"n": 3, "object": "q.png", "bucket": ALICE_BUCKET, "project_id": "spoofed"})

    ctx.task_manager.tasks_status.create_task.assert_not_called()
    (reply,) = _replies(mq)
    assert reply["code"] == "project_not_member"


def test_an_unauthorized_caller_never_reaches_the_handler(buckets):
    mq = _MQ()
    mq.get_auth_context = lambda _cid: SimpleNamespace(
        can_execute_command=lambda *_a, **_k: False, role=SimpleNamespace(value="guest"))
    ctx = _ctx(mq, buckets)
    _dispatch(ctx, {"n": 3, "object": "q.png", "bucket": ALICE_BUCKET})

    ctx.task_manager.tasks_status.create_task.assert_not_called()
    assert mq.sent == [("UNAUTHORIZED:KNN", SENDER)]


def test_an_overloaded_pool_replies_server_overloaded(buckets):
    mq = _MQ()
    ctx = _ctx(mq, buckets)
    ctx.task_manager.active_threads.launch_knn.side_effect = ServerOverloadedError("full")
    _dispatch(ctx, {"n": 3, "object": "q.png", "bucket": ALICE_BUCKET})

    assert mq.sent == [("SERVER_OVERLOADED", SENDER)]
    ctx.task_manager.tasks_status.update_status.assert_called_with(41, TaskStatus.ERROR)


def test_end_to_end_through_the_real_thread_pool(buckets, tmp_path, monkeypatch):
    """Registry -> KnnHandler -> UMRSThreads.launch_knn -> worker -> reply."""
    from diskcache import Cache
    from umrs.core.UMRSThreads import UMRSThreads

    monkeypatch.setenv("UMRS_SERVICE_LIFECYCLE", "manual")
    seen = []

    def _search(entry, request):
        seen.append((entry, request))
        return knn_worker.KnnOutcome(results=[], cached=False, n_indexed=0)

    monkeypatch.setattr(knn_worker, "search", _search)
    threads = UMRSThreads(Cache(str(tmp_path / "status")), Cache(str(tmp_path / "threads")))
    mq = _MQ()
    ctx = _ctx(mq, buckets)
    ctx.task_manager.active_threads = threads
    try:
        _dispatch(ctx, {"n": 3, "object": "q.png", "bucket": ALICE_BUCKET})
        threads.waitall()
    finally:
        threads.close()

    assert seen == [(buckets.entries[ALICE_BUCKET],
                     KnnRequest(n=3, s3_object="q.png", bucket=ALICE_BUCKET))]
    (reply,) = _replies(mq)
    assert reply["status"] == "ok"
    statuses = [c.args[1] for c in ctx.task_manager.tasks_status.update_status.call_args_list]
    assert statuses[0] == TaskStatus.LAUNCHED
    assert statuses[-1] == TaskStatus.COMPLETED


# --- worker bookkeeping -------------------------------------------------------------


class _Threads:
    def __init__(self):
        self.calls = []

    def update_status_to_done(self):
        self.calls.append("done")

    def record_status_for_users_by_key(self, users, key):
        self.calls.append(("record", tuple(users), key))

    def remove_from_running_list(self):
        self.calls.append("removed")


def _request(**kw):
    base = {"n": 2, "s3_object": "q.png", "bucket": ALICE_BUCKET}
    base.update(kw)
    return KnnRequest(**base)


def test_worker_replies_ok_and_completes(monkeypatch):
    outcome = knn_worker.KnnOutcome(
        results=[{"object": f"{ALICE_BUCKET}/a.png", "score": 0.9}], cached=True, n_indexed=4)
    monkeypatch.setattr(knn_worker, "search", lambda _entry, _req: outcome)
    threads, mq, statuses = _Threads(), _MQ(), []

    knn_worker.knn_search_worker(threads, "alice", _request(), object(), mq, SENDER, "knn",
                                 statuses.append)

    (reply,) = _replies(mq)
    assert reply["status"] == "ok"
    assert reply["results"] == outcome.results
    assert reply["n_indexed"] == 4
    assert reply["cached"] is True
    assert "k/n: 2/4" in reply["summary"]
    assert statuses == [TaskStatus.RUNNING, TaskStatus.COMPLETED]
    assert threads.calls == ["done", ("record", ("alice",), "knn"), "removed"]


@pytest.mark.parametrize("exc", [KnnError("S3 load failed for q.png: gone"), RuntimeError("boom")])
def test_worker_turns_failures_into_an_error_reply(monkeypatch, exc):
    def _fail(_entry, _req):
        raise exc

    monkeypatch.setattr(knn_worker, "search", _fail)
    threads, mq, statuses = _Threads(), _MQ(), []

    knn_worker.knn_search_worker(threads, "alice", _request(), object(), mq, SENDER, "knn",
                                 statuses.append)

    (reply,) = _replies(mq)
    assert reply["status"] == "error"
    assert str(exc) in reply["message"]
    assert statuses == [TaskStatus.RUNNING, TaskStatus.ERROR]
    assert "removed" in threads.calls


# --- wire + task + pretty print ---------------------------------------------------------


def test_request_knn_wire_format():
    from umrs.core.MessageQueue import MessageQueue

    mq = MagicMock()
    mq.role = "client"
    mq.socket.getsockopt.return_value = 8000
    sent = []
    mq.send_string = sent.append
    mq.recv_string = MagicMock(return_value=(None, '{"status": "ok"}'))

    reply = MessageQueue.request_knn(mq, "al:ice", 4, "pets/q.png", ALICE_BUCKET, subdir="pets")

    assert reply == '{"status": "ok"}'
    cmd, client, body = sent[0].split(":", 2)
    assert (cmd, client) == ("KNN", "al_ice")
    assert json.loads(body) == {"n": 4, "object": "pets/q.png", "bucket": ALICE_BUCKET,
                                "subdir": "pets"}
    mq.socket.setsockopt.assert_called_with(__import__("zmq").RCVTIMEO, 8000)


def test_request_knn_is_client_only():
    from umrs.core.MessageQueue import MessageQueue

    with pytest.raises(PermissionError):
        MessageQueue.request_knn(SimpleNamespace(role="server"), "alice", 1, "q", ALICE_BUCKET)


def test_knn_is_parsed_as_a_json_payload():
    from umrs.core.MessageQueue import MessageQueue

    mq = MessageQueue.__new__(MessageQueue)
    body = json.dumps({"n": 1, "object": "a:b.png", "bucket": ALICE_BUCKET})
    assert MessageQueue._CONTROL_PARSERS[TaskCmd.KNN] == "_parse_json_payload"
    cmd, payload = mq._parse_json_payload(["KNN", "alice", *body.split(":")], SENDER)
    assert (cmd, payload) == ("KNN", ("alice", body, SENDER))


def test_knn_creates_a_task_row():
    from umrs.core.UMRSTask import UMRSTask

    tasks = UMRSTask.__new__(UMRSTask)
    tasks.add_task = MagicMock(return_value=7)
    body = _args()
    assert tasks.create_task(TaskCmd.KNN, ("alice", body, SENDER)) == 7
    tasks.add_task.assert_called_once_with("alice", TaskCmd.KNN, body, TaskStatus.SUBMITTED)


def test_pretty_print_reports_k_over_indexed_and_one_row_per_hit():
    text = pretty_print_knn("q.png", ALICE_BUCKET, 2,
                            [{"object": f"{ALICE_BUCKET}/pets/a.png", "score": 0.95},
                             {"object": f"{ALICE_BUCKET}/b.png", "score": -0.2}],
                            0.0123, False, 40)
    assert "k/n: 2/40" in text
    assert "(embedding recomputed)" in text
    assert "0.9500" in text and "a.png" in text and "pets/" not in text.split("\n", 2)[2]
    assert "█████████░" in text
    assert "░░░░░░░░░░" in text  # a negative score is an empty bar, not a crash


def test_pretty_print_without_results():
    text = pretty_print_knn("q.png", ALICE_BUCKET, 3, [], 0.0, True)
    assert "k/n: 3 " in text
    assert "No results." in text
