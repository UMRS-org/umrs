"""upload_dir notifies the server in batches and on abort (plan U1 A3).

Port of upstream 73ab20a0 (batched PROCESS_OBJECTS) and d8b0b364 (bug fix:
flush the pending batch when the data queue fails mid-upload). Drives
``client_umrs._action_s3_transfer`` with fake control/data queues and a fake
store helper: no ZMQ, no S3.
"""

import os
from types import SimpleNamespace

import pytest

client_umrs = pytest.importorskip("client_umrs")


class FakeDataMQ:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.sent = []
        self.closed = False

    def notify_s3(self, s3_object):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("Timeout while sending message")
        self.sent.append(s3_object)

    def close(self):
        self.closed = True


class FakeControlMQ:
    def __init__(self, data_mq):
        self.data_mq = data_mq
        self.notified = []

    def get_data_queue(self, _username, **_kwargs):
        return self.data_mq

    def notify_server(self, _username, _cmd, count, blocking=False):
        self.notified.append((count, blocking))
        return True

    def close(self):
        pass


class FakeBuckets:
    def get(self, _alias, **_kwargs):
        return ("alice", "xp", "text-mod", "/nonexistent/s3cfg", {}, "bucket-x")

    def keys(self):
        return ["alice-text"]


def _store_text_file(_s3, bucket, subdir, in_path, mq, _metadata_args):
    key = os.path.basename(in_path)
    if subdir:
        key = f"{subdir}/{key}"
    mq.notify_s3(f"s3://{bucket}/{key}")
    return key


@pytest.fixture
def upload_env(monkeypatch, tmp_path):
    monkeypatch.setattr(client_umrs, "connect_to_s3", lambda *_a, **_k: (object(), None))
    monkeypatch.setattr(
        client_umrs, "check_bucket_name", lambda *_a, **_k: ("bucket-x", True)
    )
    monkeypatch.setattr(
        client_umrs, "check_and_build_metadata", lambda *_a: {"Metadata": {}}
    )
    monkeypatch.setattr(client_umrs, "modality_category", lambda _m: "text_file")
    monkeypatch.setattr(client_umrs, "s3_store_text_file", _store_text_file)
    data_dir = tmp_path / "data"
    (data_dir / "nested").mkdir(parents=True)
    for i in range(3):
        (data_dir / f"f{i}.txt").write_text(f"file {i}\n")
    for i in range(2):
        (data_dir / "nested" / f"g{i}.txt").write_text(f"nested {i}\n")
    return data_dir


def _run(data_dir, data_mq, blocking=False):
    control = FakeControlMQ(data_mq)
    args = SimpleNamespace(
        action="upload_dir",
        bucket_alias="alice-text",
        path=str(data_dir),
        k=None,
        output="json",
        verbose=False,
        blocking=blocking,
        username=None,
    )
    ctx = SimpleNamespace(
        args=args,
        buckets_cfg=FakeBuckets(),
        control_mq=control,
        admin_token=None,
        connection=None,
    )
    rc = client_umrs._action_s3_transfer(ctx)
    return rc, control


def test_abort_on_full_data_queue_still_notifies_queued_objects(upload_env):
    data_mq = FakeDataMQ(fail_after=3)
    rc, control = _run(upload_env, data_mq)
    assert rc == 1
    assert len(data_mq.sent) == 3
    # The three URIs already in the data queue must be announced, or the
    # server never reads them and the objects stay unindexed.
    assert sum(count for count, _ in control.notified) == 3


def test_upload_dir_notifies_in_batches(upload_env, monkeypatch):
    from umrs.client import upload_batching

    monkeypatch.setattr(upload_batching, "NOTIFY_BATCH_SIZE", 2)
    data_mq = FakeDataMQ()
    rc, control = _run(upload_env, data_mq, blocking=True)
    assert rc == 0
    assert len(data_mq.sent) == 5
    # --blocking applies to every batch, not just the remainder (review U1).
    assert control.notified == [(2, True), (2, True), (1, True)]


def test_upload_dir_single_notify_below_batch_size(upload_env):
    data_mq = FakeDataMQ()
    rc, control = _run(upload_env, data_mq)
    assert rc == 0
    assert control.notified == [(5, False)]
    assert "s3://bucket-x/nested/g1.txt" in data_mq.sent
