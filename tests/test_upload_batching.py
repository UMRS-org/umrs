"""Unit tests for umrs.client.upload_batching (plan U1 A3)."""

import os

import pytest
from umrs.client.upload_batching import (
    BatchedNotifier,
    collect_upload_files,
    record_uploaded,
    report_nothing_uploaded,
)


class _DataMQ:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.sent = []

    def notify_s3(self, obj):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("Timeout while sending message")
        self.sent.append(obj)


class _Notify:
    def __init__(self, ok=True, error=None):
        self.ok = ok
        self.error = error
        self.calls = []

    def __call__(self, count, blocking):
        self.calls.append((count, blocking))
        if self.error is not None:
            raise self.error
        return self.ok


def test_batches_flush_at_size_and_final_uses_blocking():
    notify = _Notify()
    notifier = BatchedNotifier(_DataMQ(), notify, batch_size=3)
    with notifier.flushing(blocking=True):
        for i in range(7):
            notifier.notify_s3(f"s3://b/{i}")
    assert notify.calls == [(3, False), (3, False), (1, True)]
    assert notifier.total == 7
    assert notifier.pending == 0
    assert notifier.ok is True


def test_no_notify_when_nothing_queued():
    notify = _Notify()
    with BatchedNotifier(_DataMQ(), notify).flushing():
        pass
    assert notify.calls == []


def test_partial_text_file_counts_only_accepted_uris():
    # A multi-line text file pushes one URI per line; the 4th push fails.
    notify = _Notify()
    notifier = BatchedNotifier(_DataMQ(fail_after=3), notify, batch_size=100)
    with pytest.raises(RuntimeError, match="Timeout"), notifier.flushing():
        for line in range(6):
            notifier.notify_s3(f"s3://b/line{line}.txt")
    assert notify.calls == [(3, False)]


def test_abort_flush_failure_does_not_mask_original_error():
    notify = _Notify(error=TimeoutError("server gone"))
    notifier = BatchedNotifier(_DataMQ(fail_after=1), notify)
    with (
        pytest.raises(RuntimeError, match="Timeout while sending"),
        notifier.flushing(),
    ):
        notifier.notify_s3("s3://b/0")
        notifier.notify_s3("s3://b/1")
    assert notify.calls == [(1, False)]
    assert notifier.ok is False


def test_rejected_batch_marks_not_ok_but_keeps_going():
    notify = _Notify(ok=False)
    notifier = BatchedNotifier(_DataMQ(), notify, batch_size=2)
    with notifier.flushing():
        for i in range(3):
            notifier.notify_s3(f"s3://b/{i}")
    assert notify.calls == [(2, False), (1, False)]
    assert notifier.ok is False


def test_collect_upload_files_maps_subdirs_and_reports_broken_links(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "a" / "mid.txt").write_text("x")
    (tmp_path / "a" / "b" / "deep.txt").write_text("x")
    os.symlink(tmp_path / "missing", tmp_path / "dangling.txt")

    entries, nb_seen, broken = collect_upload_files(str(tmp_path), "exp")
    by_name = {os.path.basename(p): sub for p, sub in entries}
    assert by_name == {"top.txt": "exp", "mid.txt": "exp/a", "deep.txt": "exp/a/b"}
    assert nb_seen == 4
    assert len(broken) == 1 and broken[0].startswith(str(tmp_path / "dangling.txt"))

    entries, _, _ = collect_upload_files(str(tmp_path), None)
    assert {sub for _, sub in entries} == {None, "a", "a/b"}


def test_record_uploaded_handles_single_list_and_none():
    keys, files = [], []
    record_uploaded("k1", 1, "/f1", keys, files)
    record_uploaded(["l1", "l2"], 2, "/f2", keys, files)
    record_uploaded(None, 0, "/f3", keys, files)
    assert keys == ["k1", "l1", "l2"]
    assert files == ["/f1", "/f2", "/f2"]


@pytest.mark.parametrize(
    ("nb_seen", "broken", "needle"),
    [
        (0, [], "contains no files"),
        (2, ["/x -> /gone"], "broken symlinks"),
        (2, [], "none were accepted for modality 'text-mod'"),
    ],
)
def test_report_nothing_uploaded(capsys, nb_seen, broken, needle):
    report_nothing_uploaded("/data", "text-mod", nb_seen, broken)
    assert needle in capsys.readouterr().out
