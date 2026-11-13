"""S3 upload retry and skip-unchanged (port of upstream 6632dfb0 + b951866f; plan U1 A2).

Retries are botocore's alone (standard mode, ``UMRS_S3_MAX_ATTEMPTS`` set in
``connect_to_s3``); ``s3_upload`` adds no loop of its own.
Hermetic: a fake client stands in for boto3, no network and no real store.
"""

import hashlib

import pytest
from botocore.exceptions import ClientError
from umrs.shared.storage import s3 as s3_mod
from umrs.shared.storage import s3_upload


def _client_error(code, status=400, op="PutObject"):
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        op,
    )


class FakeS3:
    def __init__(self, failures=(), head=None, head_error=None):
        self.failures = list(failures)
        self.head = head
        self.head_error = head_error
        self.uploads = []
        self.heads = 0

    def head_object(self, **_params):
        self.heads += 1
        if self.head_error is not None:
            raise self.head_error
        if self.head is None:
            raise _client_error("404", 404, "HeadObject")
        return self.head

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        if self.failures:
            raise self.failures.pop(0)
        self.uploads.append((Filename, Bucket, Key, ExtraArgs))


@pytest.fixture
def sleeps(monkeypatch):
    """Records any sleep, so a test can assert that no hand-written backoff ran."""
    calls = []
    monkeypatch.setattr("time.sleep", calls.append)
    monkeypatch.delenv(s3_upload.SKIP_UNCHANGED_ENV, raising=False)
    monkeypatch.delenv("UMRS_S3_MAX_ATTEMPTS", raising=False)
    return calls


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"hello umrs\n")
    return path


def _head_for(path, metadata=None, etag=None, **extra):
    data = path.read_bytes()
    head = {
        "ETag": f'"{etag or hashlib.md5(data).hexdigest()}"',
        "ContentLength": len(data),
        "Metadata": metadata or {},
    }
    head.update(extra)
    return head


# --- one retry layer: botocore's -------------------------------------------


def test_a_failed_upload_is_not_retried_on_top_of_botocore(sleeps, local_file):
    """Review U1: a hand-written loop stacked on botocore's own retries turned
    one file into (loop attempts x botocore attempts) requests."""
    fake = FakeS3(failures=[_client_error("SlowDown", 503)])
    with pytest.raises(ClientError):
        s3_upload.upload_local_file(fake, "b", "k", str(local_file), skip_unchanged=False)
    assert fake.uploads == []
    assert fake.failures == []  # exactly one upload_file call
    assert sleeps == []


def _s3cfg(tmp_path):
    path = tmp_path / "s3cfg"
    path.write_text(
        "[default]\nhost_base = store:8333\naccess_key = a\nsecret_key = s\n"
        "bucket_location = us-east-1\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def boto_clients(monkeypatch):
    built = []

    class _Client:
        def list_buckets(self):
            return {}

    def _client(*_args, **kwargs):
        built.append(kwargs)
        return _Client()

    monkeypatch.setattr(s3_mod.boto3, "client", _client)
    return built


@pytest.mark.parametrize(("raw", "expected"), [(None, 5), ("2", 2), ("10", 10)])
def test_connect_to_s3_retries_in_botocore_standard_mode(
    tmp_path, boto_clients, monkeypatch, raw, expected
):
    if raw is None:
        monkeypatch.delenv(s3_upload.MAX_ATTEMPTS_ENV, raising=False)
    else:
        monkeypatch.setenv(s3_upload.MAX_ATTEMPTS_ENV, raw)
    s3_mod.connect_to_s3(_s3cfg(tmp_path))
    (kwargs,) = boto_clients
    assert kwargs["config"].retries == {"total_max_attempts": expected, "mode": "standard"}


@pytest.mark.parametrize("raw", ["0", "11", "junk", "2.5"])
def test_attempts_knob_out_of_bounds_keeps_the_default(monkeypatch, caplog, raw):
    monkeypatch.setenv(s3_upload.MAX_ATTEMPTS_ENV, raw)
    with caplog.at_level("WARNING"):
        assert s3_upload.s3_max_attempts() == s3_upload.DEFAULT_MAX_ATTEMPTS == 5
    assert any(s3_upload.MAX_ATTEMPTS_ENV in r.getMessage() for r in caplog.records)


# --- skip unchanged -------------------------------------------------------


def test_skip_when_single_part_etag_matches(sleeps, local_file):
    meta = {"Metadata": {"Owner": "alice"}}
    fake = FakeS3(head=_head_for(local_file, metadata={"owner": "alice"}))
    sent = s3_upload.upload_local_file(fake, "b", "k", str(local_file), meta)
    assert sent is False
    assert fake.uploads == []


def test_multipart_etag_is_not_an_md5_so_upload(sleeps, local_file):
    md5 = hashlib.md5(local_file.read_bytes()).hexdigest()
    # Even when the hex part happens to equal the local MD5, "-N" means multipart.
    fake = FakeS3(head=_head_for(local_file, etag=f"{md5}-3"))
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file)) is True
    assert len(fake.uploads) == 1


def test_changed_content_uploads(sleeps, local_file):
    fake = FakeS3(head=_head_for(local_file, etag="0" * 32))
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file)) is True
    assert len(fake.uploads) == 1


def test_changed_metadata_uploads(sleeps, local_file):
    fake = FakeS3(head=_head_for(local_file, metadata={"owner": "bob"}))
    meta = {"Metadata": {"owner": "alice"}}
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file), meta) is True


def test_sse_kms_etag_uploads(sleeps, local_file):
    fake = FakeS3(head=_head_for(local_file, ServerSideEncryption="aws:kms"))
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file)) is True


def test_head_forbidden_falls_back_to_upload(sleeps, local_file):
    fake = FakeS3(head_error=_client_error("403", 403, "HeadObject"))
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file)) is True
    assert len(fake.uploads) == 1


def test_skip_can_be_disabled(sleeps, local_file, monkeypatch):
    monkeypatch.setenv(s3_upload.SKIP_UNCHANGED_ENV, "off")
    fake = FakeS3(head=_head_for(local_file))
    assert s3_upload.upload_local_file(fake, "b", "k", str(local_file)) is True
    assert fake.heads == 0


def test_missing_local_file_raises(sleeps, tmp_path):
    with pytest.raises(FileNotFoundError):
        s3_upload.upload_local_file(FakeS3(), "b", "k", str(tmp_path / "nope.txt"))


# --- s3_store_* wiring ----------------------------------------------------


class _RecordingMQ:
    def __init__(self):
        self.notified = []

    def notify_s3(self, obj):
        self.notified.append(obj)


def test_store_text_file_skips_unchanged_but_still_notifies(sleeps, local_file):
    fake = FakeS3(head=_head_for(local_file))
    mq = _RecordingMQ()
    key = s3_mod.s3_store_text_file(
        fake, "bucket", "sub", str(local_file), mq, {"Metadata": {}}
    )
    assert key == "sub/doc.txt"
    assert fake.uploads == []
    assert mq.notified == ["s3://bucket/sub/doc.txt"]
