"""S3 upload retry and skip-unchanged (port of upstream 6632dfb0 + b951866f; plan U1 A2).

Hermetic: a fake client stands in for boto3, no network and no real store.
"""

import hashlib

import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
    SSLError,
)
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


def _wrapped_upload_error(inner):
    """Mimic boto3: S3UploadFailedError raised while handling a ClientError."""
    try:
        raise inner
    except ClientError as exc:
        try:
            raise S3UploadFailedError(f"Failed to upload x to b/k: {exc}")
        except S3UploadFailedError as wrapped:
            return wrapped


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
    calls = []
    monkeypatch.setattr(s3_upload, "_sleep", calls.append)
    for name in (
        s3_upload.ATTEMPTS_ENV,
        s3_upload.BACKOFF_ENV,
        s3_upload.SKIP_UNCHANGED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
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


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        _client_error("SlowDown", 503),
        _client_error("RequestTimeout", 400),
        _client_error("IncompleteBody", 400),
        _client_error("InternalError", 500),
        _client_error("Whatever", 502),
        _client_error("TooManyRequests", 429),
        _wrapped_upload_error(_client_error("ServiceUnavailable", 503)),
        S3UploadFailedError(
            "Failed to upload: An error occurred (SlowDown) when calling"
        ),
        EndpointConnectionError(endpoint_url="http://store:8333"),
        ReadTimeoutError(endpoint_url="http://store:8333"),
    ],
)
def test_transient_errors_are_retryable(exc):
    assert s3_upload.is_transient_upload_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        _client_error("AccessDenied", 403),
        _client_error("InvalidAccessKeyId", 403),
        _client_error("SignatureDoesNotMatch", 403),
        _client_error("NoSuchBucket", 404),
        _client_error("NotImplemented", 501),
        _wrapped_upload_error(_client_error("AccessDenied", 403)),
        S3UploadFailedError("Failed to upload: An error occurred (AccessDenied)"),
        NoCredentialsError(),
        SSLError(endpoint_url="https://store", error="bad cert"),
    ],
)
def test_auth_and_client_errors_are_not_retryable(exc):
    assert not s3_upload.is_transient_upload_error(exc)


# --- retry ----------------------------------------------------------------


def test_retry_recovers_from_transient_failures(sleeps, local_file):
    fake = FakeS3(
        failures=[
            _client_error("SlowDown", 503),
            EndpointConnectionError(endpoint_url="u"),
        ]
    )
    s3_upload.upload_with_retry(fake, str(local_file), "b", "k", {"Metadata": {}})
    assert len(fake.uploads) == 1
    assert sleeps == [1.0, 2.0]


def test_retry_is_bounded_and_reraises(sleeps, local_file, monkeypatch):
    monkeypatch.setenv(s3_upload.ATTEMPTS_ENV, "2")
    monkeypatch.setenv(s3_upload.BACKOFF_ENV, "0.5")
    errors = [_client_error("SlowDown", 503) for _ in range(5)]
    fake = FakeS3(failures=errors)
    with pytest.raises(ClientError):
        s3_upload.upload_with_retry(fake, str(local_file), "b", "k")
    assert sleeps == [0.5]
    assert len(fake.failures) == 3  # exactly two attempts consumed


def test_auth_error_is_not_retried(sleeps, local_file):
    fake = FakeS3(failures=[_wrapped_upload_error(_client_error("AccessDenied", 403))])
    with pytest.raises(S3UploadFailedError):
        s3_upload.upload_with_retry(fake, str(local_file), "b", "k")
    assert sleeps == []
    assert fake.uploads == []


@pytest.mark.parametrize(
    ("raw", "expected"), [("", 3), ("1", 1), ("0", 1), ("99", 10), ("junk", 3)]
)
def test_attempts_env_is_bounded(monkeypatch, raw, expected):
    monkeypatch.setenv(s3_upload.ATTEMPTS_ENV, raw)
    assert s3_upload.upload_attempts() == expected


def test_backoff_doubles_and_caps():
    delays = [s3_upload.retry_delay_s(n, 1.0) for n in range(1, 8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


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


def test_store_text_file_retries_transient_upload(sleeps, local_file):
    fake = FakeS3(failures=[_wrapped_upload_error(_client_error("SlowDown", 503))])
    mq = _RecordingMQ()
    s3_mod.s3_store_text_file(
        fake, "bucket", None, str(local_file), mq, {"Metadata": {}}
    )
    assert [u[2] for u in fake.uploads] == ["doc.txt"]
    assert sleeps == [1.0]
