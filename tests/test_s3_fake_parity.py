"""Cycle-3 memfix #6 (2026-10-22): the in-memory S3 fake answers like the real
store on the paths the bounded knowledge reads and the conditional ingest
copy depend on.

Found by the mem review (``.stab-cycle3/memreview/memreview-v6-0``): the fakes
answered a range over an empty object 200 ``bytes 0--1/0`` where SeaweedFS
4.47 (the umrs-front pin) answers 416 ``InvalidRange``, so the
``InvalidRange`` branch of ``_fetch_artifact_prefix`` was never run; and the
fake ``copy_object`` ignored ``CopySourceIfMatch``, so the 412 path was only
reached by injecting the error.

Every case runs on ``tests.s3_ranged.FakeS3`` and, when
``UMRS_TEST_S3_ENDPOINT`` names a real store (a throwaway SeaweedFS, e.g.
``podman run --rm -p 127.0.0.1:58333:8333 docker.io/chrislusf/seaweedfs:4.47
server -dir=/data -s3``), on that store too, in a fresh bucket it deletes.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError
from tests.s3_ranged import FakeS3
from umrs.jobs import knowledge_adapters as ka

_ENDPOINT_ENV = "UMRS_TEST_S3_ENDPOINT"
_TEN = b"0123456789"


def _real_store() -> tuple[Any, str]:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ[_ENDPOINT_ENV],
        aws_access_key_id=os.environ.get("UMRS_TEST_S3_ACCESS_KEY", "any"),
        aws_secret_access_key=os.environ.get("UMRS_TEST_S3_SECRET_KEY", "any"),
        region_name="us-east-1",
    )
    bucket = f"parity-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    return client, bucket


def _empty_bucket(client: Any, bucket: str) -> None:
    listed = client.list_objects_v2(Bucket=bucket).get("Contents") or []
    for item in listed:
        client.delete_object(Bucket=bucket, Key=item["Key"])
    client.delete_bucket(Bucket=bucket)


@pytest.fixture(params=["fake", "seaweedfs"])
def store(request) -> Iterator[tuple[Any, str]]:
    if request.param == "fake":
        yield FakeS3(), "kb"
        return
    if not os.environ.get(_ENDPOINT_ENV):
        pytest.skip(f"{_ENDPOINT_ENV} names no real S3 store")
    client, bucket = _real_store()
    try:
        yield client, bucket
    finally:
        _empty_bucket(client, bucket)


def _error(call: Any) -> tuple[str, int]:
    with pytest.raises(ClientError) as caught:
        call()
    response = caught.value.response
    return response["Error"]["Code"], response["ResponseMetadata"]["HTTPStatusCode"]


@pytest.mark.parametrize("range_header", ["bytes=0-1023", "bytes=0-0"])
def test_a_range_over_an_empty_object_is_invalid_range(store, range_header):
    client, bucket = store
    client.put_object(Bucket=bucket, Key="empty", Body=b"")
    get = lambda: client.get_object(Bucket=bucket, Key="empty", Range=range_header)  # noqa: E731
    assert _error(get) == ("InvalidRange", 416)


def test_a_range_starting_past_the_end_is_invalid_range(store):
    client, bucket = store
    client.put_object(Bucket=bucket, Key="ten", Body=_TEN)
    get = lambda: client.get_object(Bucket=bucket, Key="ten", Range="bytes=10-20")  # noqa: E731
    assert _error(get) == ("InvalidRange", 416)


@pytest.mark.parametrize(
    ("range_header", "content_range", "body"),
    [
        ("bytes=0-1023", "bytes 0-9/10", _TEN),
        ("bytes=4-7", "bytes 4-7/10", b"4567"),
        ("bytes=8-100", "bytes 8-9/10", b"89"),
    ],
)
def test_a_range_names_the_part_and_the_whole_size(store, range_header, content_range, body):
    client, bucket = store
    client.put_object(Bucket=bucket, Key="ten", Body=_TEN)
    answer = client.get_object(Bucket=bucket, Key="ten", Range=range_header)
    assert answer["ContentRange"] == content_range
    assert answer["Body"].read() == body
    assert answer["ContentLength"] == len(body)


def test_a_single_part_etag_is_the_quoted_md5(store):
    client, bucket = store
    put = client.put_object(Bucket=bucket, Key="ten", Body=_TEN)
    md5 = f'"{hashlib.md5(_TEN).hexdigest()}"'  # noqa: S324 — the S3 ETag
    assert put["ETag"] == md5
    assert client.get_object(Bucket=bucket, Key="ten")["ETag"] == md5


def test_a_conditional_copy_on_the_sources_etag_copies(store):
    client, bucket = store
    etag = client.put_object(Bucket=bucket, Key="staged", Body=_TEN)["ETag"]
    client.copy_object(
        Bucket=bucket,
        Key="stored",
        CopySource={"Bucket": bucket, "Key": "staged"},
        CopySourceIfMatch=etag,
    )
    assert client.get_object(Bucket=bucket, Key="stored")["Body"].read() == _TEN


def test_a_conditional_copy_on_another_etag_is_refused_and_stores_nothing(store):
    client, bucket = store
    client.put_object(Bucket=bucket, Key="staged", Body=_TEN)
    stale = f'"{hashlib.md5(b"the bytes ingest read").hexdigest()}"'  # noqa: S324

    def copy() -> None:
        client.copy_object(
            Bucket=bucket,
            Key="stored",
            CopySource={"Bucket": bucket, "Key": "staged"},
            CopySourceIfMatch=stale,
        )

    assert _error(copy) == ("PreconditionFailed", 412)
    head = lambda: client.head_object(Bucket=bucket, Key="stored")  # noqa: E731
    assert _error(head)[1] == 404


def test_the_bounded_prefix_read_on_the_store(store):
    client, bucket = store
    client.put_object(Bucket=bucket, Key="empty", Body=b"")
    client.put_object(Bucket=bucket, Key="ten", Body=_TEN)
    assert ka._fetch_artifact_prefix(client, bucket, "empty", 1024) == (b"", 0)
    assert ka._fetch_artifact_prefix(client, bucket, "ten", 4) == (b"0123", 10)
    assert ka._fetch_artifact_prefix(client, bucket, "ten", 1024) == (_TEN, 10)
