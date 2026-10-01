"""ADR-077 live leg: the binding probes and the bound client against real stores.

Env-gated (skipped unless both are set; no ``integration`` marker — in
umrs/tests that marker means "needs UMRS_TEST_DB_URL"):

    UMRS_PROJECT_STORAGE_IT_S3_URL   http://AK:SK@host:port   (SeaweedFS/MinIO)
    UMRS_PROJECT_STORAGE_IT_PG_URI   postgresql://u:p@host:port/db (pgvector installed)

The probes create the project bucket pair when absent and leave no objects
behind; Postgres sees only a session TEMP table inside a rolled-back
transaction.
"""

from __future__ import annotations

import os
import secrets

import pytest
from umrs.shared.storage.project_storage import (
    ProjectS3Config,
    StorageBinding,
    parse_postgres_uri,
    parse_s3_url,
    probe_binding,
    probe_postgres,
    probe_s3,
)

S3_URL = os.environ.get("UMRS_PROJECT_STORAGE_IT_S3_URL", "")
PG_URI = os.environ.get("UMRS_PROJECT_STORAGE_IT_PG_URI", "")

pytestmark = pytest.mark.skipif(
    not (S3_URL and PG_URI),
    reason="set UMRS_PROJECT_STORAGE_IT_S3_URL and UMRS_PROJECT_STORAGE_IT_PG_URI",
)


def _bucket() -> str:
    return f"umrs-it-{secrets.token_hex(4)}"


def test_s3_probe_creates_bucket_pair_and_roundtrips() -> None:
    locator = parse_s3_url(S3_URL)
    bucket = _bucket()
    report = probe_s3(locator, bucket=bucket)
    assert report["ok"] is True
    assert sorted(report["buckets_created"]) == sorted([bucket, f"{bucket}-output"])
    again = probe_s3(locator, bucket=bucket)
    assert again["buckets_created"] == []  # idempotent
    client = ProjectS3Config(locator).get_client()
    listed = client.list_objects_v2(Bucket=bucket).get("Contents", [])
    assert listed == []  # the probe object was deleted


def test_bound_config_reads_and_writes_the_bound_store() -> None:
    bucket = _bucket()
    locator = parse_s3_url(f"{S3_URL.rstrip('/')}/{bucket}")
    probe_s3(locator, bucket=bucket)
    cfg = ProjectS3Config(locator)
    assert cfg.bucket == bucket
    client = cfg.get_client()
    client.put_object(Bucket=cfg.bucket, Key="opossum/clip.mp4", Body=b"frame-bytes")
    body = client.get_object(Bucket=cfg.bucket, Key="opossum/clip.mp4")["Body"].read()
    assert body == b"frame-bytes"
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": cfg.bucket, "Key": "opossum/clip.mp4"}, ExpiresIn=60
    )
    assert url.startswith(locator.endpoint_url)
    client.delete_object(Bucket=cfg.bucket, Key="opossum/clip.mp4")


def test_postgres_probe_reports_facts_and_vector_roundtrip() -> None:
    report = probe_postgres(parse_postgres_uri(PG_URI))
    assert report["ok"] is True
    assert report["pgvector"], "the IT database must have the vector extension"
    assert report["vector_roundtrip"] is True
    assert report["server_version"]


def test_probe_binding_reports_both_stores() -> None:
    binding = StorageBinding(
        "opossum-it", s3=parse_s3_url(S3_URL), postgres=parse_postgres_uri(PG_URI)
    )
    report = probe_binding(binding, default_bucket=_bucket())
    assert report["ok"] is True, report
    assert report["s3"]["ok"] and report["postgres"]["ok"]


def test_unreachable_store_is_reported_not_raised() -> None:
    binding = StorageBinding(
        "opossum-it",
        postgres=parse_postgres_uri("postgresql://u:p@127.0.0.1:1/none?connect_timeout=2"),
    )
    report = probe_binding(binding, default_bucket="unused")
    assert report["ok"] is False
    assert report["postgres"]["ok"] is False
    assert "p@" not in report["postgres"]["detail"]  # no credentials in the detail
