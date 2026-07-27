"""Regression tests for presigned-URL client caching (per-user credentials).

Bug: ``_get_presign_client`` cached the external-endpoint boto3 client keyed
by endpoint alone.  The first user to trigger presigning "won" the cache and
every subsequent user's presigned URLs were signed with the first user's
credentials, producing HTTP 403 on their own buckets.
"""

from types import SimpleNamespace

import umrs.shared.storage.s3_presign as s3_presign


def _fake_s3_client(access_key: str, secret_key: str):
    creds = SimpleNamespace(access_key=access_key, secret_key=secret_key)
    return SimpleNamespace(_request_signer=SimpleNamespace(_credentials=creds))


def test_presign_client_is_cached_per_user(monkeypatch):
    monkeypatch.setattr(s3_presign, "_presign_clients", {})

    created = []

    def fake_boto3_client(_service, **kwargs):
        created.append(kwargs["aws_access_key_id"])
        return SimpleNamespace(access_key=kwargs["aws_access_key_id"])

    monkeypatch.setattr(s3_presign.boto3, "client", fake_boto3_client)

    endpoint = "http://minio-host:19000"
    user_a = _fake_s3_client("user_a", "secret_a")
    user_b = _fake_s3_client("user_b", "secret_b")

    client_a = s3_presign._get_presign_client(user_a, endpoint_override=endpoint)
    client_b = s3_presign._get_presign_client(user_b, endpoint_override=endpoint)

    # Each user must get a client signed with their own credentials.
    assert client_a.access_key == "user_a"
    assert client_b.access_key == "user_b"
    assert created == ["user_a", "user_b"]

    # Same user + endpoint hits the cache (no new client).
    client_a2 = s3_presign._get_presign_client(user_a, endpoint_override=endpoint)
    assert client_a2 is client_a
    assert created == ["user_a", "user_b"]


def test_presign_client_no_endpoint_returns_original(monkeypatch):
    monkeypatch.setattr(s3_presign, "_presign_clients", {})
    monkeypatch.delenv("S3_PRESIGN_ENDPOINT", raising=False)
    monkeypatch.setattr(s3_presign, "_get_presign_endpoint", lambda endpoint_override=None: "")

    user_a = _fake_s3_client("user_a", "secret_a")
    assert s3_presign._get_presign_client(user_a) is user_a
