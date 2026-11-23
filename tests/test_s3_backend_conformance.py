"""ADR-055 object-store backend conformance — live endpoint only.

Runs the platform's real S3 usage patterns against whichever backend answers
at ``UMRS_S3_CONFORMANCE_ENDPOINT``: presigned PUT/GET exactly as the vision
service and files API do, multipart, bucket policy, the admin plane through
the backend adapter, and STS AssumeRole with the ADR-043 session policy.
Adding a backend later = make this file pass against it.

Gate (all skipped otherwise):
    UMRS_S3_CONFORMANCE_ENDPOINT      e.g. http://127.0.0.1:8333
    UMRS_S3_CONFORMANCE_ACCESS_KEY    admin access key
    UMRS_S3_CONFORMANCE_SECRET_KEY    admin secret key
Optional:
    UMRS_S3_CONFORMANCE_IAM_ENDPOINT  SeaweedFS IAM API (default host:8111)
    S3_BACKEND                        assert the backend instead of sniffing it

Local run (SeaweedFS 4.47), see docs/adr/055-object-store-backend-seaweedfs.md.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import uuid

import pytest

ENDPOINT = os.environ.get("UMRS_S3_CONFORMANCE_ENDPOINT", "").strip()
ACCESS_KEY = os.environ.get("UMRS_S3_CONFORMANCE_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("UMRS_S3_CONFORMANCE_SECRET_KEY", "")

# Not marked ``integration`` on purpose: in umrs/tests that marker means
# "needs UMRS_TEST_DB_URL"; this suite gates on its own endpoint variables.
pytestmark = [
    pytest.mark.skipif(
        not (ENDPOINT and ACCESS_KEY and SECRET_KEY),
        reason="set UMRS_S3_CONFORMANCE_ENDPOINT/_ACCESS_KEY/_SECRET_KEY to run",
    ),
]


def _http(url: str, method: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read()


def _denied(fn) -> str:
    from botocore.exceptions import ClientError

    try:
        fn()
    except ClientError as exc:
        return exc.response["Error"]["Code"]
    raise AssertionError("operation was allowed but must be denied")


# SeaweedFS stores identities in the filer and its S3 gateway picks up an IAM
# API change asynchronously (filer metadata subscription). A data-plane call
# made right after CreateAccessKey / PutUserPolicy / DeleteUser can still see
# the old identity set: CI 2026-11-23 (UMRS-org run 45170945890) got
# AccessDenied on the PutObject right after grant_bucket_access, and a local
# SeaweedFS 4.47 throttled to 0.05 CPU fails 6 runs in 20 the same way (plus
# InvalidAccessKeyId for the just-issued key). Production has no such
# call-after-change within the server (the s3cfg goes to a remote client), so
# the wait lives here: poll until the change is visible, bounded.
_IAM_PROPAGATION_TIMEOUT_S = 15.0


def _error_code(fn) -> str | None:
    """Run *fn*; return its S3 error code, or None when it succeeds."""
    from botocore.exceptions import ClientError

    try:
        fn()
    except ClientError as exc:
        return exc.response["Error"]["Code"]
    return None


def _settled_code(fn, settled, timeout: float = _IAM_PROPAGATION_TIMEOUT_S) -> str | None:
    """Poll *fn* until ``settled(code)`` holds or *timeout* passes; return the last code."""
    import time

    deadline = time.monotonic() + timeout
    code = _error_code(fn)
    while not settled(code) and time.monotonic() < deadline:
        time.sleep(0.1)
        code = _error_code(fn)
    return code


def test_settled_code_polls_until_the_change_is_visible():
    from botocore.exceptions import ClientError

    answers = iter(["InvalidAccessKeyId", "AccessDenied", None])

    def call():
        code = next(answers)
        if code:
            raise ClientError({"Error": {"Code": code}}, "PutObject")

    assert _settled_code(call, lambda code: code is None) is None


def test_settled_code_gives_up_at_the_timeout():
    from botocore.exceptions import ClientError

    def call():
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")

    assert _settled_code(call, lambda code: code is None, timeout=0.2) == "AccessDenied"


@pytest.fixture(scope="module")
def admin():
    from umrs.shared.storage.s3_config import S3Config

    cfg = S3Config()
    cfg.endpoint_url, cfg.access_key, cfg.secret_key = ENDPOINT, ACCESS_KEY, SECRET_KEY
    return cfg.get_client()


@pytest.fixture(scope="module")
def backend_name(admin):
    from umrs.shared.storage.backends import resolve_backend_name

    header = admin.list_buckets()["ResponseMetadata"]["HTTPHeaders"].get("server", "")
    return resolve_backend_name(header)


@pytest.fixture(scope="module")
def profile(backend_name):
    from umrs.shared.storage.backends import get_profile

    return get_profile(backend_name)


@pytest.fixture
def bucket(admin):
    # 2026-09-16: the 09-14 4.47 image treated the STS inline session policy
    # as union-with-role (session scopes alone authorized); the current 4.47
    # rebuild evaluates the role policy as the ceiling (safer — inline can
    # only narrow). The production shape is one pinned bucket with sessions
    # scoped deeper, so the conformance bucket must be role-covered: pin a
    # name via UMRS_S3_CONFORMANCE_BUCKET that matches the iam.json render
    # (S3_BUCKET), random only for evaluator-union backends.
    name = os.environ.get("UMRS_S3_CONFORMANCE_BUCKET") or f"umrs-conf-{uuid.uuid4().hex[:8]}"
    try:
        admin.create_bucket(Bucket=name)
    except admin.exceptions.BucketAlreadyOwnedByYou:
        # The pinned name survives a run cancelled before teardown (CI
        # 2026-09-24: 7 setup errors on the self-hosted runner) — reuse it,
        # emptied, instead of failing every later run.
        _empty_bucket(admin, name)
    yield name
    _empty_bucket(admin, name)
    admin.delete_bucket(Bucket=name)


def _empty_bucket(admin, name):
    for page in admin.get_paginator("list_objects_v2").paginate(Bucket=name):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            admin.delete_objects(Bucket=name, Delete={"Objects": objs, "Quiet": True})


@pytest.fixture
def admin_s3cfg(tmp_path):
    from urllib.parse import urlsplit

    parts = urlsplit(ENDPOINT)
    path = tmp_path / "admin.s3cfg"
    path.write_text(
        "[default]\n"
        f"host_base = {parts.netloc}\n"
        f"host_bucket = {parts.netloc}\n"
        "bucket_location = us-east-1\n"
        f"use_https = {parts.scheme == 'https'}\n"
        "check_ssl_certificate = False\n"
        f"access_key = {ACCESS_KEY}\n"
        f"secret_key = {SECRET_KEY}\n"
        "alias = conformance\n",
        encoding="utf-8",
    )
    iam = os.environ.get("UMRS_S3_CONFORMANCE_IAM_ENDPOINT", "").strip()
    if iam:
        os.environ["S3_IAM_ENDPOINT_URL"] = iam
    return str(path)


# --------------------------------------------------------------- data plane --
def test_health_path_answers(profile):
    statuses = []
    for path in profile.health_paths:
        try:
            statuses.append(_http(ENDPOINT + path, "GET")[0])
        except urllib.error.HTTPError as exc:
            statuses.append(exc.code)
    assert any(200 <= s < 300 for s in statuses), statuses


def test_presigned_put_signed_content_type_then_get_roundtrip(admin, bucket):
    payload = os.urandom(200_000)
    put_url = admin.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": "attempt/x/result.pkl",
            "ContentType": "application/octet-stream",
        },
        ExpiresIn=600,
    )
    assert _http(put_url, "PUT", payload, {"Content-Type": "application/octet-stream"})[0] == 200
    with pytest.raises(urllib.error.HTTPError) as exc:
        _http(put_url, "PUT", payload, {"Content-Type": "text/plain"})
    assert exc.value.code == 403  # signed Content-Type enforced

    get_url = admin.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": "attempt/x/result.pkl"}, ExpiresIn=600
    )
    status, body = _http(get_url, "GET")
    assert status == 200 and hashlib.sha256(body).digest() == hashlib.sha256(payload).digest()


def test_presigned_put_with_metadata_roundtrip(admin, bucket):
    url = admin.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": "proj/img.png",
            "ContentType": "image/png",
            "Metadata": {"project-id": "p1", "uploaded-by": "u1"},
        },
        ExpiresIn=600,
    )
    headers = {
        "Content-Type": "image/png",
        "x-amz-meta-project-id": "p1",
        "x-amz-meta-uploaded-by": "u1",
    }
    assert _http(url, "PUT", b"\x89PNG" + os.urandom(2048), headers)[0] == 200
    assert admin.head_object(Bucket=bucket, Key="proj/img.png")["Metadata"] == {
        "project-id": "p1",
        "uploaded-by": "u1",
    }


def test_presigned_get_expired_and_tampered_rejected(admin, bucket):
    import time

    admin.put_object(Bucket=bucket, Key="k", Body=b"v")
    url = admin.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": "k"}, ExpiresIn=1
    )
    time.sleep(2)
    with pytest.raises(urllib.error.HTTPError):
        _http(url, "GET")
    good = admin.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": "k"}, ExpiresIn=600
    )
    with pytest.raises(urllib.error.HTTPError):
        _http(good[:-4] + "0000", "GET")


def test_multipart_upload(admin, bucket):
    mp = admin.create_multipart_upload(Bucket=bucket, Key="big.bin")
    parts = []
    for n in (1, 2):
        r = admin.upload_part(
            Bucket=bucket,
            Key="big.bin",
            PartNumber=n,
            UploadId=mp["UploadId"],
            Body=os.urandom(5 * 1024 * 1024),
        )
        parts.append({"PartNumber": n, "ETag": r["ETag"]})
    admin.complete_multipart_upload(
        Bucket=bucket, Key="big.bin", UploadId=mp["UploadId"], MultipartUpload={"Parts": parts}
    )
    assert admin.head_object(Bucket=bucket, Key="big.bin")["ContentLength"] == 10 * 1024 * 1024


def test_bucket_policy_accepted(admin, bucket):
    import json

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}"],
            }
        ],
    }
    code = admin.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))["ResponseMetadata"][
        "HTTPStatusCode"
    ]
    assert code in (200, 204)


# -------------------------------------------------------------- admin plane --
def test_admin_plane_user_lifecycle(admin, backend_name, bucket, admin_s3cfg, tmp_path):
    import configparser
    import shutil

    from umrs.shared.storage.backends import get_backend
    from umrs.shared.storage.s3_config import S3Config

    if backend_name == "minio" and not shutil.which("mc"):
        pytest.skip("MinIO adapter needs the mc binary on PATH")
    backend = get_backend(backend_name)
    username = f"conf-{uuid.uuid4().hex[:6]}"
    other = f"{bucket}-other"
    admin.create_bucket(Bucket=other)
    try:
        _, content = backend.create_user(username, str(tmp_path / f"s3cfg_{username}"), admin_s3cfg)
        cfg = configparser.ConfigParser()
        cfg.read_string(content)
        ucfg = S3Config()
        ucfg.endpoint_url = ENDPOINT
        ucfg.access_key, ucfg.secret_key = (
            cfg.get("default", "access_key"),
            cfg.get("default", "secret_key"),
        )
        user = ucfg.get_client()

        def put_granted():
            return user.put_object(Bucket=bucket, Key="x", Body=b"1")

        # Each step waits for SeaweedFS to apply the IAM change (see above).
        key_known = _settled_code(put_granted, lambda code: code != "InvalidAccessKeyId")
        assert key_known == "AccessDenied"
        backend.grant_bucket_access(username, bucket, admin_s3cfg)
        assert _settled_code(put_granted, lambda code: code is None) is None
        assert [o["Key"] for o in user.list_objects_v2(Bucket=bucket)["Contents"]] == ["x"]
        assert _denied(lambda: user.put_object(Bucket=other, Key="x", Body=b"1")) == "AccessDenied"

        backend.delete_user(username, admin_s3cfg)
        gone = _settled_code(
            lambda: user.list_objects_v2(Bucket=bucket), lambda code: code is not None
        )
        assert gone in {"InvalidAccessKeyId", "AccessDenied"}
    finally:
        backend.remove_bucket(other, admin_s3cfg)


# ------------------------------------------------------------------- STS --
def test_sts_session_policy_scopes_attempt(admin, bucket, profile):
    from umrs.jobs.attempt_credentials import mint_attempt_credentials
    from umrs.jobs.sts_federation import assume_role_for_attempt
    from umrs.shared.storage.s3_config import S3Config

    record = mint_attempt_credentials("job-conf", "attempt-conf", 1, ttl_sec=900)
    prefix = record.scope_prefixes[0]
    admin.put_object(Bucket=bucket, Key=f"{prefix}in.bin", Body=b"in")
    admin.put_object(Bucket=bucket, Key="jobs/other/attempt_1/secret.bin", Body=b"no")

    creds = assume_role_for_attempt(
        record,
        bucket=bucket,
        endpoint_url=ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        role_arn=(os.getenv("S3_STS_ROLE_ARN") or profile.sts_role_arn),
    )
    import boto3
    from botocore.config import Config as BotoConfig

    session = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        aws_session_token=creds["session_token"],
        region_name=S3Config().region,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    assert session.get_object(Bucket=bucket, Key=f"{prefix}in.bin")["Body"].read() == b"in"
    assert (
        session.put_object(Bucket=bucket, Key=f"{prefix}out.bin", Body=b"out")["ResponseMetadata"][
            "HTTPStatusCode"
        ]
        == 200
    )
    assert (
        _denied(lambda: session.get_object(Bucket=bucket, Key="jobs/other/attempt_1/secret.bin"))
        == "AccessDenied"
    )
    assert (
        _denied(lambda: session.put_object(Bucket=bucket, Key="jobs/other/attempt_1/x", Body=b"x"))
        == "AccessDenied"
    )
    if profile.sts_list_prefix_condition_enforced:
        assert (
            _denied(lambda: session.list_objects_v2(Bucket=bucket, Prefix="jobs/other/"))
            == "AccessDenied"
        )
    # presigned GET signed by the session identity works for in-scope keys
    url = session.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": f"{prefix}in.bin"}, ExpiresIn=300
    )
    assert _http(url, "GET")[0] == 200
