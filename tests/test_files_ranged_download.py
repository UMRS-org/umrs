"""ADR-115 S1: ranged, inline-capable ``GET|HEAD /umrs/files/download`` and
the one bucket→store resolver (D-1, D-3).

Before S1 the route always answered a full 200 ``attachment`` with no
``Accept-Ranges``/``ETag``, refused HEAD (405), and the owned-bucket
download/list routes always used the server's default store, while the
workflow runner wrote a bound project's buckets in its bound store.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.server import step_staging
from umrs.shared.storage import project_storage as ps
from umrs.shared.storage import s3_config as s3_config_mod
from umrs.shared.storage import user_buckets as ub

PAYLOAD = bytes(range(256)) * 4  # 1024 bytes
N = len(PAYLOAD)
ETAG = '"0123456789abcdef"'
MODIFIED = datetime(2026, 11, 13, 9, 30, 0, tzinfo=timezone.utc)
MODIFIED_HTTP = "Fri, 13 Nov 2026 09:30:00 GMT"
OWNED = "alice.opossum.videos-output"
S3_URL = "http://AKIDEXAMPLE:s3cr3t-key@10.129.0.31:28900"


class _FakeS3:
    """Records every call; honours ``Range`` like S3 (inclusive bytes)."""

    def __init__(self, objects: dict[tuple[str, str], tuple[bytes, str]] | None = None):
        self.objects = dict(objects or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def head_object(self, **kw):
        self.calls.append(("head_object", kw))
        if (kw["Bucket"], kw["Key"]) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body, ctype = self.objects[(kw["Bucket"], kw["Key"])]
        return {
            "ContentLength": len(body),
            "ContentType": ctype,
            "ETag": ETAG,
            "LastModified": MODIFIED,
        }

    def get_object(self, **kw):
        self.calls.append(("get_object", kw))
        body, _ctype = self.objects[(kw["Bucket"], kw["Key"])]
        rng = kw.get("Range")
        if rng:
            first, last = (int(x) for x in rng.removeprefix("bytes=").split("-"))
            body = body[first : last + 1]
        return {"Body": io.BytesIO(body)}

    def list_objects_v2(self, **kw):
        self.calls.append(("list_objects_v2", kw))
        keys = [k for (b, k) in self.objects if b == kw["Bucket"]]
        return {"Contents": [{"Key": k, "Size": 1} for k in sorted(keys)], "IsTruncated": False}


class _Cfg:
    def __init__(self, s3: _FakeS3, bucket: str = "r4-data") -> None:
        self.bucket = bucket
        self._s3 = s3

    def get_client(self):
        return self._s3


class _OwnerStore:
    """User-bucket store where alice owns ``alice.opossum.videos``."""

    def get(self, _bucket):
        return None

    def owner_of(self, bucket):
        return "alice" if ub.input_bucket_of(bucket) == "alice.opossum.videos" else None

    def can_access(self, bucket, user_id):
        return self.owner_of(bucket) == user_id


class _FakeResolver(ps.ProjectStorageResolver):
    def __init__(self, bindings=None, fail=False):
        super().__init__(pool_getter=lambda: None)
        self._bindings = bindings or {}
        self._fail = fail

    def binding(self, project_id, *, pool=None):  # noqa: ARG002 — resolver signature
        if self._fail:
            raise ps.ProjectStorageUnavailable("registry unreadable")
        return self._bindings.get(project_id)


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    ub.set_user_bucket_store(_OwnerStore())  # type: ignore[arg-type]
    monkeypatch.setattr(step_staging, "_bucket_registry", None)
    ps.set_project_storage_resolver(_FakeResolver())
    yield
    ps.set_project_storage_resolver(None)
    ub.set_user_bucket_store(None)


@pytest.fixture
def default_s3(monkeypatch):
    import umrs.api.files as files_api

    fake = _FakeS3(
        {
            ("r4-data-output", "runs/wf-1/clip.mp4"): (PAYLOAD, "video/mp4"),
            ("r4-data-output", "runs/wf-1/page.html"): (b"<html>", "text/html"),
            ("r4-data-output", "runs/wf-1/run.sh"): (b"#!/bin/sh", "application/x-sh"),
            (OWNED, "runs/wf-1/clip.mp4"): (PAYLOAD, "video/mp4"),
        }
    )
    monkeypatch.setattr(files_api, "get_s3_config", lambda: _Cfg(fake))
    return fake


def _client(project: str = "default", user: str = "alice", cfg_s3: _FakeS3 | None = None):
    import umrs.api.files as files_api

    app = FastAPI()
    app.include_router(files_api.router)
    app.dependency_overrides[files_api.get_project_id] = lambda: project
    if cfg_s3 is not None:
        app.dependency_overrides[files_api.get_project_s3_config] = lambda: _Cfg(cfg_s3)
    else:  # the real project dependency, over a fake default store
        app.dependency_overrides[s3_config_mod.get_s3_config] = files_api.get_s3_config
    client = TestClient(app)
    client.headers.update({"X-User-ID": user})
    return client


def _get(s3: _FakeS3, key: str = "runs/wf-1/clip.mp4", method: str = "GET", **kw):
    params = {"bucket": "r4-data-output", **kw.pop("params", {})}
    return _client(cfg_s3=s3).request(
        method, f"/umrs/files/download/{key}", params=params, headers=kw.pop("headers", {})
    )


def _gets(s3: _FakeS3) -> list[dict[str, Any]]:
    return [kw for name, kw in s3.calls if name == "get_object"]


# -- D-1: ranges ------------------------------------------------------------------


def test_first_hundred_bytes_answer_206_from_a_ranged_s3_read(default_s3):
    resp = _get(default_s3, headers={"Range": "bytes=0-99"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-99/{N}"
    assert resp.headers["content-length"] == "100"
    assert resp.content == PAYLOAD[:100]
    assert _gets(default_s3)[0]["Range"] == "bytes=0-99"


def test_suffix_range_returns_the_last_bytes(default_s3):
    resp = _get(default_s3, headers={"Range": "bytes=-10"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes {N - 10}-{N - 1}/{N}"
    assert resp.content == PAYLOAD[-10:]


def test_suffix_longer_than_the_object_is_the_whole_object(default_s3):
    resp = _get(default_s3, headers={"Range": f"bytes=-{N * 2}"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-{N - 1}/{N}"
    assert resp.content == PAYLOAD


def test_open_and_overlong_ranges_are_clamped(default_s3):
    open_ended = _get(default_s3, headers={"Range": f"bytes={N - 4}-"})
    assert open_ended.status_code == 206 and open_ended.content == PAYLOAD[-4:]
    overlong = _get(default_s3, headers={"Range": f"bytes=1000-{N * 5}"})
    assert overlong.headers["content-range"] == f"bytes 1000-{N - 1}/{N}"


@pytest.mark.parametrize("spec", [f"bytes={N}-", f"bytes={N + 5}-{N + 9}", "bytes=-0"])
def test_unsatisfiable_range_is_416_and_never_reads_the_body(default_s3, spec):
    resp = _get(default_s3, headers={"Range": spec})
    assert resp.status_code == 416
    assert resp.headers["content-range"] == f"bytes */{N}"
    assert _gets(default_s3) == []


@pytest.mark.parametrize(
    "spec", ["bytes=0-1,5-9", "bytes=abc", "items=0-9", "bytes=9-1", "bytes=-", "bytes"]
)
def test_multi_or_malformed_range_answers_the_full_200(default_s3, spec):
    resp = _get(default_s3, headers={"Range": spec})
    assert resp.status_code == 200
    assert resp.content == PAYLOAD
    assert "Range" not in _gets(default_s3)[0]


def test_plain_get_advertises_ranges_and_validators(default_s3):
    resp = _get(default_s3)
    assert resp.status_code == 200
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["etag"] == ETAG
    assert resp.headers["last-modified"] == MODIFIED_HTTP
    assert resp.headers["content-length"] == str(N)
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "private, no-store"


def test_stale_if_range_answers_the_full_200(default_s3):
    resp = _get(default_s3, headers={"Range": "bytes=0-9", "If-Range": '"stale-etag"'})
    assert resp.status_code == 200 and resp.content == PAYLOAD
    weak = _get(default_s3, headers={"Range": "bytes=0-9", "If-Range": f"W/{ETAG}"})
    assert weak.status_code == 200
    old_date = _get(
        default_s3, headers={"Range": "bytes=0-9", "If-Range": "Thu, 01 Jan 2026 00:00:00 GMT"}
    )
    assert old_date.status_code == 200


def test_current_if_range_keeps_the_range(default_s3):
    by_tag = _get(default_s3, headers={"Range": "bytes=0-9", "If-Range": ETAG})
    assert by_tag.status_code == 206 and by_tag.content == PAYLOAD[:10]
    by_date = _get(default_s3, headers={"Range": "bytes=0-9", "If-Range": MODIFIED_HTTP})
    assert by_date.status_code == 206


def test_head_answers_headers_without_reading_the_body(default_s3):
    resp = _get(default_s3, method="HEAD")
    assert resp.status_code == 200
    assert resp.headers["content-length"] == str(N)
    assert resp.headers["accept-ranges"] == "bytes" and resp.headers["etag"] == ETAG
    assert resp.content == b""
    ranged = _get(default_s3, method="HEAD", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and ranged.headers["content-length"] == "100"
    assert _gets(default_s3) == []


# -- D-1: disposition ---------------------------------------------------------------


def test_inline_is_honoured_for_allowlisted_media(default_s3):
    resp = _get(default_s3, params={"disposition": "inline"})
    assert resp.headers["content-disposition"] == 'inline; filename="clip.mp4"'
    assert resp.headers["content-type"] == "video/mp4"


@pytest.mark.parametrize("key", ["runs/wf-1/page.html", "runs/wf-1/run.sh"])
def test_inline_is_refused_for_other_types(default_s3, key):
    resp = _get(default_s3, key=key, params={"disposition": "inline"})
    assert resp.headers["content-disposition"].startswith("attachment;")
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_media_defaults_to_attachment(default_s3):
    assert _get(default_s3).headers["content-disposition"].startswith("attachment;")


# -- authorization before S3 ------------------------------------------------------------


def test_foreign_bucket_with_a_range_is_403_before_any_s3_call(default_s3):
    resp = _client(user="bob").get(
        "/umrs/files/download/runs/wf-1/clip.mp4",
        params={"bucket": OWNED},
        headers={"Range": "bytes=0-99"},
    )
    assert resp.status_code == 403
    assert default_s3.calls == []


def test_owner_streams_a_range_from_an_owned_bucket(default_s3):
    resp = _client().get(
        "/umrs/files/download/runs/wf-1/clip.mp4",
        params={"bucket": OWNED, "disposition": "inline"},
        headers={"Range": "bytes=10-19"},
    )
    assert resp.status_code == 206 and resp.content == PAYLOAD[10:20]
    assert resp.headers["content-disposition"].startswith("inline;")


def test_no_response_carries_a_presigned_url(default_s3):
    responses = [
        _get(default_s3),
        _get(default_s3, headers={"Range": "bytes=0-9"}),
        _get(default_s3, method="HEAD"),
        _get(default_s3, headers={"Range": f"bytes={N}-"}),
    ]
    for resp in responses:
        blob = (str(dict(resp.headers)) + resp.content.decode("latin-1")).lower()
        assert "x-amz-signature" not in blob and "credential=" not in blob


# -- D-3: one bucket→store resolver ------------------------------------------------------


@pytest.fixture
def bound_s3(monkeypatch):
    bound = _FakeS3({(OWNED, "runs/wf-1/clip.mp4"): (PAYLOAD, "video/mp4")})
    monkeypatch.setattr(ps, "s3_client_for", lambda _locator, **_kw: bound)
    binding = ps.StorageBinding("opossum", s3=ps.parse_s3_url(S3_URL))
    ps.set_project_storage_resolver(_FakeResolver({"opossum": binding}))
    return bound


def test_bound_project_downloads_from_its_bound_store(default_s3, bound_s3):
    resp = _client(project="opossum").get(
        "/umrs/files/download/runs/wf-1/clip.mp4",
        params={"bucket": OWNED},
        headers={"Range": "bytes=0-3"},
    )
    assert resp.status_code == 206 and resp.content == PAYLOAD[:4]
    assert _gets(bound_s3) and not default_s3.calls


def test_bound_project_lists_its_bound_store(default_s3, bound_s3):
    resp = _client(project="opossum").get("/umrs/files/list", params={"bucket": OWNED})
    assert resp.status_code == 200
    assert [o["key"] for o in resp.json()["objects"]] == ["runs/wf-1/clip.mp4"]
    assert [c for c, _ in bound_s3.calls] == ["list_objects_v2"]
    assert not default_s3.calls


def test_unbound_project_keeps_the_default_store(default_s3, bound_s3):
    resp = _client(project="default").get("/umrs/files/list", params={"bucket": OWNED})
    assert resp.status_code == 200
    assert not bound_s3.calls and default_s3.calls


def test_unreadable_binding_is_503_and_never_falls_back(default_s3):
    ps.set_project_storage_resolver(_FakeResolver(fail=True))
    client = _client(project="opossum")
    listed = client.get("/umrs/files/list", params={"bucket": OWNED})
    got = client.get("/umrs/files/download/runs/wf-1/clip.mp4", params={"bucket": OWNED})
    assert listed.status_code == 503 and got.status_code == 503
    assert default_s3.calls == []


def test_runner_and_files_routes_resolve_the_same_store(bound_s3):
    import threading
    from unittest.mock import MagicMock

    from umrs.api.files import bucket_store
    from umrs.server.workflow_runner import _resolve_s3_client

    record = MagicMock()
    record.params = {"bucket": OWNED}
    runner_client, _bucket = _resolve_s3_client(record, threading.Lock(), project_id="opossum")
    files_client = bucket_store(OWNED, "opossum", _Cfg(_FakeS3())).get_client()
    assert runner_client is files_client is bound_s3


def test_bound_project_creates_personal_buckets_on_its_bound_store(default_s3, bound_s3):
    from umrs.api import buckets as buckets_api
    from umrs.api._verified_headers import CallerIdentity, get_verified_caller

    class _Store:
        def __init__(self):
            self.rows = {}

        def get(self, name):
            return self.rows.get(name)

        def add(self, entry):
            self.rows[entry.bucket] = entry
            return True

    made: list[str] = []

    def _head_bucket(**_kw):
        raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    bound_s3.head_bucket = _head_bucket  # type: ignore[attr-defined]
    bound_s3.create_bucket = lambda **kw: made.append(kw["Bucket"])  # type: ignore[attr-defined]
    ub.set_user_bucket_store(_Store())  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(buckets_api.router)
    app.dependency_overrides[get_verified_caller] = lambda: CallerIdentity(
        user_id="alice", project_id="opossum", is_admin=False
    )
    resp = TestClient(app).post("/umrs/buckets", json={"project": "opossum", "alias": "clips"})
    assert resp.status_code == 201, resp.text
    assert made == ["alice.opossum.clips", "alice.opossum.clips-output"]


# Live regression 2026-11-13: botocore returns LastModified with dateutil's
# tzutc(), and email.utils.format_datetime(usegmt=True) accepts only
# datetime.timezone.utc, so every download answered 500 after the S1 rollout.
def test_last_modified_from_botocore_tzutc_is_an_http_date():
    from dateutil.tz import tzutc
    from umrs.api._ranged_download import _http_date

    value = datetime(2026, 11, 13, 9, 30, 0, tzinfo=tzutc())
    assert _http_date(value) == "Fri, 13 Nov 2026 09:30:00 GMT"


def test_last_modified_naive_or_offset_is_normalised_to_gmt():
    from datetime import timedelta

    from umrs.api._ranged_download import _http_date

    assert _http_date(datetime(2026, 11, 13, 9, 30, 0)) == "Fri, 13 Nov 2026 09:30:00 GMT"
    jst = timezone(timedelta(hours=9))
    assert _http_date(datetime(2026, 11, 13, 18, 30, 0, tzinfo=jst)) == "Fri, 13 Nov 2026 09:30:00 GMT"
    assert _http_date("not a datetime") == ""
