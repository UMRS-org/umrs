"""H-1 leftovers (2026-10-18): PUT /umrs/buckets/{bucket}/access is no oracle.

A non-owner of an existing bucket got 403 bucket_denied while an unknown
bucket was 404, so a caller could probe guessable ``<user>.<project>.<alias>``
names. A caller the access list does not admit now gets files.not_found, like
GET .../access; one the list admits (it already sees the bucket) keeps 403.
Strict mode, gateway-signed tuples and a forged tuple.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.buckets import router as buckets_router
from umrs.shared.storage import user_buckets as ub

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1x-bucket-oracle-secret"
BUCKET = "alice.proj-a.videos"


def _signed(user: str, project: str = "proj-a", admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


class _Store:
    def __init__(self):
        self.rows = {BUCKET: ub.UserBucket(BUCKET, "alice", "proj-a", "videos", ("user:bob",))}
        self.writes: list[tuple[str, list[str]]] = []

    def get(self, bucket):
        return self.rows.get(ub.input_bucket_of(bucket))

    def can_access(self, bucket, user_id):
        row = self.get(bucket)
        return row is not None and (row.owner_id == user_id or f"user:{user_id}" in row.access)

    def set_access(self, bucket, access):
        self.writes.append((bucket, list(access)))
        row = self.get(bucket)
        self.rows[row.bucket] = ub.UserBucket(
            row.bucket, row.owner_id, row.project, row.alias, tuple(access)
        )
        return True


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setattr("umrs.api.audit_events.emit_audit", lambda *_a, **_k: None)
    yield
    ub.set_user_bucket_store(None)


@pytest.fixture
def store() -> _Store:
    s = _Store()
    ub.set_user_bucket_store(s)
    return s


@pytest.fixture
def client(store) -> TestClient:  # noqa: ARG001 — installs the store
    app = FastAPI()
    app.include_router(buckets_router)
    return TestClient(app)


def _put(client: TestClient, bucket: str, headers: dict):
    return client.put(f"/umrs/buckets/{bucket}/access", json={"access": []}, headers=headers)


def _detail(response, bucket: str) -> str:
    """The detail with the requested name masked (it echoes the request)."""
    return str(response.json()["detail"]).replace(repr(bucket), "<bucket>")


def test_stranger_cannot_tell_an_existing_bucket_from_an_unknown_one(client, store):
    carol = _signed("carol", "proj-b")
    existing = _put(client, BUCKET, carol)
    unknown = _put(client, "alice.proj-a.nothing", carol)
    assert existing.status_code == unknown.status_code == 404
    assert _detail(existing, BUCKET) == _detail(unknown, "alice.proj-a.nothing")
    assert store.writes == []


def test_put_matches_get_for_a_stranger(client):
    carol = _signed("carol", "proj-b")
    get = client.get(f"/umrs/buckets/{BUCKET}/access", headers=carol)
    put = _put(client, BUCKET, carol)
    assert get.status_code == put.status_code == 404
    assert _detail(get, BUCKET) == _detail(put, BUCKET) == "Bucket <bucket> not found"


def test_admitted_reader_still_gets_403(client, store):
    r = _put(client, BUCKET, _signed("bob"))
    assert r.status_code == 403, r.text
    assert "Only the owner" in r.text
    assert store.writes == []


def test_owner_and_admin_still_change_access(client, store):
    assert _put(client, BUCKET, _signed("alice")).status_code == 200
    assert _put(client, BUCKET, _signed("ops", "ops", admin=True)).status_code == 200
    assert store.writes == [(BUCKET, []), (BUCKET, [])]


def test_forged_tuple_is_401_for_existing_and_unknown(client, store):
    forged = {**_signed("carol", "proj-b"), "X-User-ID": "alice"}
    for bucket in (BUCKET, "alice.proj-a.nothing"):
        assert _put(client, bucket, forged).status_code == 401
    assert store.writes == []
