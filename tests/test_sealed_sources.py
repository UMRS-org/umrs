"""Sealed source routes (R1-1): push/list/unpush + audit + identity rules.

Dual-backend irrelevant (Postgres required) — run against the shared pool
with migration applied; fake-meta-store pattern is impossible here because
sealing is a real write. Hermetic per run: unique owner names.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.user_sources import router
from umrs.jobs.secret_broker import SecretBroker
from umrs.security.seal import generate_recipient_keypair

from r4_contracts.handshake_hmac import sign_claims

SECRET = "sealed-secret"
_MIGRATIONS = Path(__file__).resolve().parents[1] / "src/umrs/migrations"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _public_dns(host, port, family, socktype):
    return [(2, 1, 6, "", ("104.18.2.161", 0))]


@pytest.fixture
def env(monkeypatch, tmp_path):
    pytest.importorskip("psycopg2")
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("UMRS_SECRET_BROKER_BACKEND", "pg")
    priv, pub = generate_recipient_keypair()
    monkeypatch.setenv("UMRS_SEAL_PUBKEY_B64", pub)
    monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)

    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")

    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            for mig in (
                "0085_secret_leases.sql",
                "0086_user_sources_sealed.sql",
                "0124_secret_leases_venue.sql",
            ):
                cur.execute((_MIGRATIONS / mig).read_text())
    finally:
        pool.putconn(conn)

    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    app.state._test_pool = pool
    yield app
    pool.closeall()


def _owner() -> str:
    return f"test-owner-{uuid.uuid4().hex[:8]}"


def _push_body(**over):
    body = {
        "name": "openai-main",
        "provider_kind": "openai",
        "base_url": "https://api.example-provider.com/v1",
        "api_key": "sk-test-" + uuid.uuid4().hex[:24],
        "allow_paths": [],
    }
    body.update(over)
    return body


class TestPush:
    def test_push_creates_metadata_row_with_fingerprint(self, env):
        owner = _owner()
        with TestClient(env) as client:
            body = _push_body()
            response = client.post(
                "/umrs/user-sources/sealed", json=body, headers=_headers(owner, "p")
            )
            assert response.status_code == 201, response.text
            payload = response.json()
            assert payload["allow_host"] == "https://api.example-provider.com"
            assert payload["key_fingerprint"].startswith("sha256:")
            assert payload["seal_key_id"].startswith("v1:")

            # Row is unsealable with the recipient private key via the store path.
            from umrs.security.seal import unseal_credential
            from umrs.security.sealed_sources import SealedSourceStore

            store = SealedSourceStore(env.state._test_pool)
            full = store.fetch_for_unseal(payload["id"])
            assert full is not None
            # Real-store row carries every field the proxy needs (live-leg
            # lesson: a base_url omission was invisible in fake-store tests).
            assert full.base_url == "https://api.example-provider.com/v1"
            assert full.allow_host == "https://api.example-provider.com"
            assert (
                unseal_credential(
                    type(
                        "B",
                        (),
                        {
                            "ciphertext": full.ciphertext,
                            "nonce": full.nonce,
                            "dek_wrapped": full.dek_wrapped,
                            "seal_key_id": "",
                        },
                    )(),
                    privkey_b64=os.environ["UMRS_UNSEAL_PRIVKEY_B64"],
                    owner=owner,
                    name="openai-main",
                    provider_kind="openai",
                    allow_host="https://api.example-provider.com",
                )
                == body["api_key"]
            )

    def test_push_rejects_bad_base_url(self, env):
        owner = _owner()
        with TestClient(env) as client:
            response = client.post(
                "/umrs/user-sources/sealed",
                json=_push_body(base_url="http://insecure.example.org"),
                headers=_headers(owner, "p"),
            )
            assert response.status_code == 400
            assert response.json()["detail"]["reason"] == "scheme_not_https"

    def test_push_rejects_unknown_provider_kind(self, env):
        owner = _owner()
        with TestClient(env) as client:
            response = client.post(
                "/umrs/user-sources/sealed",
                json=_push_body(provider_kind="mistral"),
                headers=_headers(owner, "p"),
            )
            assert response.status_code == 400

    def test_no_identity_401(self, env):
        with TestClient(env) as client:
            assert client.post("/umrs/user-sources/sealed", json=_push_body()).status_code == 401

    def test_re_push_marks_replaced_and_only_one_live_row(self, env):
        owner = _owner()
        with TestClient(env) as client:
            first = client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner, "p")
            )
            assert first.status_code == 201
            second = client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner, "p")
            )
            assert second.status_code == 201
            assert second.json()["replaced"] is True
            listing = client.get("/umrs/user-sources/sealed", headers=_headers(owner, "p")).json()
            live = [row for row in listing if row["name"] == "openai-main"]
            assert len(live) == 1


class TestList:
    def test_list_returns_metadata_only_no_ciphertext(self, env):
        owner = _owner()
        with TestClient(env) as client:
            client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner, "p")
            )
            listing = client.get("/umrs/user-sources/sealed", headers=_headers(owner, "p")).json()
        assert len(listing) == 1
        row = listing[0]
        assert row["key_fingerprint"].startswith("sha256:")
        for forbidden in ("ciphertext", "nonce", "dek_wrapped", "api_key"):
            assert forbidden not in row
        assert row.get("active_leases", True)

    def test_list_is_owner_scoped(self, env):
        owner_a, owner_b = _owner(), _owner()
        with TestClient(env) as client:
            client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner_a, "p")
            )
            assert (
                client.get("/umrs/user-sources/sealed", headers=_headers(owner_b, "p")).json() == []
            )


class TestUnpush:
    def test_tombstone_and_revoke_cascade(self, env):
        owner = _owner()
        with TestClient(env) as client:
            pushed = client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner, "p")
            ).json()
            source_id = pushed["id"]

            # Lease referencing the source (broker pg mode, same pool).
            broker = SecretBroker(db_pool=env.state._test_pool)
            lease, _token = broker.issue_lease(
                "job-1",
                "inference-token",
                "inference",
                owner_id=owner,
                session_id="sess-1",
                source_ids=(source_id,),
                default_source_id=source_id,
                egress_ip="10.99.0.4",
            )

            response = client.delete(
                "/umrs/user-sources/sealed/openai-main", headers=_headers(owner, "p")
            )
            assert response.status_code == 200
            assert response.json()["leases_revoked"] == 1
            assert not broker.get_lease(lease.lease_id).is_valid

            listing = client.get("/umrs/user-sources/sealed", headers=_headers(owner, "p")).json()
            assert listing == []

            audit = client.get("/umrs/user-sources/audit", headers=_headers(owner, "p")).json()
            actions = [row["action"] for row in audit]
            assert actions[0] == "unpush"
            assert "push" in actions

    def test_unpush_missing_source_404s(self, env):
        owner = _owner()
        with TestClient(env) as client:
            assert (
                client.delete(
                    "/umrs/user-sources/sealed/nope", headers=_headers(owner, "p")
                ).status_code
                == 404
            )

    def test_cross_owner_unpush_404s(self, env):
        owner_a, owner_b = _owner(), _owner()
        with TestClient(env) as client:
            client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner_a, "p")
            )
            assert (
                client.delete(
                    "/umrs/user-sources/sealed/openai-main", headers=_headers(owner_b, "p")
                ).status_code
                == 404
            )


class TestSealUnconfigured:
    def test_push_503_without_pubkey(self, env, monkeypatch):
        monkeypatch.delenv("UMRS_SEAL_PUBKEY_B64", raising=False)
        from umrs.security.seal import recipient_keys

        recipient_keys()._pub = None
        owner = _owner()
        with TestClient(env) as client:
            response = client.post(
                "/umrs/user-sources/sealed", json=_push_body(), headers=_headers(owner, "p")
            )
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
        # Restore for subsequent tests in the same process.
        _, pub2 = generate_recipient_keypair()
        os.environ["UMRS_SEAL_PUBKEY_B64"] = pub2
        recipient_keys()._pub = pub2


@pytest.mark.parametrize("name", ["site", "SITE", "Site"])
def test_push_refuses_the_reserved_site_name(name):
    """ADR-092 slice 9: ``site`` names the venue's own models (slice 8b);
    refused before anything is sealed or stored (no database needed)."""
    from fastapi import FastAPI
    from umrs.api import user_sources

    app = FastAPI()
    app.include_router(user_sources.router)
    app.dependency_overrides[user_sources.get_caller_scope] = lambda: SimpleNamespace(
        user_id="alice", project_id="p", is_admin=False
    )
    with TestClient(app) as client:
        response = client.post("/umrs/user-sources/sealed", json=_push_body(name=name))
    assert response.status_code == 400
    assert "reserved" in response.text
    assert response.headers["x-platform-error-code"] == "usersources.bad_request"
