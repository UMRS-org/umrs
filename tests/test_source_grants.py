"""R3-1: source grants end-to-end (store + routes on real pool + revoke cascade)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.source_grants import router
from umrs.grants.source_store import SourceGrantStore
from umrs.jobs.secret_broker import SecretBroker
from umrs.security.seal import generate_recipient_keypair, seal_credential
from umrs.security.sealed_sources import SealedSourceStore

from r4_contracts.handshake_hmac import sign_claims

SECRET = "grants-secret"
_MIGRATIONS = Path(__file__).resolve().parents[1] / "src/umrs/migrations"


def _headers(user: str, project: str = "p", admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _public_dns(host, port, family, socktype):
    return [(2, 1, 6, "", ("104.18.2.161", 0))]


@pytest.fixture
def env(monkeypatch):
    pytest.importorskip("psycopg2")
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("UMRS_SECRET_BROKER_BACKEND", "pg")

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
                "0087_source_grants.sql",
            ):
                cur.execute((_MIGRATIONS / mig).read_text())
    finally:
        pool.putconn(conn)

    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    app.state._pool = pool
    yield app
    pool.closeall()


def _push_source(pool, owner: str, name: str = "main") -> str:
    priv, pub = generate_recipient_keypair()
    blob = seal_credential(
        "sk-x",
        pubkey_b64=pub,
        owner=owner,
        name=name,
        provider_kind="openai",
        allow_host="https://api.example-provider.com",
    )
    store = SealedSourceStore(pool)
    return store.insert(
        owner=owner,
        name=name,
        provider_kind="openai",
        base_url="https://api.example-provider.com/v1",
        allow_host="https://api.example-provider.com",
        allow_paths=("files",),
        key_fingerprint="sha256:x",
        seal_key_id=blob.seal_key_id,
        dek_wrapped=blob.dek_wrapped,
        nonce=blob.nonce,
        ciphertext=blob.ciphertext,
    )


def _ids():
    owner, grantee = (f"owner-{uuid.uuid4().hex[:6]}", f"grantee-{uuid.uuid4().hex[:6]}")
    return owner, grantee


class TestGrantLifecycle:
    def test_create_list_revoke(self, env):
        owner, grantee = _ids()
        _push_source(env.state._pool, owner)
        with TestClient(env) as client:
            created = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": grantee, "allow_paths": ["files"]},
                headers=_headers(owner),
            )
            assert created.status_code == 201, created.text
            grant_id = created.json()["grant_id"]

            mine = client.get("/umrs/user-sources/grants/mine", headers=_headers(owner)).json()
            assert any(g["grant_id"] == grant_id for g in mine)
            for_me = client.get("/umrs/user-sources/grants/for-me", headers=_headers(grantee))
            assert any(g["grant_id"] == grant_id for g in for_me.json())

            # Lease-authorization path: grantee holds a live grant.
            store = SourceGrantStore(env.state._pool)
            assert store.has_live_grant(user_id=grantee, source_id=created.json()["source_id"])

            revoked = client.post(
                f"/umrs/user-sources/grants/{grant_id}/revoke", headers=_headers(owner)
            )
            assert revoked.status_code == 200
            assert not store.has_live_grant(user_id=grantee, source_id=created.json()["source_id"])

    def test_create_only_on_own_source(self, env):
        owner_a, _ = _ids()
        _push_source(env.state._pool, owner_a)
        third = f"third-{uuid.uuid4().hex[:6]}"
        caller_no_source = f"nosrc-{uuid.uuid4().hex[:6]}"
        with TestClient(env) as client:
            result = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": third},
                headers=_headers(caller_no_source),  # caller owns no source named main
            )
            assert result.status_code == 404

    def test_allow_paths_narrowed_to_source_opt_ins(self, env):
        owner, grantee = _ids()
        _push_source(env.state._pool, owner)
        with TestClient(env) as client:
            created = client.post(
                "/umrs/user-sources/grants",
                json={
                    "source_name": "main",
                    "grantee": grantee,
                    "allow_paths": ["files", "fine_tuning"],  # fine_tuning not pushed
                },
                headers=_headers(owner),
            )
            assert created.status_code == 201
            assert created.json()["allow_paths"] == ["files"]

    def test_grant_to_self_rejected(self, env):
        owner, _ = _ids()
        with TestClient(env) as client:
            self_grant = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": owner},
                headers=_headers(owner),
            )
            assert self_grant.status_code == 400

    def test_duplicate_live_grant_rejected(self, env):
        owner, grantee = _ids()
        _push_source(env.state._pool, owner)
        with TestClient(env) as client:
            assert client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": grantee},
                headers=_headers(owner),
            ).status_code == 201
            duplicate = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": grantee},
                headers=_headers(owner),
            )
            assert duplicate.status_code == 409
            assert duplicate.json()["detail"]["error"] == "grant_exists"


class TestRevokeCascade:
    def test_revoke_kills_grantee_leases(self, env):
        owner, grantee = _ids()
        source_id = _push_source(env.state._pool, owner)
        with TestClient(env) as client:
            created = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": grantee},
                headers=_headers(owner),
            )
            grant_id = created.json()["grant_id"]

        broker = SecretBroker(db_pool=env.state._pool)
        lease, _ = broker.issue_lease(
            "job-g1", "t", "inference",
            owner_id=grantee, session_id="sess-g1",
            source_ids=(source_id,), default_source_id=source_id,
        )

        with TestClient(env) as client:
            revoked = client.post(f"/umrs/user-sources/grants/{grant_id}/revoke", headers=_headers(owner))
            assert revoked.status_code == 200
            assert revoked.json()["leases_revoked"] == 1
        assert not broker.get_lease(lease.lease_id).is_valid


class TestAuditTrail:
    def test_push_grant_revoke_all_audited_owner_scope(self, env):
        owner, grantee = _ids()
        _push_source(env.state._pool, owner)
        with TestClient(env) as client:
            created = client.post(
                "/umrs/user-sources/grants",
                json={"source_name": "main", "grantee": grantee},
                headers=_headers(owner),
            )
            grant_id = created.json()["grant_id"]
            client.post(f"/umrs/user-sources/grants/{grant_id}/revoke", headers=_headers(owner))

        store = SealedSourceStore(env.state._pool)
        conn = store._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT action FROM user_source_audit WHERE owner = %s ORDER BY created_at",
                    (owner,),
                )
                actions = [r["action"] for r in cur.fetchall()]
        finally:
            store._pool.putconn(conn)
        assert "grant_create" in actions
        assert "grant_revoke" in actions
