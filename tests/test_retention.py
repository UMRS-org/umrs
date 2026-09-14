"""R3-2: audit retention (purge + export halves, tombstone TTL)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from umrs.grants.source_store import SourceGrantStore
from umrs.security.seal import generate_recipient_keypair, seal_credential
from umrs.security.sealed_sources import SealedSourceStore

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src/umrs/migrations"


@pytest.fixture
def pool(request):
    pytest.importorskip("psycopg2")
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
            cur.execute("TRUNCATE user_source_audit, source_grants, user_sources_sealed")
    finally:
        pool.putconn(conn)
    request.addfinalizer(pool.closeall)
    return pool


def _source_id(store: SealedSourceStore, owner: str, seal_pubkey_priv) -> str:
    priv, pub = seal_pubkey_priv
    blob = seal_credential(
        "sk-x",
        pubkey_b64=pub,
        owner=owner,
        name="main",
        provider_kind="openai",
        allow_host="https://api.example-provider.com",
    )
    return store.insert(
        owner=owner,
        name="main",
        provider_kind="openai",
        base_url="https://api.example-provider.com/v1",
        allow_host="https://api.example-provider.com",
        allow_paths=(),
        key_fingerprint="sha256:x",
        seal_key_id=blob.seal_key_id,
        dek_wrapped=blob.dek_wrapped,
        nonce=blob.nonce,
        ciphertext=blob.ciphertext,
    )


class TestRetention:
    def test_purge_cmd_deletes_old_and_keeps_recent(self, pool):
        store = SealedSourceStore(pool)
        owner = f"ret-{uuid.uuid4().hex[:6]}"
        sid = _source_id(store, owner, generate_recipient_keypair())
        store.audit(action="push", owner=owner, source_id=sid, detail={})
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                # Move one row 120 days back; leave one fresh.
                cur.execute("DELETE FROM user_source_audit;")
                cur.execute(
                    "INSERT INTO user_source_audit (action, owner, source_id, detail, created_at) "
                    "VALUES ('legacy', %s, %s, '{}'::jsonb, now() - make_interval(days => 120)), "
                    "('fresh', %s, %s, '{}'::jsonb, now())",
                    (owner, sid, owner, sid),
                )
        finally:
            pool.putconn(conn)

        counts = store.purge_audit_older_than(90)
        assert counts["user_source_audit"] == 1
        exported = store.export_audit(owner=owner, since_days=3650)
        assert [r["action"] for r in exported] == ["fresh"]

    def test_tombstone_ttl_in_grants(self, pool):
        owner = f"ret-{uuid.uuid4().hex[:6]}"
        store = SourceGrantStore(pool)
        sources = SealedSourceStore(pool)
        sid = _source_id(sources, owner, generate_recipient_keypair())
        from datetime import datetime, timezone

        grant = store.create_grant(
            source_id=sid,
            owner=owner,
            grantee=f"g-{uuid.uuid4().hex[:6]}",
            allow_paths=(),
            budget_usd=None,
            expires_at=None,
            now=datetime.now(timezone.utc),
        )
        store.revoke_grant(grant.grant_id, actor=owner, now=datetime.now(timezone.utc))
        # Aged tombstone: force its timestamp into the past.
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE source_grants SET tombstoned_at = now() - make_interval(days => 200) WHERE grant_id = %s",
                    (grant.grant_id,),
                )
        finally:
            pool.putconn(conn)
        assert store.purge_tombstoned_older_than(180) == 1
        assert store.get_grant(grant.grant_id) is None

    def test_recent_tombstone_kept(self, pool):
        owner = f"ret-{uuid.uuid4().hex[:6]}"
        store = SourceGrantStore(pool)
        sources = SealedSourceStore(pool)
        sid = _source_id(sources, owner, generate_recipient_keypair())
        from datetime import datetime, timezone

        grant = store.create_grant(
            source_id=sid,
            owner=owner,
            grantee=f"g-{uuid.uuid4().hex[:6]}",
            allow_paths=(),
            budget_usd=None,
            expires_at=None,
            now=datetime.now(timezone.utc),
        )
        store.revoke_grant(grant.grant_id, actor=owner, now=datetime.now(timezone.utc))
        assert store.purge_tombstoned_older_than(180) == 0
        assert store.get_grant(grant.grant_id) is not None

    def test_export_owner_scoped(self, pool):
        store = SealedSourceStore(pool)
        owner_a, owner_b = (f"a-{uuid.uuid4().hex[:6]}", f"b-{uuid.uuid4().hex[:6]}")
        sid = _source_id(store, owner_a, generate_recipient_keypair())
        store.audit(action="push", owner=owner_a, source_id=sid, detail={})
        store.audit(action="push", owner=owner_b, source_id=sid, detail={})
        assert {r["owner"] for r in store.export_audit(owner=owner_a)} == {owner_a}
        assert len(store.export_audit()) >= 2
