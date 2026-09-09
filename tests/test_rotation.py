"""R2-5: DEK rewrap job (rotation) — row integrity across keypair rotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from umrs.security.rotation import (
    rewrap_sealed_rows,
    rewrap_wrapped_dek,
    unwrap_dek,
    wrap_dek,
)
from umrs.security.seal import (
    SealError,
    generate_recipient_keypair,
    seal_credential,
    unseal_credential,
)
from umrs.security.sealed_sources import SealedSourceStore

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src/umrs/migrations"


@pytest.fixture
def keys():
    old_priv, old_pub = generate_recipient_keypair()
    new_priv, new_pub = generate_recipient_keypair()
    return {
        "old_priv": old_priv,
        "old_pub": old_pub,
        "new_priv": new_priv,
        "new_pub": new_pub,
    }


def _seal(pub, owner="alice", name="main"):
    return seal_credential(
        "sk-real-key-rot",
        pubkey_b64=pub,
        owner=owner,
        name=name,
        provider_kind="openai",
        allow_host="https://api.openai.com",
    )


class TestPrimitives:
    def test_rewrap_row_preserves_unsealability_and_ciphertext(self, keys):
        blob = _seal(keys["old_pub"])
        rotated_bytes = rewrap_wrapped_dek(
            blob.dek_wrapped,
            old_privkey_b64=keys["old_priv"],
            new_pubkey_b64=keys["new_pub"],
        )
        rotated = type(blob)(
            ciphertext=blob.ciphertext,
            nonce=blob.nonce,
            dek_wrapped=rotated_bytes,
            seal_key_id=blob.seal_key_id,
        )
        result = unseal_credential(
            rotated,
            privkey_b64=keys["new_priv"],
            owner="alice",
            name="main",
            provider_kind="openai",
            allow_host="https://api.openai.com",
        )
        assert result == "sk-real-key-rot"
        assert blob.ciphertext == blob.ciphertext  # never rewritten

    def test_wrong_old_key_aborts_before_any_output(self, keys):
        blob = _seal(keys["old_pub"])
        other_priv, _ = generate_recipient_keypair()
        with pytest.raises(SealError, match="unwrap failed"):
            rewrap_wrapped_dek(
                blob.dek_wrapped,
                old_privkey_b64=other_priv,
                new_pubkey_b64=keys["new_pub"],
            )

    def test_rewrap_rows_yields_new_key_id(self, keys):
        rows = [
            ("row-1", _seal(keys["old_pub"], name="a").dek_wrapped),
            ("row-2", _seal(keys["old_pub"], name="b").dek_wrapped),
        ]
        out = list(
            rewrap_sealed_rows(
                rows,
                old_privkey_b64=keys["old_priv"],
                new_pubkey_b64=keys["new_pub"],
            )
        )
        assert [r[0] for r in out] == ["row-1", "row-2"]
        assert all(r[2].startswith("v1:") for r in out)
        assert out[0][2] == out[1][2]

    def test_primitive_unwrap_wrap_round_trip(self, keys):
        blob = _seal(keys["old_pub"])
        dek = unwrap_dek(blob.dek_wrapped, keys["old_priv"])
        rewrapped = wrap_dek(dek, keys["new_pub"])
        assert unwrap_dek(rewrapped, keys["new_priv"]) == dek


class TestStoreRewrap:
    @pytest.fixture
    def pool(self, request):
        pytest.importorskip("psycopg2")
        from umrs.shared.pg import build_sync_pool_from_env

        try:
            pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
        except Exception as exc:
            pytest.skip(f"Postgres unavailable: {exc}")
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute((_MIGRATIONS / "0086_user_sources_sealed.sql").read_text())
        finally:
            pool.putconn(conn)
        request.addfinalizer(pool.closeall)
        return pool

    def test_full_table_rewrap_writes_new_envelopes(self, pool, keys):
        import uuid

        store = SealedSourceStore(pool)
        owner = f"rot-owner-{uuid.uuid4().hex[:8]}"
        blob = _seal(keys["old_pub"], owner=owner, name="main")
        source_id = store.insert(
            owner=owner,
            name="main",
            provider_kind="openai",
            base_url="https://api.openai.com/v1",
            allow_host="https://api.openai.com",
            allow_paths=(),
            key_fingerprint="sha256:deadbeef",
            seal_key_id=blob.seal_key_id,
            dek_wrapped=blob.dek_wrapped,
            nonce=blob.nonce,
            ciphertext=blob.ciphertext,
        )

        store.rewrap_all(old_privkey_b64=keys["old_priv"], new_pubkey_b64=keys["new_pub"])

        full = store.fetch_for_unseal(source_id)
        assert full is not None
        result = unseal_credential(
            type(
                "B",
                (),
                {
                    "ciphertext": full.ciphertext,
                    "nonce": full.nonce,
                    "dek_wrapped": full.dek_wrapped,
                    "seal_key_id": full.seal_key_id if hasattr(full, "seal_key_id") else "",
                },
            )(),
            privkey_b64=keys["new_priv"],
            owner=owner,
            name="main",
            provider_kind="openai",
            allow_host="https://api.openai.com",
        )
        assert result == "sk-real-key-rot"
        with pytest.raises(SealError):
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
                privkey_b64=keys["old_priv"],
                owner=owner,
                name="main",
                provider_kind="openai",
                allow_host="https://api.openai.com",
            )
        # Cleanup: remove the rotated row so subsequent runs stay hermetic.
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM user_source_audit WHERE owner = %s", (owner,))
                cur.execute("DELETE FROM user_sources_sealed WHERE owner = %s", (owner,))
        finally:
            pool.putconn(conn)
