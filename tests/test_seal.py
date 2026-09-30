"""Seal envelope round trip + tamper/rotation pins (R1-1, design §3.1.2)."""

from __future__ import annotations

import pytest
from umrs.security.seal import (
    SealError,
    generate_recipient_keypair,
    key_fingerprint,
    seal_credential,
    seal_key_id,
    unseal_credential,
)


@pytest.fixture
def keypair():
    priv, pub = generate_recipient_keypair()
    return {"priv": priv, "pub": pub}


def _seal(pub, **aad):
    defaults = {
        "owner": "alice",
        "name": "openai-main",
        "provider_kind": "openai",
        "allow_host": "https://api.openai.com",
    }
    defaults.update(aad)
    return seal_credential("sk-live-secret-42", pubkey_b64=pub, **defaults), defaults


def test_round_trip(keypair):
    blob, aad = _seal(keypair["pub"])
    assert unseal_credential(blob, privkey_b64=keypair["priv"], **aad) == "sk-live-secret-42"


def test_plaintext_never_in_blob(keypair):
    blob, _ = _seal(keypair["pub"])
    serialized = (blob.ciphertext + blob.dek_wrapped + blob.nonce).hex()
    assert "sk-live-secret-42" not in serialized
    assert b"sk-live-secret-42" not in blob.ciphertext


def test_two_seals_same_key_pair_differ(keypair):
    """Fresh DEK + nonces per push — ciphertext never repeats."""
    blob1, _ = _seal(keypair["pub"])
    blob2, _ = _seal(keypair["pub"])
    assert blob1.ciphertext != blob2.ciphertext
    assert blob1.dek_wrapped != blob2.dek_wrapped
    assert blob1.seal_key_id == blob2.seal_key_id


@pytest.mark.parametrize("field", ["owner", "name", "provider_kind", "allow_host"])
def test_every_aad_field_binding(keypair, field):
    blob, aad = _seal(keypair["pub"])
    tampered = dict(aad)
    tampered[field] = "mallory" if field == "owner" else "tampered"
    with pytest.raises(SealError):
        unseal_credential(blob, privkey_b64=keypair["priv"], **tampered)


def test_wrong_recipient_key_fails(keypair):
    blob, aad = _seal(keypair["pub"])
    other_priv, _ = generate_recipient_keypair()
    with pytest.raises(SealError):
        unseal_credential(blob, privkey_b64=other_priv, **aad)


def test_corrupted_blob_rejected(keypair):
    blob, aad = _seal(keypair["pub"])
    damaged = type(blob)(
        ciphertext=blob.ciphertext[:-2] + b"xx",
        nonce=blob.nonce,
        dek_wrapped=blob.dek_wrapped,
        seal_key_id=blob.seal_key_id,
    )
    with pytest.raises(SealError):
        unseal_credential(damaged, privkey_b64=keypair["priv"], **aad)


def test_rotation_rewrap_preserves_plaintext(keypair):
    """A proxy-side rewrap job can move DEKs to a new recipient without touching ciphertext."""
    blob, aad = _seal(keypair["pub"])
    plaintext = unseal_credential(blob, privkey_b64=keypair["priv"], **aad)

    # Old → unwrap DEK → rewrap for the new recipient (simulated rotation job).
    from umrs.security.seal import _crypto, _hkdf_wrap_key

    c = _crypto()
    eph_pub_raw = blob.dek_wrapped[:32]
    wrap_nonce = blob.dek_wrapped[32:44]
    wrapped = blob.dek_wrapped[44:]
    priv = c["X25519PrivateKey"].from_private_bytes(__import__("base64").b64decode(keypair["priv"]))
    shared = priv.exchange(c["X25519PublicKey"].from_public_bytes(eph_pub_raw))
    wrap_key = _hkdf_wrap_key(shared, eph_pub_raw)
    dek = c["AESGCM"](wrap_key).decrypt(wrap_nonce, wrapped, eph_pub_raw)

    new_priv, new_pub = generate_recipient_keypair()
    eph2 = c["X25519PrivateKey"].generate()
    shared2 = eph2.exchange(
        c["X25519PublicKey"].from_public_bytes(__import__("base64").b64decode(new_pub))
    )
    raw_eph_pub2 = eph2.public_key().public_bytes(
        c["serialization"].Encoding.Raw, c["serialization"].PublicFormat.Raw
    )
    wrap_key2 = _hkdf_wrap_key(shared2, raw_eph_pub2)
    import os

    wrap_nonce2 = os.urandom(12)
    wrapped2 = c["AESGCM"](wrap_key2).encrypt(wrap_nonce2, dek, raw_eph_pub2)
    new_blob = type(blob)(
        ciphertext=blob.ciphertext,  # untouched
        nonce=blob.nonce,
        dek_wrapped=raw_eph_pub2 + wrap_nonce2 + wrapped2,
        seal_key_id=seal_key_id(new_pub),
    )
    assert unseal_credential(new_blob, privkey_b64=new_priv, **aad) == plaintext


def test_fingerprint_and_key_id_stable():
    assert key_fingerprint("sk-x").startswith("sha256:")
    assert len(key_fingerprint("sk-x")) == len("sha256:") + 8
    _, pub = generate_recipient_keypair()
    assert seal_key_id(pub).startswith("v1:")
    assert seal_key_id(pub) == seal_key_id(pub)  # deterministic
