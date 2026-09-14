"""DEK rewrap job for recipient-keypair rotation (R2-5).

Runs inside the inference proxy (the only place the unseal private half
exists): unwrap each DEK with the OLD private key, rewrap it for the NEW
recipient public key. Ciphertext bytes never change; only ``dek_wrapped``
and ``seal_key_id`` move. Safe to run row-by-row with a transaction per
batch; wrong old key → hard failure before any row is written.
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .seal import SealError, _crypto, _hkdf_wrap_key, seal_key_id

logger = logging.getLogger(__name__)


def unwrap_dek(dek_wrapped: bytes, privkey_b64: str) -> bytes:
    """AEAD-unwrap one DEK with the old recipient private key."""
    c = _crypto()
    try:
        priv = c["X25519PrivateKey"].from_private_bytes(base64.b64decode(privkey_b64))
        eph_pub_raw = dek_wrapped[:32]
        nonce = dek_wrapped[32:44]
        wrapped = dek_wrapped[44:]
        eph_pub = c["X25519PublicKey"].from_public_bytes(eph_pub_raw)
        shared = priv.exchange(eph_pub)
        wrap_key = _hkdf_wrap_key(shared, eph_pub_raw)
        return c["AESGCM"](wrap_key).decrypt(nonce, wrapped, eph_pub_raw)
    except Exception as exc:
        raise SealError("DEK unwrap failed (wrong old key or damaged row)") from exc


def wrap_dek(dek: bytes, pubkey_b64: str) -> bytes:
    """Rewrap one DEK for the new recipient public key."""
    c = _crypto()
    try:
        pub = c["X25519PublicKey"].from_public_bytes(base64.b64decode(pubkey_b64))
        eph = c["X25519PrivateKey"].generate()
        shared = eph.exchange(pub)
        raw_eph_pub = eph.public_key().public_bytes(
            c["serialization"].Encoding.Raw,
            c["serialization"].PublicFormat.Raw,
        )
        wrap_key = _hkdf_wrap_key(shared, raw_eph_pub)
        nonce = os.urandom(12)
        wrapped = c["AESGCM"](wrap_key).encrypt(nonce, dek, raw_eph_pub)
        return raw_eph_pub + nonce + wrapped
    except Exception as exc:
        raise SealError("DEK wrap failed") from exc


def rewrap_wrapped_dek(
    dek_wrapped: bytes, *, old_privkey_b64: str, new_pubkey_b64: str
) -> bytes:
    """One row: old priv unwraps the old envelope, new pub re-wraps it."""
    dek = unwrap_dek(dek_wrapped, old_privkey_b64)
    return wrap_dek(dek, new_pubkey_b64)


def rewrap_sealed_rows(
    rows: Iterable[Any],
    *,
    old_privkey_b64: str,
    new_pubkey_b64: str,
    row_attrs: Callable[[Any], tuple[Any, bytes]] | None = None,
) -> Iterator[tuple[Any, bytes, str]]:
    """Rewrap sealed-source rows one by one (caller writes per batch).

    Yields ``(row_id, new_dek_wrapped, new_seal_key_id)``. ``rows`` may be
    ``UnsealFetch``-shaped objects or ``(row_id, dek_wrapped)`` tuples.
    Raises SealError on the first failed row — the caller writes nothing
    in that case (atomicity by caller's transaction).
    """

    def _default_extract(r: Any) -> tuple[Any, bytes]:
        if isinstance(r, tuple):
            return r
        return r.id, r.dek_wrapped

    extract = row_attrs or _default_extract
    new_seal_id = seal_key_id(new_pubkey_b64)
    for row in rows:
        row_id, dek_wrapped = extract(row)
        new_wrapped = rewrap_wrapped_dek(
            dek_wrapped, old_privkey_b64=old_privkey_b64, new_pubkey_b64=new_pubkey_b64
        )
        yield row_id, new_wrapped, new_seal_id
