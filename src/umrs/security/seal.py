"""Credential envelope sealing for the sealed user-credential store (R1-1).

Recipient-keypair envelope (design D-3): the UMRS API holds the PUBLIC half
and seals at push; the inference proxy holds the PRIVATE half and unseals
per request. A compromised API cannot read keys.

Envelope format (PyNaCl-equivalent construction on `cryptography`, the
already-shipped dependency — libsodium's SealedBox is not in the dep tree):

    dek_wrapped = b64( eph_pub(32B) ‖ nonce(12B) ‖ AESGCM(wrap_key, dek, ctx) )
        shared   = X25519(eph_priv, recipient_pub)
        wrap_key = HKDF-SHA256(shared, salt=eph_pub, info=b"r4-seal-v1", 32)

Payload: `AES-256-GCM(DEK, api_key, AAD)` where AAD binds owner, name,
provider_kind and allow_host — ciphertext cannot be re-pointed at another
owner or host without failing integrity.

Key bootstrap: API reads ``UMRS_SEAL_PUBKEY_B64``; the proxy reads the
private key from ``UMRS_UNSEAL_PRIVKEY_PATH`` (podman secret mount) or
``UMRS_UNSEAL_PRIVKEY_B64`` (dev/test only). ``seal_key_id`` =
``"v1:<sha256(pub)[:6]>"`` so rotation has a deterministic rolling id.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_AAD_SEP = b"\x1f"
_HKDF_INFO = b"r4-seal-v1"
_NONCE_LEN = 12
_DEK_LEN = 32


class SealError(RuntimeError):
    """Sealing/unsealing failure — never carries key material in the message."""


def _crypto():
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return {
            "AESGCM": AESGCM,
            "HKDF": HKDF,
            "X25519PrivateKey": X25519PrivateKey,
            "X25519PublicKey": X25519PublicKey,
            "hashes": hashes,
            "serialization": serialization,
        }
    except ImportError as exc:  # pragma: no cover - dep present on server installs
        raise SealError(
            "cryptography package unavailable — install the umrs server extras"
        ) from exc


def generate_recipient_keypair() -> tuple[str, str]:
    """Fresh X25519 pair → (privkey_b64_raw, pubkey_b64_raw)."""
    c = _crypto()
    priv = c["X25519PrivateKey"].generate()
    raw_priv = priv.private_bytes(
        c["serialization"].Encoding.Raw,
        c["serialization"].PrivateFormat.Raw,
        c["serialization"].NoEncryption(),
    )
    raw_pub = priv.public_key().public_bytes(
        c["serialization"].Encoding.Raw,
        c["serialization"].PublicFormat.Raw,
    )
    return base64.b64encode(raw_priv).decode(), base64.b64encode(raw_pub).decode()


def key_fingerprint(value: str) -> str:
    """Short, non-sensitive identity of a credential (show in UIs/logs)."""
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:8]


def seal_key_id(pubkey_b64: str) -> str:
    """Stable identifier for the recipient keypair (rotation friendly)."""
    raw = base64.b64decode(pubkey_b64)
    return "v1:" + hashlib.sha256(raw).hexdigest()[:6]


def _aad(owner: str, name: str, provider_kind: str, allow_host: str) -> bytes:
    return _AAD_SEP.join(
        part.encode() for part in (owner, name, provider_kind, allow_host)
    )


def _hkdf_wrap_key(shared: bytes, salt: bytes) -> bytes:
    c = _crypto()
    kdf = c["HKDF"](
        algorithm=c["hashes"].SHA256(),
        length=32,
        salt=salt,
        info=_HKDF_INFO,
    )
    return kdf.derive(shared)


@dataclass(frozen=True)
class SealedBlob:
    """Everything needed to persist one sealed credential row."""

    ciphertext: bytes
    nonce: bytes  # nonce for the DEK↔api_key GCM
    dek_wrapped: bytes  # eph_pub ‖ nonce ‖ wrapped DEK (per format above)
    seal_key_id: str


def seal_credential(
    api_key: str,
    *,
    pubkey_b64: str,
    owner: str,
    name: str,
    provider_kind: str,
    allow_host: str,
) -> SealedBlob:
    """Seal one plaintext key for the recipient public key. Request-scope only."""
    c = _crypto()
    try:
        recipient_pub = c["X25519PublicKey"].from_public_bytes(
            base64.b64decode(pubkey_b64)
        )
        eph = c["X25519PrivateKey"].generate()
        shared = eph.exchange(recipient_pub)
        raw_eph_pub = eph.public_key().public_bytes(
            c["serialization"].Encoding.Raw,
            c["serialization"].PublicFormat.Raw,
        )
        wrap_key = _hkdf_wrap_key(shared, raw_eph_pub)

        dek = os.urandom(_DEK_LEN)
        dek_nonce = os.urandom(_NONCE_LEN)
        ciphertext = c["AESGCM"](dek).encrypt(
            dek_nonce, api_key.encode(), _aad(owner, name, provider_kind, allow_host)
        )
        wrap_nonce = os.urandom(_NONCE_LEN)
        wrapped = c["AESGCM"](wrap_key).encrypt(wrap_nonce, dek, raw_eph_pub)
        return SealedBlob(
            ciphertext=ciphertext,
            nonce=dek_nonce,
            dek_wrapped=raw_eph_pub + wrap_nonce + wrapped,
            seal_key_id=seal_key_id(pubkey_b64),
        )
    except SealError:
        raise
    except Exception as exc:
        raise SealError(f"seal failed: {type(exc).__name__}") from exc


def unseal_credential(
    blob: SealedBlob,
    *,
    privkey_b64: str,
    owner: str,
    name: str,
    provider_kind: str,
    allow_host: str,
) -> str:
    """Unseal with the recipient private key. Wrong key/AAD → SealError."""
    c = _crypto()
    try:
        priv = c["X25519PrivateKey"].from_private_bytes(base64.b64decode(privkey_b64))
        eph_pub_raw = blob.dek_wrapped[:32]
        wrap_nonce = blob.dek_wrapped[32 : 32 + _NONCE_LEN]
        wrapped = blob.dek_wrapped[32 + _NONCE_LEN :]
        eph_pub = c["X25519PublicKey"].from_public_bytes(eph_pub_raw)
        shared = priv.exchange(eph_pub)
        wrap_key = _hkdf_wrap_key(shared, eph_pub_raw)
        dek = c["AESGCM"](wrap_key).decrypt(wrap_nonce, wrapped, eph_pub_raw)
        plaintext = c["AESGCM"](dek).decrypt(
            blob.nonce,
            blob.ciphertext,
            _aad(owner, name, provider_kind, allow_host),
        )
        return plaintext.decode()
    except Exception as exc:
        raise SealError(
            "unseal failed (wrong key, tampered row, or corrupted blob)"
        ) from exc


class RecipientKeys:
    """Bootstrap holder for sealing (API) / unsealing (proxy) keys."""

    def __init__(self) -> None:
        self._pub: str | None = None
        self._priv: str | None = None

    def seal_pubkey(self) -> str:
        if self._pub is None:
            env = os.environ.get("UMRS_SEAL_PUBKEY_B64", "").strip()
            if not env:
                raise SealError(
                    "UMRS_SEAL_PUBKEY_B64 not set (sealed store unavailable)"
                )
            self._pub = env
        return self._pub

    def unseal_privkey(self) -> str:
        if self._priv is None:
            path = os.environ.get("UMRS_UNSEAL_PRIVKEY_PATH", "").strip()
            if path:
                try:
                    with open(path, "rb") as fh:
                        self._priv = fh.read().decode().strip()
                except OSError as exc:
                    raise SealError(f"unseal private key unreadable at {path}") from exc
            else:
                env = os.environ.get("UMRS_UNSEAL_PRIVKEY_B64", "").strip()
                if not env:
                    raise SealError(
                        "no unseal private key (set UMRS_UNSEAL_PRIVKEY_PATH or dev-var)"
                    )
                self._priv = env
        return self._priv


_recipient_keys = RecipientKeys()


def recipient_keys() -> RecipientKeys:
    return _recipient_keys
