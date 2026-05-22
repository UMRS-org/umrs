from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CURVE_KEY_LEN = 40


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name)
    if raw:
        return raw.strip()

    file_path = env.get(f"{name}_FILE")
    if not file_path:
        return None
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Could not read {name}_FILE={file_path}: {exc}") from exc


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _validate_curve_key(name: str, value: str | None) -> bytes:
    if not value:
        raise ValueError(f"{name} is required when UMRS_ZMQ_AUTH_TYPE=curve")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be ASCII Z85 text") from exc
    if len(encoded) != CURVE_KEY_LEN:
        raise ValueError(
            f"{name} must be {CURVE_KEY_LEN} characters of Z85-encoded CURVE key material"
        )
    return encoded


@dataclass(frozen=True)
class ZMQSecurityConfig:
    """ZMQ socket security configuration.

    ``curve`` uses ZMTP CURVE with the RFC 27 ZAP authenticator. Server sockets
    require either a trusted client certificate directory or an explicit
    allow-any development escape hatch.
    """

    auth_type: str = "none"
    server_public_key: str | None = None
    server_secret_key: str | None = None
    client_public_key: str | None = None
    client_secret_key: str | None = None
    client_keys_dir: str | None = None
    allow_any_client_key: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ZMQSecurityConfig":
        source = os.environ if env is None else env
        auth_type = source.get("UMRS_ZMQ_AUTH_TYPE", "none").strip().lower()
        return cls(
            auth_type=auth_type,
            server_public_key=_env_value(source, "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY"),
            server_secret_key=_env_value(source, "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY"),
            client_public_key=_env_value(source, "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY"),
            client_secret_key=_env_value(source, "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY"),
            client_keys_dir=_env_value(source, "UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR"),
            allow_any_client_key=_env_flag(
                source,
                "UMRS_ZMQ_CURVE_ALLOW_ANY_CLIENT_KEY",
                False,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.auth_type != "none"

    def apply_to_socket(self, socket: Any, *, server: bool, context: Any = None) -> Any | None:
        """Apply configured security options before bind/connect.

        Returns a started authenticator when the server path needs one. Callers
        should retain and stop it when the owning socket is closed.
        """
        if self.auth_type == "none":
            return None
        if self.auth_type != "curve":
            raise ValueError(
                "Unsupported UMRS_ZMQ_AUTH_TYPE={!r}; expected 'none' or 'curve'".format(
                    self.auth_type
                )
            )

        if server:
            return self._apply_curve_server(socket, context=context)
        self._apply_curve_client(socket)
        return None

    def _apply_curve_server(self, socket: Any, *, context: Any = None) -> Any | None:
        authenticator = self._start_curve_authenticator(context)
        socket.curve_publickey = _validate_curve_key(
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY",
            self.server_public_key,
        )
        socket.curve_secretkey = _validate_curve_key(
            "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY",
            self.server_secret_key,
        )
        socket.curve_server = True
        return authenticator

    def _apply_curve_client(self, socket: Any) -> None:
        socket.curve_serverkey = _validate_curve_key(
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY",
            self.server_public_key,
        )
        socket.curve_publickey = _validate_curve_key(
            "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY",
            self.client_public_key,
        )
        socket.curve_secretkey = _validate_curve_key(
            "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY",
            self.client_secret_key,
        )

    def _start_curve_authenticator(self, context: Any) -> Any:
        if context is None:
            raise ValueError("A ZMQ context is required for CURVE server authentication")
        if not self.client_keys_dir and not self.allow_any_client_key:
            raise ValueError(
                "UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR is required for authenticated CURVE servers "
                "unless UMRS_ZMQ_CURVE_ALLOW_ANY_CLIENT_KEY=true is set for development"
            )

        from zmq.auth import CURVE_ALLOW_ANY
        from zmq.auth.thread import ThreadAuthenticator

        authenticator = ThreadAuthenticator(context)
        authenticator.start()
        location = CURVE_ALLOW_ANY if self.allow_any_client_key else self.client_keys_dir
        authenticator.configure_curve(domain="*", location=location)
        logger.info(
            "ZMQ CURVE authentication enabled (%s)",
            "allow-any development mode" if self.allow_any_client_key else self.client_keys_dir,
        )
        return authenticator


# ---------------------------------------------------------------------------
# CURVE key ↔ user-identity mapping
# ---------------------------------------------------------------------------


@dataclass
class CurveKeyMap:
    """Maps Z85-encoded CURVE public keys to UMRS usernames.

    When ``UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR`` is used, the ZAP authenticator
    loads ``*.key`` files from that directory.  ``CurveKeyMap`` mirrors the
    same directory and builds a ``{public_key_z85 → username}`` mapping by
    deriving the username from the filename (``<username>.key``).

    Lifecycle: created once at server startup, call :meth:`reload` if files
    change at runtime (e.g. after ``umrs-server admin zmq-keygen``).
    """

    _key_to_user: dict[str, str] = field(default_factory=dict)
    _dir: str | None = None

    @classmethod
    def from_directory(cls, keys_dir: str | None) -> "CurveKeyMap":
        km = cls(_dir=keys_dir)
        km.reload()
        return km

    def reload(self) -> None:
        self._key_to_user.clear()
        if not self._dir:
            return
        keys_path = Path(self._dir)
        if not keys_path.is_dir():
            logger.warning("CURVE client keys directory does not exist: %s", self._dir)
            return
        for key_file in sorted(keys_path.glob("*.key")):
            # Skip secret files; we only need the public key files
            if key_file.name.endswith(".key_secret"):
                continue
            username = key_file.stem  # e.g. "alice.key" → "alice"
            try:
                from zmq.auth import load_certificate

                pub_key, _ = load_certificate(str(key_file))
                z85 = pub_key.decode("ascii") if isinstance(pub_key, bytes) else pub_key
                self._key_to_user[z85] = username
                logger.debug("Loaded CURVE key for user %r from %s", username, key_file)
            except Exception:
                logger.warning("Failed to load CURVE key from %s", key_file, exc_info=True)
        logger.info(
            "CurveKeyMap loaded %d client key(s) from %s", len(self._key_to_user), self._dir
        )

    def username_for_key(self, z85_public_key: str) -> str | None:
        """Return the UMRS username mapped to *z85_public_key*, or ``None``."""
        return self._key_to_user.get(z85_public_key)

    def __len__(self) -> int:
        return len(self._key_to_user)


# ---------------------------------------------------------------------------
# Keypair generation utility
# ---------------------------------------------------------------------------


def generate_curve_keypair(
    keys_dir: str, name: str, *, metadata: dict[str, str] | None = None
) -> tuple[str, str]:
    """Generate a CURVE keypair and write it to *keys_dir*.

    Returns ``(public_key_file, secret_key_file)`` absolute paths.
    Uses :func:`zmq.auth.create_certificates` under the hood.
    """
    from zmq.auth import create_certificates

    Path(keys_dir).mkdir(parents=True, exist_ok=True)
    pub_path, sec_path = create_certificates(keys_dir, name, metadata=metadata)
    # Restrict permissions on the secret key
    Path(sec_path).chmod(0o600)
    return pub_path, sec_path


def load_public_key_z85(key_file: str) -> str:
    """Read a ``.key`` file and return the Z85-encoded public key string."""
    from zmq.auth import load_certificate

    pub_key, _ = load_certificate(key_file)
    return pub_key.decode("ascii") if isinstance(pub_key, bytes) else pub_key