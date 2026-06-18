"""Session-scoped CURVE credentials provider for ZeroMQ ZAP authentication.

Implements pyzmq's ``configure_curve_callback`` protocol with an in-memory
session registry.  Each entry binds a Z85-encoded CURVE public key to a
session identity (session_id, user_id, expiry).  A background thread sweeps
expired entries.

Usage on the server::

    provider = SessionCredentialsProvider(max_ttl_seconds=86400, max_per_user=5)
    provider.start()

    # During login (HTTP endpoint):
    entry = provider.register(pubkey_z85, session_id, user_id, ttl_seconds=3600)

    # Wire into ZAP:
    authenticator.configure_curve_callback(domain="*", credentials_provider=provider)

    # Shutdown:
    provider.stop()
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CURVE_KEY_LEN = 40


@dataclass(frozen=True)
class SessionGrants:
    """Authorization grants bound to a CURVE session.

    Captured from JWT claims at registration time so every ZMQ request
    carries identity + authorization context without an additional lookup.
    """

    projects: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("user",)
    email: str = ""
    # Capability-scoped session limits (empty = unrestricted for role)
    max_gpus: int = 0  # 0 means no cap (use site default)
    max_walltime_sec: int = 0  # 0 means no cap
    allowed_operations: tuple[str, ...] = ()  # empty = all ops for role

    def has_project(self, project_id: str) -> bool:
        """Check if session has access to a project."""
        if not self.projects:
            return True  # No project restriction
        return project_id in self.projects

    def has_role(self, role: str) -> bool:
        """Check if session carries a specific role."""
        return role in self.roles

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def effective_max_gpus(self, site_default: int = 8) -> int:
        """Return effective GPU cap (session cap or site default)."""
        return self.max_gpus if self.max_gpus > 0 else site_default

    def effective_max_walltime(self, site_default: int = 172800) -> int:
        """Return effective walltime cap in seconds (session cap or site default 48h)."""
        return self.max_walltime_sec if self.max_walltime_sec > 0 else site_default


@dataclass(frozen=True)
class SessionCurveEntry:
    """A session-scoped CURVE credential binding."""

    pubkey_z85: str
    session_id: str
    user_id: str
    created_at: float  # time.time()
    expires_at: float  # time.time()
    grants: SessionGrants = field(default_factory=SessionGrants)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())


class SessionCredentialsProvider:
    """In-memory session → CURVE public key registry.

    Implements the pyzmq ``credentials_provider`` callback protocol:
    ``callback(domain, key) -> bool``.

    Thread-safe: all mutations and lookups are guarded by an ``RLock``.

    Parameters
    ----------
    max_ttl_seconds:
        Hard cap on session lifetime.  ``register()`` clamps the requested
        TTL to this value.  Default 24 hours.
    max_per_user:
        Maximum concurrent sessions per ``user_id``.  Oldest sessions are
        evicted when the limit is reached.  Default 5.
    sweep_interval_seconds:
        How often the background thread removes expired entries.  Default 60 s.
    """

    def __init__(
        self,
        *,
        max_ttl_seconds: int = 86400,
        max_per_user: int = 5,
        sweep_interval_seconds: float = 60.0,
    ) -> None:
        self._max_ttl = max_ttl_seconds
        self._max_per_user = max_per_user
        self._sweep_interval = sweep_interval_seconds

        # pubkey_z85 → SessionCurveEntry
        self._by_key: dict[str, SessionCurveEntry] = {}
        # session_id → pubkey_z85  (reverse index for revocation)
        self._by_session: dict[str, str] = {}

        self._lock = threading.RLock()
        self._sweep_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # pyzmq callback protocol
    # ------------------------------------------------------------------

    def callback(self, domain: str, key: bytes) -> bool:
        """ZAP callback invoked by ``ThreadAuthenticator`` for every
        incoming CURVE handshake.

        Parameters
        ----------
        domain:
            ZAP domain (always ``"*"`` in our setup).
        key:
            Z85-encoded client public key as ``bytes``.

        Returns
        -------
        bool
            ``True`` if the key is registered and the session is not expired.
        """
        z85 = key.decode("ascii") if isinstance(key, bytes) else str(key)
        with self._lock:
            entry = self._by_key.get(z85)
            if entry is None:
                logger.debug("ZAP reject: unknown CURVE pubkey %.8s…", z85)
                return False
            if entry.is_expired:
                logger.info(
                    "ZAP reject: session %s for user %r expired",
                    entry.session_id,
                    entry.user_id,
                )
                self._remove_entry_unlocked(entry)
                return False
            logger.debug(
                "ZAP allow: session %s user %r (TTL %.0fs)",
                entry.session_id,
                entry.user_id,
                entry.ttl_remaining,
            )
            return True

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def register(
        self,
        pubkey_z85: str,
        user_id: str,
        ttl_seconds: int,
        *,
        session_id: Optional[str] = None,
        grants: Optional[SessionGrants] = None,
    ) -> SessionCurveEntry:
        """Register an ephemeral CURVE public key for a session.

        Parameters
        ----------
        pubkey_z85:
            Z85-encoded client public key (40 ASCII chars).
        user_id:
            Authenticated user identity (from JWT ``sub`` claim).
        ttl_seconds:
            Requested session lifetime.  Clamped to ``max_ttl_seconds``.
        session_id:
            Optional caller-supplied session ID.  Generated if omitted.
        grants:
            Authorization grants captured from JWT claims at registration.
            If ``None``, a default ``SessionGrants()`` with role=user is used.

        Returns
        -------
        SessionCurveEntry
            The created entry.

        Raises
        ------
        ValueError
            If *pubkey_z85* is malformed or *ttl_seconds* is not positive.
        """
        if not pubkey_z85 or len(pubkey_z85) != CURVE_KEY_LEN:
            raise ValueError(
                f"pubkey_z85 must be {CURVE_KEY_LEN} Z85 characters, "
                f"got {len(pubkey_z85) if pubkey_z85 else 0}"
            )
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        effective_ttl = min(ttl_seconds, self._max_ttl)
        now = time.time()
        sid = session_id or uuid.uuid4().hex
        entry = SessionCurveEntry(
            pubkey_z85=pubkey_z85,
            session_id=sid,
            user_id=user_id,
            created_at=now,
            expires_at=now + effective_ttl,
            grants=grants or SessionGrants(),
        )

        with self._lock:
            # Evict oldest sessions for this user if at capacity
            self._enforce_user_limit_unlocked(user_id)

            # If the pubkey was already registered (e.g. re-register), remove old
            old = self._by_key.get(pubkey_z85)
            if old is not None:
                self._remove_entry_unlocked(old)

            self._by_key[pubkey_z85] = entry
            self._by_session[sid] = pubkey_z85

        logger.info(
            "CURVE session registered: session=%s user=%r ttl=%ds",
            sid,
            user_id,
            effective_ttl,
        )
        return entry

    def revoke(self, session_id: str) -> bool:
        """Revoke a single session by ID.

        Returns ``True`` if the session existed and was removed.
        """
        with self._lock:
            pubkey = self._by_session.get(session_id)
            if pubkey is None:
                return False
            entry = self._by_key.get(pubkey)
            if entry is not None:
                self._remove_entry_unlocked(entry)
            else:
                del self._by_session[session_id]
            logger.info("CURVE session revoked: session=%s", session_id)
            return True

    def revoke_user(self, user_id: str) -> int:
        """Revoke all sessions for a user.  Returns the count removed."""
        with self._lock:
            to_remove = [e for e in self._by_key.values() if e.user_id == user_id]
            for entry in to_remove:
                self._remove_entry_unlocked(entry)
        if to_remove:
            logger.info(
                "CURVE sessions revoked for user %r: %d removed",
                user_id,
                len(to_remove),
            )
        return len(to_remove)

    def refresh(self, session_id: str, ttl_seconds: int) -> Optional[SessionCurveEntry]:
        """Extend session TTL.  Returns the updated entry, or ``None``."""
        if ttl_seconds <= 0:
            return None
        effective_ttl = min(ttl_seconds, self._max_ttl)
        with self._lock:
            pubkey = self._by_session.get(session_id)
            if pubkey is None:
                return None
            old = self._by_key.get(pubkey)
            if old is None:
                return None
            if old.is_expired:
                self._remove_entry_unlocked(old)
                return None
            # Replace with updated expiry (frozen dataclass → create new)
            updated = SessionCurveEntry(
                pubkey_z85=old.pubkey_z85,
                session_id=old.session_id,
                user_id=old.user_id,
                created_at=old.created_at,
                expires_at=time.time() + effective_ttl,
                grants=old.grants,
            )
            self._by_key[pubkey] = updated
        logger.debug(
            "CURVE session refreshed: session=%s new_ttl=%ds", session_id, effective_ttl
        )
        return updated

    def get_session(self, session_id: str) -> Optional[SessionCurveEntry]:
        """Look up a session by ID."""
        with self._lock:
            pubkey = self._by_session.get(session_id)
            if pubkey is None:
                return None
            entry = self._by_key.get(pubkey)
            if entry is not None and entry.is_expired:
                self._remove_entry_unlocked(entry)
                return None
            return entry

    def user_id_for_key(self, pubkey_z85: str) -> Optional[str]:
        """Return the user identity bound to a CURVE public key."""
        with self._lock:
            entry = self._by_key.get(pubkey_z85)
            if entry is None or entry.is_expired:
                return None
            return entry.user_id

    def lookup(self, pubkey_z85: str) -> Optional[SessionCurveEntry]:
        """Return the full session entry for a CURVE public key.

        This is the primary method for identity resolution after ZAP
        authentication.  Returns ``None`` if the key is unknown or expired.
        """
        with self._lock:
            entry = self._by_key.get(pubkey_z85)
            if entry is None or entry.is_expired:
                return None
            return entry

    def active_session_count(self, user_id: Optional[str] = None) -> int:
        """Return the number of active (non-expired) sessions."""
        with self._lock:
            if user_id is None:
                return sum(1 for e in self._by_key.values() if not e.is_expired)
            return sum(
                1
                for e in self._by_key.values()
                if e.user_id == user_id and not e.is_expired
            )

    def list_sessions(self, user_id: Optional[str] = None) -> list[SessionCurveEntry]:
        """List active sessions, optionally filtered by user."""
        with self._lock:
            entries = list(self._by_key.values())
        result = [e for e in entries if not e.is_expired]
        if user_id is not None:
            result = [e for e in result if e.user_id == user_id]
        return sorted(result, key=lambda e: e.created_at)

    # ------------------------------------------------------------------
    # Background sweep
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background expiry sweep thread."""
        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            return
        self._stop_event.clear()
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop,
            name="curve-session-sweep",
            daemon=True,
        )
        self._sweep_thread.start()
        logger.info(
            "Session CURVE provider started (max_ttl=%ds max_per_user=%d sweep=%ds)",
            self._max_ttl,
            self._max_per_user,
            self._sweep_interval,
        )

    def stop(self) -> None:
        """Stop the background sweep thread."""
        self._stop_event.set()
        if self._sweep_thread is not None:
            self._sweep_thread.join(timeout=5.0)
            self._sweep_thread = None
        logger.info("Session CURVE provider stopped")

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count removed."""
        with self._lock:
            expired = [e for e in self._by_key.values() if e.is_expired]
            for entry in expired:
                self._remove_entry_unlocked(entry)
        if expired:
            logger.info("Swept %d expired CURVE session(s)", len(expired))
        return len(expired)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_entry_unlocked(self, entry: SessionCurveEntry) -> None:
        """Remove an entry from both indexes.  Caller must hold ``_lock``."""
        self._by_key.pop(entry.pubkey_z85, None)
        self._by_session.pop(entry.session_id, None)

    def _enforce_user_limit_unlocked(self, user_id: str) -> None:
        """Evict oldest sessions if *user_id* has reached ``_max_per_user``."""
        user_entries = sorted(
            (e for e in self._by_key.values() if e.user_id == user_id),
            key=lambda e: e.created_at,
        )
        while len(user_entries) >= self._max_per_user:
            oldest = user_entries.pop(0)
            logger.info(
                "Evicting oldest CURVE session %s for user %r (limit %d)",
                oldest.session_id,
                user_id,
                self._max_per_user,
            )
            self._remove_entry_unlocked(oldest)

    def _sweep_loop(self) -> None:
        """Background loop that periodically purges expired sessions."""
        while not self._stop_event.wait(timeout=self._sweep_interval):
            try:
                self.cleanup_expired()
            except Exception:
                logger.exception("Error in CURVE session sweep")


# ---------------------------------------------------------------------------
# Process-wide shared provider singleton
# ---------------------------------------------------------------------------
# In session mode the same :class:`SessionCredentialsProvider` instance must be
# shared by (a) the HTTP session API (``umrs.api.zmq_sessions``) that registers
# client public keys and (b) every server-side ZMQ socket's ZAP authenticator
# that validates handshakes against that registry. This neutral core module is
# the single source of truth so neither ``core`` nor ``api`` has to import the
# other across the architecture boundary.

_shared_session_provider: "SessionCredentialsProvider | None" = None


def set_shared_session_provider(
    provider: "SessionCredentialsProvider | None",
) -> None:
    """Install (or clear) the process-wide session CURVE provider singleton."""
    global _shared_session_provider
    _shared_session_provider = provider


def get_shared_session_provider() -> "SessionCredentialsProvider | None":
    """Return the process-wide session CURVE provider, or ``None`` if unset."""
    return _shared_session_provider
