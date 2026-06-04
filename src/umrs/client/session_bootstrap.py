"""Client-side CURVE session bootstrap for ZMQ authentication.

Implements the client half of the hybrid HTTP+CURVE auth flow:
1. GET /umrs/zmq/server-key  → retrieve server's CURVE public key
2. Generate ephemeral CURVE keypair locally
3. POST /umrs/zmq/session    → register pubkey (JWT-authenticated)
4. Configure ZMQ env vars    → MessageQueue picks up CURVE config

Usage::

    from umrs.client.session_bootstrap import bootstrap_curve_session

    # Before creating MessageQueue:
    session = bootstrap_curve_session(
        gateway_url="https://data0.ai.r-ccs.riken.jp/rccs",
        jwt_token="eyJ...",
        ttl_seconds=3600,
    )
    # ZMQ env vars are now set; MessageQueue will use CURVE.
    # session.session_id can be used for later revocation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurveSessionInfo:
    """Result of a successful CURVE session bootstrap."""

    session_id: str
    server_public_key: str
    client_public_key: str
    client_secret_key: str
    expires_at: float
    ttl_seconds: int
    grants: dict[str, Any]


class SessionBootstrapError(Exception):
    """Raised when session bootstrap fails."""
    pass


def bootstrap_curve_session(
    gateway_url: str,
    jwt_token: str,
    *,
    ttl_seconds: int = 3600,
    max_gpus: int = 0,
    max_walltime_sec: int = 0,
    set_env: bool = True,
    timeout: float = 15.0,
) -> CurveSessionInfo:
    """Bootstrap a CURVE session via the HTTP registration flow.

    This function performs the full client-side bootstrap:
    1. Fetches the server's CURVE public key (unauthenticated).
    2. Generates an ephemeral CURVE keypair.
    3. Registers the public key with the server (JWT-authenticated).
    4. Optionally sets ZMQ environment variables for MessageQueue.

    Parameters
    ----------
    gateway_url:
        Base URL of the UMRS API (e.g. "https://data0.ai.r-ccs.riken.jp/rccs").
        The /umrs/zmq/* endpoints must be accessible at this base.
    jwt_token:
        Valid JWT bearer token for authentication.
    ttl_seconds:
        Requested session lifetime (60–86400 seconds).
    max_gpus:
        Optional session GPU capability cap (0 = unrestricted).
    max_walltime_sec:
        Optional session walltime cap (0 = unrestricted).
    set_env:
        If True, set UMRS_ZMQ_CURVE_* env vars so MessageQueue auto-configures.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    CurveSessionInfo
        Session details including keys and expiry.

    Raises
    ------
    SessionBootstrapError
        On network errors, auth failures, or server-side rejections.
    """
    import requests

    base = gateway_url.rstrip("/")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    # Step 1: Get server CURVE public key (unauthenticated, best-effort).
    # In session mode the server may not expose a static key endpoint (404);
    # the server public key is returned by POST /umrs/zmq/session instead.
    server_pub = ""
    try:
        resp = requests.get(
            f"{base}/umrs/zmq/server-key",
            timeout=timeout,
        )
        if resp.status_code == 404:
            raise SessionBootstrapError(
                "ZMQ CURVE sessions not enabled on server (server-key returned 404)"
            )
        else:
            resp.raise_for_status()
            server_pub = resp.json().get("server_public_key", "")
    except requests.RequestException as exc:
        raise SessionBootstrapError(
            f"Failed to fetch server CURVE key: {exc}"
        ) from exc

    # Step 2: Generate ephemeral CURVE keypair
    try:
        import zmq
        client_pub, client_sec = zmq.curve_keypair()
        # zmq.curve_keypair() returns bytes (Z85-encoded)
        client_pub_str = client_pub.decode("ascii") if isinstance(client_pub, bytes) else str(client_pub)
        client_sec_str = client_sec.decode("ascii") if isinstance(client_sec, bytes) else str(client_sec)
    except ImportError:
        raise SessionBootstrapError(
            "pyzmq is required for CURVE session bootstrap. Install with: pip install pyzmq"
        )

    # Step 3: Register session (JWT-authenticated)
    try:
        resp = requests.post(
            f"{base}/umrs/zmq/session",
            headers=headers,
            json={
                "client_public_key": client_pub_str,
                "ttl_seconds": ttl_seconds,
                "max_gpus": max_gpus,
                "max_walltime_sec": max_walltime_sec,
            },
            timeout=timeout,
        )
        if resp.status_code == 401:
            raise SessionBootstrapError(
                "JWT authentication failed during session registration (401)"
            )
        if resp.status_code == 503:
            raise SessionBootstrapError(
                "CURVE session mode not available on server (503)"
            )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise SessionBootstrapError(
            f"Failed to register CURVE session: {exc}"
        ) from exc

    session_info = CurveSessionInfo(
        session_id=data["session_id"],
        server_public_key=data.get("server_public_key", server_pub),
        client_public_key=client_pub_str,
        client_secret_key=client_sec_str,
        expires_at=data["expires_at"],
        ttl_seconds=data["ttl_seconds"],
        grants=data.get("grants", {}),
    )

    # Step 4: Set environment variables for MessageQueue
    if set_env:
        os.environ["UMRS_ZMQ_AUTH_TYPE"] = "session"
        os.environ["UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY"] = session_info.server_public_key
        os.environ["UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY"] = session_info.client_public_key
        os.environ["UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY"] = session_info.client_secret_key
        logger.info(
            "CURVE session bootstrapped: session=%s expires_at=%.0f grants=%s",
            session_info.session_id,
            session_info.expires_at,
            session_info.grants,
        )

    return session_info


def revoke_curve_session(
    gateway_url: str,
    jwt_token: str,
    session_id: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Revoke a previously bootstrapped CURVE session.

    Parameters
    ----------
    gateway_url:
        Base URL of the UMRS API.
    jwt_token:
        Valid JWT bearer token (must be the session owner).
    session_id:
        Session ID to revoke (from CurveSessionInfo.session_id).
    timeout:
        HTTP request timeout.

    Returns
    -------
    bool
        True if revocation succeeded, False otherwise.
    """
    import requests

    base = gateway_url.rstrip("/")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        resp = requests.delete(
            f"{base}/umrs/zmq/session/{session_id}",
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code == 204:
            logger.info("CURVE session revoked: %s", session_id)
            return True
        logger.warning(
            "Session revocation returned %d: %s",
            resp.status_code,
            resp.text,
        )
        return False
    except requests.RequestException as exc:
        logger.warning("Failed to revoke CURVE session %s: %s", session_id, exc)
        return False


def refresh_curve_session(
    gateway_url: str,
    jwt_token: str,
    session_id: str,
    *,
    ttl_seconds: int = 3600,
    timeout: float = 10.0,
) -> float | None:
    """Refresh a CURVE session's TTL.

    Returns the new expires_at timestamp, or None on failure.
    """
    import requests

    base = gateway_url.rstrip("/")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        resp = requests.post(
            f"{base}/umrs/zmq/session/{session_id}/refresh",
            headers=headers,
            json={"ttl_seconds": ttl_seconds},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            new_expiry = data.get("expires_at", 0)
            logger.info(
                "CURVE session refreshed: %s → expires_at=%.0f",
                session_id, new_expiry,
            )
            return new_expiry
        logger.warning(
            "Session refresh returned %d: %s",
            resp.status_code,
            resp.text,
        )
        return None
    except requests.RequestException as exc:
        logger.warning("Failed to refresh CURVE session %s: %s", session_id, exc)
        return None
