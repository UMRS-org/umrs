"""ZMQ session CURVE credential management API.

Provides HTTP endpoints for clients to register/revoke ephemeral CURVE
sessions when ``UMRS_ZMQ_AUTH_TYPE=session`` is active.

Registration flow:
1. Client authenticates via JWT (same as all other UMRS endpoints).
2. Client generates a one-time CURVE keypair locally.
3. Client POSTs the public key to ``/umrs/zmq/session``.
4. Server registers the key in the :class:`SessionCredentialsProvider`
   with identity grants extracted from the JWT (projects, roles, email).
5. Client can now connect via ZMQ CURVE using the ephemeral key.
6. On logout/expiry the session is revoked automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/zmq", tags=["zmq-session"])

# ---------------------------------------------------------------------------
# Module-level singleton — set by MessageQueue or http_sidecar at startup
# ---------------------------------------------------------------------------
_session_provider: Any | None = None


def set_session_curve_provider(provider: Any) -> None:
    """Inject the :class:`SessionCredentialsProvider` singleton."""
    global _session_provider
    _session_provider = provider
    logger.info("ZMQ session CURVE provider set for HTTP API")


def get_session_curve_provider() -> Any:
    """FastAPI dependency — returns the provider or raises 503."""
    if _session_provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ZMQ session CURVE provider not available (UMRS_ZMQ_AUTH_TYPE != session)",
        )
    return _session_provider


# ---------------------------------------------------------------------------
# JWT claims container for session registration
# ---------------------------------------------------------------------------

@dataclass
class _JWTIdentity:
    """Parsed JWT identity with all claims needed for session grants."""

    user_id: str
    email: str
    projects: list[str]
    roles: list[str]


# ---------------------------------------------------------------------------
# Shared auth dependency: extract full identity from JWT
# ---------------------------------------------------------------------------

def _extract_jwt_identity(authorization: str | None = Header(None)) -> _JWTIdentity:
    """Extract full identity (sub, email, projects, roles) from the JWT bearer token."""
    from umrs.shared.auth import JWTError, resolve_jwt_secret, validate_jwt

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'",
        )
    token = parts[1]
    try:
        claims = validate_jwt(token, resolve_jwt_secret())
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user_id = claims.get("sub") or claims.get("username") or claims.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT missing 'sub' claim",
        )

    # Extract projects: support both list and single-value claim
    projects_claim = claims.get("projects") or claims.get("project_id")
    if isinstance(projects_claim, str):
        projects = [projects_claim] if projects_claim else []
    elif isinstance(projects_claim, list):
        projects = [str(p) for p in projects_claim]
    else:
        projects = []

    # Extract roles: support both list and single-value
    roles_claim = claims.get("roles") or claims.get("role")
    if isinstance(roles_claim, str):
        roles = [roles_claim] if roles_claim else ["user"]
    elif isinstance(roles_claim, list):
        roles = [str(r) for r in roles_claim]
    else:
        roles = ["user"]

    email = str(claims.get("email", ""))

    return _JWTIdentity(
        user_id=str(user_id),
        email=email,
        projects=projects,
        roles=roles,
    )


def _require_user_id(identity: _JWTIdentity = Depends(_extract_jwt_identity)) -> str:
    """Backward-compatible dependency that returns just user_id."""
    return identity.user_id


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SessionRegisterRequest(BaseModel):
    """Client sends its freshly generated CURVE public key."""
    client_public_key: str = Field(
        ...,
        min_length=40,
        max_length=40,
        description="Z85-encoded CURVE public key (40 ASCII chars)",
    )
    ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Requested session lifetime (60–86400 s)",
    )
    # Optional capability restrictions (empty = full access for role)
    max_gpus: int = Field(default=0, ge=0, description="Session GPU cap (0=unrestricted)")
    max_walltime_sec: int = Field(default=0, ge=0, description="Session walltime cap (0=unrestricted)")


class SessionRegisterResponse(BaseModel):
    session_id: str
    server_public_key: str
    expires_at: float
    ttl_seconds: int
    grants: dict = Field(default_factory=dict, description="Session grants summary")


class SessionInfoResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ttl_remaining: float


class SessionRefreshRequest(BaseModel):
    ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="New TTL from now (60–86400 s)",
    )


class ServerKeyResponse(BaseModel):
    server_public_key: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/server-key",
    response_model=ServerKeyResponse,
    summary="Get server CURVE public key",
)
async def get_server_key() -> ServerKeyResponse:
    """Return the server's CURVE public key for client configuration.

    This endpoint is unauthenticated — the public key is not secret.
    """
    from umrs.core.zmq_security import ZMQSecurityConfig

    cfg = ZMQSecurityConfig.from_env()
    if not cfg.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CURVE is not enabled on this server",
        )
    if not cfg.server_public_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server CURVE public key is not configured",
        )
    return ServerKeyResponse(server_public_key=cfg.server_public_key)


@router.post(
    "/session",
    response_model=SessionRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a CURVE session",
)
async def register_session(
    body: SessionRegisterRequest,
    identity: _JWTIdentity = Depends(_extract_jwt_identity),
    provider: Any = Depends(get_session_curve_provider),
) -> SessionRegisterResponse:
    """Register the client's ephemeral CURVE public key.

    The client must first generate a keypair locally, then POST the
    public key here.  The server responds with its own public key and
    the session ID.  Authorization grants from the JWT are bound to the
    session for identity-aware ZMQ communication.
    """
    from umrs.core.session_curve_provider import SessionGrants
    from umrs.core.zmq_security import ZMQSecurityConfig

    # Build session grants from JWT claims + request capabilities
    grants = SessionGrants(
        projects=tuple(identity.projects),
        roles=tuple(identity.roles),
        email=identity.email,
        max_gpus=body.max_gpus,
        max_walltime_sec=body.max_walltime_sec,
    )

    entry = provider.register(
        pubkey_z85=body.client_public_key,
        user_id=identity.user_id,
        ttl_seconds=body.ttl_seconds,
        grants=grants,
    )

    cfg = ZMQSecurityConfig.from_env()
    server_pub = cfg.server_public_key or ""

    return SessionRegisterResponse(
        session_id=entry.session_id,
        server_public_key=server_pub,
        expires_at=entry.expires_at,
        ttl_seconds=int(entry.ttl_remaining),
        grants={
            "user_id": identity.user_id,
            "projects": identity.projects,
            "roles": identity.roles,
            "max_gpus": body.max_gpus,
            "max_walltime_sec": body.max_walltime_sec,
        },
    )


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a CURVE session",
)
async def revoke_session(
    session_id: str,
    user_id: str = Depends(_require_user_id),
    provider: Any = Depends(get_session_curve_provider),
) -> None:
    """Revoke a specific session.

    Only the session owner can revoke it.
    """
    entry = provider.get_session(session_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    if entry.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot revoke another user's session",
        )
    provider.revoke(session_id)


@router.post(
    "/session/{session_id}/refresh",
    response_model=SessionInfoResponse,
    summary="Refresh session TTL",
)
async def refresh_session(
    session_id: str,
    body: SessionRefreshRequest,
    user_id: str = Depends(_require_user_id),
    provider: Any = Depends(get_session_curve_provider),
) -> SessionInfoResponse:
    """Extend the session's expiry time."""
    existing = provider.get_session(session_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    if existing.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot refresh another user's session",
        )
    updated = provider.refresh(session_id, body.ttl_seconds)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} expired or not found",
        )
    return SessionInfoResponse(
        session_id=updated.session_id,
        user_id=updated.user_id,
        created_at=updated.created_at,
        expires_at=updated.expires_at,
        ttl_remaining=updated.ttl_remaining,
    )


@router.get(
    "/session",
    response_model=list[SessionInfoResponse],
    summary="List active CURVE sessions",
)
async def list_sessions(
    user_id: str = Depends(_require_user_id),
    provider: Any = Depends(get_session_curve_provider),
) -> list[SessionInfoResponse]:
    """List the calling user's active CURVE sessions."""
    entries = provider.list_sessions(user_id=user_id)
    return [
        SessionInfoResponse(
            session_id=e.session_id,
            user_id=e.user_id,
            created_at=e.created_at,
            expires_at=e.expires_at,
            ttl_remaining=e.ttl_remaining,
        )
        for e in entries
    ]
