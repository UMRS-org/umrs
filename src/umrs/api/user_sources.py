"""Sealed user-credential routes (R1-1, design §3.6).

    POST   /umrs/user-sources/sealed          push (seal + metadata)
    GET    /umrs/user-sources/sealed          list-mine (metadata + fingerprints)
    DELETE /umrs/user-sources/sealed/{name}   unpush (tombstone + revoke cascade)

Security rules:
- Every route requires a verified caller identity (G-6 discipline; the
  router-wide dependency enforces the envelope).
- ``owner`` comes from the verified caller — never from the body.
- ``base_url`` is validated server-side by ``provider_url`` (the client-side
  check is untrusted).
- The request body (which carries the plaintext key on push) is never
  logged by this module, and push fails soft when sealing is not configured
  (``UMRS_SEAL_PUBKEY_B64``) with a named 503 — no plaintext fallback write.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from umrs.knowledge.access_scope import CallerScope
from umrs.security.provider_url import ProviderUrlError, validate_provider_base_url
from umrs.security.seal import SealError, key_fingerprint, seal_credential
from umrs.security.sealed_sources import SealedSourceStore, SealedStoreUnavailable

from ._scope_identity import get_caller_scope
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/umrs/user-sources",
    tags=["user-sources"],
    dependencies=[Depends(get_caller_scope)],
)

_VALID_PROVIDER_KINDS = {
    "openai",
    "openai_compat",
    "anthropic",
    "google",
    "azure_openai",
}


def _store(request: Request) -> SealedSourceStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.usersources(
                "unavailable",
                "sealed source store requires the shared umrs_db pool",
            )
        )
    try:
        return SealedSourceStore(pool)
    except SealedStoreUnavailable as exc:
        raise_typed(PlatformError.usersources("unavailable", str(exc)), cause=exc)


def _seal_pubkey() -> str:
    from umrs.security.seal import recipient_keys

    try:
        return recipient_keys().seal_pubkey()
    except SealError as exc:
        raise_typed(
            PlatformError.usersources(
                "unavailable",
                (
                    "sealed store not configured on this server: "
                    "UMRS_SEAL_PUBKEY_B64 missing (push unavailable, no plaintext fallback)"
                ),
            ),
            cause=exc,
        )


class PushSourceRequest(BaseModel):
    """Push body. The api_key crosses the wire exactly once on TLS (design F-1)."""

    name: str = Field(
        ..., min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    provider_kind: str = Field(
        ..., description="openai | openai_compat | anthropic | google | azure_openai"
    )
    base_url: str = Field(..., min_length=1, max_length=2048)
    api_key: str = Field(..., min_length=1, max_length=4096)
    allow_paths: list[str] = Field(default_factory=list)


class PushSourceResponse(BaseModel):
    id: str
    allow_host: str
    key_fingerprint: str
    seal_key_id: str
    replaced: bool


@router.post(
    "/sealed", response_model=PushSourceResponse, status_code=status.HTTP_201_CREATED
)
async def push_source(
    body: PushSourceRequest,
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> PushSourceResponse:
    if body.provider_kind not in _VALID_PROVIDER_KINDS:
        raise_typed(
            PlatformError.usersources(
                "bad_request", f"unknown provider_kind {body.provider_kind!r}"
            )
        )
    owner = caller.user_id
    if not owner:
        raise_typed(
            PlatformError.usersources(
                "unauthorized", "user identity required to push a source"
            )
        )
    try:
        normalized = validate_provider_base_url(body.base_url)
    except ProviderUrlError as exc:
        raise HTTPException(
            status_code=400, detail={"error": "invalid_base_url", "reason": exc.reason}
        ) from exc

    pubkey = _seal_pubkey()
    try:
        blob = seal_credential(
            body.api_key,
            pubkey_b64=pubkey,
            owner=owner,
            name=body.name,
            provider_kind=body.provider_kind,
            allow_host=normalized.allow_host,
        )
    except SealError as exc:
        raise_typed(PlatformError.usersources("failure", "sealing failed"), cause=exc)

    store = _store(request)
    try:
        replaced = store.tombstone(owner, body.name) is not None
        source_id = store.insert(
            owner=owner,
            name=body.name,
            provider_kind=body.provider_kind,
            base_url=normalized.base_url,
            allow_host=normalized.allow_host,
            allow_paths=tuple(body.allow_paths),
            key_fingerprint=key_fingerprint(body.api_key),
            seal_key_id=blob.seal_key_id,
            dek_wrapped=blob.dek_wrapped,
            nonce=blob.nonce,
            ciphertext=blob.ciphertext,
        )
        store.audit(
            action="push",
            owner=owner,
            source_id=source_id,
            detail={"allow_host": normalized.allow_host, "replaced": replaced},
        )
    except Exception:
        logger.exception(
            "sealed source push failed for owner=%s name=%s", owner, body.name
        )
        raise_typed(PlatformError.usersources("failure", "sealed source write failed"))

    return PushSourceResponse(
        id=source_id,
        allow_host=normalized.allow_host,
        key_fingerprint=key_fingerprint(body.api_key),
        seal_key_id=blob.seal_key_id,
        replaced=replaced,
    )


@router.get("/sealed", response_model=list[dict[str, Any]])
async def list_my_sources(
    request: Request,
    with_leases: bool = Query(
        False, description="include active lease counts (TUI badge)"
    ),
    caller: CallerScope = Depends(get_caller_scope),
) -> list[dict[str, Any]]:
    if not caller.user_id:
        raise_typed(PlatformError.usersources("unauthorized", "user identity required"))
    store = _store(request)
    return [
        meta.to_dict()
        for meta in store.list_mine(caller.user_id, with_lease_counts=with_leases)
    ]


@router.delete("/sealed/{name}", response_model=dict[str, Any])
async def unpush_source(
    name: str,
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> dict[str, Any]:
    if not caller.user_id:
        raise_typed(PlatformError.usersources("unauthorized", "user identity required"))
    store = _store(request)
    source_id = store.tombstone(caller.user_id, name)
    if source_id is None:
        raise_typed(
            PlatformError.usersources("not_found", f"Source {name!r} not found")
        )

    # Cascade: revoke every lease still referencing the source (D-6 instant deny).
    from ..jobs.secret_broker import build_default_broker

    broker = build_default_broker()
    revoked = broker.revoke_by_source(source_id, actor=caller.user_id)
    store.audit(
        action="unpush",
        owner=caller.user_id,
        source_id=source_id,
        detail={"leases_revoked": revoked},
    )
    return {
        "name": name,
        "source_id": source_id,
        "tombstoned": True,
        "leases_revoked": revoked,
    }


def user_source_audit_rows(
    store: SealedSourceStore, owner: str, source_id: str | None, limit: int
) -> list[dict[str, Any]]:
    """Owner-filtered audit read (used by /audit route, R2 surfaces)."""
    conn = store._pool.getconn()
    try:
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_id:
                cur.execute(
                    "SELECT created_at, action, actor_type, source_id, detail "
                    "FROM user_source_audit WHERE owner = %s AND source_id = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (owner, source_id, limit),
                )
            else:
                cur.execute(
                    "SELECT created_at, action, actor_type, source_id, detail "
                    "FROM user_source_audit WHERE owner = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (owner, limit),
                )
            rows = cur.fetchall() or []
        return [
            {
                "created_at": r["created_at"].isoformat()
                if hasattr(r["created_at"], "isoformat")
                else r["created_at"],
                "action": r["action"],
                "actor_type": r["actor_type"],
                "source_id": str(r["source_id"]) if r["source_id"] else None,
                "detail": r["detail"],
            }
            for r in rows
        ]
    finally:
        store._pool.putconn(conn)


@router.get("/audit", response_model=list[dict[str, Any]])
async def list_audit(
    request: Request,
    source_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    caller: CallerScope = Depends(get_caller_scope),
) -> list[dict[str, Any]]:
    if not caller.user_id:
        raise_typed(PlatformError.usersources("unauthorized", "user identity required"))
    store = _store(request)
    return user_source_audit_rows(store, caller.user_id, source_id, limit)
