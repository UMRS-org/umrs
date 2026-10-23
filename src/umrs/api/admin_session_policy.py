"""Session policy rules: admin routes and the project-owner layer (ADR-099 D-3).

    GET    /umrs/admin/session-policy                    rules (?venue=&scope=)
    GET    /umrs/admin/session-policy/effective          ?user=&project=&venue=
    PUT    /umrs/admin/session-policy/{scope}/{subject}  write an admin rule
    DELETE /umrs/admin/session-policy/{scope}/{subject}  remove one (?venue=&layer=)

``subject`` is ``_`` for the ``site`` scope. ``venue`` is the site the rule
binds: absent = this site, ``""`` = every site. Access: the admin credential
of the admin session routes (``admin_sessions.require_session_admin``: an
admin JWT or a gateway-verified admin identity).

The project owner's layer (``PUT /umrs/projects/{id}/session-policy``) is
served from ``api/projects.py`` with the helpers below: an owner rule may
only narrow what the admin layer allows (``check_owner_rule``).

Every write is audited as ``venue.policy.set`` / ``venue.policy.delete``
(actor, key, new doc). Refusals are typed ``sessionpolicy.*`` errors; an
unreadable store is ``sessionpolicy.unavailable`` (503).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.session_policy import (
    SCOPES,
    PolicyError,
    PolicyRule,
    PolicyRuleStore,
    check_owner_rule,
    effective_limits,
    effective_limits_for,
    site_defaults,
    validate_rule_doc,
)
from umrs.session_policy.model import check_rule_key
from umrs.venues.session_audit import audit_venue

from .admin_sessions import require_session_admin
from .companion_venue import policy_typed
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/admin/session-policy", tags=["admin-session-policy"])

SITE_SUBJECT = "_"


# ── wiring ─────────────────────────────────────────────────────────────────


def _unavailable(message: str, cause: Exception | None = None) -> NoReturn:
    raise_typed(PlatformError.sessionpolicy("unavailable", message), cause=cause)


def get_policy_store(request: Request) -> PolicyRuleStore:
    """Process-cached rule store (tests inject ``app.state.session_policy_store``)."""
    cached = getattr(request.app.state, "session_policy_store", None)
    if cached is not None:
        return cached
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        _unavailable("session policy store unavailable: shared umrs_db pool absent")
    store = PolicyRuleStore(pool)
    request.app.state.session_policy_store = store
    return store


def this_site(request: Request) -> str:
    """The site this server's rules bind by default (the venue service's)."""
    service = getattr(request.app.state, "companion_venue_service", None)
    if service is not None:
        return str(service.site_name or "")
    from .companion_venue import _site_name

    return _site_name()


def _guarded(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a store/resolver call; refusals typed, store failures a 503."""
    try:
        return fn(*args, **kwargs)
    except PolicyError as exc:
        policy_typed(exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("session policy store call failed", exc_info=True)
        _unavailable(f"session policy rules unreadable: {type(exc).__name__}", exc)


def _audit(event: str, actor: str, rule: PolicyRule, doc: Any) -> None:
    audit_venue(
        f"policy.{event}",
        actor=actor,
        session_id="",
        project=rule.subject if rule.scope == "project" else "",
        scope="system",
        key=_key(rule),
        doc=doc,
    )


# ── the owner layer (served from api/projects.py) ──────────────────────────


def _owner_rule_doc(
    store: PolicyRuleStore, venue: str, project_id: str
) -> dict[str, Any] | None:
    rule = store.get(venue, "project", project_id, "owner") or store.get(
        "", "project", project_id, "owner"
    )
    return dict(rule.doc) if rule is not None else None


def owner_policy_view(request: Request, project_id: str, caller: str) -> dict[str, Any]:
    """``{"owner_rule", "limits"}``: the owner layer and the caller's limits."""
    store = get_policy_store(request)
    venue = this_site(request)
    return {
        "project_id": project_id,
        "venue": venue,
        "owner_rule": _guarded(_owner_rule_doc, store, venue, project_id),
        "limits": _guarded(
            effective_limits_for,
            store,
            venue=venue,
            user_id=caller,
            project_id=project_id,
        ),
    }


def _admin_limits(
    store: PolicyRuleStore, venue: str, project_id: str
) -> dict[str, Any]:
    """The project's limits from the admin layer alone (no user, no owner)."""
    rules = [
        r
        for r in store.applicable(venue=venue, user_id="", project_id=project_id)
        if r.layer == "admin"
    ]
    return effective_limits(
        rules, venue=venue, user_id="", project_id=project_id, defaults=site_defaults()
    )


def _put_owner(
    store: PolicyRuleStore, venue: str, project_id: str, doc: Any, note: str, actor: str
) -> PolicyRule:
    clean = validate_rule_doc(doc, scope="project", layer="owner")
    check_owner_rule(clean, _admin_limits(store, venue, project_id))
    return store.put(
        PolicyRule(
            venue=venue,
            scope="project",
            subject=project_id,
            layer="owner",
            doc=clean,
            set_by=actor,
            note=note[:500],
        )
    )


def put_owner_rule(
    request: Request, project_id: str, doc: Any, note: str, actor: str
) -> dict[str, Any]:
    """Write the project owner's narrowing rule at this site (audited)."""
    store = get_policy_store(request)
    rule = _guarded(_put_owner, store, this_site(request), project_id, doc, note, actor)
    _audit("set", actor, rule, rule.doc)
    return rule.to_dict()


# ── admin routes ───────────────────────────────────────────────────────────


class AdminRuleBody(BaseModel):
    doc: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    venue: str | None = Field(
        default=None,
        description="site the rule binds: absent = this site, '' = every site",
    )


def _checked_scope(scope: str) -> str:
    if scope not in SCOPES:
        policy_typed(
            PolicyError(
                "invalid", f"scope must be one of {', '.join(SCOPES)}", field="scope"
            )
        )
    return scope


def _subject(scope: str, subject: str) -> str:
    """The stored subject: ``_`` names the site scope's empty subject."""
    value = "" if subject == SITE_SUBJECT else subject.strip()
    try:
        check_rule_key(scope, value, "admin")
    except PolicyError as exc:
        policy_typed(exc)
    return value


def _venue(request: Request, venue: str | None) -> str:
    return this_site(request) if venue is None else venue.strip()


@router.get("", response_model=dict[str, Any])
async def admin_list_policy_rules(
    request: Request,
    venue: str | None = Query(
        None, description="'' = the all-sites rules; absent = all"
    ),
    scope: str | None = Query(None, description="site|project|user"),
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Every stored limit rule (both layers).
    # auth: admin(jwt|verified-admin)
    """
    del admin
    if scope is not None:
        _checked_scope(scope)
    store = get_policy_store(request)
    rules = await asyncio.to_thread(_guarded, store.list, venue=venue, scope=scope)
    return {"rules": [r.to_dict() for r in rules], "site": this_site(request)}


@router.get("/effective", response_model=dict[str, Any])
async def admin_effective_limits(
    request: Request,
    user: str = Query(..., description="the session owner"),
    project: str = Query("", description="the session's project"),
    venue: str | None = Query(None, description="site (default: this site)"),
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """What a start of ``user`` in ``project`` would get (before the choice).
    # auth: admin(jwt|verified-admin)
    """
    del admin
    store = get_policy_store(request)
    return await asyncio.to_thread(
        _guarded,
        effective_limits_for,
        store,
        venue=_venue(request, venue),
        user_id=user.strip(),
        project_id=project.strip(),
    )


def _put_admin(store: PolicyRuleStore, rule: PolicyRule) -> PolicyRule:
    clean = validate_rule_doc(rule.doc, scope=rule.scope, layer="admin")
    return store.put(replace(rule, doc=clean))


@router.put("/{scope}/{subject}", response_model=dict[str, Any])
async def admin_put_policy_rule(
    scope: str,
    subject: str,
    body: AdminRuleBody,
    request: Request,
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Write (replace) the admin rule at ``(venue, scope, subject)``.
    # auth: admin(jwt|verified-admin)
    """
    scope = _checked_scope(scope)
    wanted = PolicyRule(
        venue=_venue(request, body.venue),
        scope=scope,
        subject=_subject(scope, subject),
        layer="admin",
        doc=body.doc,
        set_by=admin,
        note=body.note[:500],
    )
    store = get_policy_store(request)
    rule = await asyncio.to_thread(_guarded, _put_admin, store, wanted)
    _audit("set", admin, rule, rule.doc)
    return rule.to_dict()


@router.delete("/{scope}/{subject}", response_model=dict[str, Any])
async def admin_delete_policy_rule(
    scope: str,
    subject: str,
    request: Request,
    venue: str | None = Query(None, description="site (default: this site; '' = all)"),
    layer: str = Query("admin", description="admin|owner"),
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Remove one rule (an admin may also remove a project's owner layer).
    # auth: admin(jwt|verified-admin)
    """
    scope = _checked_scope(scope)
    key = PolicyRule(
        venue=_venue(request, venue),
        scope=scope,
        subject=_subject(scope, subject),
        layer=layer,
    )
    _guarded(check_rule_key, key.scope, key.subject, key.layer)
    store = get_policy_store(request)
    deleted = await asyncio.to_thread(_guarded, store.delete, *key.key)
    if not deleted:
        policy_typed(
            PolicyError("not_found", "no such session policy rule", **_key(key))
        )
    _audit("delete", admin, key, None)
    return {"deleted": True}


def _key(rule: PolicyRule) -> dict[str, str]:
    return {
        "venue": rule.venue,
        "scope": rule.scope,
        "subject": rule.subject,
        "layer": rule.layer,
    }
