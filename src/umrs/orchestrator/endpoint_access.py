"""Endpoint access scopes — who may route inference to an endpoint (ADR-078).

An endpoint row carries ``metadata["access"]``: a list of scope strings in the
canonical grammar (``umrs.shared.scope_grammar``)::

    user:<user_id>        that user only
    project:<project_id>  callers whose verified project is that project
    group:<group_id>      members of that (active) group
    system                every authenticated caller

A row without the key keeps the historical rule: ``project:<owning project>``
only — nothing becomes shared by accident. The owning project always keeps
access (it is where the endpoint was provisioned and is billed).

Access governs *use* (routing, model listing, warm reuse). Lifecycle writes
(deprovision, refresh, re-scoping) stay with the owning project or an admin.
Sharing spends only the owner's GPU, so the owner may name any user, project
or group; ``system`` (everyone's model list) is admin-only.

Foundation module: stdlib + ``umrs.shared`` only; the caller's groups come in
through a lazy resolver so a request that never meets a group-scoped
endpoint never touches the database.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..shared.scope_grammar import Scope, ScopeError, parse

ACCESS_KEY = "access"
MAX_ACCESS_ENTRIES = 16
_KINDS = ("system", "project", "group", "user")


class AccessError(ValueError):
    """An access list the caller may not set (or that is malformed)."""


@dataclass
class EndpointCaller:
    """The routing caller: verified identity + lazily resolved groups."""

    user_id: str
    project_id: str
    is_admin: bool = False
    groups_resolver: Callable[[], Iterable[str]] | None = None
    _groups: frozenset[str] | None = field(default=None, init=False, repr=False)

    def groups(self) -> frozenset[str]:
        if self._groups is None:
            resolver = self.groups_resolver
            self._groups = frozenset(resolver() if resolver and self.user_id else ())
        return self._groups


def _parse_entry(raw: Any) -> Scope:
    try:
        scope = parse(str(raw))
    except ScopeError as exc:
        raise AccessError(str(exc)) from exc
    if scope.kind not in _KINDS:
        raise AccessError(f"endpoint access cannot use {scope.kind!r} scopes")
    return scope


def endpoint_scopes(endpoint: Any) -> list[Scope]:
    """The endpoint's effective access list (owning project always included)."""
    raw = (getattr(endpoint, "metadata", None) or {}).get(ACCESS_KEY)
    owner = Scope(kind="project", id=str(endpoint.project_id or ""))
    scopes: list[Scope] = [owner]
    for entry in raw if isinstance(raw, list) else ():
        try:
            scope = _parse_entry(entry)
        except AccessError:
            continue  # a malformed stored entry grants nothing
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def _scope_admits(scope: Scope, caller: EndpointCaller) -> bool:
    if scope.kind == "system":
        return True
    if scope.kind == "project":
        return bool(scope.id) and scope.id == caller.project_id
    if scope.kind == "user":
        return bool(scope.id) and scope.id == caller.user_id
    return scope.id in caller.groups()


def can_use(endpoint: Any, caller: EndpointCaller) -> bool:
    """May this caller route requests to the endpoint?"""
    scopes = endpoint_scopes(endpoint)
    # Cheap kinds first so a group lookup happens only when it decides.
    ordered = sorted(scopes, key=lambda s: s.kind == "group")
    return any(_scope_admits(scope, caller) for scope in ordered)


def is_shared_with(endpoint: Any, caller: EndpointCaller) -> bool:
    """Usable by the caller through a grant other than owning the project."""
    return endpoint.project_id != caller.project_id and can_use(endpoint, caller)


def _may_grant(scope: Scope, caller: EndpointCaller) -> bool:
    return caller.is_admin or scope.kind != "system"


def validate_access(raw: Any, caller: EndpointCaller) -> list[str]:
    """Normalize a requested access list; AccessError when it is not grantable.

    ``None`` → ``[]`` (owning project only). Duplicates collapse; order kept.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise AccessError("access must be a list of scopes")
    if len(raw) > MAX_ACCESS_ENTRIES:
        raise AccessError(f"at most {MAX_ACCESS_ENTRIES} access entries")
    normalized: list[str] = []
    for entry in raw:
        scope = _parse_entry(entry)
        if not _may_grant(scope, caller):
            raise AccessError(f"not allowed to share an endpoint with {scope}")
        if str(scope) not in normalized:
            normalized.append(str(scope))
    return normalized


def usable_endpoints(
    endpoints: Iterable[Any], caller: EndpointCaller, *, shared_only: bool = False
) -> list[Any]:
    """Filter registry rows to those the caller may use (optionally only shared ones)."""
    check = is_shared_with if shared_only else can_use
    return [ep for ep in endpoints if check(ep, caller)]
