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

The policy itself is the shared one-guard module
(``umrs.shared.access_scopes``, ADR-080); this module keeps the
endpoint-specific surface (metadata key, owning-project implicit scope, the
EndpointCaller/AccessError names ADR-078 landed with).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..shared.access_scopes import (
    MAX_ACCESS_ENTRIES,
    AccessListError,
    ResourceCaller,
    scopes_admit,
    stored_access_scopes,
)
from ..shared.access_scopes import (
    validate_access as _validate_access,
)
from ..shared.scope_grammar import Scope

ACCESS_KEY = "access"

__all__ = [
    "ACCESS_KEY",
    "MAX_ACCESS_ENTRIES",
    "AccessError",
    "EndpointCaller",
    "can_use",
    "endpoint_scopes",
    "is_shared_with",
    "usable_endpoints",
    "validate_access",
]


class AccessError(AccessListError):
    """An access list the caller may not set (or that is malformed)."""


class EndpointCaller(ResourceCaller):
    """The routing caller: verified identity + lazily resolved groups."""


def endpoint_scopes(endpoint: Any) -> list[Scope]:
    """The endpoint's effective access list (owning project always included)."""
    owner = Scope(kind="project", id=str(endpoint.project_id or ""))
    scopes: list[Scope] = [owner]
    raw = (getattr(endpoint, "metadata", None) or {}).get(ACCESS_KEY)
    for scope in stored_access_scopes(raw):
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def can_use(endpoint: Any, caller: EndpointCaller) -> bool:
    """May this caller route requests to the endpoint?"""
    return scopes_admit(endpoint_scopes(endpoint), caller)


def is_shared_with(endpoint: Any, caller: EndpointCaller) -> bool:
    """Usable by the caller through a grant other than owning the project."""
    return endpoint.project_id != caller.project_id and can_use(endpoint, caller)


def validate_access(raw: Any, caller: EndpointCaller) -> list[str]:
    """Normalize a requested access list; AccessError when it is not grantable.

    ``None`` → ``[]`` (owning project only). Duplicates collapse; order kept.
    """
    try:
        return _validate_access(raw, caller)
    except AccessListError as exc:
        raise AccessError(str(exc)) from exc


def usable_endpoints(
    endpoints: Iterable[Any], caller: EndpointCaller, *, shared_only: bool = False
) -> list[Any]:
    """Filter registry rows to those the caller may use (optionally only shared ones)."""
    check = is_shared_with if shared_only else can_use
    return [ep for ep in endpoints if check(ep, caller)]
