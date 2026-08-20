"""Three-tier access boundary for knowledge graphs and the science catalogue.

Single source of truth for read/write visibility — mirrors the model
registry's system/project/user pattern (development.instructions.md):

+----------+---------+-----------------------------+
| Scope    | Owner   | Access                      |
+----------+---------+-----------------------------+
| system   | Admin   | All users (read)            |
| project  | Project | Project members             |
| user     | User    | Owner only                  |
+----------+---------+-----------------------------+

Namespace conventions (no schema change required):
- ``kind`` on ``knowledge_graphs``: "system" | "project" | "user"
  (legacy "research"/"group"/"team" kinds behave as project-scoped)
- ``project_id`` namespaces on catalogue rows and graphs:
  ``system`` — platform-shared datasets/graphs (read for every caller),
  ``user:<uid>`` — a user's personal project (only that user),
  anything else — a collaborative project (members = callers verified
  into the same project).

Admin callers (``is_admin`` from gateway-verified headers) bypass all scope
checks. Anonymous callers (mvp mode, no identity headers) stay permissive —
this preserves legacy dev-loop behavior; in strict mode identity absence is
rejected upstream (401) before these predicates ever run. Enforcement binds
everywhere verified identity exists: member/owner deltas are loud 403/404.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_SPACE = "system"
USER_PREFIX = "user:"

_PROJECT_KINDS = {"project", "research", "group", "team", ""}
_USER_KINDS = {"user", "personal"}
_SYSTEM_KINDS = {"system"}


@dataclass(frozen=True)
class CallerScope:
    """Trust-gated caller for scope decisions (None when anonymous)."""

    user_id: str = ""
    project_id: str = ""
    is_admin: bool = False
    anonymous: bool = False

    @classmethod
    def from_identity(cls, identity: object | None) -> "CallerScope":
        """Adapt a verified-headers CallerIdentity (or None = anonymous)."""
        if identity is None:
            return cls(anonymous=True)
        return cls(
            user_id=str(getattr(identity, "user_id", "") or ""),
            project_id=str(getattr(identity, "project_id", "") or ""),
            is_admin=bool(getattr(identity, "is_admin", False)),
        )


def personal_project(user_id: str) -> str:
    """Personal project namespace for a user (personal uploads/graphs)."""
    return f"{USER_PREFIX}{user_id}"


def _is_user_owned(project_id: str) -> bool:
    return project_id.startswith(USER_PREFIX)


def project_readable(project_id: str, caller: CallerScope) -> bool:
    """May the caller read rows under this project namespace?"""
    if caller.is_admin or caller.anonymous:
        return True
    if not project_id:
        return True  # global rows are readable by any authenticated tier
    if project_id == SYSTEM_SPACE:
        return True
    if _is_user_owned(project_id):
        return project_id == personal_project(caller.user_id)
    return project_id == caller.project_id


def project_writable(project_id: str, caller: CallerScope) -> bool:
    """May the caller write rows under this project namespace?"""
    if caller.is_admin or caller.anonymous:
        return True
    if project_id == SYSTEM_SPACE:
        return False  # system space is admin-managed
    if not project_id:
        return False
    if _is_user_owned(project_id):
        return project_id == personal_project(caller.user_id)
    return project_id == caller.project_id


def graph_readable(
    *, kind: str, project_id: str | None, owner_id: str, caller: CallerScope
) -> bool:
    """May the caller read one knowledge graph?

    - system graphs → readable by everyone (shared public layer)
    - user/personal graphs → owner only
    - project-like kinds → project namespace rule
    """
    if caller.is_admin or caller.anonymous:
        return True
    reduced = (kind or "").strip().lower()
    if reduced in _SYSTEM_KINDS:
        return True
    if reduced in _USER_KINDS:
        return bool(owner_id) and owner_id == caller.user_id
    return project_readable(project_id or "", caller)


def graph_writable(
    *, kind: str, project_id: str | None, owner_id: str, caller: CallerScope
) -> bool:
    """May the caller update/archive one knowledge graph?

    Lifecycle is admin for system graphs, owner for personal graphs,
    project members for project graphs. Agent-originated writes stay
    blocked separately by the registry mutation guard.
    """
    if caller.is_admin or caller.anonymous:
        return True
    reduced = (kind or "").strip().lower()
    if reduced in _SYSTEM_KINDS:
        return False
    if reduced in _USER_KINDS:
        return bool(owner_id) and owner_id == caller.user_id
    return project_writable(project_id or "", caller)
