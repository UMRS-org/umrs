"""Canonical scope semantics for the R4/UMRS governance model.

This module defines the formal scope hierarchy, visibility rules,
inheritance semantics, and sharing model for all governed resources.

Scope Hierarchy
===============

    system  →  project  →  group  →  user

- **system**: Admin-managed, visible to all authenticated users.
  Created by admins, owned by the platform. Cannot be deleted by
  non-admins. Examples: shared inference endpoints, system models.

- **project**: Owned by a project lead, visible to project members.
  Resources scoped to a project inherit the project's access list.
  Examples: team endpoints, project RAG indexes, project MCP servers.

- **group**: Owned by a group (lab, collaboration, institution).
  Visible to all group members across projects. Groups enable
  cross-project resource sharing without elevating to system scope.
  Examples: shared datasets, lab-wide MCP tools, institutional models.

- **user**: Owned by an individual user, visible only to the owner
  (and admins). Examples: personal endpoints, user experiments.

Visibility Rules
================

A user can see a resource if ANY of these conditions hold:

1. The resource is ``system`` scope (all authenticated users).
2. The resource is ``project`` scope AND the user is a member of
   that project.
3. The resource is ``group`` scope AND the user is a member of
   that group.
4. The resource is ``user`` scope AND the user is the owner.
5. The resource has explicit shares granting the user access.
6. The user is an admin (can see all resources).

Inheritance
===========

- Project-scoped resources inherit the project's member list for
  visibility. No explicit per-resource ACL is needed at this level.
- Group-scoped resources inherit the group's member list.
- System-scoped resources are implicitly visible to all. No
  inheritance needed.
- User-scoped resources have no inheritance — owner-only.

Sharing
=======

Resources can be *promoted* to a broader scope:

    user → group → project → system

Demotion (narrowing scope) is also supported but requires admin
action to prevent accidental access revocation.

Resources can also be explicitly shared with individual users or
groups via the resource_shares table, without changing scope.

Promotion/demotion always emits a governance audit event.

Ownership Transfer
==================

- Within the same scope: direct transfer (admin or owner).
- Across scopes: promotion/demotion (always requires admin).
- Orphaned resources: detected via ``find_orphans()`` in
  ``governance.py``, cleaned up by admin.

Deletion Rights
===============

- ``user`` scope: owner or admin can delete.
- ``group`` scope: group admin or platform admin can delete.
- ``project`` scope: project admin or platform admin can delete.
- ``system`` scope: platform admin only.

Retention
=========

- Resources are retained as long as their owning entity exists.
- When a user is deactivated: user-scoped resources are marked
  ``orphaned`` but not auto-deleted (admin cleanup required).
- When a group is disbanded: group-scoped resources are frozen
  until admin decides to delete, transfer, or re-scope.
- When a project is archived: project-scoped resources are frozen
  (read-only) until admin decides to delete or transfer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Scope(str, Enum):
    """Canonical scope tiers for governed resources."""

    SYSTEM = "system"
    PROJECT = "project"
    GROUP = "group"
    USER = "user"

    @classmethod
    def from_str(cls, value: str) -> "Scope":
        """Parse a scope string, defaulting to USER for unknown values."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.USER

    def is_broader_than(self, other: "Scope") -> bool:
        """Return True if this scope is broader (more visible) than other."""
        order = {Scope.SYSTEM: 0, Scope.PROJECT: 1, Scope.GROUP: 2, Scope.USER: 3}
        return order[self] < order[other]


# Visibility rank: lower number = broader visibility
_SCOPE_RANK = {Scope.SYSTEM: 0, Scope.PROJECT: 1, Scope.GROUP: 2, Scope.USER: 3}


def can_view_resource(
    *,
    resource_scope: str,
    resource_owner_id: str,
    resource_project_id: str,
    resource_group_id: str = "",
    viewer_id: str,
    viewer_projects: list[str],
    viewer_groups: list[str] | None = None,
    is_admin: bool = False,
) -> bool:
    """Determine whether a viewer can see a resource.

    Parameters
    ----------
    resource_scope : str
        The scope tier of the resource ("system", "project", "group", "user").
    resource_owner_id : str
        The owner of the resource.
    resource_project_id : str
        The project the resource belongs to (empty if unscoped).
    resource_group_id : str
        The group the resource belongs to (empty if unscoped).
    viewer_id : str
        The authenticated user viewing.
    viewer_projects : list[str]
        Projects the viewer is a member of.
    viewer_groups : list[str] | None
        Groups the viewer is a member of. None treated as empty.
    is_admin : bool
        Whether the viewer has admin privileges.

    Returns
    -------
    bool
        True if the viewer is allowed to see this resource.
    """
    if is_admin:
        return True

    scope = Scope.from_str(resource_scope)

    if scope == Scope.SYSTEM:
        return True  # All authenticated users can see system resources

    if scope == Scope.PROJECT:
        return resource_project_id in viewer_projects

    if scope == Scope.GROUP:
        groups = viewer_groups or []
        return resource_group_id in groups

    if scope == Scope.USER:
        return viewer_id == resource_owner_id

    return False


def can_delete_resource(
    *,
    resource_scope: str,
    resource_owner_id: str,
    actor_id: str,
    is_admin: bool = False,
    is_project_admin: bool = False,
    is_group_admin: bool = False,
) -> bool:
    """Determine whether an actor can delete a resource.

    Parameters
    ----------
    resource_scope : str
        The scope tier of the resource.
    resource_owner_id : str
        The owner of the resource.
    actor_id : str
        The user attempting deletion.
    is_admin : bool
        Whether the actor is a platform admin.
    is_project_admin : bool
        Whether the actor is a project admin for the resource's project.
    is_group_admin : bool
        Whether the actor is a group admin for the resource's group.
    """
    if is_admin:
        return True

    scope = Scope.from_str(resource_scope)

    if scope == Scope.SYSTEM:
        return False  # Only platform admins

    if scope == Scope.PROJECT:
        return is_project_admin

    if scope == Scope.GROUP:
        return is_group_admin

    if scope == Scope.USER:
        return actor_id == resource_owner_id

    return False


def validate_promotion(
    current_scope: str,
    target_scope: str,
    *,
    is_admin: bool = False,
) -> tuple[bool, str]:
    """Check if a scope promotion/demotion is allowed.

    Returns (allowed, reason).
    """
    current = Scope.from_str(current_scope)
    target = Scope.from_str(target_scope)

    if current == target:
        return True, "no change"

    # Promotion (user → project → system): owner or admin
    if target.is_broader_than(current):
        # Promoting to system always requires admin
        if target == Scope.SYSTEM and not is_admin:
            return False, "promotion to system scope requires admin"
        return True, "promotion allowed"

    # Demotion (system → project → user): always requires admin
    if not is_admin:
        return False, "demotion requires admin"
    return True, "demotion allowed (admin)"


def filter_by_visibility(
    resources: list[dict[str, Any]],
    *,
    viewer_id: str,
    viewer_projects: list[str],
    viewer_groups: list[str] | None = None,
    is_admin: bool = False,
    scope_key: str = "tier",
    owner_key: str = "owner_id",
    project_key: str = "project_id",
    group_key: str = "group_id",
) -> list[dict[str, Any]]:
    """Filter a list of resource dicts to only those visible to the viewer.

    Generic utility: works with any dict that has scope, owner, project,
    and group fields. Field names are configurable via *_key parameters.
    """
    return [
        r
        for r in resources
        if can_view_resource(
            resource_scope=r.get(scope_key, "user"),
            resource_owner_id=r.get(owner_key, ""),
            resource_project_id=r.get(project_key, ""),
            resource_group_id=r.get(group_key, ""),
            viewer_id=viewer_id,
            viewer_projects=viewer_projects,
            viewer_groups=viewer_groups,
            is_admin=is_admin,
        )
    ]
