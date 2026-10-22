"""Row-carried access scopes — ONE guard for every shareable resource (ADR-080).

A shareable row (endpoint, knowledge graph, scientific catalog, workflow,
bucket, RAG corpus) carries an ``access`` list of scope strings in the
canonical grammar (``umrs.shared.scope_grammar``)::

    user:<user_id>        that user only
    project:<project_id>  callers verified into that project
    circle:<project_id>/<circle_id>
                          members of that (active) circle — a user may
                          hold many memberships; any match admits
    system                every authenticated caller

Semantics (uniform across families, generalized from ADR-078 endpoints):

- An absent/empty list keeps the historical posture — nothing becomes
  shared by accident; malformed stored entries grant nothing.
- The list governs *use*: reads and runs. Lifecycle writes (re-scoping,
  retire, delete) stay with each family's existing write ladder.
- Setting the list: the write-tier owner (or an admin) may name any
  ``user:`` / ``project:`` / ``circle:`` scope; ``system`` is admin-only.
- Circle membership resolves through the single authority (migration
  0119; retired circles never resolve) and lazily — a request that never
  meets a circle-scoped row never touches the database.
- ``group:`` entries (superseded by ADR-086) parse but admit no one and
  are rejected on writes.

Foundation module: stdlib + ``umrs.shared`` only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .scope_grammar import Scope, ScopeError, parse

#: Scope kinds an access entry may use (``session:`` never shares a resource;
#: ``group:`` is superseded by ``circle:`` — ADR-086).
ACCESS_KINDS = ("system", "project", "circle", "user")
#: Cap on one row's access list (parsing/lookup stays bounded).
MAX_ACCESS_ENTRIES = 16


class AccessListError(ValueError):
    """An access list the caller may not set (or that is malformed)."""


@dataclass
class ResourceCaller:
    """The caller an access decision is made for.

    ``circles_resolver`` is called at most once per instance and only when a
    ``circle:`` scope actually decides the outcome.
    """

    user_id: str
    project_id: str = ""
    is_admin: bool = False
    circles_resolver: Callable[[], Iterable[str]] | None = None
    #: ``project/circle`` → the active circle it names (exact id), or the
    #: one it differs from only in letter case, or None; None = no lookup.
    circle_finder: Callable[[str], str | None] | None = None
    _circles: frozenset[str] | None = field(default=None, init=False, repr=False)

    def circles(self) -> frozenset[str]:
        """Active circle refs (``project/circle``) of the caller (lazy, memoized)."""
        if self._circles is None:
            resolver = self.circles_resolver
            self._circles = frozenset(resolver() if resolver and self.user_id else ())
        return self._circles


def parse_access_entry(raw: Any) -> Scope:
    """One stored/requested entry → Scope; AccessListError on bad kinds."""
    try:
        scope = parse(str(raw))
    except ScopeError as exc:
        raise AccessListError(str(exc)) from exc
    if scope.kind not in ACCESS_KINDS:
        raise AccessListError(f"access cannot use {scope.kind!r} scopes")
    return scope


def stored_access_scopes(raw: Any) -> list[Scope]:
    """Parse a stored access list; malformed entries grant nothing."""
    scopes: list[Scope] = []
    for entry in raw if isinstance(raw, list) else ():
        try:
            scope = parse_access_entry(entry)
        except AccessListError:
            continue
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def scope_admits(scope: Scope, caller: ResourceCaller) -> bool:
    """Does this one scope admit the caller?"""
    if scope.kind == "system":
        return True
    if scope.kind == "project":
        return bool(scope.id) and scope.id == caller.project_id
    if scope.kind == "user":
        return bool(scope.id) and scope.id == caller.user_id
    if scope.kind == "circle":
        return scope.id in caller.circles()
    # group: (superseded, ADR-086) and anything else admit no one.
    return False


def scopes_admit(scopes: Iterable[Scope], caller: ResourceCaller) -> bool:
    """Any scope admits; cheap kinds first so a circle lookup decides last."""
    ordered = sorted(scopes, key=lambda s: s.kind == "circle")
    return any(scope_admits(scope, caller) for scope in ordered)


def access_admits(raw: Any, caller: ResourceCaller) -> bool:
    """May the caller use a row carrying this stored access list?

    Direct personal access does NOT widen by project or admin — the owning
    tier already passes through each family's own ladder; this predicate is
    exactly the stored list.
    """
    return scopes_admit(stored_access_scopes(raw), caller)


def _require_known_circle(scope: Scope, caller: ResourceCaller) -> None:
    """A ``circle:`` entry must name an active circle exactly: matching stays
    exact, so ``circle:BDR/opossum`` never admitted the members of
    ``bdr/opossum`` (2026-10-22) and must not be stored as if it did."""
    if scope.kind != "circle" or caller.circle_finder is None:
        return
    found = caller.circle_finder(scope.id)
    if found == scope.id:
        return
    hint = f" — did you mean circle:{found}?" if found else ""
    raise AccessListError(f"no active circle {scope.id}{hint}")


def validate_access(raw: Any, caller: ResourceCaller) -> list[str]:
    """Normalize a requested access list; AccessListError when not grantable.

    ``None`` → ``[]`` (private again). A bare string is one entry.
    Duplicates collapse; order kept; ``system`` entries are admin-only.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise AccessListError("access must be a list of scopes")
    if len(raw) > MAX_ACCESS_ENTRIES:
        raise AccessListError(f"at most {MAX_ACCESS_ENTRIES} access entries")
    normalized: list[str] = []
    for entry in raw:
        scope = parse_access_entry(entry)
        if scope.kind == "system" and not caller.is_admin:
            raise AccessListError(f"not allowed to share with {scope}")
        _require_known_circle(scope, caller)
        if str(scope) not in normalized:
            normalized.append(str(scope))
    return normalized
