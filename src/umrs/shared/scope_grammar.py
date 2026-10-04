"""Canonical scope grammar — single module for every store that records a scope.

Both current consumers dr i f t e d  already (2026-09-02 review round):
the companion trust store spoke ``global`` where the asset index spoke
``system``, and only one of them knew ``group:`` while only the other knew
``session:``. Any new store that inventes a third dialect gets rejected by
the same validator both use here.

Grammar
-------

    system                  platform-wide (was 'global' in the trust store)
    project:<project_id>    one project
    circle:<project_id>/<circle_id>
                            one circle (project-scoped context/audience
                            cell, ADR-086)
    group:<group_id>        legacy — superseded by circle: (ADR-086);
                            parseable, grants nothing on access lists
    user:<user_id>          one user, owner-only
    session:<session_id>    one companion/research session (trust-only today)

'global' is accepted on READ paths as a legacy alias of 'system' and
REJECTED on writes — stores must normalize; the API layer fails loudly on
legacy writes so a mixed dialect never lands in the data envelope.

Scope lookup precedence for permission-style consumers (trust checker):
    session > project > group > user > system
The helper :func:`fallback_chain` returns that chain for a given scope so
consumers share exactly one precedence rule instead of re-inventing it.
"""

from __future__ import annotations

from dataclasses import dataclass

BARE_SCOPES = frozenset({"system"})
PREFIX_SCOPES = frozenset({"project:", "circle:", "group:", "user:", "session:"})
# Legacy alias accepted on read/normalize, rejected on write.
LEGACY_ALIASES = {"global": "system"}

# Most-specific-first precedence chain for permission fallbacks.
_PRECEDENCE = ("session", "project", "circle", "group", "user", "system")


class ScopeError(ValueError):
    """Raised when a scope string violates the grammar."""


def _validate_circle_id(raw: str) -> None:
    """Circle ids are ``<project_id>/<circle_id>`` — one slash, non-empty parts."""
    project, sep, circle = raw.partition("/")
    if not sep or not project or not circle or "/" in circle:
        raise ScopeError(
            f"circle scope wants circle:<project_id>/<circle_id>, got {raw!r}"
        )


@dataclass(frozen=True)
class Scope:
    """A validated scope: bare kind + optional id."""

    kind: str  # system | project | circle | group | user | session
    id: str = ""

    def __post_init__(self) -> None:
        if self.kind == "circle":
            _validate_circle_id(self.id)

    def __str__(self) -> str:
        return (
            self.kind
            if self.kind == "system" or not self.id
            else f"{self.kind}:{self.id}"
        )

    def parent(self) -> Scope | None:
        """Next-broader scope in the precedence chain, or None at 'system'."""
        if self.kind == "system":
            return None
        return Scope(kind="system")

    def fallback_chain(self) -> list[Scope]:
        """Most-specific-first chain ending at system (never empty)."""
        chain: list[Scope] = [self]
        current = self
        while True:
            parent = current.parent()
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain


def parse(raw: str, *, allow_legacy: bool = False) -> Scope:
    """Validate+parse a scope string; ScopeError on garbage or legacy-writes."""
    text = (raw or "").strip()
    if not text:
        raise ScopeError("empty scope")
    if text in BARE_SCOPES:
        return Scope(kind="system")
    if text in LEGACY_ALIASES:
        if allow_legacy:
            return Scope(kind=LEGACY_ALIASES[text])
        raise ScopeError(
            f"legacy scope alias {text!r} not accepted on writes (use {LEGACY_ALIASES[text]!r})"
        )
    for prefix in PREFIX_SCOPES:
        if text.startswith(prefix) and len(text) > len(prefix):
            # circle shape rides Scope.__post_init__ (raises ScopeError here)
            return Scope(kind=prefix[:-1], id=text[len(prefix) :])
    raise ScopeError(
        f"invalid scope {text!r} — want system | project:<id> |"
        " circle:<project>/<id> | user:<id> | session:<id>"
    )


def parse_lenient(raw: str | None) -> Scope:
    """Read-path parse: legacy aliases normalized, missing → system."""
    if not raw:
        return Scope(kind="system")
    return parse(raw, allow_legacy=True)


def fallback_chain(raw: str) -> list[str]:
    """Stringly chain helper: scope → precedence list of scope strings."""
    return [str(scope) for scope in parse_lenient(raw).fallback_chain()]


def is_descendant_of(child: str, parent: str) -> bool:
    """True when child's fallback chain contains parent."""
    try:
        return parse_lenient(parent) in parse_lenient(child).fallback_chain()
    except ScopeError:
        return False


# ---------------------------------------------------------------------------
# Authorization ladder (ADR-041, P10-C1) — one ladder for every asset type
#
# The label grammar above names trust/asset-tag consumers. Asset *records*
# (model rows, knowledge graphs, endpoints) carry their scope as an
# (representation, owner_id, project_id) triple — the divergence G-P10-2
# records. ``normalize_asset_scope`` maps every representation into one
# ``AssetScope`` and the two predicates below are the ONLY read/write ladder;
# asset stores delegate here instead of re-inventing tier checks.
# ---------------------------------------------------------------------------

SYSTEM_NAMESPACE = "system"
USER_NAMESPACE_PREFIX = "user:"

_ASSET_SYSTEM_KINDS = {"system"}
_ASSET_USER_KINDS = {"user", "personal"}


@dataclass(frozen=True)
class AssetScope:
    """Normalized scope of one asset record."""

    kind: str  # system | project | user
    owner_id: str = ""
    project_id: str = ""


def normalize_asset_scope(
    representation: str,
    *,
    owner_id: str = "",
    project_id: str = "",
) -> AssetScope:
    """Map an asset-specific scope representation into the grammar.

    Accepted representations (case-insensitive): ``system`` → system;
    ``user`` / ``personal`` → user; everything else (``project``, ``research``,
    ``group``, ``team``, …) → project. Mirrors the label grammar's kinds.
    """
    reduced = (representation or "").strip().lower()
    if reduced in _ASSET_SYSTEM_KINDS:
        kind = "system"
    elif reduced in _ASSET_USER_KINDS:
        kind = "user"
    else:
        kind = "project"
    return AssetScope(kind=kind, owner_id=owner_id, project_id=project_id)


def scope_readable(
    ref: AssetScope,
    *,
    user_id: str,
    project_id: str,
    is_admin: bool = False,
    anonymous: bool = False,
) -> bool:
    """May the caller read an asset with this scope? (single ladder)

    A project-scoped asset with no owning project is a global project row:
    readable by any authenticated caller (the knowledge-side rule; model
    scope invariants ensure project models always name a project, so the
    branches coincide for model-grant checks). The ``system`` namespace is
    shared-readable; a ``user:<uid>`` namespace is that user's alone (the
    label grammar's user scope).
    """
    if is_admin or anonymous:
        return True
    if ref.kind == "system":
        return True
    if ref.kind == "user":
        return bool(ref.owner_id) and ref.owner_id == user_id
    if not ref.project_id:
        return True  # global project row, any authenticated tier
    if ref.project_id == SYSTEM_NAMESPACE:
        return True  # shared system namespace
    if ref.project_id.startswith(USER_NAMESPACE_PREFIX):
        return ref.project_id == f"{USER_NAMESPACE_PREFIX}{user_id}"
    return ref.project_id == project_id


def scope_writable(
    ref: AssetScope,
    *,
    user_id: str,
    project_id: str,
    is_admin: bool = False,
    anonymous: bool = False,
) -> bool:
    """May the caller modify an asset with this scope? (single ladder)

    The ``system`` kind and the ``system`` project namespace are both
    admin-managed: no verified project membership grants write there.
    """
    if is_admin or anonymous:
        return True
    if ref.kind == "system":
        return False
    if ref.kind == "user":
        return bool(ref.owner_id) and ref.owner_id == user_id
    if not ref.project_id or ref.project_id == SYSTEM_NAMESPACE:
        return False
    if ref.project_id.startswith(USER_NAMESPACE_PREFIX):
        return ref.project_id == f"{USER_NAMESPACE_PREFIX}{user_id}"
    return ref.project_id == project_id
