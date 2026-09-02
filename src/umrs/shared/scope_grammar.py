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
    group:<group_id>        one group (lab/collab tier, scopes.py §Hierarchy)
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
PREFIX_SCOPES = frozenset({"project:", "group:", "user:", "session:"})
# Legacy alias accepted on read/normalize, rejected on write.
LEGACY_ALIASES = {"global": "system"}

# Most-specific-first precedence chain for permission fallbacks.
_PRECEDENCE = ("session", "project", "group", "user", "system")


class ScopeError(ValueError):
    """Raised when a scope string violates the grammar."""


@dataclass(frozen=True)
class Scope:
    """A validated scope: bare kind + optional id."""

    kind: str  # system | project | group | user | session
    id: str = ""

    def __str__(self) -> str:
        return self.kind if self.kind == "system" or not self.id else f"{self.kind}:{self.id}"

    def parent(self) -> "Scope | None":
        """Next-broader scope in the precedence chain, or None at 'system'."""
        if self.kind == "system":
            return None
        return Scope(kind="system")

    def fallback_chain(self) -> list["Scope"]:
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
            return Scope(kind=prefix[:-1], id=text[len(prefix):])
    raise ScopeError(
        f"invalid scope {text!r} — want system | project:<id> | group:<id> | user:<id> | session:<id>"
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
