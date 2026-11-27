"""The scope overlay's write rules (ADR-108 D-4, slice S-W3a). Pure: no I/O.

A work level (loop, iteration, teamup member) acts under a signed scope
token whose path names it, e.g. ``s-ab12/loop-7/iter-3``. What it writes
lands **proposed** and **tagged** with that path, in the ``scope_path``
column migration 0150 adds to ``knowledge_artifacts``, ``knowledge_nodes``
and ``knowledge_edges``. ``NULL`` is the root scope: a root write is
exactly as before 0150.

Where the path comes from, and only from there:

- ``POST /umrs/k2/ingest`` and the upload route put the *verified* token's
  path on the job spec (``knowledge_v2._scope_fields``);
- a scoped workflow run puts its recorded path on the ingest of its outputs
  (``knowledge_adapters._run_scope_path``);
- a client-built ``knowledge_ingest``/``knowledge_understand`` spec that
  names ``scope_path`` is refused (``session_turns.client_spec_refusal``).

The tier rule (the integrator reviews this): **a scoped write is never
established.** ``scoped_tier`` and ``scoped_tier_stamp`` force the proposed
tier whatever the origin said, so a scoped workflow run's outputs (origin
``workflow``, not ``agent``) land proposed too. Establishing stays the
human decision of ADR-084; merging a child into its parent re-tags (S-W3c).

Visibility (who may read a tagged row) is S-W3b's
``umrs.scopes.visibility``; it builds on ``scope_lineage``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from umrs.knowledge.actors import AGENT_ACTOR
from umrs.scopes.tokens import ScopeError, check_path

#: The job-spec key (and the artifact-descriptor key) carrying the path.
SCOPE_KEY = "scope_path"
PROPOSED_TIER = "proposed"
ESTABLISHED_TIER = "established"


class ScopeOverlayError(ValueError):
    """A non-empty scope path that is not a valid level path.

    Never read as "no scope": a malformed path fails the write instead of
    landing it untagged (and so visible) at the root.
    """


def clean_scope_path(raw: Any) -> str | None:
    """The checked level path, or None for the root scope.

    None, ``""`` and whitespace are the root scope. Anything else must pass
    ``umrs.scopes.tokens.check_path`` (1-8 segments of letters, digits,
    ``.``, ``_``, ``-``; each segment 1-64 chars, starting alphanumeric) or
    :class:`ScopeOverlayError` is raised. Surrounding whitespace is
    stripped; nothing else is rewritten.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ScopeOverlayError(f"scope path must be text, not {type(raw).__name__}")
    text = raw.strip()
    if not text:
        return None
    try:
        return check_path(text)
    except ScopeError as exc:
        raise ScopeOverlayError(str(exc)) from exc


def scoped_spec_path(spec: Any) -> str | None:
    """The job spec's checked ``scope_path`` (None: root or no spec)."""
    if not isinstance(spec, Mapping):
        return None
    return clean_scope_path(spec.get(SCOPE_KEY))


def scoped_tier(status: str, scope_path: str | None) -> str:
    """The tier a row is written in: ``proposed`` under any scope, else
    ``status`` unchanged."""
    return PROPOSED_TIER if scope_path else status


def scoped_tier_stamp(
    proposed: bool, stamp: dict[str, Any], scope_path: str | None, job_id: str
) -> tuple[bool, dict[str, Any]]:
    """``session_turns.tier_stamp``'s ``(proposed, stamp)`` with the tier rule
    applied: a scoped job that would have written established rows writes
    proposed rows stamped like an agent's (``origin_actor`` agent, the job
    id), and any ``review`` (a vouch) is dropped. Root, or already proposed:
    unchanged."""
    if not scope_path or proposed:
        return proposed, stamp
    return True, {"origin_actor": AGENT_ACTOR, "job_id": job_id}


def scope_lineage(path: str) -> tuple[str, ...]:
    """The path's ancestors and the path itself, root-most first.

    ``"a/b/c"`` gives ``("a", "a/b", "a/b/c")``; ``"a"`` gives ``("a",)``.
    ``path`` is checked first with :func:`clean_scope_path`: an empty path
    (the root scope) gives ``()``, a malformed one raises
    :class:`ScopeOverlayError`. A reader in scope P sees the rows whose
    ``scope_path`` is in ``scope_lineage(P)`` (plus established and root
    rows; S-W3b).
    """
    raise NotImplementedError("s-w3a-01")


def parent_scope(path: str) -> str | None:
    """The path one level up, or None when ``path`` is a top-level scope.

    ``"a/b/c"`` gives ``"a/b"``; ``"a"`` gives None (its parent is the
    root scope). ``path`` is checked with :func:`clean_scope_path`; the root
    scope (empty) and a malformed path raise :class:`ScopeOverlayError`
    (the root has no parent). Merging a child re-tags its rows to this
    (S-W3c).
    """
    raise NotImplementedError("s-w3a-01")


__all__ = [
    "ESTABLISHED_TIER",
    "PROPOSED_TIER",
    "SCOPE_KEY",
    "ScopeOverlayError",
    "clean_scope_path",
    "parent_scope",
    "scope_lineage",
    "scoped_spec_path",
    "scoped_tier",
    "scoped_tier_stamp",
]
