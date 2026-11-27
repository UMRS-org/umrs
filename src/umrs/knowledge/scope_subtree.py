r"""SQL for "a scope and everything beneath it" (ADR-108 D-5, for S-W3c).

Discarding a worktree archives the proposed rows of its scope **and of every
scope beneath it**; a revocation covers the same set
(``umrs.scopes.tokens.revoked_by``). In SQL that is
``scope_path = $n OR scope_path LIKE $m ESCAPE '\'`` where ``$m`` is the
path with LIKE's metacharacters escaped, followed by ``/%``. A level path
may hold ``_`` (a LIKE wildcard), so an unescaped pattern for ``a_b`` would
also match ``axb/...``. Pure: no I/O; the caller binds the parameters.

In the docstrings below, examples are Python literals.
"""

from __future__ import annotations

#: The columns a subtree condition may name (never caller text in SQL).
SCOPE_COLUMNS = frozenset({"scope_path", "a.scope_path", "n.scope_path", "e.scope_path"})


def subtree_like_pattern(path: str) -> str:
    r"""The LIKE pattern matching every scope strictly beneath ``path``.

    Each backslash in ``path`` becomes two backslashes, then each ``%``
    becomes backslash-percent and each ``_`` becomes backslash-underscore,
    then ``/%`` is appended. Examples (Python literals):
    ``subtree_like_pattern("a/b") == "a/b/%"``,
    ``subtree_like_pattern("a_b") == "a\\_b/%"``,
    ``subtree_like_pattern("50%") == "50\\%/%"``,
    ``subtree_like_pattern("x\\y") == "x\\\\y/%"``.
    An empty or whitespace-only ``path`` raises ``ValueError`` (the root
    scope's subtree is every row, never a pattern). ``path`` is otherwise
    used as given (no strip, no validation beyond emptiness).
    """
    raise NotImplementedError("s-w3a-03")


def subtree_condition(column: str, first_param: int) -> str:
    r"""The SQL condition "column is the path, or lies beneath it".

    The caller binds ``$n`` (``n = first_param``) to the path and ``$n+1``
    to :func:`subtree_like_pattern` of it. ``column`` must be one of
    :data:`SCOPE_COLUMNS` and ``first_param`` an ``int`` (not a ``bool``)
    of at least 1, else ``ValueError``. Example (Python literal):
    ``subtree_condition("scope_path", 3) ==
    "(scope_path = $3 OR scope_path LIKE $4 ESCAPE '\\')"``, i.e. the
    returned SQL holds exactly one backslash between the two quotes.
    """
    raise NotImplementedError("s-w3a-03")


__all__ = ["SCOPE_COLUMNS", "subtree_condition", "subtree_like_pattern"]
