"""Slurm nodelist parsing helpers.

Slurm reports allocated nodes in a compressed form (e.g. ``ng-dgx-m3-[14-15]``
or ``nodeA,nodeB``). That compressed string is NOT a resolvable hostname, so any
code that turns a Slurm ``%N`` / ``NodeList`` value into a host for tunneling or
readiness probing must first reduce it to a concrete node — the head node where
rank 0 (the serving process) runs.

These helpers are pure-Python so they work in the front-end UMRS process, which
reaches compute nodes over SSH and therefore cannot rely on ``scontrol show
hostnames`` being available locally.
"""

from __future__ import annotations

_NULL_TOKENS = frozenset({"", "(null)", "none", "n/a"})


def first_node_hostname(nodelist: str | None) -> str:
    """Return the first (head) hostname from a Slurm nodelist expression.

    Examples
    --------
    ``ng-dgx-m3-14``          -> ``ng-dgx-m3-14``
    ``ng-dgx-m3-[14-15]``     -> ``ng-dgx-m3-14``
    ``ng-dgx-m3-[14,17-19]``  -> ``ng-dgx-m3-14``
    ``nodeA,nodeB``           -> ``nodeA``
    ``ng-dgx-m3-[014-015]``   -> ``ng-dgx-m3-014`` (zero-padding preserved)

    Empty / null-ish inputs are returned unchanged so callers can keep their
    existing sentinel handling (``pending``, ``unknown``, etc.).
    """
    if nodelist is None:
        return ""

    text = nodelist.strip()
    if text.lower() in _NULL_TOKENS:
        return text

    # Take the first top-level group (split on commas that are outside brackets,
    # since a bracket list may itself contain commas).
    depth = 0
    end = len(text)
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            end = i
            break
    group = text[:end]

    if "[" not in group:
        return group

    prefix, _, rest = group.partition("[")
    inner = rest.split("]", 1)[0]
    first = inner.split(",")[0].split("-")[0].strip()
    return f"{prefix}{first}"
