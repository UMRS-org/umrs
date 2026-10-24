"""The closed actor vocabulary of knowledge writes (review of K1.1, 2026-10-24).

A request names its actor in ``x-r4-actor`` and an ingest job carries it as
``spec.origin_actor``. Only the exact string ``agent`` used to count as an
agent, so ``companion``, ``loop`` or ``agent-team`` were recorded verbatim
yet got the human tier and no proposal check. Now there are two values: a
person (``user``: the header absent, empty or ``user``) and everything else,
which is an agent (fail closed). Client-honest, as before (K-12).
"""

from __future__ import annotations

AGENT_ACTOR = "agent"
HUMAN_ACTOR = "user"


def normalize_actor(raw: object) -> str:
    """``user`` for an absent, empty or ``user`` value; ``agent`` otherwise."""
    value = str(raw or "").strip().lower()
    return HUMAN_ACTOR if value in ("", HUMAN_ACTOR) else AGENT_ACTOR
