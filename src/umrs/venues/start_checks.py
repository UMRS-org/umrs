"""The check a venue start runs before it reserves anything (ADR-099 D-6).

``FrontendVenueService`` calls its ``quota_check`` with one
:class:`StartCheck` after the session policy is resolved and before the
data-class gate, the subnet and the job: a refusal leaves nothing behind.
The check raises a typed ``HTTPException`` to refuse (the concurrent-session
cap, ``quota.concurrent_sessions_exceeded``).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StartCheck:
    owner: str
    project: str
    venue: str  # site name
    session_id: str
    policy: dict[str, Any] = field(default_factory=dict)  # resolved policy (ADR-099)


StartCheckFn = Callable[[StartCheck], None]

#: The venue's start lock (site name -> held while entered; ADR-099 D-6):
#: the check, the job and the row of one start before the next start counts.
StartLockFn = Callable[[str], AbstractContextManager[None]]

__all__ = ["StartCheck", "StartCheckFn", "StartLockFn"]
