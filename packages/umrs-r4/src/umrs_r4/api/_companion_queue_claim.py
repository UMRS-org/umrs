"""ADR-057 D-8 (slice SH-J3): exactly-once queued turns on the worker rail.

``pop`` removed a queued turn before it ran, so a worker that died between
the pop and the reply lost the turn. The rail now offers the turn with
``claim`` and removes it only on ``ack``:

* ``claim`` stamps the first queued turn with a server-minted ``claim_id``,
  ``claimed_at`` and an ``attempts`` count and returns it; the entry stays in
  ``metadata.chat_queue``. An unacked entry is offered again on the next
  claim: the venue has one writer, the lease holder (ADR-057 D-9), so a
  second claim means the earlier attempt died or its ack was lost.
* ``ack`` removes the entry by ``queue_id`` (the entry ``id``) when the
  ``claim_id`` is the latest one; an older claim is ``superseded`` and an
  already-removed entry is ``absent`` (idempotent). Neither writes.
* The worker stamps ``metadata.queue_id`` on the turn and its reply in the
  journal and skips an entry whose reply is already there, so a redelivery
  never runs a turn twice.

ADR-114 session messages keep their own read state: ``claim`` marks the
first unnotified message exactly as ``pop`` does, and ``ack`` never removes
one (``not_a_turn``). ``pop`` stays as the legacy alias for workers that
predate ``claim`` (claim and ack in one step).

This module owns no schema or store: it mutates the owner-scoped
``companion_session_snapshots`` row on the caller's connection, through the
same row lock as every other queue transition.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from umrs_r4.api._companion_session_bus import (
    MESSAGE_KIND,
    _finish,
    _iso,
    _locked,
    _now,
    _parts,
    _prune_expired,
    _write_metadata,
)

#: Entry ids an older server minted from the queue length (``q-<n>``); they
#: repeat after a pop, so a claim re-ids them before they become queue_ids.
_LEGACY_ID_RE = re.compile(r"^q-\d+$")


class QueueAckBody(BaseModel):
    """Acknowledge one claimed turn: its entry id and the claim being acked."""

    queue_id: str = Field(min_length=1, max_length=128)
    claim_id: str = Field(min_length=1, max_length=128)


def new_queue_id() -> str:
    """A queue entry id that never repeats (the worker's idempotency key)."""
    return f"q-{uuid.uuid4().hex[:16]}"


def _attempts(entry: dict[str, Any]) -> int:
    try:
        return max(0, int(entry.get("attempts") or 0))
    except (TypeError, ValueError):
        return 0


def _claim_turn(entry: dict[str, Any], now: str) -> dict[str, Any]:
    if _LEGACY_ID_RE.match(str(entry.get("id") or "")):
        entry["id"] = new_queue_id()
    entry["claim_id"] = f"c-{uuid.uuid4().hex[:16]}"
    entry["claimed_at"] = now
    entry["attempts"] = _attempts(entry) + 1
    return entry


def claim_front(entries: list, now: str) -> dict[str, Any] | None:
    """Stamp and return the next entry to run; None when nothing is offered.

    The first queued turn wins whether or not it was claimed before (a
    claimed, unacked turn is re-offered); a session message is offered once,
    by its ``notified_at`` mark, exactly like the legacy pop.
    """
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != MESSAGE_KIND:
            return _claim_turn(entry, now)
        if not (entry.get("read_at") or entry.get("notified_at")):
            entry["notified_at"] = now
            return entry
    return None


def ack_entry(entries: list, *, queue_id: str, claim_id: str) -> str:
    """Remove the acked turn in place → acked | superseded | not_a_turn | absent."""
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("id") != queue_id:
            continue
        if entry.get("kind") == MESSAGE_KIND:
            return "not_a_turn"
        if entry.get("claim_id") != claim_id:
            return "superseded"
        entries.pop(index)
        return "acked"
    return "absent"


def worker_claim_row(conn: Any, uid: str, session_id: str) -> dict[str, Any]:
    """Claim the next queued turn (or unnotified message) for the lease holder."""
    now = _now()
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        blob, metadata, entries = _parts(row["payload"])
        pruned = _prune_expired(entries, now)
        front = claim_front(entries, _iso(now))
        if front is None and not pruned:
            conn.rollback()
            return {"applied": True, "queue": entries, "front": None}
        metadata["chat_queue"] = entries
        _write_metadata(blob, metadata)
        _finish(conn, cur, uid, session_id, blob)
    return {"applied": True, "queue": entries, "front": front}


def worker_ack_row(
    conn: Any, uid: str, session_id: str, body: QueueAckBody
) -> dict[str, Any]:
    """Remove one claimed turn; a stale or repeated ack changes nothing."""
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        blob, metadata, entries = _parts(row["payload"])
        outcome = ack_entry(entries, queue_id=body.queue_id, claim_id=body.claim_id)
        if outcome != "acked":
            conn.rollback()
            return {"applied": False, "outcome": outcome, "queue": entries}
        metadata["chat_queue"] = entries
        _write_metadata(blob, metadata)
        _finish(conn, cur, uid, session_id, blob)
    return {"applied": True, "outcome": outcome, "queue": entries}
