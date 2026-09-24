"""Parallel hypothesis tracks (ADR-066 amendment 2, gap row D-34 second leg).

A loop started with ``max_tracks > 1`` pursues several hypotheses at once:
each track pins one hypothesis and walks the plan's experiment sequence with
its own turn counter; iterations serve ACTIVE tracks round-robin, so a
question pending on one track never idles the others. Tracks are durable in
``loop.metadata["tracks"]`` (the record survives restarts); a rejected track
closes without failing the loop.

Pure functions over the loop record — the engine owns the phase pipeline.
"""

from __future__ import annotations

from typing import Any

MAX_TRACKS_LIMIT = 4
TRACK_ACTIVE = "active"
TRACK_REJECTED = "rejected"


def max_tracks(loop: Any) -> int:
    """The loop's track budget (1 = the single-branch loop)."""
    try:
        value = int(loop.metadata.get("max_tracks") or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_TRACKS_LIMIT, value))


def tracks_enabled(loop: Any) -> bool:
    """Tracks apply when the budget allows ≥2 and the plan has ≥2 hypotheses."""
    plan = getattr(loop, "plan", None)
    return max_tracks(loop) > 1 and bool(plan) and len(plan.hypotheses) > 1


def tracks(loop: Any) -> list[dict[str, Any]]:
    return list(loop.metadata.get("tracks") or [])


def needs_choice(loop: Any) -> bool:
    """More candidates than tracks: which ones to pursue is a real choice."""
    return len(loop.plan.hypotheses) > max_tracks(loop)


def hypothesis_id_of(loop: Any, hyp: Any) -> str:
    """The hypothesis id the single-branch loop would record for ``hyp``."""
    hid = getattr(hyp, "hypothesis_id", "") or ""
    return hid or f"hyp-{loop.plan.hypotheses.index(hyp)}"


def open_tracks(loop: Any, ranked: list[Any]) -> list[dict[str, Any]]:
    """One track per distinct top-ranked hypothesis, up to the budget."""
    opened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hyp in ranked:
        hid = hypothesis_id_of(loop, hyp)
        if hid in seen:
            continue
        seen.add(hid)
        opened.append(
            {
                "track_id": f"track-{len(opened) + 1}",
                "hypothesis_id": hid,
                "hypothesis": getattr(hyp, "statement", "") or "",
                "turns": 0,
                "status": TRACK_ACTIVE,
            }
        )
        if len(opened) >= max_tracks(loop):
            break
    loop.metadata["tracks"] = opened
    loop.metadata["track_cursor"] = 0
    return opened


def next_track(loop: Any) -> dict[str, Any] | None:
    """The next ACTIVE track round-robin (None when every track is closed).

    Returns the stored dict itself, so callers update it in place.
    """
    stored = loop.metadata.get("tracks") or []
    if not stored:
        return None
    cursor = int(loop.metadata.get("track_cursor") or 0)
    for step in range(len(stored)):
        track = stored[(cursor + step) % len(stored)]
        if track.get("status") == TRACK_ACTIVE:
            loop.metadata["track_cursor"] = (cursor + step + 1) % len(stored)
            return track
    return None


def take_turn(loop: Any, track: dict[str, Any]) -> Any | None:
    """The track's next experiment (its own sequence) and bump its turns."""
    experiments = list(getattr(loop.plan, "experiments", None) or [])
    turn = int(track.get("turns") or 0)
    track["turns"] = turn + 1
    return experiments[turn % len(experiments)] if experiments else None


def close_track(loop: Any, track_id: str, status: str = TRACK_REJECTED) -> None:
    for track in loop.metadata.get("tracks") or []:
        if track.get("track_id") == track_id:
            track["status"] = status


__all__ = [
    "MAX_TRACKS_LIMIT",
    "TRACK_ACTIVE",
    "TRACK_REJECTED",
    "close_track",
    "hypothesis_id_of",
    "max_tracks",
    "needs_choice",
    "next_track",
    "open_tracks",
    "take_turn",
    "tracks",
    "tracks_enabled",
]
