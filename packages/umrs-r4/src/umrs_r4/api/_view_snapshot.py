"""A ``session.snapshot`` view op built from the session's journal row.

A reader the view ring cannot serve (a new stream with no emitter snapshot
held, a ``Last-Event-ID`` older than the ring, an emitter restart) starts
from the journal row, the durable truth (ADR-057 D-6). The row carries no
blocks beyond the text of each message, so earlier messages become closed
turns of text blocks, the same rule as r4's native emitter
(``r4/companion/ui/view_ops_session.history_turns``): a user message starts
a turn, an assistant message with text appends a block.

Every key below is one the view contract v1 names
(``contracts/view/v1/view.schema.json``); the snapshot carries no other.
"""

from __future__ import annotations

import json
from typing import Any

_MODES = frozenset({"chat", "plan", "autopilot"})


def _obj(raw: Any) -> dict[str, Any]:
    """A dict, or a JSON string of one; anything else is empty."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("messages")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict)]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def history_turns(sid: str, messages: list[dict[str, Any]], at: int) -> list[dict[str, Any]]:
    """Earlier messages as closed turns, in order (see the module docstring)."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        body = _text(message.get("content"))
        role = message.get("role")
        if role == "user":
            turn_id = _text(_obj(message.get("metadata")).get("turn_id"))
            turns.append(_turn(sid, turn_id or f"history-{len(turns) + 1}", body, at, len(turns)))
        elif role == "assistant" and body:
            if not turns:
                turns.append(_turn(sid, "history-1", "", at, 0))
            _add_block(turns[-1], body, at)
    return turns


def _turn(sid: str, turn_id: str, user: str, at: int, index: int) -> dict[str, Any]:
    return {
        "id": turn_id,
        "sessionId": sid,
        "seq": index + 1,
        "user": user,
        "blocks": [],
        "status": "done",
        "startedAt": at,
        "endedAt": at,
    }


def _add_block(turn: dict[str, Any], body: str, at: int) -> None:
    blocks = turn["blocks"]
    blocks.append(
        {
            "id": f"{turn['id']}:b{len(blocks) + 1}",
            "turnId": turn["id"],
            "kind": "text",
            "state": "ok",
            "startedAt": at,
            "endedAt": at,
            "summary": body.splitlines()[0][:120] if body else "",
            "body": body,
        }
    )


def _run_state(row_state: str) -> str:
    return "error" if row_state == "error" else "idle"


def session_noun(detail: dict[str, Any], payload: dict[str, Any], native: bool) -> dict[str, Any]:
    """The Session noun of a journal row (``snapshot_detail_row`` shape)."""
    metadata = _obj(payload.get("metadata"))
    mode = _text(metadata.get("interaction_mode"))
    project = _text(metadata.get("project_id")) or _text(metadata.get("project"))
    return {
        "id": str(detail.get("session_id") or ""),
        "project": project,
        "title": _text(detail.get("name")) or _text(payload.get("name")),
        "venue": _text(detail.get("venue")) or "local",
        "model": _text(payload.get("model")),
        "mode": mode if mode in _MODES else "chat",
        "state": _run_state(_text(detail.get("state"))),
        "ctxPct": 0,
        "detail": "full" if native else "limited",
    }


def snapshot_op(detail: dict[str, Any], *, seq: int, at: int, native: bool) -> dict[str, Any]:
    """One ``session.snapshot`` op at ``seq`` from a journal detail row.

    ``native``: the session's emitter posts ops (``detail: full``); else the
    stream is the row alone (``detail: limited``).
    """
    payload = _obj(detail.get("payload"))
    sid = str(detail.get("session_id") or "")
    return {
        "v": 1,
        "sid": sid,
        "seq": max(seq, 1),
        "at": at,
        "op": "session.snapshot",
        "session": session_noun(detail, payload, native),
        "turns": history_turns(sid, _messages(payload), at),
        "tasks": [],
        "asks": [],
        "artifacts": [],
    }
