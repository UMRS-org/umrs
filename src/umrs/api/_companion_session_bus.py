"""ADR-114 S2.1 storage transitions inside ADR-046's existing turn queue.

This module owns no schema or store. Every function mutates the caller's
owner-scoped ``companion_session_snapshots`` row on the caller's connection.
Unread/notification/read state for a session message stays in that row's
``metadata.chat_queue`` list; remote presence is ``metadata.session_bus``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from .typed_errors import PlatformError, raise_typed

MESSAGE_KIND = "session_message"
PRESENCE_KEY = "session_bus"
DEFAULT_MESSAGE_TTL_SECONDS = 86400
PRESENCE_TTL_SECONDS = 90
MAX_UNREAD = 100
MAX_BODY_CHARS = 8000
_LABEL_DROP_RE = re.compile(r"[^\w .:@/-]+")
_ACTIVE_STATES = frozenset({"", "ongoing"})
_REFUSAL_STATES = frozenset({"archived", "completed", "error", "stopped"})


class SessionBusBindBody(BaseModel):
    """The local bus identity and display labels, never a second durable id."""

    bus_session_id: str = Field(min_length=1, max_length=128)
    name: str = Field(default="", max_length=256)
    host: str = Field(default="", max_length=256)
    cwd: str = Field(default="", max_length=4096)
    # Project is derived from the durable payload, never used as routing input.
    model: str = Field(default="", max_length=256)
    mode: str = Field(default="", max_length=64)


class SessionBusSendBody(BaseModel):
    """One idempotent send into another of this owner's durable sessions."""

    message_id: str = Field(pattern=r"^msg-[A-Za-z0-9][A-Za-z0-9-]{7,40}$")
    recipient_id: str = Field(min_length=1, max_length=128)
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    sender_name: str = Field(default="", max_length=256)
    sender_kind: str = Field(default="session", pattern="^(session|user)$")
    reply_to: str = Field(default="", max_length=64)


class SessionBusInboxBody(BaseModel):
    """Peek or atomically mark selected unread entries read."""

    action: str = Field(pattern="^(peek|take)$")
    message_ids: list[str] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=20, ge=1, le=50)


def _clean_label(value: Any) -> str:
    text = _LABEL_DROP_RE.sub(" ", str(value or "")).strip()
    return " ".join(text.split())[:60]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _expired(value: Any, now: datetime) -> bool:
    expires = _datetime(value)
    return expires is None or expires <= now


def _ttl_seconds() -> int:
    raw = os.environ.get("UMRS_SESSION_MESSAGE_TTL_SECONDS", "")
    if not raw:
        return DEFAULT_MESSAGE_TTL_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MESSAGE_TTL_SECONDS


def _refuse(code: str, message: str) -> None:
    raise_typed(PlatformError.companionsessions(code, message))


def _json_dict(raw: Any) -> dict[str, Any]:
    from .companion_sessions import _json_dict as parse

    return parse(raw) or {}


def _parts(payload: Any) -> tuple[dict[str, Any], dict[str, Any], list]:
    from .companion_sessions import _queue_entries

    return _queue_entries(payload)


def _write_metadata(blob: dict[str, Any], metadata: dict[str, Any]) -> None:
    from .companion_sessions import _set_metadata

    _set_metadata(blob, metadata)


def _lock_snapshot(cur: Any, uid: str, session_id: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT s.name, s.payload, s.venue, s.state, a.archived_at
        FROM companion_session_snapshots s
        LEFT JOIN companion_session_archives a ON a.session_id = s.session_id
        WHERE s.user_id = %s AND s.session_id = %s
        FOR UPDATE
        """,
        (uid, session_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    name, payload, venue, state, archived_at = row
    return {
        "name": name,
        "payload": payload,
        "venue": venue,
        "state": "archived" if archived_at else state,
    }


def _locked(cur: Any, uid: str, session_id: str) -> dict[str, Any]:
    row = _lock_snapshot(cur, uid, session_id)
    if row is None:
        _refuse("not_found", "snapshot not found")
    return row


def _project_label(metadata: dict[str, Any]) -> str:
    config = metadata.get("config") or {}
    return str(config.get("project_id") or metadata.get("project_id") or "")


def _bindable(row: dict[str, Any], bus_session_id: str, session_id: str) -> None:
    blob = _json_dict(row["payload"])
    _, metadata, _ = _parts(row["payload"])
    if metadata.get("session_kind") == "subagent":
        _refuse(
            "receiver_refused",
            "subagent sessions are read-only and do not join the session bus",
        )
    identities_match = (
        bus_session_id == session_id and str(blob.get("id") or "") == session_id
    )
    if not identities_match:
        _refuse(
            "identity_mismatch", "session bus identity does not match durable session"
        )
    if row["state"] in _REFUSAL_STATES:
        _refuse("receiver_refused", "session is not accepting session-bus presence")


def _store_payload(cur: Any, uid: str, session_id: str, blob: dict[str, Any]) -> None:
    import json

    cur.execute(
        """
        UPDATE companion_session_snapshots
        SET payload = %s, updated_at = now()
        WHERE user_id = %s AND session_id = %s
        """,
        (json.dumps(blob), uid, session_id),
    )


def _finish(
    conn: Any, cur: Any, uid: str, session_id: str, blob: dict[str, Any]
) -> None:
    _store_payload(cur, uid, session_id, blob)
    conn.commit()


def _presence_view(row: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    fields = ("bus_session_id", "name", "host", "cwd", "project", "model", "mode")
    answer = {key: str(binding.get(key) or "") for key in fields}
    return {
        "session_id": answer["bus_session_id"],
        "bus_session_id": answer.pop("bus_session_id"),
        **answer,
        "venue": str(row.get("venue") or "local"),
        "pid": 0,
        "started_at": float(binding.get("started_at") or 0.0),
        "expires_at": str(binding.get("expires_at") or ""),
    }


def bind_presence_row(
    conn: Any, uid: str, session_id: str, body: SessionBusBindBody
) -> dict[str, Any]:
    """Bind exactly the local/durable/session-rail id; any mismatch refuses."""
    body = body.model_copy(update={"name": _clean_label(body.name)})
    now = _now()
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        _bindable(row, body.bus_session_id, session_id)
        blob, metadata, _entries = _parts(row["payload"])
        before = metadata.get(PRESENCE_KEY) or {}
        started_at = before.get("started_at") or now.timestamp()
        binding = body.model_dump()
        binding["project"] = _project_label(metadata)
        binding.update(
            {
                "started_at": started_at,
                "expires_at": _iso(now + timedelta(seconds=PRESENCE_TTL_SECONDS)),
            }
        )
        binding.pop("withdrawn_at", None)
        metadata[PRESENCE_KEY] = binding
        _write_metadata(blob, metadata)
        _finish(conn, cur, uid, session_id, blob)
        return {"bound": True, **_presence_view(row, binding)}


def withdraw_presence_row(conn: Any, uid: str, session_id: str) -> dict[str, Any]:
    """Expire presence but retain its address for offline delivery."""
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        blob, metadata, _entries = _parts(row["payload"])
        binding = metadata.get(PRESENCE_KEY)
        if isinstance(binding, dict) and binding.get("bus_session_id") == session_id:
            binding["expires_at"] = _iso(_now())
            binding["withdrawn_at"] = _iso(_now())
            _write_metadata(blob, metadata)
            _finish(conn, cur, uid, session_id, blob)
        else:
            conn.rollback()
    return {"bound": False, "session_id": session_id}


def _live_binding(row: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    _blob, metadata, _entries = _parts(row["payload"])
    binding = metadata.get(PRESENCE_KEY)
    if not isinstance(binding, dict) or row["state"] not in _ACTIVE_STATES:
        return None
    if binding.get("bus_session_id") != row["session_id"]:
        return None
    expires = _datetime(binding.get("expires_at"))
    return binding if expires is not None and expires > now else None


def presence_rows(conn: Any, uid: str) -> dict[str, Any]:
    """Only this owner's unexpired, identity-checked bindings."""
    now = _now()
    found: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.session_id, s.name, s.payload, s.venue, s.state, a.archived_at
            FROM companion_session_snapshots s
            LEFT JOIN companion_session_archives a ON a.session_id = s.session_id
            WHERE s.user_id = %s
            """,
            (uid,),
        )
        for sid, name, payload, venue, state, archived_at in cur.fetchall():
            row = {
                "session_id": sid,
                "name": name,
                "payload": payload,
                "venue": venue,
                "state": "archived" if archived_at else state,
            }
            binding = _live_binding(row, now)
            if binding is not None:
                found.append(_presence_view(row, binding))
    found.sort(key=lambda row: (row["started_at"], row["session_id"]))
    return {"sessions": found}


def _prune_expired(entries: list, now: datetime) -> bool:
    kept = [
        entry
        for entry in entries
        if not (
            isinstance(entry, dict)
            and entry.get("kind") == MESSAGE_KIND
            and _expired(entry.get("expires_at"), now)
        )
    ]
    changed = len(kept) != len(entries)
    entries[:] = kept
    return changed


def _message_view(entry: dict[str, Any]) -> dict[str, Any]:
    added = _datetime(entry.get("added_at")) or datetime.fromtimestamp(0, timezone.utc)
    return {
        "message_id": str(entry.get("id") or ""),
        "sender_id": str(entry.get("sender_id") or ""),
        "sender_name": _clean_label(entry.get("sender_name")),
        "recipient_id": str(entry.get("recipient_id") or ""),
        "body": str(entry.get("content") or ""),
        "sent_at": added.timestamp(),
        "reply_to": str(entry.get("reply_to") or ""),
        "sender_kind": str(entry.get("sender_kind") or "session"),
    }


def _read_payload(cur: Any, uid: str, session_id: str) -> Any:
    cur.execute(
        "SELECT payload FROM companion_session_snapshots WHERE session_id = %s AND user_id = %s",
        (session_id, uid),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _sender_binding(cur: Any, uid: str, sender_id: str) -> dict[str, Any] | None:
    payload = _read_payload(cur, uid, sender_id)
    if payload is None:
        return None
    blob, metadata, _entries = _parts(payload)
    blob_id = str(blob.get("id") or "")
    binding = metadata.get(PRESENCE_KEY)
    if blob_id != sender_id or not isinstance(binding, dict):
        return None
    if binding.get("bus_session_id") != sender_id:
        return None
    expires = _datetime(binding.get("expires_at"))
    return binding if expires is not None and expires > _now() else None


def _validate_send(
    cur: Any, uid: str, sender_id: str, row: dict[str, Any], body: SessionBusSendBody
) -> None:
    if sender_id == body.recipient_id:
        _refuse("invalid_recipient", "a session cannot message itself")
    if row["state"] in _REFUSAL_STATES:
        _refuse("receiver_refused", "recipient session is not accepting messages")
    blob, metadata, _entries = _parts(row["payload"])
    if metadata.get("session_kind") == "subagent":
        _refuse(
            "receiver_refused",
            "subagent sessions are read-only and cannot receive session messages",
        )
    if str(blob.get("id") or "") != body.recipient_id:
        _refuse(
            "identity_mismatch", "recipient identity does not match the durable session"
        )
    binding = metadata.get(PRESENCE_KEY)
    if (
        not isinstance(binding, dict)
        or binding.get("bus_session_id") != body.recipient_id
    ):
        _refuse("receiver_refused", "recipient is not on the session bus")
    if _sender_binding(cur, uid, sender_id) is None:
        _refuse(
            "identity_mismatch",
            "sender does not have a live matching session-bus binding",
        )


def _fingerprint(sender_id: str, body: SessionBusSendBody) -> tuple[str, ...]:
    return (
        sender_id,
        body.recipient_id,
        body.body.strip(),
        body.reply_to,
        body.sender_kind,
    )


def _text(value: Any, default: str = "") -> str:
    return str(value) if value else default


def _entry_fingerprint(entry: dict[str, Any]) -> tuple[str, ...]:
    return (
        _text(entry.get("sender_id")),
        _text(entry.get("recipient_id")),
        _text(entry.get("content")),
        _text(entry.get("reply_to")),
        _text(entry.get("sender_kind"), "session"),
    )


def _replay(
    entries: list, body: SessionBusSendBody, fingerprint: tuple[str, ...]
) -> dict[str, Any] | None:
    matches = (
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("kind") == MESSAGE_KIND
        and entry.get("id") == body.message_id
    )
    entry = next(matches, None)
    if entry is None:
        return None
    if _entry_fingerprint(entry) != fingerprint:
        _refuse("replay_conflict", "message id replay does not match the original")
    return entry


def _reply_valid(entries: list, body: SessionBusSendBody) -> bool:
    if not body.reply_to:
        return True
    for entry in entries:
        if entry.get("id") != body.reply_to or entry.get("kind") != MESSAGE_KIND:
            continue
        return (
            bool(entry.get("read_at")) and entry.get("sender_id") == body.recipient_id
        )
    _refuse(
        "invalid_reply", "reply_to is not a read message received from the recipient"
    )
    return False


def _validate_reply(
    cur: Any, uid: str, sender_id: str, body: SessionBusSendBody
) -> None:
    if not body.reply_to:
        return
    payload = _read_payload(cur, uid, sender_id)
    _blob, _metadata, entries = _parts(payload)
    _reply_valid(entries, body)


def send_message_row(
    conn: Any, uid: str, sender_id: str, body: SessionBusSendBody
) -> dict[str, Any]:
    """Append one validated/idempotent message; never create a second inbox."""
    content = body.body.strip()
    if not content:
        _refuse("invalid_value", "the message is empty")
    body = body.model_copy(
        update={"body": content, "sender_name": _clean_label(body.sender_name)}
    )
    now = _now()
    with conn.cursor() as cur:
        recipient = _locked(cur, uid, body.recipient_id)
        _validate_send(cur, uid, sender_id, recipient, body)
        blob, metadata, entries = _parts(recipient["payload"])
        _prune_expired(entries, now)
        fingerprint = _fingerprint(sender_id, body)
        existing = _replay(entries, body, fingerprint)
        if existing is not None:
            conn.rollback()
            return {"duplicate": True, "message": _message_view(existing)}
        unread = [
            e for e in entries if e.get("kind") == MESSAGE_KIND and not e.get("read_at")
        ]
        if len(unread) >= MAX_UNREAD:
            _refuse("inbox_full", f"recipient inbox is full ({MAX_UNREAD} unread)")
        _validate_reply(cur, uid, sender_id, body)
        entry = {
            "id": body.message_id,
            "kind": MESSAGE_KIND,
            "content": content,
            "client_id": sender_id,
            "added_at": _iso(now),
            "expires_at": _iso(now + timedelta(seconds=_ttl_seconds())),
            "sender_id": sender_id,
            "sender_name": body.sender_name,
            "sender_kind": body.sender_kind,
            "recipient_id": body.recipient_id,
            "reply_to": body.reply_to,
            "notified_at": "",
            "read_at": "",
        }
        entries.append(entry)
        metadata["chat_queue"] = entries
        _write_metadata(blob, metadata)
        _finish(conn, cur, uid, body.recipient_id, blob)
        return {"duplicate": False, "message": _message_view(entry)}


def _unread_entries(entries: list) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if entry.get("kind") == MESSAGE_KIND and not entry.get("read_at")
    ]


def _select_inbox(
    unread: list[dict[str, Any]], body: SessionBusInboxBody, now: datetime
) -> tuple[list[dict[str, Any]], bool]:
    wanted = set(body.message_ids)
    selected = [entry for entry in unread if not wanted or entry.get("id") in wanted]
    if body.action != "take":
        return selected, False
    selected = selected[: body.limit]
    for entry in selected:
        entry["read_at"] = _iso(now)
    return selected, bool(selected)


def inbox_row(
    conn: Any, uid: str, session_id: str, body: SessionBusInboxBody
) -> dict[str, Any]:
    """Read unread rows FIFO; take atomically stamps their dedup markers."""
    now = _now()
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        blob, metadata, entries = _parts(row["payload"])
        expired = _prune_expired(entries, now)
        unread = _unread_entries(entries)
        selected, marked = _select_inbox(unread, body, now)
        changed = expired or marked
        metadata["chat_queue"] = entries
        if changed:
            _write_metadata(blob, metadata)
            _finish(conn, cur, uid, session_id, blob)
        else:
            conn.rollback()
        return {
            "messages": [_message_view(entry) for entry in selected],
            "unread": len(unread),
        }


def worker_pop_row(conn: Any, uid: str, session_id: str) -> dict[str, Any]:
    """ADR-046 worker pop extended for durable, non-user-prompt messages."""
    now = _now()
    with conn.cursor() as cur:
        row = _locked(cur, uid, session_id)
        blob, metadata, entries = _parts(row["payload"])
        changed = _prune_expired(entries, now)
        front: dict[str, Any] | None = None
        for index, entry in enumerate(entries):
            if entry.get("kind") == MESSAGE_KIND:
                if entry.get("read_at") or entry.get("notified_at"):
                    continue
                entry["notified_at"] = _iso(now)
                front = entry
            else:
                front = entries.pop(index)
            changed = True
            break
        metadata["chat_queue"] = entries
        if changed:
            _write_metadata(blob, metadata)
            _finish(conn, cur, uid, session_id, blob)
        else:
            conn.rollback()
        return {"applied": True, "queue": entries, "front": front}
