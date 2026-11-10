"""Server-bound remote subagent snapshots and paged event history.

Child rows reuse ``companion_session_snapshots``; no second session table or
writer exists. A venue lease is valid only for its root worker. Every write
rechecks that root lease, its running venue row and the exact immediate parent
(the root conversation or an already-bound child) before updating the child.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ._rail_identity import RailPrincipal
from .typed_errors import PlatformError, raise_typed

_SNAPSHOT_CAP = 4 * 1024 * 1024
_EVENT_CAP = 65_536
_SERVER_FIELDS = ("subagent_event_log", "subagent_binding")


class SubagentSnapshotBody(BaseModel):
    """One full R4 child record; the server owns binding and event fields."""

    payload: str = ""


class SubagentEventBody(BaseModel):
    """An R4 event-log row and its stable local id for replay suppression."""

    source_event_id: int
    event: dict[str, Any]


def _conflict(message: str) -> None:
    raise_typed(PlatformError.companionvenue("conflict", message))


def _bad_request(message: str) -> None:
    raise_typed(PlatformError.companionvenue("bad_request", message))


def _not_found() -> None:
    raise_typed(
        PlatformError.companionvenue("not_found", "subagent snapshot not found")
    )


def _json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    return _json_object(record.get("metadata")) or {}


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


def subagent_payload_kind(payload: Any) -> str:
    """Read the kind without accepting a child record through ordinary mirrors."""
    record = _json_object(payload)
    if record is None:
        return ""
    return str(_metadata(record).get("session_kind") or "")


def without_subagent_event_log(payload: str) -> str:
    """Ordinary snapshot reads never return the child's unbounded event vector."""
    record = _json_object(payload)
    if record is None or subagent_payload_kind(record) != "subagent":
        return payload
    stripped = dict(record)
    for key in _SERVER_FIELDS:
        stripped.pop(key, None)
    return json.dumps(stripped)


def _detail(conn: Any, uid: str, session_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.session_id, s.name, s.payload, s.update_marker,
                   s.venue, s.state, a.archived_at
            FROM companion_session_snapshots s
            LEFT JOIN companion_session_archives a ON a.session_id = s.session_id
            WHERE s.user_id = %s AND s.session_id = %s
            """,
            (uid, session_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    sid, name, payload, marker, venue, state, archived = row
    return {
        "session_id": sid,
        "name": name,
        "payload": payload,
        "updated_at": marker,
        "venue": venue,
        "state": state,
        "archived_at": archived,
    }


def _locked_detail(conn: Any, uid: str, session_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT payload FROM companion_session_snapshots
            WHERE user_id = %s AND session_id = %s
            FOR UPDATE
            """,
            (uid, session_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    outer = _detail(conn, uid, session_id)
    return {**(outer or {}), "payload": row[0]}


def _running_venue(conn: Any, principal: RailPrincipal) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT venue, state, lease_id FROM companion_venue_sessions
            WHERE user_id = %s AND session_id = %s
            """,
            (principal.owner, principal.session_id),
        )
        row = cur.fetchone()
    if row is None:
        _conflict("subagent parent venue is unbound")
    venue, state, lease_id = row
    if str(lease_id) != principal.lease_id:
        _conflict("subagent lease is not bound to the parent venue")
    if state != "running":
        _conflict("subagent parent venue is not running")
    return str(venue)


def _child_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(record)
    required = ("session_kind", "parent_session_id", "parent_agent_id", "venue")
    if any(not metadata.get(key) for key in required):
        _conflict("subagent immediate parent and venue binding are required")
    if metadata.get("session_kind") != "subagent":
        _conflict("subagent session kind must be subagent")
    if not metadata.get("agent_id"):
        _conflict("subagent agent identity is required")
    return metadata


def _metadata_revision(metadata: dict[str, Any]) -> int:
    revision = metadata.get("subagent_revision")
    valid = (
        isinstance(revision, int) and not isinstance(revision, bool) and revision >= 1
    )
    if not valid:
        _conflict("subagent revision must be a positive integer")
    return int(revision)


def _request_record(
    body: SubagentSnapshotBody, child_id: str
) -> tuple[dict, dict, int]:
    if len(body.payload.encode("utf-8")) > _SNAPSHOT_CAP:
        _bad_request("subagent snapshot payload exceeds the bounded request limit")
    record = _json_object(body.payload)
    if record is None:
        _bad_request("subagent snapshot payload must be a JSON object")
    if _text(record.get("id")) != child_id:
        _conflict("subagent session id does not match path")
    metadata = _child_metadata(record)
    return record, metadata, _metadata_revision(metadata)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    public = dict(record)
    for key in _SERVER_FIELDS:
        public.pop(key, None)
    return public


def _stored_child(detail: dict[str, Any], child_id: str) -> tuple[dict, dict, dict]:
    record = _json_object(detail.get("payload"))
    if record is None or _text(record.get("id")) != child_id:
        _not_found()
    if subagent_payload_kind(record) != "subagent":
        _not_found()
    binding = record.get("subagent_binding")
    if not isinstance(binding, dict):
        _conflict("subagent snapshot is unbound")
    required = (
        "root_session_id",
        "parent_session_id",
        "parent_agent_id",
        "agent_id",
        "lease_id",
        "venue",
    )
    if any(not binding.get(key) for key in required):
        _conflict("subagent snapshot has incomplete venue binding")
    return record, _metadata(record), binding


def _validate_nested_parent(
    detail: dict[str, Any], binding: dict[str, Any], principal: RailPrincipal
) -> None:
    parent_id = str(binding["parent_session_id"])
    _, _, parent_binding = _stored_child(detail, parent_id)
    if parent_binding.get("agent_id") != binding["parent_agent_id"]:
        _conflict("subagent parent agent identity mismatch")
    if parent_binding.get("root_session_id") != principal.session_id:
        _conflict("subagent immediate parent belongs to another venue")
    if parent_binding.get("lease_id") != principal.lease_id:
        _conflict("subagent immediate parent belongs to another lease")


def _validate_immediate_parent(
    conn: Any,
    principal: RailPrincipal,
    binding: dict[str, Any],
) -> None:
    parent_id = str(binding["parent_session_id"])
    detail = _detail(conn, principal.owner, parent_id)
    if detail is None:
        _conflict("subagent immediate parent snapshot not found")
    parent = _json_object(detail.get("payload"))
    if parent is None or _text(parent.get("id")) != parent_id:
        _conflict("subagent immediate parent identity mismatch")
    metadata = _metadata(parent)
    if _text(metadata.get("agent_id")) != str(binding["parent_agent_id"]):
        _conflict("subagent parent agent identity mismatch")
    if parent_id == principal.session_id:
        if metadata.get("session_kind") == "subagent":
            _conflict("subagent root parent must be the venue conversation")
        return
    _validate_nested_parent(detail, binding, principal)


def _validate_binding(
    conn: Any, principal: RailPrincipal, binding: dict[str, Any]
) -> None:
    if binding.get("root_session_id") != principal.session_id:
        _conflict("subagent root venue binding mismatch")
    venue = _running_venue(conn, principal)
    if binding.get("lease_id") != principal.lease_id:
        _conflict("subagent lease is not bound to the parent venue")
    if binding.get("venue") != venue:
        _conflict("subagent venue binding mismatch")
    _validate_immediate_parent(conn, principal, binding)


def _new_binding(
    conn: Any,
    principal: RailPrincipal,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    binding = {
        "root_session_id": principal.session_id,
        "parent_session_id": str(metadata["parent_session_id"]),
        "parent_agent_id": str(metadata["parent_agent_id"]),
        "agent_id": str(metadata["agent_id"]),
        "lease_id": principal.lease_id,
        "venue": str(metadata["venue"]),
    }
    _validate_binding(conn, principal, binding)
    return binding


def _check_existing_lineage(metadata: dict[str, Any], binding: dict[str, Any]) -> None:
    fields = ("parent_session_id", "parent_agent_id", "agent_id", "venue")
    if any(_text(metadata.get(key)) != _text(binding.get(key)) for key in fields):
        _conflict("subagent lineage does not match its server binding")


def _state(metadata: dict[str, Any]) -> str:
    states = {
        "running": "ongoing",
        "completed": "completed",
        "failed": "error",
        "cancelled": "stopped",
    }
    status = str(metadata.get("subagent_status") or "running")
    if status not in states:
        _conflict("subagent lifecycle state is invalid")
    return states[status]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _receipt(child_id: str, binding: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "session_id": child_id,
        "parent_session_id": binding["parent_session_id"],
        "agent_id": binding["agent_id"],
        **fields,
    }


def _insert(
    conn: Any, principal: RailPrincipal, child_id: str, record: dict, binding: dict
) -> bool:
    stored = dict(record, subagent_event_log=[], subagent_binding=binding)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO companion_session_snapshots
                (user_id, session_id, name, payload, update_marker, venue, state)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                principal.owner,
                child_id,
                str(record.get("name") or ""),
                json.dumps(stored),
                str(record.get("updated_at") or ""),
                binding["venue"],
                _state(_metadata(record)),
            ),
        )
        applied = cur.rowcount > 0
    conn.commit()
    return applied


def _update(
    conn: Any,
    principal: RailPrincipal,
    child_id: str,
    record: dict,
    stored_record: dict,
    binding: dict,
) -> bool:
    updated = dict(
        record,
        subagent_event_log=stored_record.get("subagent_event_log") or [],
        subagent_binding=binding,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE companion_session_snapshots
            SET name = %s, payload = %s, update_marker = %s, state = %s, updated_at = now()
            WHERE user_id = %s AND session_id = %s
            """,
            (
                str(record.get("name") or ""),
                json.dumps(updated),
                str(record.get("updated_at") or ""),
                _state(_metadata(record)),
                principal.owner,
                child_id,
            ),
        )
        applied = cur.rowcount > 0
    if applied:
        conn.commit()
    else:
        conn.rollback()
    return applied


def put_subagent_snapshot(
    conn: Any,
    principal: RailPrincipal,
    child_id: str,
    body: SubagentSnapshotBody,
) -> dict[str, Any]:
    """Create or advance one bound child, with revision/replay protection."""
    record, metadata, revision = _request_record(body, child_id)
    existing = _locked_detail(conn, principal.owner, child_id)
    if existing is None:
        binding = _new_binding(conn, principal, metadata)
        applied = _insert(conn, principal, child_id, record, binding)
        return _receipt(
            child_id,
            binding,
            created=True,
            applied=applied,
            replay=False,
            revision=revision,
        )
    stored, stored_metadata, binding = _stored_child(existing, child_id)
    _check_existing_lineage(metadata, binding)
    _new_binding(conn, principal, binding)
    current = int(stored_metadata.get("subagent_revision") or 0)
    if revision < current:
        conn.rollback()
        return _receipt(
            child_id,
            binding,
            created=False,
            applied=False,
            stale=True,
            replay=False,
            revision=current,
        )
    if revision == current and _canonical(record) == _canonical(_public_record(stored)):
        conn.rollback()
        return _receipt(
            child_id,
            binding,
            created=False,
            applied=False,
            replay=True,
            revision=current,
        )
    if revision == current:
        _conflict("subagent snapshot replay conflict at the same revision")
    applied = _update(conn, principal, child_id, record, stored, binding)
    return _receipt(
        child_id,
        binding,
        created=False,
        applied=applied,
        replay=False,
        revision=revision,
    )


def _event_lineage(
    event: dict[str, Any], child_id: str, binding: dict[str, Any]
) -> None:
    if _text(event.get("session_id")) != child_id:
        _conflict("subagent event session id does not match its bound child")
    data = event.get("data")
    if not isinstance(data, dict):
        _conflict("subagent event data must be an object")
    expected = {
        "child_session_id": child_id,
        "parent_session_id": binding["parent_session_id"],
        "parent_agent_id": binding["parent_agent_id"],
        "agent_id": binding["agent_id"],
    }
    if any(_text(data.get(key)) != str(value) for key, value in expected.items()):
        _conflict("subagent event lineage does not match its server binding")
    for key, bound in (
        ("agent_id", binding["agent_id"]),
        ("parent_agent_id", binding["parent_agent_id"]),
    ):
        if event.get(key) is not None and str(event.get(key)) != str(bound):
            _conflict("subagent event agent columns do not match its binding")


def _event_replay(
    events: list[dict], source_id: int, event: dict[str, Any]
) -> dict | None:
    existing = next(
        (row for row in events if row.get("source_event_id") == source_id), None
    )
    if existing is None:
        return None
    comparable = {
        key: value
        for key, value in existing.items()
        if key not in {"id", "source_event_id"}
    }
    if _canonical(comparable) != _canonical(event):
        _conflict("subagent event replay conflict for the same source id")
    return existing


def append_subagent_event(
    conn: Any,
    principal: RailPrincipal,
    child_id: str,
    body: SubagentEventBody,
) -> dict[str, Any]:
    """Append one bounded event in source order, idempotently under the child row lock."""
    source_id = body.source_event_id
    if isinstance(source_id, bool) or source_id < 1:
        _bad_request("subagent source event id must be positive")
    if (
        len(
            _canonical({"source_event_id": source_id, "event": body.event}).encode(
                "utf-8"
            )
        )
        > _EVENT_CAP
    ):
        _bad_request("subagent event payload exceeds the bounded request limit")
    detail = _locked_detail(conn, principal.owner, child_id)
    if detail is None:
        _not_found()
    stored, _, binding = _stored_child(detail, child_id)
    _validate_binding(conn, principal, binding)
    _event_lineage(body.event, child_id, binding)
    events = list(stored.get("subagent_event_log", []))
    replay = _event_replay(events, source_id, body.event)
    if replay is not None:
        conn.rollback()
        return _receipt(
            child_id,
            binding,
            applied=False,
            replay=True,
            id=replay["id"],
            source_event_id=source_id,
        )
    last_source = max((int(row.get("source_event_id", 0)) for row in events), default=0)
    if source_id <= last_source:
        _conflict("subagent event source order is older than the remote log")
    next_id = max((int(row.get("id", 0)) for row in events), default=0) + 1
    event = dict(body.event, id=next_id, source_event_id=source_id)
    stored["subagent_event_log"] = [*events, event]
    applied = _update(
        conn, principal, child_id, _public_record(stored), stored, binding
    )
    return _receipt(
        child_id,
        binding,
        applied=applied,
        replay=False,
        id=next_id,
        source_event_id=source_id,
    )


def read_subagent(
    conn: Any,
    uid: str,
    child_id: str,
    *,
    after_id: int,
    limit: int,
) -> dict[str, Any]:
    """Owner-only historical read; a stopped worker never erases this history."""
    detail = _detail(conn, uid, child_id)
    if detail is None:
        _not_found()
    stored, _, _ = _stored_child(detail, child_id)
    events = list(stored.get("subagent_event_log", []))
    remaining = [row for row in events if int(row.get("id", 0)) > after_id]
    page = remaining[:limit]
    return {
        "record": _public_record(stored),
        "session_id": child_id,
        "events": page,
        "next_after_id": int(page[-1]["id"]) if page else after_id,
        "has_more": len(remaining) > limit,
        "total_events": len(events),
        "limit": limit,
    }


def subagent_summary(conn: Any, uid: str, child_id: str) -> dict[str, Any]:
    """Lineage projection for the opt-in remote child inventory."""
    detail = _detail(conn, uid, child_id)
    if detail is None:
        return {}
    record, metadata, _ = _stored_child(detail, child_id)
    return {
        "created_at": _text(record.get("created_at")),
        "model": _text(record.get("model")),
        "message_count": len(record.get("messages", [])),
        "session_kind": "subagent",
        "parent_session_id": _text(metadata.get("parent_session_id")),
        "parent_agent_id": _text(metadata.get("parent_agent_id")),
        "agent_id": _text(metadata.get("agent_id")),
        "subagent_status": _text(metadata.get("subagent_status")),
    }


__all__ = [
    "SubagentEventBody",
    "SubagentSnapshotBody",
    "append_subagent_event",
    "put_subagent_snapshot",
    "read_subagent",
    "subagent_payload_kind",
    "subagent_summary",
    "without_subagent_event_log",
]
