"""``session_turns``: keep chat turns, verbatim, in project knowledge (K1.6).

Knowledge plan 2026-10-24, K1.6 (``/save``). R4 sends the exact text of the
turns a person chose, read from its local session store by turn id (never
re-typed by the model), plus a title, tags and a back-link to the session.
The server cannot check the text against a store it does not hold, so the
artifact records ``attested_by_client: true`` (a client attestation, like
the actor header: K-12).

Tier (K-1, ADR-084 D-2, decided 2026-10-24):

* the person's own text (``role: user``) is human-authored: established;
* an assistant answer is the model's text the person kept: **proposed**,
  origin :data:`KEPT_ORIGIN`, unless the person vouches for it ("I vouch
  for this"): then established, origin :data:`VOUCHED_ORIGIN`, with the
  person recorded as its establisher;
* an agent-initiated save (``x-r4-actor`` not ``user``) is an agent write:
  proposed, origin ``agent``; an agent cannot vouch.

Before anything is queued the text goes through the egress scan
(``egress_fetch.scan_content``) with token shapes added: a credential or
token-shaped string refuses the save by name, showing only a masked span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .actors import AGENT_ACTOR, normalize_actor

SESSION_TURNS_KIND = "session_turns"
#: An assistant answer the person kept without vouching (proposed).
KEPT_ORIGIN = "agent_authored_user_kept"
#: An assistant answer the person vouched for (established, they establish it).
VOUCHED_ORIGIN = "agent_authored_user_vouched"
#: The labels a session save may record beside ``user`` / ``agent``.
SAVE_ORIGINS = frozenset({KEPT_ORIGIN, VOUCHED_ORIGIN})

MAX_TURNS = 20
MAX_TEXT_BYTES = 2 * 1024 * 1024
_ROLES = frozenset({"user", "assistant"})
_TURN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

#: Token shapes on top of the egress scan's ``key: value`` patterns: a bare
#: token pasted into a chat has no ``token:`` in front of it.
TOKEN_SHAPES: tuple[tuple[str, str], ...] = (
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("github_token", r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}"),
    ("openai_style_key", r"\bsk-[A-Za-z0-9_-]{20,}"),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class SessionSaveRefused(ValueError):
    """The save is refused by name (``code``); nothing was queued."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionSave:
    """What the ingest route queues for one session save."""

    source: dict[str, Any]
    origin: str  # user | agent | KEPT_ORIGIN | VOUCHED_ORIGIN
    established_by: str = ""
    spec_extra: dict[str, Any] = field(default_factory=dict)


def _mask(span: str) -> str:
    """The first four characters and the length: enough to find it, not to use it."""
    return f"{span[:4]}… ({len(span)} chars)"


def _token_hit(text: str) -> tuple[str, str] | None:
    for name, pattern in TOKEN_SHAPES:
        match = re.search(pattern, text)
        if match:
            return name, match.group(0)
    return None


def scan_turn_text(text: str) -> None:
    """Refuse (:class:`SessionSaveRefused`, ``secret_detected``) text with a
    credential or token-shaped string: the egress scan's patterns, then the
    token shapes. The message names the pattern and a masked span."""
    from .egress_fetch import scan_content

    scan = scan_content(text.encode("utf-8"))
    hit: tuple[str, str] | None = None
    if not scan.passed and scan.matched_patterns:
        pattern = scan.matched_patterns[0]
        found = re.search(pattern, text)
        hit = (f"pattern {pattern}", found.group(0) if found else "")
    hit = hit or _token_hit(text)
    if hit is None:
        return
    name, span = hit
    raise SessionSaveRefused(
        "secret_detected",
        f"refused: the text holds a credential-like string ({name}: {_mask(span)}); "
        "nothing was saved. Remove it and save again.",
    )


def _turns(source: dict[str, Any]) -> list[dict[str, str]]:
    raw = source.get("turns")
    if not isinstance(raw, list) or not raw:
        raise SessionSaveRefused("bad_request", "session_turns needs at least one turn")
    if len(raw) > MAX_TURNS:
        raise SessionSaveRefused("bad_request", f"at most {MAX_TURNS} turns per save")
    return [_turn(index, item) for index, item in enumerate(raw)]


def _turn(index: int, item: Any) -> dict[str, str]:
    turn = item if isinstance(item, dict) else {}
    turn_id = str(turn.get("turn_id") or "")
    role = str(turn.get("role") or "")
    text = turn.get("text")
    if not _TURN_ID.match(turn_id) or role not in _ROLES:
        raise SessionSaveRefused(
            "bad_request", f"turns[{index}] needs a turn_id and a role (user|assistant)"
        )
    if not isinstance(text, str) or not text.strip():
        raise SessionSaveRefused("bad_request", f"turns[{index}] has no text")
    return {"turn_id": turn_id, "role": role, "text": text}


def _origin(turns: list[dict[str, str]], actor: str, vouch: bool) -> str:
    if normalize_actor(actor) == AGENT_ACTOR:
        return AGENT_ACTOR
    if all(t["role"] == "user" for t in turns):
        return "user"
    return VOUCHED_ORIGIN if vouch else KEPT_ORIGIN


def _content(turns: list[dict[str, str]]) -> str:
    """One turn: its text exactly. Several: each under a ``## <turn> (<role>)``
    heading, in the order given."""
    if len(turns) == 1:
        return turns[0]["text"]
    return "\n\n".join(f"## {t['turn_id']} ({t['role']})\n\n{t['text']}" for t in turns)


def _labels(source: dict[str, Any]) -> tuple[str, list[str]]:
    title = " ".join(str(source.get("title") or "").split())[:200]
    tags = [str(t).strip()[:64] for t in (source.get("tags") or []) if str(t).strip()][
        :20
    ]
    return title, tags


def _metadata(
    source: dict[str, Any],
    turns: list[dict[str, str]],
    origin: str,
    title: str,
    tags: list[str],
) -> dict[str, Any]:
    session_id = str(source.get("session_id") or "")[:128]
    return {
        "source_name": title,
        "title": title,
        "tags": tags,
        "session_id": session_id,
        "turn_ids": [t["turn_id"] for t in turns],
        "roles": sorted({t["role"] for t in turns}),
        "attested_by_client": True,
        "saved_origin": origin,
    }


def plan_session_save(
    source: dict[str, Any], *, actor: str, user_id: str
) -> SessionSave:
    """Validate and scan one ``session_turns`` source; the inline source the
    ingest job stores and the tier fields its spec carries."""
    turns = _turns(source)
    content = _content(turns)
    if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
        raise SessionSaveRefused("bad_request", "the saved text exceeds 2 MiB")
    scan_turn_text(content)
    origin = _origin(turns, actor, bool(source.get("vouch")))
    title, tags = _labels(source)
    title = (
        title
        or f"session {source.get('session_id') or ''} {turns[0]['turn_id']}".strip()
    )
    stored = {
        "kind": SESSION_TURNS_KIND,
        "uri": f"session-turns/{'-'.join(t['turn_id'] for t in turns)}.md",
        "content": content,
        "metadata": _metadata(source, turns, origin, title, tags),
    }
    established_by = user_id if origin in ("user", VOUCHED_ORIGIN) else ""
    extra: dict[str, Any] = {}
    if origin in SAVE_ORIGINS:
        extra["kept_origin"] = origin
    if origin == VOUCHED_ORIGIN:
        extra["established_by"] = user_id
    return SessionSave(stored, origin, established_by, extra)


def tier_stamp(spec: dict[str, Any], job_id: str) -> tuple[bool, dict[str, Any]]:
    """``(proposed, stamp)`` for the rows an ingest job's understand stage
    writes: an agent origin or a kept (unvouched) answer lands proposed; a
    vouched answer lands established and names its establisher."""
    if normalize_actor(spec.get("origin_actor")) == AGENT_ACTOR:
        return True, {"origin_actor": AGENT_ACTOR, "job_id": job_id}
    kept = spec.get("kept_origin")
    if kept == KEPT_ORIGIN:
        return True, {"origin_actor": KEPT_ORIGIN, "job_id": job_id}
    if kept == VOUCHED_ORIGIN:
        review = {
            "established_by": str(spec.get("established_by") or ""),
            "vouched": True,
        }
        return False, {"origin_actor": VOUCHED_ORIGIN, "review": review}
    return False, {}


def tier_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """The tier fields an ingest spec hands its chained understand job."""
    return {k: spec[k] for k in ("kept_origin", "established_by") if spec.get(k)}


def recorded_origin(spec: dict[str, Any]) -> str | None:
    """The label a contribution records for a session save, else ``None``
    (the request actor applies)."""
    kept = spec.get("kept_origin")
    if (
        kept in SAVE_ORIGINS
        and normalize_actor(spec.get("origin_actor")) != AGENT_ACTOR
    ):
        return str(kept)
    return None


__all__ = [
    "KEPT_ORIGIN",
    "MAX_TURNS",
    "SAVE_ORIGINS",
    "SESSION_TURNS_KIND",
    "TOKEN_SHAPES",
    "VOUCHED_ORIGIN",
    "SessionSave",
    "SessionSaveRefused",
    "plan_session_save",
    "recorded_origin",
    "scan_turn_text",
    "tier_fields",
    "tier_stamp",
]
