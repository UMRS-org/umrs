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

Before anything is queued the text, its title and its tags go through
:func:`scan_turn_text`: a credential or token-shaped string refuses the save
by name, showing only a masked span. (Review of K1 part B, 2026-10-24: the
egress scan's key-name patterns refused ordinary code answers such as
``pwd = os.getcwd()`` and missed ``{"password": "…"}``, URL credentials and
anything past its 500 000-character window; the save scan matches a key
only with a value that looks like a secret, and reads the whole text.)

Only the ingest route builds a save. A raw ``knowledge_ingest`` or
``knowledge_understand`` spec submitted through ``POST /umrs/jobs`` or the
ZMQ submit may not carry the tier fields or a ``session_turns`` source
(:func:`client_spec_refusal`), and the establisher of a vouched save is the
job's owner, never a spec field.
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

#: Token shapes: a bare token pasted into a chat has no ``token:`` in front.
TOKEN_SHAPES: tuple[tuple[str, str], ...] = (
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("github_token", r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}"),
    ("gitlab_token", r"\bglpat-[A-Za-z0-9_-]{20,}"),
    ("openai_style_key", r"\bsk-[A-Za-z0-9_-]{20,}"),
    ("stripe_key", r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("url_credentials", r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@"),
    ("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
)

#: ``key: value`` / ``key = value`` / ``"key": "value"``: a credential key
#: name, an optional closing quote, then a value. The key alone is not a hit
#: (``pwd = os.getcwd()``, ``max_token: 512``): :func:`_secret_value` judges
#: the value.
_KEY_VALUE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<key>password|passwd|pwd|passphrase|secret|"
    r"secret_key|client_secret|api_key|apikey|api-key|access_key|"
    r"aws_secret_access_key|private_key|token|auth_token|access_token|"
    r"refresh_token)[\"']?\s*[:=]\s*[\"']?(?P<value>[^\s\"'`,;(){}\[\]<>]+)"
    r"(?=$|[\s\"'`,;)}\]])"
)
_PASSWORD_KEYS = frozenset({"password", "passwd", "pwd", "passphrase"})
_PLACEHOLDER = re.compile(
    r"^(?:\$\{?\w+\}?|%\(?\w+\)?s?|\*+|x+|\.+|null|none|true|false)$", re.IGNORECASE
)


def _secret_value(key: str, value: str) -> bool:
    """A value that looks like a credential: a password of 6+ characters,
    any other key's value of 12+, with a digit in it (or 24+ characters
    without one); never a placeholder (``${TOKEN}``, ``***``, ``xxxx``)."""
    if _PLACEHOLDER.match(value):
        return False
    if len(value) >= 24:
        return True
    floor = 6 if key.lower() in _PASSWORD_KEYS else 12
    return len(value) >= floor and any(c.isdigit() for c in value)


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


def _key_value_hit(text: str) -> tuple[str, str] | None:
    for match in _KEY_VALUE.finditer(text):
        if _secret_value(match.group("key"), match.group("value")):
            return f"{match.group('key').lower()}=…", match.group("value")
    return None


def scan_turn_text(text: str) -> None:
    """Refuse (:class:`SessionSaveRefused`, ``secret_detected``) text with a
    credential or token-shaped string, anywhere in it: a token shape, else a
    credential key with a secret-looking value. The message names what
    matched and a masked span, never the secret."""
    hit = _token_hit(text) or _key_value_hit(text)
    if hit is None:
        return
    name, span = hit
    raise SessionSaveRefused(
        "secret_detected",
        f"refused: the text holds a credential-like string ({name}: {_mask(span)}); "
        "nothing was saved. Save a turn without it, or upload an edited copy "
        "(r4 knowledge upload).",
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
    labels: tuple[str, str, list[str]],
) -> dict[str, Any]:
    """The artifact's save metadata. ``saved_by`` lets a listing show the
    back-link only to callers who may see that person's save (the artifact
    is shared by content hash: ``contributions.public_metadata``)."""
    origin, title, tags = labels
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
        "saved_by": str(source.get("_saved_by") or ""),
    }


def plan_session_save(
    source: dict[str, Any], *, actor: str, user_id: str
) -> SessionSave:
    """Validate and scan one ``session_turns`` source; the inline source the
    ingest job stores and the tier field its spec carries."""
    turns = _turns(source)
    content = _content(turns)
    if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
        raise SessionSaveRefused("bad_request", "the saved text exceeds 2 MiB")
    title, tags = _labels(source)
    scan_turn_text("\n".join([content, title, *tags]))
    origin = _origin(turns, actor, bool(source.get("vouch")))
    title = (
        title
        or f"session {source.get('session_id') or ''} {turns[0]['turn_id']}".strip()
    )
    stored = {
        "kind": SESSION_TURNS_KIND,
        "uri": f"session-turns/{'-'.join(t['turn_id'] for t in turns)}.md",
        "content": content,
        "metadata": _metadata(
            {**source, "_saved_by": user_id}, turns, (origin, title, tags)
        ),
    }
    established_by = user_id if origin in ("user", VOUCHED_ORIGIN) else ""
    extra = {"kept_origin": origin} if origin in SAVE_ORIGINS else {}
    return SessionSave(stored, origin, established_by, extra)


def planned_tier(origin: str) -> str:
    """The tier a save with *origin* lands in (K-1)."""
    return "proposed" if origin in (AGENT_ACTOR, KEPT_ORIGIN) else "established"


#: Origins whose rows are already established: a vouch over them changes nothing.
_ESTABLISHED_ORIGINS = frozenset({"user", VOUCHED_ORIGIN})


def vouch_refusal(
    save: SessionSave, stored_origins: set[str]
) -> SessionSaveRefused | None:
    """A vouched save of text already in the target graph through a proposed
    contribution (kept, agent, ...) is refused by name: the bytes deduplicate
    to the stored artifact, its contribution keeps its first origin and its
    rows are not re-extracted, so the vouch would change nothing while the
    receipt said "established" (review of K1 part B, 2026-10-24)."""
    if save.origin != VOUCHED_ORIGIN or not stored_origins - _ESTABLISHED_ORIGINS:
        return None
    return SessionSaveRefused(
        "already_saved",
        "this text is already in the graph as proposed (kept or agent-written); "
        "a vouch does not change a stored item. Establish its rows with "
        "r4 knowledge review (establish --run <run> or <row-id>).",
    )


def tier_stamp(
    spec: dict[str, Any], job_id: str, user_id: str
) -> tuple[bool, dict[str, Any]]:
    """``(proposed, stamp)`` for the rows an ingest job's understand stage
    writes: an agent origin or a kept (unvouched) answer lands proposed; a
    vouched answer lands established and names its establisher, the job's
    owner (*user_id*): never a spec field, which a raw job spec could set."""
    if normalize_actor(spec.get("origin_actor")) == AGENT_ACTOR:
        return True, {"origin_actor": AGENT_ACTOR, "job_id": job_id}
    kept = spec.get("kept_origin")
    if kept == KEPT_ORIGIN:
        return True, {"origin_actor": KEPT_ORIGIN, "job_id": job_id}
    if kept == VOUCHED_ORIGIN:
        review = {"established_by": user_id, "vouched": True}
        return False, {"origin_actor": VOUCHED_ORIGIN, "review": review}
    return False, {}


def tier_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """The tier field an ingest spec hands its chained understand job."""
    return {"kept_origin": spec["kept_origin"]} if spec.get("kept_origin") else {}


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


#: Spec fields only the ingest route sets (a /save's tier).
SERVER_ONLY_SPEC_FIELDS = ("kept_origin", "established_by")


def client_spec_refusal(spec: Any) -> str | None:
    """Why a client-built ``knowledge_ingest`` / ``knowledge_understand``
    spec (``POST /umrs/jobs``, the ZMQ submit) may not be queued, or None:
    the /save tier fields and ``session_turns`` sources come only from the
    ingest route, which scans the text and takes the vouch from the caller
    (review of K1 part B, 2026-10-24: a raw spec stored unscanned text as
    "vouched by" any user it named)."""
    if not isinstance(spec, dict):
        return None
    named = [f for f in SERVER_ONLY_SPEC_FIELDS if f in spec]
    if named:
        return f"spec.{named[0]} is set by POST /umrs/k2/ingest only (a /save)"
    if SESSION_TURNS_KIND in _source_kinds(spec.get("sources")):
        return "session_turns sources are saved through POST /umrs/k2/ingest (/save)"
    return None


def _source_kinds(sources: Any) -> set[str]:
    """The ``kind`` of each source object in a spec's ``sources`` list."""
    rows = sources if isinstance(sources, list) else []
    return {str(src.get("kind") or "") for src in rows if isinstance(src, dict)}


__all__ = [
    "KEPT_ORIGIN",
    "MAX_TURNS",
    "SAVE_ORIGINS",
    "SERVER_ONLY_SPEC_FIELDS",
    "SESSION_TURNS_KIND",
    "TOKEN_SHAPES",
    "VOUCHED_ORIGIN",
    "SessionSave",
    "SessionSaveRefused",
    "client_spec_refusal",
    "plan_session_save",
    "planned_tier",
    "recorded_origin",
    "scan_turn_text",
    "tier_fields",
    "tier_stamp",
    "vouch_refusal",
]
