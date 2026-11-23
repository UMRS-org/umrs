"""Security-watch audit families and their redaction (ADR-088 W0, D-3).

ADR-088 D-3: no detector is written for an event the audit stream does not
carry. This module is the one place that names those events and the fields
each may carry. Every call site goes through :func:`emit_security`, which
keeps only the fields of the event's schema, checks each value against the
field's kind, and writes the result through the existing emitter
(``umrs.api.audit_events.emit_audit``: log line, ring buffer, Postgres
``audit_events``). No second store.

Redaction is by construction, not by pattern hunting: a field the schema
does not name is dropped (``dropped_keys`` counts them), and a value that is
not of its field's kind becomes ``"<redacted>"``. So a call site that hands
in free text, a prompt, a file name, a token or an address by mistake writes
none of it. Identifiers that look like a credential (a JWT, an admin token,
a provider key) or that carry an IP address are redacted as well.

Families (``event`` column):

    auth.failed               edge or UMRS refused a request's identity
    zmq.session_minted        a CURVE session (or account key) was registered
    zmq.session_revoked       a CURVE session ended before its TTL
    egress.denied             the egress policy refused a fetch
    mcp.invoke_refused        a tool call was refused at call time
    scanner.finding           the recipe/skill scanner reported findings
    export.requested          a knowledge bundle export was queued
    export.refused            an export request was refused
    admin.role_changed        an account's role changed
    admin.account_state_changed  an account was disabled or re-enabled
    admin.quota_changed       a user's quota or a project's budget changed
    admin.access_system       a resource's access list now holds ``system``

    security.incident         reserved for the watch (W1): one incident row
    security.action           reserved for the watch (W1/W3): a T1/T2 action
    security.watch_degraded   reserved for the watch: it runs degraded

The ``security.*`` names are the schema the watch writes from W1 on; nothing
emits them yet. ``step.refused`` is deferred to a second tranche (its refusal
sites live in modules another lane owns).

The emitter lives in ``umrs.api``; its import is lazy and function-local so
foundation callers (``umrs.core``, ``umrs.knowledge``, ``umrs.security``)
never import ``umrs.api`` at module load. An audit failure is logged and
never changes the decision it describes (ADR-088 D-2).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

SECURITY_SCHEMA_VERSION = "1"
REDACTED = "<redacted>"
REDACTED_IP = "<ip>"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+/-]{0,95}$")
_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}\.)*[a-z0-9-]{1,63}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_IPV4_IN_TEXT = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_TOKEN_PREFIXES = ("eyj", "bearer", "umrs-admin-", "sk-", "hf_", "ghp_", "gho_")
_MAX_LIST = 32
_DENIED_SUFFIXES = frozenset({"failed", "denied", "invoke_refused", "refused"})

# event -> {field: kind}. A field not named here never reaches the stream.
SECURITY_FAMILIES: dict[str, dict[str, str]] = {
    "auth.failed": {"surface": "code", "reason": "code", "route": "route"},
    "zmq.session_minted": {
        "session_id": "id",
        "kind": "code",
        "ttl_seconds": "int",
        "max_gpus": "int",
        "max_walltime_sec": "int",
    },
    "zmq.session_revoked": {"session_id": "id", "cause": "code", "count": "int"},
    "egress.denied": {"host": "host", "reason": "code"},
    "mcp.invoke_refused": {
        "server_id": "id",
        "tool": "id",
        "reason": "code",
        "trust_class": "code",
    },
    "scanner.finding": {
        "target_kind": "code",
        "target_name": "id",
        "errors": "int",
        "warnings": "int",
        "infos": "int",
        "categories": "codes",
    },
    "export.requested": {
        "job_id": "id",
        "include_raw": "bool",
        "chunk_text": "bool",
        "allow_unsigned": "bool",
    },
    "export.refused": {"reason": "code", "requested_project": "id"},
    "admin.role_changed": {"username": "id", "old_role": "code", "new_role": "code"},
    "admin.account_state_changed": {"username": "id", "disabled": "bool"},
    "admin.quota_changed": {
        "subject_kind": "code",
        "subject": "id",
        "fields": "codes",
        "by_admin": "bool",
    },
    "admin.access_system": {
        "resource_kind": "code",
        "resource_id": "id",
        "source_event": "code",
    },
    "security.incident": {
        "incident_id": "id",
        "rule_id": "id",
        "severity": "code",
        "tier": "code",
        "evidence_ids": "ids",
        "count": "int",
    },
    "security.action": {
        "incident_id": "id",
        "rule_id": "id",
        "action": "code",
        "target_id": "id",
        "mode": "code",
    },
    "security.watch_degraded": {"reason": "code"},
}


def _token_like(text: str) -> bool:
    """A JWT (``eyJ...``), an admin token or a provider key by its prefix."""
    return text.lower().startswith(_TOKEN_PREFIXES)


def _carries_ip(text: str) -> bool:
    if _IPV4_IN_TEXT.search(text):
        return True
    try:
        ipaddress.ip_address(text.strip("[]"))
    except ValueError:
        return False
    return True


def _as_id(value: Any) -> str:
    text = str(value) if isinstance(value, (str, int)) else ""
    if not _ID_RE.fullmatch(text) or _token_like(text) or _carries_ip(text):
        return REDACTED
    return text


def _as_code(value: Any) -> str:
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if _CODE_RE.fullmatch(text) else "other"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_host(value: Any) -> str:
    text = value.strip().lower().rstrip(".") if isinstance(value, str) else ""
    if _carries_ip(text):
        return REDACTED_IP
    return text if _HOST_RE.fullmatch(text) else REDACTED


def _as_route(value: Any) -> str:
    """The route family only: the first two path segments, never a file name,
    an id or a query string (``/umrs/jobs/abc?x=1`` -> ``/umrs/jobs``)."""
    path = value.split("?", 1)[0] if isinstance(value, str) else ""
    parts = [p for p in path.split("/") if p][:2]
    return "/" + "/".join(p if _SEGMENT_RE.fullmatch(p) else "*" for p in parts)


def _as_list(value: Any, item: Callable[[Any], Any]) -> list[Any]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return sorted({item(v) for v in list(value)[:_MAX_LIST]})


_KINDS: dict[str, Callable[[Any], Any]] = {
    "id": _as_id,
    "code": _as_code,
    "int": _as_int,
    "bool": _as_bool,
    "host": _as_host,
    "route": _as_route,
    "codes": lambda v: _as_list(v, _as_code),
    "ids": lambda v: _as_list(v, _as_id),
}


def redact_details(event: str, details: dict[str, Any]) -> dict[str, Any]:
    """``details`` reduced to the event's schema, each value of its kind.

    Unknown events keep nothing. The result always names the schema
    version; ``dropped_keys`` counts the fields the schema does not know.
    """
    schema = SECURITY_FAMILIES.get(event, {})
    clean: dict[str, Any] = {
        field: _KINDS[kind](details[field])
        for field, kind in schema.items()
        if field in details
    }
    dropped = len([key for key in details if key not in schema])
    clean["security_schema"] = SECURITY_SCHEMA_VERSION
    if dropped:
        clean["dropped_keys"] = dropped
    return clean


def _default_outcome(event: str) -> str:
    return "denied" if event.split(".", 1)[-1] in _DENIED_SUFFIXES else "success"


def emit_security(
    event: str,
    *,
    actor: str = "",
    project_id: str = "",
    outcome: str = "",
    scope: str = "system",
    **details: Any,
) -> dict[str, Any] | None:
    """Write one security-family row; return the details written.

    Never raises: an unknown event is refused with a warning (None), an
    emitter failure is logged. Actor and project are identifiers and go
    through the same check as an ``id`` field.
    """
    if event not in SECURITY_FAMILIES:
        logger.warning("security audit: %r is not a known family; not written", event)
        return None
    clean = redact_details(event, details)
    try:
        from umrs.api.audit_events import emit_audit

        emit_audit(
            event,
            actor=_as_id(actor) if actor else "",
            project_id=_as_id(project_id) if project_id else "",
            details=clean,
            outcome=outcome or _default_outcome(event),
            scope=scope,
        )
    except Exception:  # audit never changes the decision it describes
        logger.warning("security audit %s not written", event, exc_info=True)
    return clean


def _access_system_source(record: dict[str, Any]) -> bool:
    event = str(record.get("event") or "")
    access = (record.get("details") or {}).get("access")
    return (
        event.endswith("access_changed")
        and isinstance(access, (list, tuple))
        and "system" in access
    )


_RESOURCE_ID_KEYS = ("job_id", "graph_id", "catalog_id", "bucket", "workflow_name")


def _derived_parties(record: dict[str, Any]) -> tuple[str, str, str]:
    """``(actor, project, resource id)`` of an ``*access_changed`` row."""
    details = record.get("details") or {}
    project = str(record.get("project_id") or "")
    resource_id = next((details[k] for k in _RESOURCE_ID_KEYS if details.get(k)), "")
    actor = details.get("by") or record.get("actor") or ""
    return str(actor), project, str(resource_id or project)


def derive_access_system(record: dict[str, Any]) -> dict[str, Any] | None:
    """``admin.access_system`` for an ``*access_changed`` row that grants
    ``system`` (ADR-088 D-3 "access -> system"); None otherwise.

    Hooked into ``emit_audit`` so every resource family that writes an
    ``*.access_changed`` row is covered, written after the change it reports.
    """
    if not _access_system_source(record):
        return None
    actor, project, resource_id = _derived_parties(record)
    event = str(record["event"])
    return emit_security(
        "admin.access_system",
        actor=actor,
        project_id=project,
        outcome="success",
        resource_kind=event.split(".", 1)[0],
        resource_id=resource_id,
        source_event=event,
    )
