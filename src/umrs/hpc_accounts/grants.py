"""Standing permissions (grants) at the server (ADR-112 wave C; plan §6
C/D.0 C1 and C4).

A grant is a document the person signed (``umrs-grant-v1@<domain>``) that
lets UMRS start runs of the listed workflow versions within caps, until it
expires. The person's gate is the authority: it verifies the signature and
keeps the counters at every submit. The server keeps a mirror (migration
0137): the signed bytes, so every step hands them to the gate, plus a status
and a display copy of the gate's usage.

Storing a grant (:func:`check_grant`) runs the C4 checks in order; each
failure is :class:`GrantInvalid` with a reason (``hpcaccess.grant_invalid``)
or :class:`GrantGateStale` (``hpcaccess.gate_version``) or a
``BudgetExceeded`` from the ledger (``hpcaccess.budget_exceeded``). The
server never verifies the signature; the gate does (the store's last check
is a live ``grant-check``).

The view never carries the signature. ``covers_now`` and ``stale`` (K11)
are computed at read time from the server's own digests, images and links.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The contract's grant payload cap (``hpc_gate.GRANT_MAX_BYTES``, 2 MiB).
GRANT_MAX_BYTES = 2 * 1024 * 1024
SIGNATURE_MAX_BYTES = 16 * 1024
#: The store accepts ``issued`` within ``now ± 300 s`` (K8).
ISSUED_SKEW = 300
#: At most this many active grants per person and venue (C4 step 6).
MAX_ACTIVE = 32
#: Grants need gate version 3 (K18).
GATE_VERSION_FOR_GRANTS = 3

STATUSES = ("active", "cancelled", "expired")
ENDED_REASONS = ("", "cancelled", "account_revoked", "expired")
GATE_CANCELS = ("", "pending", "recorded")
INPUTS_MODES = ("listed", "chained", "any")
ACCOUNT_REVOKED = "account_revoked"

#: ``hpcaccess.grant_invalid`` reasons (C4; K18 dropped ``gate_stale``).
REASONS = (
    "bad_envelope", "too_large", "bad_schema", "wrong_site", "wrong_principal",
    "wrong_login", "not_linked", "workflow_unknown", "workflow_mismatch",
    "image_unknown", "expired", "cancelled", "too_many", "conflict", "not_covered",
)

COLUMNS = (
    "venue", "grant_id", "username", "site_login", "payload_b64", "signature",
    "grant_sha256", "projects", "accounts", "workflows", "inputs_mode", "caps",
    "status", "ended_reason", "gate_cancel", "not_before", "expires_at", "usage",
    "usage_at", "created_at", "updated_at",
)
_JSON_COLUMNS = ("projects", "accounts", "workflows", "caps", "usage")
#: The grant view (never the payload or the signature), then K8/K11 fields.
VIEW = (
    "venue", "grant_id", "grant_sha256", "username", "site_login", "status",
    "ended_reason", "gate_cancel", "projects", "accounts", "workflows",
    "inputs_mode", "caps", "not_before", "expires_at", "created_at", "usage",
    "usage_at",
)
_ARMOR_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_ARMOR_END = "-----END SSH SIGNATURE-----"
_B64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")
_GRANT_ID = re.compile(r"[0-9a-f]{32}")


class GrantInvalid(ValueError):
    """``hpcaccess.grant_invalid``: *reason* is one of :data:`REASONS`."""

    code = "hpcaccess.grant_invalid"

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class GrantGateStale(ValueError):
    """``hpcaccess.gate_version`` (reason ``grants_need_gate_3``, K18)."""

    code = "hpcaccess.gate_version"
    reason = "grants_need_gate_3"


class GrantNotFound(LookupError):
    """``hpcaccess.grant_not_found`` (404)."""

    code = "hpcaccess.grant_not_found"


def _gate() -> Any:
    from r4_contracts import hpc_gate

    return hpc_gate


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(seconds: Any) -> datetime | None:
    return datetime.fromtimestamp(int(seconds), timezone.utc) if seconds else None


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


# ── the signed grant ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignedGrant:
    """A parsed grant: the exact payload bytes, its body and signature."""

    payload: bytes
    body: dict[str, Any]
    signature: str

    @property
    def grant_id(self) -> str:
        return str(self.body["grant_id"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def b64(self) -> str:
        return base64.b64encode(self.payload).decode("ascii")


def _decode_b64(text: Any) -> bytes:
    if not isinstance(text, str) or not text or not _B64.fullmatch(text):
        raise GrantInvalid("bad_envelope", "payload is not base64")
    if len(text) > (GRANT_MAX_BYTES * 4) // 3 + 4:
        raise GrantInvalid("too_large", f"the grant exceeds {GRANT_MAX_BYTES} bytes")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GrantInvalid("bad_envelope", "payload is not base64") from exc
    if len(data) > GRANT_MAX_BYTES:
        raise GrantInvalid("too_large", f"the grant exceeds {GRANT_MAX_BYTES} bytes")
    return data


def _signature(text: Any) -> str:
    """Shape only: one armored SSHSIG block (the gate verifies it)."""
    if not isinstance(text, str) or len(text) > SIGNATURE_MAX_BYTES:
        raise GrantInvalid("bad_envelope", "signature is not an armored SSHSIG")
    body = text.strip()
    if not (body.startswith(_ARMOR_BEGIN) and body.endswith(_ARMOR_END)):
        raise GrantInvalid("bad_envelope", "signature is not an armored SSHSIG")
    return text


def _parse(payload: bytes) -> dict[str, Any]:
    gate = _gate()
    try:
        body = gate.parse_strict(payload, json_values=False)
        gate.validate_grant(body)
    except (ValueError, gate.GateError) as exc:
        raise GrantInvalid("bad_schema", str(exc)[:300]) from exc
    return body


def decode_grant(payload_b64: Any, signature: Any, *, now: float | None = None) -> SignedGrant:
    """C4 step 1: decode, parse with the gate's own parser and schema, and
    ``issued`` within ``now ± 300 s`` (K8)."""
    payload = _decode_b64(payload_b64)
    sig = _signature(signature)
    body = _parse(payload)
    current = time.time() if now is None else now
    if abs(int(body["issued"]) - int(current)) > ISSUED_SKEW:
        raise GrantInvalid("bad_schema", "issued is not within 300 s of the server's clock")
    return SignedGrant(payload=payload, body=body, signature=sig)


# ── digests the grant pins (X1) ──────────────────────────────────────────────


def steps_digest(workflow_json: dict[str, Any]) -> str:
    """``steps_sha256``: sha256 of the canonical skeleton (json values)."""
    gate = _gate()
    skeleton = gate.workflow_steps_skeleton(workflow_json)
    return hashlib.sha256(gate.canonical_bytes(skeleton, json_values=True)).hexdigest()


def skeleton_bytes(workflow_json: dict[str, Any]) -> bytes:
    """The canonical skeleton bytes a grant run's envelope carries."""
    gate = _gate()
    return gate.canonical_bytes(gate.workflow_steps_skeleton(workflow_json), json_values=True)


def manifest_digest(manifest: dict[str, Any]) -> str:
    """``src_manifest_sha256``: sha256 of the canonical manifest (no floats)."""
    return hashlib.sha256(_gate().canonical_bytes(manifest, json_values=False)).hexdigest()


def server_digests(source_path: str, workflow_json: dict[str, Any]) -> tuple[str, str]:
    """``(src_manifest_sha256, steps_sha256)`` of a registered version."""
    from umrs.server.intent_proposal import src_manifest

    raw = [s for s in workflow_json.get("steps") or [] if isinstance(s, dict)]
    names = [str(g[0]) for g in _gate().step_graph(raw)]
    manifest = src_manifest(source_path, dict(zip(names, raw)))
    return manifest_digest(manifest), steps_digest(workflow_json)


# ── the store ────────────────────────────────────────────────────────────────


def row_from_grant(venue: str, username: str, grant: SignedGrant, usage: dict[str, Any]) -> dict[str, Any]:
    body = grant.body
    now = _now()
    return {
        "venue": venue, "grant_id": grant.grant_id, "username": username,
        "site_login": str(body["login"]), "payload_b64": grant.b64,
        "signature": grant.signature, "grant_sha256": grant.sha256,
        "projects": list(body["projects"]), "accounts": list(body["accounts"]),
        "workflows": [{"name": w["name"], "version": w["version"]} for w in body["workflows"]],
        "inputs_mode": str(body["inputs"]["mode"]), "caps": dict(body["caps"]),
        "status": "active", "ended_reason": "", "gate_cancel": "",
        "not_before": _ts(body["not_before"]), "expires_at": _ts(body["expires"]),
        "usage": dict(usage or {}), "usage_at": now if usage else None,
        "created_at": now, "updated_at": now,
    }


def _expired(row: dict[str, Any], now: datetime) -> bool:
    expires = row.get("expires_at")
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires)
    return row.get("status") == "active" and isinstance(expires, datetime) and expires < now


class GrantStoreBase:
    """Shared rules of the two stores (one venue each, I-11)."""

    venue: str

    def insert(self, row: dict[str, Any]) -> dict[str, Any]:
        """The stored row; the same bytes are idempotent, other bytes under
        the same id are ``conflict``."""
        existing = self.get(row["username"], row["grant_id"]) or self._any(row["grant_id"])
        if existing is not None:
            if existing["grant_sha256"] != row["grant_sha256"] or existing["username"] != row["username"]:
                raise GrantInvalid("conflict", "another grant with this id is stored")
            return existing
        return self._put(row)

    def list(self, username: str, *, status: str | None = None) -> list[dict[str, Any]]:
        """The person's grants here, newest first; expired rows are marked
        lazily (``expired``)."""
        now = _now()
        rows = self._list(username)
        for row in rows:
            if _expired(row, now):
                self.set_status(username, row["grant_id"], "expired", "expired")
                row.update(status="expired", ended_reason="expired")
        return [r for r in rows if status is None or r["status"] == status]

    def count_active(self, username: str) -> int:
        return len(self.list(username, status="active"))

    def end_for_account(self, username: str, reason: str = ACCOUNT_REVOKED) -> int:
        """Every active grant of the person here: ``cancelled`` (*reason*)."""
        ended = 0
        for row in self.list(username, status="active"):
            ended += bool(self.set_status(username, row["grant_id"], "cancelled", reason))
        return ended

    # subclass hooks
    def get(self, username: str, grant_id: str) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError

    def _any(self, grant_id: str) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError

    def _put(self, row: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def _list(self, username: str) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def set_status(  # pragma: no cover
        self, username: str, grant_id: str, status: str, reason: str, gate_cancel: str | None = None
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def set_usage(self, username: str, grant_id: str, usage: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class InMemoryGrantStore(GrantStoreBase):
    """``hpc_grants`` over a dict, shared through ``tables`` (the site
    account store's tables: a grant goes with its account)."""

    venue: str
    tables: dict[str, Any] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    @property
    def _rows(self) -> dict[tuple[str, str], dict[str, Any]]:
        return self.tables.setdefault("grants", {})

    def get(self, username: str, grant_id: str) -> dict[str, Any] | None:
        row = self._rows.get((self.venue, grant_id))
        return copy.deepcopy(row) if row and row["username"] == username else None

    def _any(self, grant_id: str) -> dict[str, Any] | None:
        row = self._rows.get((self.venue, grant_id))
        return copy.deepcopy(row) if row else None

    def _put(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._rows[(self.venue, row["grant_id"])] = copy.deepcopy({**row, "venue": self.venue})
        return copy.deepcopy(self._rows[(self.venue, row["grant_id"])])

    def _list(self, username: str) -> list[dict[str, Any]]:
        rows = [r for (v, _g), r in self._rows.items() if v == self.venue and r["username"] == username]
        return [copy.deepcopy(r) for r in sorted(rows, key=lambda r: r["created_at"], reverse=True)]

    def set_status(
        self, username: str, grant_id: str, status: str, reason: str, gate_cancel: str | None = None
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get((self.venue, grant_id))
            if row is None or row["username"] != username:
                return None
            row.update(status=status, ended_reason=reason, updated_at=_now())
            if gate_cancel is not None:
                row["gate_cancel"] = gate_cancel
            return copy.deepcopy(row)

    def set_usage(self, username: str, grant_id: str, usage: dict[str, Any]) -> None:
        with self._lock:
            row = self._rows.get((self.venue, grant_id))
            if row is not None and row["username"] == username:
                row.update(usage=dict(usage), usage_at=_now())


_SELECT = f"SELECT {', '.join(COLUMNS)} FROM hpc_grants"
_INSERT = (
    f"INSERT INTO hpc_grants ({', '.join(COLUMNS)}) VALUES "
    f"({', '.join('%s::jsonb' if c in _JSON_COLUMNS else '%s' for c in COLUMNS)})"
    " ON CONFLICT (venue, grant_id) DO NOTHING"
)
_STATUS = (
    "UPDATE hpc_grants SET status = %s, ended_reason = %s,"
    " gate_cancel = COALESCE(%s, gate_cancel), updated_at = now()"
    f" WHERE venue = %s AND username = %s AND grant_id = %s RETURNING {', '.join(COLUMNS)}"
)
_USAGE = (
    "UPDATE hpc_grants SET usage = %s::jsonb, usage_at = now(), updated_at = now()"
    " WHERE venue = %s AND username = %s AND grant_id = %s"
)


class PgGrantStore(GrantStoreBase):
    """``hpc_grants`` of one venue over a psycopg2 pool."""

    def __init__(self, pool: Any, venue: str) -> None:
        if not venue:
            raise ValueError("a grant store needs its venue (site name)")
        self._pool = pool
        self.venue = venue

    def _rows_of(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
        return [dict(zip(COLUMNS, r)) for r in rows]

    def get(self, username: str, grant_id: str) -> dict[str, Any] | None:
        rows = self._rows_of(_SELECT + " WHERE venue = %s AND username = %s AND grant_id = %s", (self.venue, username, grant_id))
        return rows[0] if rows else None

    def _any(self, grant_id: str) -> dict[str, Any] | None:
        rows = self._rows_of(_SELECT + " WHERE venue = %s AND grant_id = %s", (self.venue, grant_id))
        return rows[0] if rows else None

    def _put(self, row: dict[str, Any]) -> dict[str, Any]:
        values = tuple(
            json.dumps(row[c]) if c in _JSON_COLUMNS else (self.venue if c == "venue" else row[c])
            for c in COLUMNS
        )
        self._rows_of(_INSERT, values)
        stored = self._any(row["grant_id"])
        if stored is None or stored["grant_sha256"] != row["grant_sha256"]:
            raise GrantInvalid("conflict", "another grant with this id is stored")
        return stored

    def _list(self, username: str) -> list[dict[str, Any]]:
        sql = _SELECT + " WHERE venue = %s AND username = %s ORDER BY created_at DESC"
        return self._rows_of(sql, (self.venue, username))

    def set_status(
        self, username: str, grant_id: str, status: str, reason: str, gate_cancel: str | None = None
    ) -> dict[str, Any] | None:
        rows = self._rows_of(_STATUS, (status, reason, gate_cancel, self.venue, username, grant_id))
        return rows[0] if rows else None

    def set_usage(self, username: str, grant_id: str, usage: dict[str, Any]) -> None:
        self._rows_of(_USAGE, (json.dumps(usage), self.venue, username, grant_id))


def grant_store_for(account_store: Any) -> GrantStoreBase:
    """The grant store beside a site account store (same venue and backing)."""
    pool = getattr(account_store, "_pool", None)
    if pool is not None:
        return PgGrantStore(pool, account_store.venue)
    return InMemoryGrantStore(account_store.venue, account_store.tables)


def end_account_grants(account_store: Any, username: str) -> int:
    """A revoke, the account cascade or a suspension ends every active grant
    of the person at this venue (``account_revoked``). Best-effort count."""
    try:
        return grant_store_for(account_store).end_for_account(username)
    except Exception:
        import logging

        logging.getLogger(__name__).warning("grants of %s not ended", username, exc_info=True)
        return 0


# ── the store's checks (C4 steps 2-7) ────────────────────────────────────────


@dataclass(frozen=True)
class GrantContext:
    """What the server knows that a grant must match."""

    venue: str
    principal: str
    account: dict[str, Any]
    links: list[dict[str, Any]]
    memberships: frozenset[str] | None
    #: ``(name, version) -> (src_manifest_sha256, steps_sha256)`` or None.
    digests_for: Callable[[str, str], tuple[str, str] | None]
    #: Every ``(ref, image_id, archive_sha256)`` the site publishes.
    images: frozenset[tuple[str, str, str]]


def check_binding(body: dict[str, Any], ctx: GrantContext) -> None:
    """C4 step 2: site, principal, login; the gate is version 3 or newer."""
    for field_, actual, reason in (
        ("site", ctx.venue, "wrong_site"),
        ("principal", ctx.principal, "wrong_principal"),
        ("login", str(ctx.account.get("verified_login") or ""), "wrong_login"),
    ):
        if body.get(field_) != actual:
            raise GrantInvalid(reason, f"{field_}: the grant says {str(body.get(field_))[:80]!r}")
    if _version(ctx.account.get("gate_version")) < GATE_VERSION_FOR_GRANTS:
        raise GrantGateStale("grants_need_gate_3: your gate predates grants: run 'r4 hpc add' again")


def _version(value: Any) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0


def active_links(ctx: GrantContext) -> dict[str, str]:
    """``project -> slurm_account`` of the person's active links here whose
    project they are a live member of."""
    live = ctx.memberships
    return {
        str(link["project_id"]): str(link["slurm_account"])
        for link in ctx.links
        if link.get("status") == "active" and (live is None or link["project_id"] in live)
    }


def check_links(body: dict[str, Any], ctx: GrantContext) -> None:
    """C4 step 3: each project linked (and a live membership); accounts ⊆ links."""
    links = active_links(ctx)
    missing = [p for p in body["projects"] if p not in links]
    if missing:
        raise GrantInvalid("not_linked", f"project {missing[0]!r} has no active link of yours here")
    allowed = {links[p] for p in body["projects"]}
    extra = [a for a in body["accounts"] if a not in allowed]
    if extra:
        raise GrantInvalid("not_linked", f"account {extra[0]!r} is not linked to the grant's projects")


def workflow_problem(entry: dict[str, Any], ctx: GrantContext) -> tuple[str, str] | None:
    """``(reason, detail)`` when the server's digests of a version differ."""
    ref = f"{entry['name']}@{entry['version']}"
    found = ctx.digests_for(str(entry["name"]), str(entry["version"]))
    if found is None:
        return "workflow_unknown", ref
    if found != (entry["src_manifest_sha256"], entry["steps_sha256"]):
        return "workflow_mismatch", ref
    return None


def check_workflows(body: dict[str, Any], ctx: GrantContext) -> None:
    """C4 step 4: each version registered and visible; both digests equal."""
    for entry in body["workflows"]:
        problem = workflow_problem(entry, ctx)
        if problem is not None:
            raise GrantInvalid(*problem)


def _image_key(image: dict[str, Any]) -> tuple[str, str, str]:
    return (str(image["ref"]), str(image["image_id"]), str(image["archive_sha256"]))


def check_images(body: dict[str, Any], ctx: GrantContext) -> None:
    """C4 step 5: every image is one the site publishes."""
    for image in body["images"]:
        if _image_key(image) not in ctx.images:
            raise GrantInvalid("image_unknown", str(image["image_id"])[:12])


def check_budgets(body: dict[str, Any], ledger: Any) -> None:
    """C4 step 7 (D3, K14): a grant that could never fit a project's budget
    is refused; a grant reserves nothing."""
    from .usage_ledger import BudgetExceeded, budget_seconds

    want_s = int(body["caps"]["max_gpu_hours"]) * 3600
    for project in body["projects"]:
        budget_s = budget_seconds(ledger.max_gpu_hours(project))
        if budget_s <= 0:
            continue
        sums = ledger.project_totals(project)
        used_s, res_s = int(sums["gpu_seconds"]), int(sums["reserved_gpu_seconds"])
        if want_s > budget_s - used_s - res_s:
            raise BudgetExceeded(project, used_s, res_s, want_s, budget_s)


def check_grant(grant: SignedGrant, ctx: GrantContext, store: GrantStoreBase, ledger: Any) -> None:
    """C4 steps 2-7, in order (the live ``grant-check`` is the caller's)."""
    body = grant.body
    check_binding(body, ctx)
    check_links(body, ctx)
    check_workflows(body, ctx)
    check_images(body, ctx)
    existing = store.get(ctx.principal, grant.grant_id)
    if existing is None and store.count_active(ctx.principal) >= MAX_ACTIVE:
        raise GrantInvalid("too_many", f"at most {MAX_ACTIVE} active grants per person here")
    check_budgets(body, ledger)


# ── the view (K11) ───────────────────────────────────────────────────────────


def signed_body(row: dict[str, Any]) -> dict[str, Any]:
    """The stored grant's body (it was parsed strictly when it was stored)."""
    return json.loads(base64.b64decode(row["payload_b64"]))


def stale_reasons(row: dict[str, Any], ctx: GrantContext) -> list[str]:
    """What the grant no longer covers here, computed now (K11)."""
    body = signed_body(row)
    links = active_links(ctx)
    reasons = [f"workflow_mismatch: {w['name']}@{w['version']}" for w in body["workflows"] if workflow_problem(w, ctx)]
    reasons += [f"image_unknown: {str(i['image_id'])[:12]}" for i in body["images"] if _image_key(i) not in ctx.images]
    reasons += [f"link_inactive: {p}" for p in body["projects"] if p not in links]
    return reasons


def grant_view(row: dict[str, Any], ctx: GrantContext | None = None) -> dict[str, Any]:
    """The wire view; ``covers_now`` / ``stale`` when *ctx* is given."""
    view = {k: _iso(row.get(k)) for k in VIEW}
    stale = stale_reasons(row, ctx) if ctx is not None else []
    view["covers_now"] = row.get("status") == "active" and ctx is not None and not stale
    view["stale"] = stale
    return view


def candidates(
    rows: Iterable[dict[str, Any]], ctx: GrantContext, *, project: str, workflow: tuple[str, str], account: str
) -> list[dict[str, Any]]:
    """The admission's candidates (C4 step 2): active grants that list the
    project, the version and the account, and cover now (K11)."""
    found = []
    for row in rows:
        body = signed_body(row)
        listed = any((w["name"], w["version"]) == workflow for w in body["workflows"])
        if row["status"] != "active" or not listed:
            continue
        if project in body["projects"] and account in body["accounts"] and not stale_reasons(row, ctx):
            found.append(row)
    return found


def check_grant_id(grant_id: str) -> str:
    if not _GRANT_ID.fullmatch(str(grant_id or "")):
        raise GrantNotFound(f"no grant {str(grant_id)[:40]!r} here")
    return str(grant_id)


__all__ = [
    "ACCOUNT_REVOKED",
    "COLUMNS",
    "ENDED_REASONS",
    "GATE_CANCELS",
    "GRANT_MAX_BYTES",
    "INPUTS_MODES",
    "MAX_ACTIVE",
    "REASONS",
    "STATUSES",
    "GrantContext",
    "GrantGateStale",
    "GrantInvalid",
    "GrantNotFound",
    "GrantStoreBase",
    "InMemoryGrantStore",
    "PgGrantStore",
    "SignedGrant",
    "active_links",
    "candidates",
    "check_grant",
    "check_grant_id",
    "decode_grant",
    "end_account_grants",
    "grant_store_for",
    "grant_view",
    "manifest_digest",
    "row_from_grant",
    "server_digests",
    "signed_body",
    "skeleton_bytes",
    "stale_reasons",
    "steps_digest",
]
