"""The caller's own HPC account at this site (ADR-112 D-3/D-7, wave B1).

Six routes under ``/umrs/sites/me/hpc`` (plan §4 B1.5a, the binding wire
contract shared with the gate and r4). The caller is always the verified
gateway identity and the site is always this front's venue: nobody reads or
changes another person's account here.

- ``POST /transport-key`` issues (or, with ``rotate``, replaces) the
  person's transport key. The answer carries the public key only.
- ``POST /verify`` reaches the person's gate with that key: ``probe`` must
  be refused by the gate itself (``gate_probe_refusal``, D-7: not a site
  ``ForceCommand``), then ``hello`` must report the claimed login and the
  gate this server ships (sha256). Only then is the account ``active``, with
  the site's own report of the login and its Slurm associations stored as
  ``verified_*``. A verify also suspends any link whose allocation the site
  no longer lists (D-7), and drops transport keys a rotation replaced.
- ``GET`` shows the account and links; ``POST /links`` links one of the
  caller's projects (live membership, ADR-097) to an allocation the site
  listed for them; ``DELETE /links/{project_id}`` and ``DELETE`` (revoke:
  the transport key is destroyed) end them.

Registration (the three POST routes) is open only where the site's
``scheduler.hpc_access.modes_allowed`` lists ``user_account``; elsewhere it
refuses ``hpcaccess.mode_undeclared``. Reading and revoking stay open, so a
person can always see and end what they registered. Nothing here runs work
under a user's login (wave B2).

Every route that changes the account is the person's act (ADR-112 D-8): a
request marked ``x-r4-actor`` as anything but the person (an agent; r4 marks
its Companion's calls) is refused ``hpcaccess.person_required``. A verify
that fails on an active account suspends it (D-7: active only while the
server reaches the gate), and one person runs one verify at a time.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import threading
from collections.abc import Iterator
from typing import Any, NoReturn

import anyio.from_thread
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from umrs.hpc_accounts import gate_client, transport_keys
from umrs.hpc_accounts.revoke import revoke_account
from umrs.hpc_accounts.store import SiteAccountStore

from ._verified_headers import CallerIdentity, get_verified_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/sites/me/hpc", tags=["sites"])

USER_ACCOUNT = "user_account"
ASSOC_KEYS = ("cluster", "account", "partition", "qos")
_ASSOC_CAP = 256
_SIGNER_CAP = 16
_FIELD_CAP = 128
# The venue keys site_accounts / project_site_allocations as VARCHAR(32), the
# width every venue column has (jobs 0088, users 0113).
VENUE_MAX = 32
# B1.3 version floor, checked on the site's own OpenSSH (hello.openssh): the
# gate's ``ssh-keygen -Y verify`` needs 8.1, and 8.2 for sk- signing keys.
OPENSSH_FLOOR = (8, 1)
OPENSSH_SK_FLOOR = (8, 2)
_OPENSSH = re.compile(r"OpenSSH_(\d+)\.(\d+)")

# Test seam: an async ``(argv, stdin, timeout) -> (code, out, err)``; None = ssh.
_gate_runner: gate_client.Runner | None = None

# One verify per person at a time (each holds a worker for two gate calls).
_VERIFY_GUARD = threading.Lock()
_VERIFY_LOCKS: dict[tuple[str, str], threading.Lock] = {}


class TransportKeyRequest(BaseModel):
    rotate: bool = False


class VerifyRequest(BaseModel):
    login: str = Field(..., pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
    signers: list[str] = Field(default_factory=list, max_length=_SIGNER_CAP)
    uv_required: bool = False
    allow_non_sk: bool = False


class LinkRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=_FIELD_CAP)
    slurm_account: str = Field(..., min_length=1, max_length=_FIELD_CAP)


# ── seams ────────────────────────────────────────────────────────────────


def _load_site() -> Any:
    from .sites import _load_site as load

    return load()


def _make_store(request: Request, venue: str) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.projects(
                "unavailable",
                "site account store unavailable: shared umrs_db pool absent",
            )
        )
    return SiteAccountStore(pool, venue)


def _gate_manifest() -> tuple[str, str]:
    """(version, sha256) of the gate this server ships (r4-contracts)."""
    from r4_contracts import hpc_gate

    return str(hpc_gate.GATE_VERSION), hpc_gate.gate_sha256()


def _live_memberships(request: Request, user: str) -> list[str] | None:
    """ADR-097 D-1 live memberships over the shared pool; None = unreadable."""
    from .projects import ProjectStore, live_memberships

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return None
    conn = pool.getconn()
    try:
        return live_memberships(ProjectStore(conn), user)
    finally:
        try:
            conn.rollback()  # a read must not leave the conn idle in transaction
        finally:
            pool.putconn(conn)


# ── site and mode ────────────────────────────────────────────────────────


def _venue_site() -> tuple[Any, str]:
    site = _load_site()
    venue = str(getattr(site, "name", "") or "")
    if not venue:
        raise_typed(
            PlatformError.hpcaccess(
                "mode_undeclared", "this server has no site configuration"
            )
        )
    return site, venue


def _registration_site() -> tuple[Any, str, Any]:
    """(site, venue, user_account block) where registration is open."""
    site, venue = _venue_site()
    hpc = getattr(getattr(site, "scheduler", None), "hpc_access", None)
    user_account = getattr(hpc, "user_account", None)
    if (
        USER_ACCOUNT not in (getattr(hpc, "modes_allowed", None) or [])
        or user_account is None
    ):
        raise_typed(
            PlatformError.hpcaccess(
                "mode_undeclared",
                f"site {venue} does not allow user accounts (hpc_access.modes_allowed)",
                site=venue,
            )
        )
    _require_gate_site_name(venue)
    return site, venue, user_account


def _require_gate_site_name(venue: str) -> None:
    """The venue lands in the gate's ``--site`` and in the transport comment;
    r4 and the gate accept only ``[a-z0-9-]{1,40}`` there, and the tables key
    it as ``VARCHAR(32)``, so registration is not offered under any other
    name (2026-11-08 integration and review)."""
    from r4_contracts.hpc_gate import is_site_name

    if is_site_name(venue) and len(venue) <= VENUE_MAX:
        return
    raise_typed(
        PlatformError.hpcaccess(
            "mode_undeclared",
            f"site name {venue!r} cannot name a gate: user accounts need a site "
            f"name of 1-{VENUE_MAX} lowercase letters, digits or dashes",
            site=venue,
        )
    )


def _require_person(request: Request) -> None:
    """D-8: the account's trust anchors move only on the person's own act:
    never on a request an agent marked (``x-r4-actor``) or one made inside a
    work level (an ADR-108 ``x-r4-scope`` token: loop or teamup work)."""
    from umrs.knowledge.actors import AGENT_ACTOR, normalize_actor

    agent = normalize_actor(request.headers.get("x-r4-actor")) == AGENT_ACTOR
    if not agent and not (request.headers.get("x-r4-scope") or "").strip():
        return
    raise_typed(
        PlatformError.hpcaccess(
            "person_required",
            "this change to your HPC account is your own act: run the r4 hpc "
            "command yourself, not through the assistant",
        )
    )


def _not_enrolled(venue: str, why: str) -> NoReturn:
    raise_typed(PlatformError.hpcaccess("not_enrolled", why, site=venue))


# ── transport key ────────────────────────────────────────────────────────


def _load_key(venue: str, user: str, key_id: str) -> Any:
    """The stored key, None when gone; an unreadable one is not_enrolled."""
    try:
        return transport_keys.load(venue, user, key_id)
    except transport_keys.TransportKeyError as exc:
        _not_enrolled(venue, f"the transport key is unusable: {exc}")


def _recorded_key_id(account: Any) -> str:
    if account is None or account["status"] == "revoked":
        return ""
    return str(account["transport_key_id"] or "")


def _current_key(account: Any, venue: str, user: str, rotate: bool) -> Any:
    """The account's live key when it may be reused, else None."""
    key_id = _recorded_key_id(account)
    if rotate or not key_id:
        return None
    return _load_key(venue, user, key_id)


def _new_key(store: Any, user: str, previous: str) -> Any:
    """A new transport key on disk and its id/fingerprint in the store.

    At most two keys stay: the new one and the one it replaces (kept until
    the gate answers on the new one); older unverified ones go."""
    try:
        key = transport_keys.generate(store.venue, user)
    except transport_keys.TransportKeyError as exc:
        _not_enrolled(store.venue, f"no transport key can be issued: {exc}")
    store.issue_key(user, key.key_id, key.fingerprint)
    transport_keys.destroy_others(store.venue, user, keep={key.key_id, previous})
    return key


@router.post("/transport-key")
def issue_transport_key(
    request: Request,
    body: TransportKeyRequest | None = None,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Issue or rotate the caller's transport key at this site.

    # auth: verified
    """
    _require_person(request)
    _site, venue, user_account = _registration_site()
    store = _make_store(request, venue)
    account = store.get_account(caller.user_id)
    key = _current_key(account, venue, caller.user_id, bool(body and body.rotate))
    if key is None:
        key = _new_key(store, caller.user_id, _recorded_key_id(account))
    version, sha = _gate_manifest()
    return {
        "key_id": key.key_id,
        "public_key": key.public_key,
        "site": venue,
        "domain": user_account.signing_domain,
        "from": user_account.from_address,
        "gate": {"version": version, "sha256": sha},
    }


# ── verify ───────────────────────────────────────────────────────────────


def _call_gate(site: Any, login: str, key: Any, verb: str) -> Any:
    call = functools.partial(
        gate_client.run_verb, site, login, key.path, verb, runner=_gate_runner
    )
    try:
        return anyio.from_thread.run(call)
    except ValueError as exc:
        raise_typed(
            PlatformError.hpcaccess("gate_refused", str(exc), gate="bad_request")
        )


def _probe_state(answer: Any) -> str:
    """``refused`` only for the gate's own refusal: its name on exit 4."""
    if isinstance(answer, gate_client.GateRefusal):
        if (
            answer.error == gate_client.GATE_PROBE_REFUSAL
            and answer.exit_code == gate_client.EXIT_REFUSED
        ):
            return "refused"
        if answer.error in (gate_client.UNREACHABLE, gate_client.BAD_ANSWER):
            return "unknown"
    return "bypassed"


def _require_probe_refused(store: Any, user: str, answer: Any) -> None:
    """D-7: the gate itself refuses ``probe``; anything else is recorded."""
    state = _probe_state(answer)
    store.record_probe(user, state)
    if state == "refused":
        return
    name = getattr(answer, "error", "answered")
    raise_typed(
        PlatformError.hpcaccess(
            "gate_refused",
            f"the probe was not refused by the gate ({name}); a site ForceCommand or "
            "a missing authorized_keys line answers instead",
            site=store.venue,
            gate=name,
            probe=state,
        )
    )


def _require_hello(answer: Any, login: str, venue: str) -> dict[str, Any]:
    if isinstance(answer, gate_client.GateRefusal):
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                f"hello refused: {answer.detail}",
                site=venue,
                gate=answer.error,
            )
        )
    data = answer.data
    if data.get("login") != login:
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                f"the gate runs as {data.get('login')!r}, not {login!r}",
                site=venue,
                gate="login_mismatch",
            )
        )
    if data.get("site") != venue:
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                f"the gate is configured for site {data.get('site')!r}, not {venue!r}",
                site=venue,
                gate="wrong_site",
            )
        )
    return dict(data)


def _openssh_version(text: Any) -> tuple[int, int] | None:
    found = _OPENSSH.search(str(text or ""))
    return (int(found.group(1)), int(found.group(2))) if found else None


def _has_sk_signer(hello: dict[str, Any]) -> bool:
    signers = hello.get("signers") if isinstance(hello.get("signers"), list) else []
    return any(
        isinstance(s, dict) and str(s.get("type", "")).startswith("sk-") for s in signers
    )


def _require_site_openssh(hello: dict[str, Any], venue: str) -> None:
    """B1.3 floor on the site's OpenSSH, the one that verifies signatures."""
    floor = OPENSSH_SK_FLOOR if _has_sk_signer(hello) else OPENSSH_FLOOR
    found = _openssh_version(hello.get("openssh"))
    if found is not None and found >= floor:
        return
    raise_typed(
        PlatformError.hpcaccess(
            "gate_version",
            f"the site's OpenSSH ({_clip(hello.get('openssh')) or 'unknown'}) is older than "
            f"{floor[0]}.{floor[1]}: the gate cannot verify your signatures there",
            site=venue,
            openssh=_clip(hello.get("openssh")),
            min_openssh=f"{floor[0]}.{floor[1]}",
        )
    )


def _version_number(value: Any) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0


def _require_gate_release(hello: dict[str, Any], user_account: Any, venue: str) -> None:
    """The installed gate is the one this server ships, and new enough."""
    _version, sha = _gate_manifest()
    installed = _version_number(hello.get("gate_version"))
    if hello.get("gate_sha256") == sha and installed >= user_account.gate_min_version:
        return
    raise_typed(
        PlatformError.hpcaccess(
            "gate_version",
            "the installed gate is not this server's release: run 'r4 hpc add' again",
            site=venue,
            installed_sha256=str(hello.get("gate_sha256") or ""),
            installed_version=str(hello.get("gate_version") or ""),
            expected_sha256=sha,
            min_version=user_account.gate_min_version,
        )
    )


def _require_uv_policy(hello: dict[str, Any], user_account: Any, venue: str) -> None:
    """A site that requires a PIN accepts only a gate that requires one, as
    the gate itself reports it from ``gate.json`` (never the client's claim)."""
    enforced = hello.get("uv_required") is True and hello.get("allow_non_sk") is False
    if not user_account.uv_required or enforced:
        return
    raise_typed(
        PlatformError.hpcaccess(
            "gate_refused",
            "this site requires user verification (a FIDO key with a PIN) on every signature",
            site=venue,
            gate="verification_required",
        )
    )


def _clip(value: Any) -> str:
    return str(value if value is not None else "")[:_FIELD_CAP]


def _assocs(raw: Any) -> list[dict[str, str]]:
    rows = raw if isinstance(raw, list) else []
    return [
        {key: _clip(row.get(key)) for key in ASSOC_KEYS}
        for row in rows[:_ASSOC_CAP]
        if isinstance(row, dict)
    ]


def _verified_fields(body: VerifyRequest, hello: dict[str, Any]) -> dict[str, Any]:
    """What the gate reported; the client's ``signers`` are display only and
    ``uv_required`` is the gate's own (``gate.json``), not the request's."""
    return {
        "site_login": body.login,
        "verified_login": _clip(hello.get("login")),
        "verified_assocs": _assocs(hello.get("assocs")),
        "gate_version": _clip(hello.get("gate_version")),
        "gate_sha256": _clip(hello.get("gate_sha256")),
        "signer_fingerprints": [_clip(s) for s in body.signers],
        "uv_required": hello.get("uv_required") is True,
    }


def _verify_key(store: Any, user: str) -> Any:
    """The account's current transport key file, or not_enrolled."""
    key_id = _recorded_key_id(store.get_account(user))
    if not key_id:
        _not_enrolled(store.venue, "no transport key here: POST transport-key first")
    key = _load_key(store.venue, user, key_id)
    if key is None:
        _not_enrolled(
            store.venue, "the transport key is gone: POST transport-key again"
        )
    return key


@contextlib.contextmanager
def _verify_slot(venue: str, user: str) -> Iterator[None]:
    """One verify per person: a second one at once is refused, not queued."""
    with _VERIFY_GUARD:
        lock = _VERIFY_LOCKS.setdefault((venue, user), threading.Lock())
    if not lock.acquire(blocking=False):
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                "a verify of this account is already running: retry when it ends",
                site=venue,
                gate="busy",
            )
        )
    try:
        yield
    finally:
        lock.release()


def _checked_hello(
    store: Any, user: str, site: Any, body: VerifyRequest, key: Any, user_account: Any
) -> dict[str, Any]:
    """``hello`` and every check on it; any failure suspends an active
    account (D-7) before the refusal goes out."""
    try:
        hello = _require_hello(
            _call_gate(site, body.login, key, "hello"), body.login, store.venue
        )
        _require_gate_release(hello, user_account, store.venue)
        _require_site_openssh(hello, store.venue)
        _require_uv_policy(hello, user_account, store.venue)
    except Exception:
        store.suspend_account(user)
        raise
    return hello


def _drop_replaced_keys(store: Any, user: str, verified: str) -> None:
    """Keys the verified one replaced go; the account's current key (a
    rotation that landed meanwhile) and anything newer stay."""
    current = _recorded_key_id(store.get_account(user))
    transport_keys.destroy_others(store.venue, user, keep={verified, current})


def _suspend_vanished_links(
    store: Any, user: str, assocs: list[dict[str, str]]
) -> None:
    """D-7: a link whose allocation the site no longer lists is suspended."""
    listed = {row.get("account") for row in assocs}
    for link in store.list_links(user):
        if link["status"] == "active" and link["slurm_account"] not in listed:
            store.suspend_links(user, "allocation_gone", project_id=link["project_id"])


@router.post("/verify")
def verify_account(
    body: VerifyRequest,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Reach the caller's gate; activate the account on the site's answer.

    # auth: verified
    """
    _require_person(request)
    site, venue, user_account = _registration_site()
    store = _make_store(request, venue)
    with _verify_slot(venue, caller.user_id):
        key = _verify_key(store, caller.user_id)
        _require_probe_refused(
            store, caller.user_id, _call_gate(site, body.login, key, "probe")
        )
        hello = _checked_hello(store, caller.user_id, site, body, key, user_account)
        account = store.record_verified(
            caller.user_id, key.key_id, _verified_fields(body, hello)
        )
        if account is None:
            _not_enrolled(venue, "the transport key changed during verify: verify again")
        _drop_replaced_keys(store, caller.user_id, key.key_id)
        _suspend_vanished_links(store, caller.user_id, account["verified_assocs"])
    return {"account": account}


# ── read, links, revoke ──────────────────────────────────────────────────


@router.get("")
def get_account(
    request: Request, caller: CallerIdentity = Depends(get_verified_caller)
) -> dict[str, Any]:
    """The caller's account and links here (public information only).

    # auth: verified
    """
    _site, venue = _venue_site()
    store = _make_store(request, venue)
    return {
        "account": store.get_account(caller.user_id),
        "links": store.list_links(caller.user_id),
    }


def _require_live_member(request: Request, user: str, project_id: str) -> None:
    """ADR-097 D-1: the caller is a live member of ``project_id``."""
    projects = _live_memberships(request, user)
    if projects is None:
        raise_typed(
            PlatformError.projects(
                "unavailable", "live project memberships are unreadable"
            )
        )
    if project_id not in projects:
        raise_typed(
            PlatformError.projects(
                "forbidden",
                f"not a live member of project '{project_id}'",
                project_id=project_id,
            )
        )


def _active_account(store: Any, user: str) -> dict[str, Any]:
    account = store.get_account(user)
    if account is None or account["status"] != "active":
        _not_enrolled(store.venue, "no verified account here: run 'r4 hpc add' first")
    return dict(account or {})


@router.post("/links")
def link_project(
    body: LinkRequest,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Link one of the caller's projects to an allocation the site lists.

    # auth: verified
    """
    _require_person(request)
    _site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    account = _active_account(store, caller.user_id)
    _require_live_member(request, caller.user_id, body.project_id)
    listed = {row.get("account") for row in account["verified_assocs"]}
    if body.slurm_account not in listed:
        raise_typed(
            PlatformError.hpcaccess(
                "allocation_not_at_site",
                f"the site lists no allocation '{body.slurm_account}' for your login",
                site=venue,
                slurm_account=body.slurm_account,
            )
        )
    return {
        "link": store.upsert_link(caller.user_id, body.project_id, body.slurm_account)
    }


@router.delete("/links/{project_id}")
def unlink_project(
    project_id: str,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Remove the caller's own link for ``project_id`` at this site.

    # auth: verified
    """
    _require_person(request)
    _site, venue = _venue_site()
    if not _make_store(request, venue).delete_link(caller.user_id, project_id):
        raise_typed(
            PlatformError.projects(
                "not_found", f"no link for project '{project_id}' here"
            )
        )
    return {"status": "deleted"}


@router.delete("")
def revoke_my_account(
    request: Request, caller: CallerIdentity = Depends(get_verified_caller)
) -> dict[str, Any]:
    """Revoke the caller's account here: the transport key is destroyed.

    # auth: verified
    """
    _require_person(request)
    _site, venue = _venue_site()
    store = _make_store(request, venue)
    if store.get_account(caller.user_id) is None:
        _not_enrolled(venue, "no account here")
    revoke_account(store, caller.user_id)
    return {"status": "revoked"}
