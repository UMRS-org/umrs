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

Wave B2 (plan §5 B2.0 §2 and §10.3) adds three routes and one check:

- ``GET /site-profile`` publishes the gate site profile (``{profile, sha256,
  bytes_b64}``, read only) that r4 shows the person and installs as the
  gate's ``site.json``;
- ``GET /gate`` runs a live ``hello`` with no state change and says whether
  the installed gate and site profile are the ones this server ships;
- ``POST /control`` relays a signed ``revoke`` / ``rotate`` envelope to the
  person's gate (the producer B1 deferred), the person's own act;
- ``POST /verify`` also refuses ``hpcaccess.gate_version``
  (``site_profile_missing``) when the gate has no site profile, and answers
  ``site_profile_current``: an account whose installed profile differs is
  still activated, and a proposal then refuses until ``r4 hpc profile``.

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
from umrs.hpc_accounts.grants import end_account_grants
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


class GrantWorkflowRef(BaseModel):
    name: str = Field(..., min_length=1, max_length=_FIELD_CAP)
    version: str | None = Field(default=None, max_length=64)


class GrantProposalRequest(BaseModel):
    """``POST /grant-proposal`` (plan §6 C/D.0 C4, X6)."""

    project_id: str = Field(..., min_length=1, max_length=_FIELD_CAP)
    workflows: list[GrantWorkflowRef] = Field(..., min_length=1, max_length=16)


class SignedEnvelope(BaseModel):
    """A signed grant, or a signed ``grant-cancel`` control (base64 + SSHSIG)."""

    payload: str = Field(..., min_length=1, max_length=3 * 1024 * 1024)
    signature: str = Field(..., min_length=1, max_length=16 * 1024)


class ControlRequest(BaseModel):
    """A signed control envelope for the person's gate (B1.5a)."""

    op: str = Field(..., pattern=r"^(revoke|rotate)$")
    payload: str = Field(..., min_length=1, max_length=64 * 1024)
    signature: str = Field(..., min_length=1, max_length=16 * 1024)


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


def _call_gate(
    site: Any, login: str, key: Any, verb: str, body: dict[str, Any] | None = None
) -> Any:
    call = functools.partial(
        gate_client.run_verb, site, login, key.path, verb, body, runner=_gate_runner
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
    end_account_grants(store, user)
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


def _require_site_profile(hello: dict[str, Any], venue: str) -> None:
    """B2.0 §2: a gate without its ``site.json`` cannot run gated steps."""
    if hello.get("site_profile_sha256"):
        return
    raise_typed(
        PlatformError.hpcaccess(
            "gate_version",
            "site_profile_missing: the gate has no site profile installed: run 'r4 hpc profile' "
            "(or 'r4 hpc add' again)",
            site=venue,
            reason="site_profile_missing",
        )
    )


def _published_profile_sha(site: Any) -> str:
    """The sha256 of the profile this server publishes ("" when none)."""
    from umrs.hpc_accounts.site_profile import SiteProfileUnavailable, published

    try:
        return published(site)[2]
    except SiteProfileUnavailable:
        return ""


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
        _require_site_profile(hello, store.venue)
    except Exception:
        if store.suspend_account(user):
            end_account_grants(store, user)
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
    current = hello.get("site_profile_sha256") == _published_profile_sha(site)
    return {"account": account, "site_profile_current": current}


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


# ── wave B2: the published site profile, a live hello, signed controls ──


@router.get("/site-profile")
def get_site_profile(
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """The gate site profile this server publishes (read only).

    # auth: verified
    """
    import base64

    from umrs.hpc_accounts.site_profile import SiteProfileUnavailable, published

    site, venue, _user_account = _registration_site()
    try:
        profile, data, sha = published(site)
    except SiteProfileUnavailable as exc:
        raise_typed(
            PlatformError.hpcaccess(
                "gate_version", str(exc), site=venue, reason="site_profile_unavailable"
            )
        )
    return {
        "profile": profile,
        "sha256": sha,
        "bytes_b64": base64.b64encode(data).decode("ascii"),
    }


def _active_key(store: Any, user: str) -> tuple[dict[str, Any], Any]:
    """The caller's active account and its current transport key."""
    account = _active_account(store, user)
    key = _load_key(store.venue, user, str(account.get("transport_key_id") or ""))
    if key is None:
        _not_enrolled(store.venue, "the transport key is gone: run 'r4 hpc rotate'")
    return account, key


@router.get("/gate")
def get_gate(
    request: Request, caller: CallerIdentity = Depends(get_verified_caller)
) -> dict[str, Any]:
    """A live ``hello`` of the caller's gate; nothing is stored.

    # auth: verified
    """
    site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    account, key = _active_key(store, caller.user_id)
    login = str(account.get("verified_login") or "")
    hello = _require_hello(_call_gate(site, login, key, "hello"), login, venue)
    _version, sha = _gate_manifest()
    profile_sha = str(hello.get("site_profile_sha256") or "")
    return {
        "gate_version": str(hello.get("gate_version") or ""),
        "gate_sha256": str(hello.get("gate_sha256") or ""),
        "site_profile_sha256": profile_sha,
        "log_head": hello.get("log_head"),
        "current": {
            "gate": hello.get("gate_sha256") == sha,
            "site_profile": bool(profile_sha) and profile_sha == _published_profile_sha(site),
        },
    }


@router.post("/control")
def relay_control(
    body: ControlRequest,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Relay the caller's signed ``revoke`` / ``rotate`` to their gate.

    # auth: verified
    """
    _require_person(request)
    site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    account, key = _active_key(store, caller.user_id)
    login = str(account.get("verified_login") or "")
    answer = _call_gate(
        site, login, key, body.op, {"payload": body.payload, "signature": body.signature}
    )
    if isinstance(answer, gate_client.GateRefusal):
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                f"{answer.error}: {answer.detail}".rstrip(": "),
                site=venue,
                gate=answer.error,
            )
        )
    return {"result": answer.data}


@router.delete("")
def revoke_my_account(
    request: Request, caller: CallerIdentity = Depends(get_verified_caller)
) -> dict[str, Any]:
    """Revoke the caller's account here: the transport key is destroyed.
    The person's live gated jobs here are cancelled through their gate
    first, while the key still reaches it (as the account cascade does,
    plan §5 B2.0 §10.6); ``gated_jobs_cancelled`` counts them.

    # auth: verified
    """
    _require_person(request)
    _site, venue = _venue_site()
    store = _make_store(request, venue)
    if store.get_account(caller.user_id) is None:
        _not_enrolled(venue, "no account here")
    cancelled = _cancel_my_gated_jobs(venue, caller.user_id)
    revoke_account(store, caller.user_id)
    return {"status": "revoked", "gated_jobs_cancelled": cancelled}


def _cancel_my_gated_jobs(venue: str, user_id: str) -> int:
    """Best-effort (fixer finding 2026-11-08): without it a self-revoke left
    running jobs on the person's allocation with nothing to collect them;
    an unreachable gate cancels none (they still end by ``--time``)."""
    try:
        from umrs.server.gated_step_lane import cancel_person_jobs
        from umrs.server.workflow_execution_registry import get_execution_registry

        records = list(get_execution_registry().list(user_id=user_id))
        return cancel_person_jobs(venue, user_id, records)
    except Exception:
        logger.warning("revoke: gated jobs of %s not cancelled", user_id, exc_info=True)
        return 0


# ── wave C: grants (plan §6 C/D.0 C4) ────────────────────────────────────


def _grant_store(store: Any) -> Any:
    from umrs.hpc_accounts.grants import grant_store_for

    return grant_store_for(store)


def _usage_ledger() -> Any:
    from umrs.hpc_accounts.usage_ledger import LedgerUnavailable, get_usage_ledger

    try:
        return get_usage_ledger()
    except LedgerUnavailable as exc:
        raise_typed(PlatformError.hpcaccess("ledger_unavailable", str(exc)))


def _grant_invalid(exc: Any) -> NoReturn:
    raise_typed(
        PlatformError.hpcaccess("grant_invalid", str(exc), reason=exc.reason), cause=exc
    )


def _grant_typed(exc: Exception) -> NoReturn:
    """The typed refusal of a grant check (C4 error codes)."""
    from umrs.hpc_accounts.grants import GrantGateStale, GrantInvalid
    from umrs.hpc_accounts.usage_ledger import BudgetExceeded

    if isinstance(exc, GrantInvalid):
        _grant_invalid(exc)
    if isinstance(exc, GrantGateStale):
        raise_typed(
            PlatformError.hpcaccess("gate_version", str(exc), reason=exc.reason), cause=exc
        )
    if isinstance(exc, BudgetExceeded):
        raise_typed(
            PlatformError.hpcaccess("budget_exceeded", str(exc), project_id=exc.project), cause=exc
        )
    raise exc


def _call_grant_verb(site: Any, login: str, key: Any, verb: str, body: dict[str, Any]) -> Any:
    """``grant-check`` / ``grant-cancel`` with the verb's own timeout (the
    gate settles with one ``sacct`` before it answers)."""
    call = functools.partial(
        gate_client.run_verb, site, login, key.path, verb, body,
        runner=_gate_runner, timeout=gate_client.verb_timeout(verb),
    )
    return anyio.from_thread.run(call)


def _gate_refused(answer: Any, venue: str) -> NoReturn:
    raise_typed(
        PlatformError.hpcaccess(
            "gate_refused", f"{answer.error}: {answer.detail}".rstrip(": "),
            site=venue, gate=answer.error,
        )
    )


def _grant_context(request: Request, store: Any, user: str, site: Any) -> Any:
    from umrs.hpc_accounts.grant_service import build_context

    return build_context(store.venue, user, store, _live_memberships(request, user), site)


def _live_check(site: Any, account: dict[str, Any], key: Any, row: dict[str, Any]) -> Any:
    login = str(account.get("verified_login") or "")
    body = {"payload": row["payload_b64"], "signature": row["signature"]}
    return _call_grant_verb(site, login, key, "grant-check", body)


def _require_live_state(answer: Any, venue: str) -> dict[str, Any]:
    """C4 step 8: a refusal is ``gate_refused``; ``expired`` / ``cancelled``
    are ``grant_invalid`` with that reason."""
    from umrs.hpc_accounts.grants import GrantInvalid

    if isinstance(answer, gate_client.GateRefusal):
        _gate_refused(answer, venue)
    state = str(answer.data.get("state") or "")
    if state in ("expired", "cancelled"):
        _grant_invalid(GrantInvalid(state, f"the gate reports this grant {state}"))
    return dict(answer.data.get("usage") or {})


@router.post("/grant-proposal")
def propose_grant(
    body: GrantProposalRequest,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """What a grant for these workflows would pin (advisory; r4 recomputes).

    # auth: verified
    """
    from umrs.hpc_accounts.grants import GrantInvalid
    from umrs.hpc_accounts.site_profile import SiteProfileUnavailable, published

    site, venue, user_account = _registration_site()
    store = _make_store(request, venue)
    account = _active_account(store, caller.user_id)
    _require_live_member(request, caller.user_id, body.project_id)
    link = _active_link(store, caller.user_id, body.project_id)
    try:
        profile, _data, profile_sha = published(site)
        answer = _proposal_body(body, caller, site, profile)
    except GrantInvalid as exc:
        _grant_invalid(exc)
    except SiteProfileUnavailable as exc:
        raise_typed(
            PlatformError.hpcaccess("gate_version", str(exc), reason="site_profile_unavailable")
        )
    return {
        "site": venue,
        "domain": user_account.signing_domain,
        "login": str(account.get("verified_login") or ""),
        "accounts": [link["slurm_account"]],
        **answer,
        "site_profile_sha256": profile_sha,
        "budget": _usage_ledger().budget(body.project_id),
    }


def _active_link(store: Any, user: str, project_id: str) -> dict[str, Any]:
    from umrs.hpc_accounts.grants import GrantInvalid

    for link in store.list_links(user):
        if link["project_id"] == project_id and link["status"] == "active":
            return dict(link)
    _grant_invalid(
        GrantInvalid("not_linked", f"project {project_id!r} has no active link of yours here")
    )


def _proposal_body(
    body: GrantProposalRequest, caller: CallerIdentity, site: Any, profile: Any
) -> dict[str, Any]:
    from umrs.hpc_accounts.grant_service import proposal_versions, proposal_workflow

    refs = [ref.model_dump() for ref in body.workflows]
    versions = proposal_versions(refs, caller.user_id, body.project_id, is_admin=caller.is_admin)
    return {"workflows": [proposal_workflow(v, site, profile) for v in versions]}


@router.post("/grants")
def store_grant(
    body: SignedEnvelope,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Store the caller's signed grant after the C4 checks and a live
    ``grant-check`` at their gate.

    # auth: verified
    """
    from umrs.hpc_accounts.grants import (
        GrantGateStale,
        GrantInvalid,
        check_grant,
        decode_grant,
        grant_view,
        row_from_grant,
    )
    from umrs.hpc_accounts.usage_ledger import BudgetExceeded

    _require_person(request)
    site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    account, key = _active_key(store, caller.user_id)
    grants, ctx = _grant_store(store), _grant_context(request, store, caller.user_id, site)
    if ctx.memberships is None:
        raise_typed(
            PlatformError.projects("unavailable", "live project memberships are unreadable")
        )
    try:
        grant = decode_grant(body.payload, body.signature)
        check_grant(grant, ctx, grants, _usage_ledger())
    except (GrantInvalid, GrantGateStale, BudgetExceeded) as exc:
        _grant_typed(exc)
    row = row_from_grant(venue, caller.user_id, grant, {})
    usage = _require_live_state(_live_check(site, account, key, row), venue)
    try:
        stored = grants.insert({**row, "usage": usage})
    except GrantInvalid as exc:
        _grant_invalid(exc)
    return {"grant": grant_view(stored, ctx)}


_REFRESH_CAP = 20


def _refresh_usage(grants: Any, rows: list[dict[str, Any]], site: Any, store: Any, user: str) -> None:
    """``refresh=1``: ``grant-check`` for at most 20 active grants; the
    mirror is updated (display only); a gate that cannot answer leaves it."""
    found = _refresh_key(store, user)
    if found is None:
        return
    account, key = found
    for row in [r for r in rows if r["status"] == "active"][:_REFRESH_CAP]:
        answer = _live_check(site, account, key, row)
        if isinstance(answer, gate_client.GateResult):
            grants.set_usage(user, row["grant_id"], dict(answer.data.get("usage") or {}))


def _refresh_key(store: Any, user: str) -> tuple[dict[str, Any], Any] | None:
    """The active account and its transport key, else None (no refresh)."""
    account = store.get_account(user) or {}
    if account.get("status") != "active":
        return None
    key = _load_key(store.venue, user, str(account.get("transport_key_id") or ""))
    return None if key is None else (account, key)


@router.get("/grants")
def list_grants(
    request: Request,
    refresh: int = 0,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """The caller's grants here, with ``covers_now`` and ``stale`` (K11).

    # auth: verified
    """
    from umrs.hpc_accounts.grants import grant_view

    site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    grants = _grant_store(store)
    rows = grants.list(caller.user_id)
    if refresh:
        _refresh_usage(grants, rows, site, store, caller.user_id)
        rows = grants.list(caller.user_id)  # the mirror as stored (usage_at)
    ctx = _grant_context(request, store, caller.user_id, site)
    return {"grants": [grant_view(r, ctx) for r in rows]}


def _cancel_target(grants: Any, user: str, grant_id: str) -> dict[str, Any]:
    from umrs.hpc_accounts.grants import GrantNotFound, check_grant_id

    try:
        row = grants.get(user, check_grant_id(grant_id))
    except GrantNotFound as exc:
        raise_typed(PlatformError.hpcaccess("grant_not_found", str(exc)), cause=exc)
    if row is None:
        raise_typed(
            PlatformError.hpcaccess("grant_not_found", f"no grant {grant_id} of yours here")
        )
    return dict(row)


def _signed_cancel_names(body: SignedEnvelope, grant_id: str) -> None:
    """The signed control names this grant (the gate checks the signature)."""
    import base64
    import json

    try:
        control = json.loads(base64.b64decode(body.payload, validate=True))
    except ValueError:
        control = None
    names = isinstance(control, dict) and control.get("op") == "grant-cancel"
    if not names or control.get("grant_id") != grant_id:
        raise_typed(
            PlatformError.hpcaccess(
                "gate_refused",
                "the signed control is not a grant-cancel of this grant",
                gate="bad_request",
            )
        )


def _unreachable(answer: Any) -> bool:
    """A transport failure (ssh exit 255 or a timeout): not the gate's word."""
    return (
        isinstance(answer, gate_client.GateRefusal)
        and answer.error == gate_client.UNREACHABLE
    )


@router.post("/grants/{grant_id}/cancel")
def cancel_grant(
    grant_id: str,
    body: SignedEnvelope,
    request: Request,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Relay the caller's signed ``grant-cancel`` to their gate, then end
    the grant here. An unreachable gate still ends it here, with
    ``gate_cancel: pending`` and ``relayed: false`` (K8); a gate refusal
    leaves the row unchanged.

    # auth: verified
    """
    from umrs.hpc_accounts.grants import grant_view

    _require_person(request)
    site, venue, _user_account = _registration_site()
    store = _make_store(request, venue)
    grants = _grant_store(store)
    _cancel_target(grants, caller.user_id, grant_id)
    _signed_cancel_names(body, grant_id)
    account, key = _active_key(store, caller.user_id)
    login = str(account.get("verified_login") or "")
    answer = _call_grant_verb(
        site, login, key, "grant-cancel", {"payload": body.payload, "signature": body.signature}
    )
    relayed = not _unreachable(answer)
    if relayed and isinstance(answer, gate_client.GateRefusal):
        _gate_refused(answer, venue)
    row = grants.set_status(
        caller.user_id, grant_id, "cancelled", "cancelled", "recorded" if relayed else "pending"
    )
    ctx = _grant_context(request, store, caller.user_id, site)
    result: dict[str, Any] = {"grant": grant_view(row or {}, ctx), "relayed": relayed}
    if relayed:
        result["result"] = answer.data
    return result
