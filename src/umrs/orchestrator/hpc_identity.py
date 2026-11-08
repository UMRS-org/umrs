"""One resolver for the HPC identity of every lane that reaches a cluster.

ADR-112 D-1/D-2 (wave A, 2026-11-08). Each site declares
``scheduler.hpc_access``; this module turns that declaration, the lane and
the verified caller into the identity the lane runs under, or refuses with a
typed ``hpcaccess.*`` reason. **No lane ever retries under another identity**
(invariant I-1): a refusal is final.

* ``Lane.PLATFORM`` (endpoints and serving, the service relay, tunnels,
  janitor, reaper, readiness and health) always resolves to the site's
  platform login (``extra.ssh_user``) in mode ``service_account``. Those job
  environments carry operator secrets, so a platform lane never resolves to a
  user's identity (I-2) and never refuses.
* ``Lane.USER_WORK`` (staged steps, ``hpc_script``, cluster loops) resolves
  per person and project (I-3). On a ``service_account`` site whose
  ``consent_ref`` is empty, user work runs only for admins and for the
  projects in ``allowed_projects``.

**Enrolment precedence (wave B2, plan §5 B2.0 §10.1).** Where the site allows
``user_account`` and the person has a link for the verified project, the
person's own account decides: an active account with an active link runs
as that login and Slurm account; an inactive account is ``not_enrolled``;
an inactive link is ``allocation_not_linked``. Otherwise ``default_mode``
decides. The enrolment is read before anything runs; it is never a
fallback, and a refusal is never retried under another mode (I-1).

``project_id`` must be the **verified** project; callers never pass the
claimed one. ``None``, ``""`` and ``"default"`` (no verified project: the
default bucket) match a list only when it names ``default`` explicitly
(maintainer decision 2026-11-09: users who work in ``default`` keep the
user-work lanes on sites that list it).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

SERVICE_ACCOUNT = "service_account"
USER_ACCOUNT = "user_account"
REFUSAL_REASONS = (
    "mode_undeclared",
    "project_not_allowed",
    "not_enrolled",
    # Wave B2 (plan §5 B2.0 §10.1).
    "allocation_not_linked",
    "lane_not_gated",
)
_ACTIVE = "active"
_UNVERIFIED_PROJECTS = frozenset({"", "default"})


class Lane(str, Enum):
    """The two lane classes of ADR-112 D-2."""

    PLATFORM = "platform"
    USER_WORK = "user_work"


@dataclass(frozen=True)
class HpcIdentity:
    """The identity one lane attempt runs under (recorded per attempt, I-10)."""

    site: str
    lane: Lane
    mode: str
    login: str
    slurm_account: str | None
    personal: bool
    declared: bool
    #: The UMRS user a ``user_account`` identity belongs to (its gate's
    #: principal); "" for the platform login.
    principal: str = ""

    def to_metrics(self) -> dict[str, Any]:
        """The attempt-metrics record (``attempt.metrics["hpc_identity"]``)."""
        return {
            "site": self.site,
            "lane": self.lane.value,
            "access_mode": self.mode,
            "site_login": self.login,
            "slurm_account": self.slurm_account,
            "personal": self.personal,
            "declared": self.declared,
        }


@dataclass(frozen=True)
class Enrolment:
    """A person's site account and their link for one project at this venue
    (B1 store views; ``None`` where there is none). Looked up by the caller
    (``shared.hpc_user_work_access``) before resolving."""

    account: dict[str, Any] | None = None
    link: dict[str, Any] | None = None


class HpcAccessRefused(Exception):
    """A lane cannot be served under its resolved identity (never retried)."""

    def __init__(self, reason: str, site: str, detail: str) -> None:
        if reason not in REFUSAL_REASONS:
            raise ValueError(f"unknown hpcaccess refusal reason: {reason!r}")
        super().__init__(f"hpcaccess.{reason}: {detail}")
        self.reason = reason
        self.site = site
        self.detail = detail

    @property
    def code(self) -> str:
        """The typed error code (r4-contracts ``HPCACCESS_ERRORS``)."""
        return f"hpcaccess.{self.reason}"


def refusal_message(exc: HpcAccessRefused) -> str:
    """One user-facing line: the detail, then the typed code."""
    return f"{exc.detail} ({exc.code})"


def platform_error(exc: HpcAccessRefused) -> Any:
    """The r4-contracts ``PlatformError`` for a refusal (HTTP routes)."""
    from r4_contracts.errors import PlatformError

    return PlatformError.hpcaccess(exc.reason, refusal_message(exc), site=exc.site)


def _site_name(site: Any) -> str:
    return str(getattr(site, "name", "") or "")


def _platform_login(site: Any) -> str:
    extra = getattr(site, "extra", None)
    value = extra.get("ssh_user") if isinstance(extra, dict) else None
    return str(value or "")


def _hpc_access(site: Any) -> Any:
    return getattr(getattr(site, "scheduler", None), "hpc_access", None)


def _personal(hpc: Any) -> bool:
    service = getattr(hpc, "service_account", None)
    return bool(getattr(service, "personal", False))


def _service_identity(
    site: Any, lane: Lane, slurm_account: str | None, *, declared: bool
) -> HpcIdentity:
    hpc = _hpc_access(site)
    return HpcIdentity(
        site=_site_name(site),
        lane=lane,
        mode=SERVICE_ACCOUNT,
        login=_platform_login(site),
        slurm_account=slurm_account or None,
        personal=_personal(hpc),
        declared=declared,
    )


def _project_listed(project_id: str | None, listed: list[str]) -> bool:
    """A verified project in the list; a run without one (the default bucket)
    only when the list names ``default`` explicitly."""
    project = str(project_id or "").strip()
    if project in _UNVERIFIED_PROJECTS:
        return "default" in listed
    return project in listed


def _project_verified(project_id: str | None) -> bool:
    return str(project_id or "").strip() not in _UNVERIFIED_PROJECTS


def _service_account_allows(service: Any, project_id: str | None) -> bool:
    """The G-8 rule: a listed project, or consent with no list at all."""
    listed = list(getattr(service, "allowed_projects", None) or [])
    if listed:
        return _project_listed(project_id, listed)
    consented = bool(getattr(service, "consent_ref", ""))
    return consented and _project_verified(project_id)


def _not_allowed_detail(site: str, service: Any, project_id: str | None) -> str:
    project = str(project_id or "").strip() or "(none)"
    login = (
        "a personal account of the operator"
        if getattr(service, "personal", False)
        else "the site's service account"
    )
    if not getattr(service, "allowed_projects", None):
        return (
            f"Site {site} runs user work under one login, {login}; no project "
            f"is listed, so only administrators may use it until the operator "
            f"consents (project {project})."
        )
    return (
        f"Site {site} runs user work under one login, {login}; only the listed "
        f"projects may use it until the operator consents, and project "
        f"{project} is not listed."
    )


def _resolve_service_account(
    site: Any, project_id: str | None, is_admin: bool, slurm_account: str | None
) -> HpcIdentity:
    service = getattr(_hpc_access(site), "service_account", None)
    if is_admin or _service_account_allows(service, project_id):
        return _service_identity(site, Lane.USER_WORK, slurm_account, declared=True)
    name = _site_name(site)
    raise HpcAccessRefused(
        "project_not_allowed", name, _not_allowed_detail(name, service, project_id)
    )


def _resolve_undeclared(
    site: Any, is_admin: bool, slurm_account: str | None
) -> HpcIdentity:
    if is_admin:
        return _service_identity(site, Lane.USER_WORK, slurm_account, declared=False)
    name = _site_name(site)
    raise HpcAccessRefused(
        "mode_undeclared",
        name,
        f"Site {name} declares no HPC access mode (scheduler.hpc_access); "
        f"only administrators may run user work here.",
    )


def _user_account_allowed(hpc: Any) -> bool:
    return USER_ACCOUNT in list(getattr(hpc, "modes_allowed", None) or [])


def _status(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("status") or "")


def _not_enrolled(name: str, why: str) -> HpcAccessRefused:
    return HpcAccessRefused(
        "not_enrolled",
        name,
        f"Site {name} runs this work under your own account there, and {why}: "
        f"run 'r4 hpc add {name}' (or 'r4 hpc verify {name}').",
    )


def _not_linked(name: str, project_id: str | None, why: str) -> HpcAccessRefused:
    project = str(project_id or "").strip() or "(none)"
    return HpcAccessRefused(
        "allocation_not_linked",
        name,
        f"Site {name} runs this work under your own account there, and {why} "
        f"for project {project}: run 'r4 hpc link {name} {project} <allocation>'.",
    )


def _enrolled_identity(
    site: Any, enrolment: Enrolment, project_id: str | None, user_id: str | None
) -> HpcIdentity:
    """The person's own login and allocation, or the refusal naming why not."""
    name = _site_name(site)
    if _status(enrolment.account) != _ACTIVE:
        raise _not_enrolled(name, "your account there is not active")
    if _status(enrolment.link) != _ACTIVE:
        raise _not_linked(name, project_id, "your link is suspended")
    return _user_account_identity(name, enrolment, user_id)


def _user_account_identity(
    name: str, enrolment: Enrolment, user_id: str | None
) -> HpcIdentity:
    account, link = enrolment.account or {}, enrolment.link or {}
    return HpcIdentity(
        site=name,
        lane=Lane.USER_WORK,
        mode=USER_ACCOUNT,
        login=str(account.get("verified_login") or ""),
        slurm_account=str(link.get("slurm_account") or "") or None,
        personal=False,
        declared=True,
        principal=str(user_id or account.get("username") or ""),
    )


def _resolve_default_user_account(
    site: Any, enrolment: Enrolment | None, project_id: str | None, user_id: str | None
) -> HpcIdentity:
    """``default_mode: user_account`` for a person without a link here."""
    name = _site_name(site)
    if enrolment is None or enrolment.account is None:
        raise _not_enrolled(name, "you have no account registered there")
    if _status(enrolment.account) != _ACTIVE:
        raise _not_enrolled(name, "your account there is not active")
    raise _not_linked(name, project_id, "no allocation is linked")


def _resolve_user_work(
    site: Any,
    project_id: str | None,
    is_admin: bool,
    slurm_account: str | None,
    enrolment: Enrolment | None = None,
    user_id: str | None = None,
) -> HpcIdentity:
    hpc = _hpc_access(site)
    if hpc is None:
        return _resolve_undeclared(site, is_admin, slurm_account)
    if _user_account_allowed(hpc) and enrolment is not None and enrolment.link:
        return _enrolled_identity(site, enrolment, project_id, user_id)
    mode = str(getattr(hpc, "default_mode", "") or "")
    if mode == SERVICE_ACCOUNT:
        return _resolve_service_account(site, project_id, is_admin, slurm_account)
    if mode == USER_ACCOUNT:
        return _resolve_default_user_account(site, enrolment, project_id, user_id)
    return _resolve_undeclared(site, is_admin, slurm_account)


def resolve_hpc_identity(
    site: Any,
    *,
    lane: Lane | str,
    user_id: str | None,
    project_id: str | None,
    is_admin: bool,
    slurm_account: str | None = None,
    enrolment: Enrolment | None = None,
) -> HpcIdentity:
    """The identity *lane* runs under at *site*, or :class:`HpcAccessRefused`.

    ``project_id`` is the verified project. ``enrolment`` is the person's
    site account and their link for that project (wave B2); a PLATFORM lane
    ignores it. ``slurm_account`` is passed through to a ``service_account``
    record unchanged; a ``user_account`` identity charges the link's own.
    """
    if Lane(lane) is Lane.PLATFORM:
        hpc = _hpc_access(site)
        return _service_identity(
            site, Lane.PLATFORM, slurm_account, declared=hpc is not None
        )
    return _resolve_user_work(
        site, project_id, bool(is_admin), slurm_account, enrolment, user_id
    )


__all__ = [
    "REFUSAL_REASONS",
    "SERVICE_ACCOUNT",
    "USER_ACCOUNT",
    "Enrolment",
    "HpcAccessRefused",
    "HpcIdentity",
    "Lane",
    "platform_error",
    "refusal_message",
    "resolve_hpc_identity",
]
