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
  projects in ``allowed_projects``. ``user_account`` is refused as
  ``not_enrolled`` until wave B lands.

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
REFUSAL_REASONS = ("mode_undeclared", "project_not_allowed", "not_enrolled")
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


def _resolve_user_work(
    site: Any, project_id: str | None, is_admin: bool, slurm_account: str | None
) -> HpcIdentity:
    hpc = _hpc_access(site)
    if hpc is None:
        return _resolve_undeclared(site, is_admin, slurm_account)
    mode = str(getattr(hpc, "default_mode", "") or "")
    if mode == SERVICE_ACCOUNT:
        return _resolve_service_account(site, project_id, is_admin, slurm_account)
    name = _site_name(site)
    if mode == USER_ACCOUNT:
        raise HpcAccessRefused(
            "not_enrolled",
            name,
            f"Site {name} runs user work under each user's own account; "
            f"registering an account is not available yet.",
        )
    return _resolve_undeclared(site, is_admin, slurm_account)


def resolve_hpc_identity(
    site: Any,
    *,
    lane: Lane | str,
    user_id: str | None,
    project_id: str | None,
    is_admin: bool,
    slurm_account: str | None = None,
) -> HpcIdentity:
    """The identity *lane* runs under at *site*, or :class:`HpcAccessRefused`.

    ``project_id`` is the verified project. ``user_id`` is recorded by the
    caller for attribution; wave A resolves per project, wave B per person.
    ``slurm_account`` is passed through to the record unchanged.
    """
    del user_id  # wave B: per-person site accounts
    if Lane(lane) is Lane.PLATFORM:
        hpc = _hpc_access(site)
        return _service_identity(
            site, Lane.PLATFORM, slurm_account, declared=hpc is not None
        )
    return _resolve_user_work(site, project_id, bool(is_admin), slurm_account)


__all__ = [
    "REFUSAL_REASONS",
    "SERVICE_ACCOUNT",
    "USER_ACCOUNT",
    "HpcAccessRefused",
    "HpcIdentity",
    "Lane",
    "platform_error",
    "refusal_message",
    "resolve_hpc_identity",
]
