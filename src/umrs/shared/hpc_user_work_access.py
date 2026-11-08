"""This server's USER_WORK identity checks (ADR-112 D-2, wave A §A.3).

:mod:`umrs.orchestrator.hpc_identity` is the pure resolver. This module binds
it to this server: the site config it reads and the user record that says
who is an administrator. It is a foundation module (ADR-053), so the durable
job backends in ``umrs.jobs`` (``hpc_script``, cluster loops) and the
``umrs.api`` / ``umrs.server`` admission checks share one binding.

**One admin source.** Admission (HTTP job create, loop start, workflow
trigger, ZMQ run, plan quote) and the later checks (durable submit, the
workflow runner) all read the administrator bit from the user record's role
(plan §3 A.3.2), never from a token claim alone. A caller admitted at the
door is therefore not refused later for its admin status, and the reverse
holds too (review 2026-11-08).

Callers always pass the **verified** project. An unreadable site config is an
undeclared site and an unreadable user record is a non-admin, so every
failure here refuses (fail closed); nothing retries under another identity.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.orchestrator.hpc_identity import (
    HpcAccessRefused,
    HpcIdentity,
    Lane,
    resolve_hpc_identity,
)

logger = logging.getLogger(__name__)


def current_site() -> Any:
    """This server's site config, or None (an undeclared site)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        return load_site_config()
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("site config unreadable; user work is refused", exc_info=True)
        return None


def _user_record(user_id: str | None) -> Any:
    """The enabled user record for *user_id*, or None (absent, unreadable)."""
    if not user_id:
        return None
    try:
        from umrs.admin.user_store import UserStore

        record = UserStore().resolve_user(user_id)
    except Exception:
        logger.debug("user record unreadable for %s", user_id, exc_info=True)
        return None
    return None if getattr(record, "disabled", False) else record


def user_is_admin(user_id: str | None) -> bool:
    """Admin by the user record's role here; absent or unreadable is not."""
    return getattr(_user_record(user_id), "role", "") == "admin"


def user_in_project(user_id: str | None, project_id: str | None) -> bool:
    """Whether the user record lists *project_id* (absent or unreadable: no)."""
    project = str(project_id or "").strip()
    projects = getattr(_user_record(user_id), "projects", None) or []
    return bool(project) and project in projects


def site_slurm_account() -> str | None:
    """The site's ``scheduler.account`` (what a loop wrapper job charges)."""
    scheduler = getattr(current_site(), "scheduler", None)
    return str(getattr(scheduler, "account", "") or "") or None


def resolve_user_work(
    *,
    user_id: str | None,
    project_id: str | None,
    slurm_account: str | None = None,
) -> HpcIdentity:
    """The USER_WORK identity at this site, or :class:`HpcAccessRefused`.

    ``is_admin`` comes from the user record (the one admin source)."""
    return resolve_hpc_identity(
        current_site(),
        lane=Lane.USER_WORK,
        user_id=user_id,
        project_id=project_id,
        is_admin=user_is_admin(user_id),
        slurm_account=slurm_account,
    )


def resolve_job_user_work(job: Any, *, slurm_account: str | None = None) -> HpcIdentity:
    """The USER_WORK identity of a durable *job* at submit (ADR-112 A.3).

    From the job's own user and project (recorded at creation from the
    verified caller). *slurm_account* is the account the submit charges, for
    the attempt record (I-10). Raises :class:`HpcAccessRefused`.
    """
    return resolve_user_work(
        user_id=getattr(job, "user_id", None),
        project_id=getattr(job, "project_id", None),
        slurm_account=slurm_account,
    )


def is_user_work_job(job_type: Any, spec: Any) -> bool:
    """Whether a durable job of *job_type* with *spec* runs on the site's
    login (ADR-112 D-2 USER_WORK): ``hpc_script``, and a
    ``research_loop_job`` on a ``cluster:<site>`` venue."""
    kind = str(getattr(job_type, "value", job_type) or "")
    if kind == "hpc_script":
        return True
    venue = str((spec or {}).get("venue") or "") if isinstance(spec, dict) else ""
    return kind == "research_loop_job" and venue.strip().startswith("cluster:")


def user_work_refusal(
    *, user_id: str | None, project_id: str | None
) -> HpcAccessRefused | None:
    """The refusal for this caller's user work here, or None when allowed."""
    try:
        resolve_user_work(user_id=user_id, project_id=project_id)
    except HpcAccessRefused as exc:
        return exc
    return None


__all__ = [
    "current_site",
    "is_user_work_job",
    "resolve_job_user_work",
    "resolve_user_work",
    "site_slurm_account",
    "user_in_project",
    "user_is_admin",
    "user_work_refusal",
]
