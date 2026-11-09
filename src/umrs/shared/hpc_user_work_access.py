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

**Enrolment (wave B2, plan §5 B2.0 §10.1).** Where the site allows
``user_account``, the person's site account and their link for the verified
project are read from the B1 tables (this venue only) and handed to the
resolver, which lets them take precedence over ``default_mode``. Unreadable
tables refuse ``not_enrolled``: a person who enrolled never silently runs
under the platform login instead. Only the staged step lane has a gated
path in B2 (``gated=True``); every other user-work lane that resolves to
``user_account`` refuses ``lane_not_gated`` before anything reaches the
cluster.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from umrs.orchestrator.hpc_identity import (
    USER_ACCOUNT,
    Enrolment,
    HpcAccessRefused,
    HpcIdentity,
    Lane,
    resolve_hpc_identity,
)

logger = logging.getLogger(__name__)

#: ``venue -> site account store`` (the B1 ``SiteAccountStore`` API); None
#: builds one over a lazily created shared ``umrs_db`` pool. Tests install
#: an ``InMemorySiteAccountStore`` factory.
_store_factory: Callable[[str], Any] | None = None
_pool_lock = threading.Lock()
_pool: Any = None


def set_site_account_store_factory(factory: Callable[[str], Any] | None) -> None:
    """Install (tests) or forget the site account store factory."""
    global _store_factory
    _store_factory = factory


def _shared_pool() -> Any:
    global _pool
    with _pool_lock:
        if _pool is None:
            from umrs.shared.pg import build_durable_job_db_pool_from_env

            _pool = build_durable_job_db_pool_from_env()
        return _pool


def site_account_store(venue: str) -> Any:
    """This venue's site account store (raises when unreadable)."""
    if _store_factory is not None:
        return _store_factory(venue)
    from umrs.hpc_accounts.store import SiteAccountStore

    return SiteAccountStore(_shared_pool(), venue)


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


def _allows_user_account(site: Any) -> bool:
    hpc = getattr(getattr(site, "scheduler", None), "hpc_access", None)
    return USER_ACCOUNT in list(getattr(hpc, "modes_allowed", None) or [])


def _project_link(store: Any, user_id: str, project_id: str | None) -> Any:
    project = str(project_id or "").strip()
    if not project:
        return None
    links = store.list_links(user_id)
    return next((link for link in links if link.get("project_id") == project), None)


def read_enrolment(
    site: Any, user_id: str | None, project_id: str | None
) -> Enrolment | None:
    """The person's account and link for *project_id* at *site*, or None
    where the site does not allow ``user_account`` (nothing to read).

    Raises :class:`HpcAccessRefused` (``not_enrolled``) when the tables
    cannot be read: the enrolment is decided before anything runs."""
    if site is None or not user_id or not _allows_user_account(site):
        return None
    venue = str(getattr(site, "name", "") or "")
    try:
        store = site_account_store(venue)
        account = store.get_account(user_id)
        link = _project_link(store, user_id, project_id) if account else None
    except Exception as exc:
        logger.warning("site accounts unreadable at %s", venue, exc_info=True)
        raise HpcAccessRefused(
            "not_enrolled",
            venue,
            f"Site {venue}: your HPC account there cannot be read now "
            f"({type(exc).__name__}); nothing ran.",
        ) from exc
    return Enrolment(account=account, link=link)


def site_slurm_account() -> str | None:
    """The site's ``scheduler.account`` (what a loop wrapper job charges)."""
    scheduler = getattr(current_site(), "scheduler", None)
    return str(getattr(scheduler, "account", "") or "") or None


def _lane_not_gated(identity: HpcIdentity) -> HpcAccessRefused:
    return HpcAccessRefused(
        "lane_not_gated",
        identity.site,
        f"Site {identity.site} runs your work under your own account there, "
        f"through the gate; only workflow steps have a gated path so far, so "
        f"this kind of job cannot run there yet (nothing ran).",
    )


def resolve_user_work(
    *,
    user_id: str | None,
    project_id: str | None,
    slurm_account: str | None = None,
    gated: bool = False,
) -> HpcIdentity:
    """The USER_WORK identity at this site, or :class:`HpcAccessRefused`.

    ``is_admin`` comes from the user record (the one admin source). The
    person's enrolment is read first (:func:`read_enrolment`). *gated* says
    the caller is a lane with a gated path (staged steps); any other lane
    that resolves to ``user_account`` refuses ``lane_not_gated`` (I-1: it
    never runs under the platform login instead)."""
    site = current_site()
    identity = resolve_hpc_identity(
        site,
        lane=Lane.USER_WORK,
        user_id=user_id,
        project_id=project_id,
        is_admin=user_is_admin(user_id),
        slurm_account=slurm_account,
        enrolment=read_enrolment(site, user_id, project_id),
    )
    if identity.mode == USER_ACCOUNT and not gated:
        raise _lane_not_gated(identity)
    return identity


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
    *, user_id: str | None, project_id: str | None, gated: bool = False
) -> HpcAccessRefused | None:
    """The refusal for this caller's user work here, or None when allowed."""
    try:
        resolve_user_work(user_id=user_id, project_id=project_id, gated=gated)
    except HpcAccessRefused as exc:
        return exc
    return None


__all__ = [
    "current_site",
    "is_user_work_job",
    "read_enrolment",
    "resolve_job_user_work",
    "resolve_user_work",
    "set_site_account_store_factory",
    "site_account_store",
    "site_slurm_account",
    "user_in_project",
    "user_is_admin",
    "user_work_refusal",
]
