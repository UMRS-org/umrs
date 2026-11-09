"""The gate site profile this server publishes (ADR-112 B2; plan §5 B2.0 §2).

The gate composes a gated step's batch script only from the person's signed
intent and its own ``~/.umrs/gate/site.json``. r4 writes that file, after
showing it to the person, from what this server publishes at
``GET /umrs/sites/me/hpc/site-profile``; every intent names the sha256 of
the profile bytes it was proposed against, and the gate refuses a submit
whose sha differs from its installed file (``profile_mismatch``).

:func:`build_site_profile` reads the site file: the step lane's partitions
and their profiles, its GPU form and flag, its container posture, the
runtime's storage and network directories, the queue timeout, the staging
caps, the reply cap and the forwarder caps; then the site's
``hpc_access.user_account.profile`` block overrides the matching keys (a
mapping is merged one level deep). The site must name ``image_archive_dir``
there. The gate validates the whole profile (``validate_site_profile``) and
the bytes are its canonical encoding (``site_profile_bytes``).
"""

from __future__ import annotations

import hashlib
from typing import Any

#: The job's watchdog period and the gate's default caps (plan §5 B2.0 §2).
#: ``/usr/bin`` only: ``/bin`` is a link on usrmerge hosts and the gate refuses a
#: linked ``slurm.path`` entry (fixer finding 2026-11-08).
DEFAULT_SLURM_PATH = "/usr/bin"
DEFAULT_MAX_WORKDIRS = 64
LOG_TAIL_BYTES = 64 * 1024
_ARCHES = ("x86_64", "aarch64")


class SiteProfileUnavailable(ValueError):
    """The site publishes no valid gate profile (a site configuration gap)."""


def _seconds(text: Any, default: int) -> int:
    from umrs.server.slurm_step_executor import slurm_duration_seconds

    value = slurm_duration_seconds(str(text or ""))
    return int(value) if value else default


def _partition(site: Any, name: str) -> dict[str, Any]:
    from umrs.orchestrator import partition_choice as pc

    se = site.step_executor
    profile = site.get_partition_profile(name)
    arch = str(getattr(profile, "arch", "") or "x86_64")
    return {
        "arch": arch if arch in _ARCHES else "x86_64",
        "max_cpus": int(se.cpus),
        "max_gpus": (pc.partition_gpus(profile) or int(se.gpus)) if se.gpu else 0,
        "max_time": _seconds(se.time_limit, 3600),
        "mem": bool(str(se.memory or "").strip()),
        "gpu_request": str(getattr(profile, "gpu_request", "") or ""),
        "gres": str(getattr(profile, "gres", "") or ""),
        "gpu_flag": str(se.gpu_flag or "gres"),
        "exclusive": bool(getattr(profile, "exclusive", False)),
    }


def _caps() -> dict[str, int]:
    from umrs.api.workflow_export import MAX_EXPORT_BYTES
    from umrs.server.remote_slurm_step_executor import REPLY_CAP_BYTES
    from umrs.server.step_staging import DEFAULT_CAPS as c

    return {
        "max_object_bytes": c.max_object_bytes,
        "max_in_bytes": c.max_in_bytes,
        "max_in_objects": c.max_in_objects,
        "max_src_bytes": min(c.max_src_bytes, MAX_EXPORT_BYTES),
        "max_src_files": c.max_src_files,
        "max_out_files": c.max_out_files,
        "max_out_bytes": c.max_out_bytes,
        "max_out_file_bytes": c.max_out_file_bytes,
        "min_free_bytes": c.min_free_bytes,
        "reply_bytes": REPLY_CAP_BYTES,
        "log_tail_bytes": LOG_TAIL_BYTES,
        "max_workdirs": DEFAULT_MAX_WORKDIRS,
    }


def _endpoint_caps(se: Any) -> dict[str, int]:
    from umrs.server import step_endpoint_forwarder as fw

    return {
        "max_request_bytes": int(
            se.endpoint_max_request_bytes or fw.DEFAULT_MAX_REQUEST_BYTES
        ),
        "max_response_bytes": int(
            se.endpoint_max_response_bytes or fw.DEFAULT_MAX_RESPONSE_BYTES
        ),
        "max_total_bytes": int(
            se.endpoint_max_total_bytes or fw.DEFAULT_MAX_TOTAL_BYTES
        ),
    }


def _base_profile(site: Any, domain: str) -> dict[str, Any]:
    from umrs.orchestrator import partition_choice as pc
    from umrs.server.slurm_step_executor import WALL_GRACE_SEC, WATCH_SECONDS

    se = site.step_executor
    names = pc.split_partitions(se.partition) or (
        [se.partition] if se.partition else []
    )
    return {
        "v": 1,
        "site": str(site.name),
        "domain": domain,
        "slurm": {
            "path": DEFAULT_SLURM_PATH,
            "owner_uids": [],
            "queue_seconds": _seconds(se.queue_timeout, 6 * 3600),
            "grace_seconds": WALL_GRACE_SEC,
        },
        "partitions": {name: _partition(site, name) for name in names},
        "gpu": {"enabled": bool(se.gpu), "form": str(se.gpu_form or "cdi")},
        "podman": {
            "pids_limit": int(se.pids_limit),
            "cgroups": str(se.cgroups or "enabled"),
            "nproc_limit": int(se.nproc_limit),
        },
        "venue": {
            "root": "",
            "storage_root": str(site.runtime.storage_root or ""),
            "network_config_dir": str(site.runtime.network_config_dir or ""),
            "network_backend": "",
        },
        "image_archive_dir": "",
        "caps": _caps(),
        "endpoint": _endpoint_caps(se),
        "watch_seconds": WATCH_SECONDS,
    }


def _override(profile: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """The site's ``profile:`` block wins; a mapping merges one level deep."""
    merged = dict(profile)
    for key, value in overrides.items():
        base = merged.get(key)
        merged[key] = (
            {**base, **value}
            if isinstance(base, dict) and isinstance(value, dict)
            else value
        )
    return merged


def _user_account(site: Any) -> Any:
    hpc = getattr(getattr(site, "scheduler", None), "hpc_access", None)
    user_account = getattr(hpc, "user_account", None)
    if user_account is None or not getattr(user_account, "signing_domain", ""):
        raise SiteProfileUnavailable(
            f"site {getattr(site, 'name', '?')} declares no user_account signing domain"
        )
    return user_account


def service_pricing_profile(site: Any) -> dict[str, Any]:
    """The site's K4 pricing context without gated-user requirements.

    Ordinary service-account jobs use the same partition resources as the
    published gate profile, but need no signing domain, user profile, image
    manifest or gate-only validation.
    """
    return _base_profile(site, "")


def build_site_profile(site: Any) -> dict[str, Any]:
    """The gate site profile of *site* (validated by the gate's own rules)."""
    from r4_contracts import hpc_gate

    user_account = _user_account(site)
    profile = _override(
        _base_profile(site, user_account.signing_domain),
        dict(user_account.profile or {}),
    )
    if not profile.get("image_archive_dir"):
        raise SiteProfileUnavailable(
            "the site names no image_archive_dir in hpc_access.user_account.profile"
        )
    try:
        hpc_gate.validate_site_profile(profile)
    except (ValueError, hpc_gate.GateError) as exc:
        raise SiteProfileUnavailable(
            f"the site's gate profile is invalid: {exc}"
        ) from exc
    return profile


def profile_bytes(profile: dict[str, Any]) -> bytes:
    from r4_contracts import hpc_gate

    return hpc_gate.site_profile_bytes(profile)


def profile_sha256(profile: dict[str, Any]) -> str:
    return hashlib.sha256(profile_bytes(profile)).hexdigest()


def published(site: Any) -> tuple[dict[str, Any], bytes, str]:
    """``(profile, bytes, sha256)`` this server publishes for *site*."""
    profile = build_site_profile(site)
    data = profile_bytes(profile)
    return profile, data, hashlib.sha256(data).hexdigest()


__all__ = [
    "SiteProfileUnavailable",
    "build_site_profile",
    "profile_bytes",
    "profile_sha256",
    "published",
    "service_pricing_profile",
]
