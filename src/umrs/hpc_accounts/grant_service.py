"""What the server knows that a grant must match (ADR-112 wave C; plan §6
C/D.0 C4): the registered versions' digests, the site's step images, the
person's links, and the grant proposal (X6). FastAPI-free, so the routes,
the run admission and the tests share it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .grants import GrantContext, GrantInvalid, server_digests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Version:
    """A registered workflow version the caller may see."""

    name: str
    version: str
    entry: dict[str, Any]
    source_path: str
    workflow_json: dict[str, Any]


def _read_workflow_json(source_path: str) -> dict[str, Any]:
    path = os.path.join(source_path, "workflow.json")
    if not source_path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def find_version(
    name: str, version: str | None, user: str, *, project: str | None = None, is_admin: bool = False
) -> Version | None:
    """The version *user* may see (owner-scoped, ADR-074 D-8), else None."""
    from umrs.server.workflow_registry_handler import resolve_workflow_version

    try:
        resolved = resolve_workflow_version(
            name, user, is_admin=is_admin, caller_project=project, version=version or None
        )
    except ValueError:
        return None
    if resolved is None:
        return None
    entry = dict(resolved.entry)
    source = str(entry.get("source_path") or "")
    return Version(str(entry.get("name") or name), str(entry.get("version") or ""), entry, source, _read_workflow_json(source))


def digests_lookup(user: str, project: str | None = None, is_admin: bool = False) -> Callable[[str, str], tuple[str, str] | None]:
    """``(name, version) -> (src_manifest_sha256, steps_sha256)`` of the
    registered version (None: unknown or not visible), cached per call."""
    cache: dict[tuple[str, str], tuple[str, str] | None] = {}

    def lookup(name: str, version: str) -> tuple[str, str] | None:
        if (name, version) not in cache:
            cache[(name, version)] = _digests(name, version, user, project, is_admin)
        return cache[(name, version)]

    return lookup


def _digests(name: str, version: str, user: str, project: str | None, is_admin: bool) -> tuple[str, str] | None:
    found = find_version(name, version, user, project=project, is_admin=is_admin)
    if found is None or found.version != version or not found.workflow_json:
        return None
    try:
        return server_digests(found.source_path, found.workflow_json)
    except Exception:
        logger.warning("grant: digests of %s@%s unreadable", name, version, exc_info=True)
        return None


def site_images(site: Any) -> frozenset[tuple[str, str, str]]:
    """Every ``(ref, image_id, archive_sha256)`` of the site's published
    step image digests (empty when the site publishes none)."""
    from umrs.server.intent_proposal import load_image_digests

    try:
        digests = load_image_digests(site)
    except Exception:  # noqa: BLE001 - no image digests: none known
        return frozenset()
    found = set()
    for ref, arches in digests.items():
        for entry in (arches or {}).values() if isinstance(arches, dict) else []:
            if isinstance(entry, dict):
                found.add((str(ref), str(entry.get("image_id") or ""), str(entry.get("archive_sha256") or "")))
    return frozenset(found)


def build_context(
    venue: str, principal: str, store: Any, memberships: frozenset[str] | None, site: Any, *, is_admin: bool = False
) -> GrantContext:
    """The context for *principal* at *venue* (account, links, digests, images)."""
    return GrantContext(
        venue=venue,
        principal=principal,
        account=dict(store.get_account(principal) or {}),
        links=list(store.list_links(principal)),
        memberships=memberships,
        digests_for=digests_lookup(principal, is_admin=is_admin),
        images=site_images(site),
    )


# ── the grant proposal (X6) ──────────────────────────────────────────────────


def _hints(step: dict[str, Any]) -> dict[str, Any]:
    hints = step.get("routing_hints")
    return hints if isinstance(hints, dict) else {}


def _step_arch(step: dict[str, Any], profile: dict[str, Any], cpus: int) -> str:
    from umrs.server.intent_proposal import _needs_gpu, _partition

    gpus = int(_hints(step).get("gpu_count") or 1) if _needs_gpu(step, profile) else 0
    return str(profile["partitions"][_partition(profile, gpus)]["arch"])


def _images_of(version: Version, site: Any, profile: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """The step images (per partition arch, as the B2 proposal picks them)
    and the run's trusted image; ``no_image_digests`` is ``image_unknown``."""
    from umrs.hpc_accounts.intents import IntentInvalid

    try:
        return _step_images(version, site, profile)
    except IntentInvalid as exc:
        raise GrantInvalid("image_unknown", exc.detail or exc.reason) from exc


def _step_images(version: Version, site: Any, profile: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    from umrs.server.intent_proposal import _image, load_image_digests, platform_image

    steps = [s for s in version.workflow_json.get("steps") or [] if isinstance(s, dict)]
    digests, ref = load_image_digests(site), platform_image(site)
    arches = [_step_arch(s, profile, 1) for s in steps]
    images = {}
    for step, arch in zip(steps, arches):
        image = _image(digests, str(_hints(step).get("image") or ref), arch)
        images[image["image_id"]] = image
    trusted = _image(digests, ref, arches[0] if arches else "x86_64")
    return [images[k] for k in sorted(images)], trusted


def _parameter_schema(workflow_json: dict[str, Any]) -> dict[str, Any]:
    from umrs.api.workflows import _workflow_parameter_schema

    return _workflow_parameter_schema(workflow_json)


def proposal_workflow(version: Version, site: Any, profile: dict[str, Any]) -> dict[str, Any]:
    """One ``workflows[]`` entry of the proposal."""
    src_sha, steps_sha = server_digests(version.source_path, version.workflow_json)
    images, trusted = _images_of(version, site, profile)
    return {
        "name": version.name,
        "version": version.version,
        "src_manifest_sha256": src_sha,
        "steps_sha256": steps_sha,
        "parameter_schema": _parameter_schema(version.workflow_json),
        "images": images,
        "trusted_image": trusted,
    }


def proposal_versions(
    refs: list[dict[str, Any]], user: str, project: str, *, is_admin: bool = False
) -> list[Version]:
    """Each named version (the default one when no version is named);
    ``workflow_unknown`` names the first the caller cannot see."""
    found = []
    for ref in refs:
        name, version = str(ref.get("name") or ""), ref.get("version")
        hit = find_version(name, version, user, project=project, is_admin=is_admin)
        if hit is None or not hit.workflow_json:
            raise GrantInvalid("workflow_unknown", f"{name}@{version or 'default'}")
        found.append(hit)
    return found


__all__ = [
    "Version",
    "build_context",
    "digests_lookup",
    "find_version",
    "proposal_versions",
    "proposal_workflow",
    "site_images",
]
