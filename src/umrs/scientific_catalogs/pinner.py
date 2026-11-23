"""Loop finding pins onto a Scientific Catalog (D-43/D-52, DEC-D39 S-D39b).

A loop may name the catalog its findings pin onto as ``catalog_id@version``
(``@version`` optional). The named catalog is resolved once at loop start
(refused by name when it cannot be), stored on the loop, and every real
finding pins there. A loop that names none keeps the old rule as the
fallback: exactly one active catalog in the project, else no pin (never a
guess between several).

Behaves the same on a hosted front and on a self-hosted stand-alone UMRS:
the store is the one ``scientific_catalogs`` table, no site specifics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from umrs.scientific_catalogs.store import CatalogNotFound

# Reasons a named catalog is refused at loop start (stable, machine-readable).
CATALOG_NOT_FOUND = "catalog_not_found"
CATALOG_RETIRED = "catalog_retired"
CATALOG_VERSION_NOT_FOUND = "catalog_version_not_found"


class CatalogPinRefused(Exception):
    """A named catalog cannot be pinned; ``reason`` is one of the names above."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def parse_catalog_ref(ref: str) -> tuple[str, str]:
    """``"scat-1@0.2.0"`` → ``("scat-1", "0.2.0")``; no ``@`` → version ``""``."""
    catalog_id, _, version = (ref or "").strip().partition("@")
    return catalog_id.strip(), version.strip()


def resolve_catalog_pin(store: Any, project_id: str, ref: str) -> dict[str, Any]:
    """The catalog row ``ref`` names in ``project_id``, or CatalogPinRefused.

    ``catalog_id@version``: the id names a catalog (any version row of it),
    the version picks the row of that name; a missing ``@version`` pins the
    named row itself. A row of another project reads as not found (no
    disclosure across projects); a retired row is refused by name.
    """
    catalog_id, version = parse_catalog_ref(ref)
    row = _project_row(store, project_id, catalog_id)
    if version and row.get("version") != version:
        row = _version_row(store, row, version)
    if row.get("status") == "retired":
        raise CatalogPinRefused(
            CATALOG_RETIRED,
            f"catalog {row['catalog_id']} is retired; name another version",
        )
    return row


def _project_row(store: Any, project_id: str, catalog_id: str) -> dict[str, Any]:
    try:
        row = store.get(catalog_id) if catalog_id else None
    except CatalogNotFound:
        row = None
    if row is None or row.get("project_id") != project_id:
        raise CatalogPinRefused(
            CATALOG_NOT_FOUND,
            f"catalog '{catalog_id}' not found in project {project_id}",
        )
    return row


def _version_row(store: Any, row: dict[str, Any], version: str) -> dict[str, Any]:
    for candidate in store.versions(project_id=row["project_id"], name=row["name"]):
        if candidate.get("version") == version:
            return candidate
    raise CatalogPinRefused(
        CATALOG_VERSION_NOT_FOUND,
        f"catalog '{row['name']}' has no version {version}",
    )


def pin_target(store: Any, project_id: str, catalog_id: str = "") -> str:
    """The catalog id findings pin onto: the loop's named one, else the
    single active catalog of the project (fallback), else ``""`` (none or
    several: nothing to pin, never a guess)."""
    if catalog_id:
        return catalog_id
    catalogs = store.list(project_id=project_id)
    return catalogs[0]["catalog_id"] if len(catalogs) == 1 else ""


def pin_findings(
    store: Any,
    project_id: str,
    pins: Iterable[tuple[str, str]],
    catalog_id: str = "",
) -> str:
    """Append ``kind=finding`` pins (deduped by ref, append-only).

    Returns the catalog id pinned onto (``""`` when nothing was pinned).
    """
    target = pin_target(store, project_id, catalog_id)
    if not target:
        return ""
    already = {
        p.get("ref")
        for p in (store.get(target).get("provenance") or {}).get("pins", [])
    }
    for ref, note in pins:
        if ref not in already:
            store.add_pin(target, kind="finding", ref=ref, note=note)
            already.add(ref)
    return target
