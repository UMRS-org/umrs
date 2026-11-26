"""Catalog identity block for the bundle manifest (DEC-D39, slice S-D39a).

The export puts one ``catalog`` block in ``manifest.json``. Before S-D39a
it held only the descriptive fields of the project's latest active
Scientific Catalog row. This module adds the identity a second installation
needs to re-register the same catalog version:

- ``catalog_id``          the row's id here (the importer records it as
                          ``origin_catalog_id``; it never reuses it as its own id)
- ``origin_fingerprint``  this installation's fingerprint
                          (``catalog_origin.installation_fingerprint``)
- ``change_notes``        the version chain up to this version
- ``members``             member refs by content identity
                          (``catalog_member_identity.member_identities``)

Pure module: no database, no I/O, never mutates its arguments.
"""

from __future__ import annotations

from typing import Any


def _chain_entry(row: dict[str, Any]) -> dict[str, str]:
    """One chain entry: a new dict with both fields read as strings."""
    return {
        "version": str(row.get("version") or ""),
        "change_note": str(row.get("change_note") or ""),
    }


def change_note_chain(
    versions: list[dict[str, Any]], upto_version: str
) -> list[dict[str, str]]:
    """The change-note chain of one catalog, oldest first, up to a version.

    ``versions`` are the rows of one (project, name) catalog, already
    ordered oldest first. Each row has ``version`` and ``change_note``
    (either may be missing or None; read both as "" then).

    Returns ``[{"version": str, "change_note": str}, ...]`` in the input
    order, stopping after (and including) the first row whose ``version``
    equals ``upto_version``. When no row has that version, every row is
    returned. Rows that are not dicts are skipped.
    """
    chain: list[dict[str, str]] = []
    for row in versions:
        if not isinstance(row, dict):
            continue
        entry = _chain_entry(row)
        chain.append(entry)
        if entry["version"] == upto_version:
            break
    return chain


def catalog_block(
    record: dict[str, Any],
    chain: list[dict[str, str]],
    members: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """The manifest ``catalog`` block: ``record`` plus its identity keys.

    ``record`` is the descriptive catalog record the export already builds
    (name, version, description, ontology, licence, contributors,
    provenance, redaction_rules) and may carry ``catalog_id``.

    Returns a NEW dict (``record`` is not mutated) with every key of
    ``record`` plus:

    - ``"catalog_id"``: ``str(record.get("catalog_id") or "")``
    - ``"change_notes"``: ``[dict(entry) for entry in chain]`` (copies)
    - ``"members"``: a copy of ``members`` (``dict(members)``)
    - ``"origin_fingerprint"``: ``str(fingerprint or "")``

    An empty ``record`` (the project has no catalog) returns ``{}`` and
    ignores the other arguments: the block is never invented.
    """
    if not record:
        return {}
    block = dict(record)
    block["catalog_id"] = str(record.get("catalog_id") or "")
    block["change_notes"] = [dict(entry) for entry in chain]
    block["members"] = dict(members)
    block["origin_fingerprint"] = str(fingerprint or "")
    return block
