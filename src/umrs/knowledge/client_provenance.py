"""Where an ingested file came from on the user's workstation (ADR-108 D-6).

The client sends, per source, the file's path relative to its git repository,
the repository's identity, branch and commit, and whether the file had
uncommitted changes. Only basenames reached UMRS before, so nothing tied an
artifact to the commit a result was produced from.

These are the client's report, not a server fact: they are kept under
``metadata.provenance`` with ``client_reported: true``, bounded, and
anything else is dropped. A missing or malformed block is no provenance,
never a failed ingest.
"""

from __future__ import annotations

import re
from typing import Any

PROVENANCE_KEY = "provenance"
_STR_FIELDS = {"path": 1024, "repo": 512, "branch": 255}
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _clean_str(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit] if text and "\x00" not in text else ""


def clean_provenance(raw: Any) -> dict[str, Any] | None:
    """The bounded, known fields of a client provenance block, or None."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for key, limit in _STR_FIELDS.items():
        text = _clean_str(raw.get(key), limit)
        if text:
            out[key] = text
    commit = str(raw.get("commit") or "").strip().lower()
    if _COMMIT_RE.match(commit):
        out["commit"] = commit
    if isinstance(raw.get("dirty"), bool):
        out["dirty"] = raw["dirty"]
    if not out:
        return None
    return {**out, "client_reported": True}


def attach_scope_path(descs: list[dict[str, Any]], scope_path: Any) -> None:
    """Stamp the work level's scope path (ADR-108 D-3) on each artifact.

    S-W3a: the checked path also goes on the descriptor itself
    (``desc["scope_path"]``), which ``_store_one_artifact`` writes to the
    0150 column with ``tier='proposed'``; the metadata copy stays for
    display. The root scope removes any descriptor key, so only this call
    decides a row's scope. A malformed path raises
    ``scope_overlay.ScopeOverlayError`` (never an untagged root write).
    """
    from umrs.knowledge.scope_overlay import SCOPE_KEY, clean_scope_path

    path = clean_scope_path(scope_path)
    if path is None:
        for desc in descs:
            desc.pop(SCOPE_KEY, None)
        return
    for desc in descs:
        desc[SCOPE_KEY] = path
        meta = desc.get("metadata")
        if isinstance(meta, dict):
            meta["scope_path"] = path
        else:
            desc["metadata"] = {"scope_path": path}


def attach_provenance(descs: list[dict[str, Any]], src_spec: dict[str, Any]) -> None:
    """Put the source's provenance on the artifacts it produced."""
    clean = clean_provenance((src_spec or {}).get(PROVENANCE_KEY))
    if clean is None:
        return
    for desc in descs:
        meta = desc.get("metadata")
        if isinstance(meta, dict):
            meta[PROVENANCE_KEY] = dict(clean)
        else:
            desc["metadata"] = {PROVENANCE_KEY: dict(clean)}
