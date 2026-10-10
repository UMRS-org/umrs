"""Gate for opt-in raw-materials export (ADR-032 raw-materials amendment).

Raw bytes leaving the installation are a data release, so:

- only the project owner or an admin may request them;
- a project whose data class is not ``non_sensitive`` (ADR-062 default:
  sensitive; an unknown class counts as sensitive) needs an explicit
  acknowledgement, which is audited and recorded in the manifest;
- the project's Scientific Catalog must state a licence.

The facts come from one read of the shared registry. When the registry is
unreachable the gate fails closed (raw export refused, reason named).

The same facts gate the opt-in chunk text of ADR-093 D-12 (MD-5): a
sensitive project's chunk bodies leave only on its owner's explicit flag.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

_FACTS_SQL = (
    "SELECT p.owner_id, p.data_class,"
    " (SELECT c.licence FROM scientific_catalogs c"
    "   WHERE c.project_id = p.project_id AND c.status = 'active'"
    "   ORDER BY c.created_at DESC LIMIT 1)"
    " FROM projects p WHERE p.project_id = %s"
)


def _project_facts(pool: Any, project_id: str) -> tuple[str, str, str] | None:
    """(owner_id, data_class, catalog licence) or None when unknown."""
    if pool is None:
        return None
    try:
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(_FACTS_SQL, (project_id,))
            row = cur.fetchone()
        finally:
            pool.putconn(conn)
    except Exception:  # fail closed below, reason logged
        logger.warning(
            "raw-export facts lookup failed for %s", project_id, exc_info=True
        )
        return None
    if row is None:
        return None
    return str(row[0] or ""), str(row[1] or ""), str(row[2] or "")


def _require_owner_and_licence(
    project_id: str, user_id: str, is_admin: bool, owner: str, licence: str
) -> None:
    if not is_admin and (not user_id or user_id != owner):
        raise_typed(
            PlatformError.knowledge(
                "raw_forbidden",
                f"raw export of {project_id!r} is limited to the project owner or an admin",
            )
        )
    if not licence.strip():
        raise_typed(
            PlatformError.knowledge(
                "raw_licence_required",
                "raw export needs the project's Scientific Catalog to state a licence "
                "(r4 catalog patch … --licence)",
            )
        )


def authorize_raw_export(
    pool: Any,
    *,
    project_id: str,
    user_id: str,
    is_admin: bool,
    acknowledge_sensitive: bool,
) -> dict[str, Any]:
    """Refuse (typed) or return the acknowledgement record ({} when the
    project is non-sensitive)."""
    facts = _project_facts(pool, project_id)
    if facts is None:
        raise_typed(
            PlatformError.knowledge(
                "unavailable",
                "raw export needs the project registry (owner, data class, "
                "catalog licence) and it is unreachable",
            )
        )
    owner, data_class, licence = facts
    _require_owner_and_licence(project_id, user_id, is_admin, owner, licence)
    if data_class == "non_sensitive":
        return {}
    if not acknowledge_sensitive:
        raise_typed(
            PlatformError.knowledge(
                "raw_unacknowledged",
                f"project {project_id!r} is {data_class or 'sensitive (unset)'}: raw "
                "export needs an explicit acknowledgement (--acknowledge-sensitive)",
            )
        )
    return {
        "by": user_id,
        "data_class": data_class or "sensitive",
        "at": datetime.now(timezone.utc).isoformat(),
    }


def authorize_chunk_text_export(
    pool: Any, *, project_id: str, user_id: str
) -> dict[str, Any]:
    """ADR-093 D-12 / MD-5: full chunk text (r4-kb v1.3) of a project whose
    class is not ``non_sensitive`` leaves only on its owner's explicit flag
    (an admin is not the owner). Refuse (typed, fail closed on an unreadable
    registry) or return the acknowledgement ({} when non-sensitive)."""
    facts = _project_facts(pool, project_id)
    if facts is None:
        raise_typed(
            PlatformError.knowledge(
                "unavailable",
                f"chunk-text export needs a readable project registry row (owner, "
                f"data class) for {project_id!r}: the project is unregistered or "
                "the registry is unreachable",
            )
        )
    owner, data_class, _licence = facts
    if data_class == "non_sensitive":
        return {}
    if not user_id or user_id != owner:
        raise_typed(
            PlatformError.knowledge(
                "raw_forbidden",
                f"project {project_id!r} is {data_class or 'sensitive (unset)'}: its "
                "chunk text is exported only on its owner's explicit --chunk-text",
            )
        )
    return {
        "by": user_id,
        "data_class": data_class or "sensitive",
        "at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["authorize_chunk_text_export", "authorize_raw_export"]
