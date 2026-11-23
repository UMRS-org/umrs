"""ADR-062 locality of an endpoint-registry row (DEC-063, gap D-63).

One definition of "this registry row is a public passthrough" and of the
class gate over it, shared by every seam that hands a project a registry
row: the gated resolver (``umrs.orchestrator.registry_route``) used by the
server consumers, and umrs-infer's site source (``infer_proxy/site_models``
re-exports :func:`is_public_row`).

A row is PUBLIC when its metadata says ``locality: public`` or names a
public provider kind (ADR-062 D-51). A sensitive project never reaches such
a row; self-deployed rows (the site's own models) are its own route. The
rule is identical on the hosted fleet and on a self-hosted UMRS
(stand-alone, own HPC): it reads only the row and the project's
``data_class``.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.shared.project_data_class import PUBLIC_PROVIDER_KINDS, data_class_for_project

logger = logging.getLogger(__name__)

#: The refusal reason every class-gated seam names (umrs-infer answers it too).
SOURCE_CLASS_REFUSED = "source_class_refused"


def is_public_row(endpoint: Any) -> bool:
    """True when the registry row passes requests through to a public provider."""
    metadata = getattr(endpoint, "metadata", None) or {}
    return (
        metadata.get("locality") == "public"
        or metadata.get("provider_kind") in PUBLIC_PROVIDER_KINDS
    )


def row_admitted(endpoint: Any, data_class: str) -> bool:
    """ADR-062: a sensitive project is refused every public row."""
    return data_class != "sensitive" or not is_public_row(endpoint)


def _default_pool() -> Any:
    """The process's shared Postgres pool, or None when none is configured."""
    try:
        from umrs.shared.pg import resolve_satellite_pool

        return resolve_satellite_pool("auto", None, "endpoint locality")
    except Exception:  # the lookup never blocks its caller
        logger.warning("endpoint locality: no Postgres pool", exc_info=True)
        return None


def project_data_class(project_id: str, pool: Any = None) -> str:
    """The project's ``data_class`` (``""`` when unknown, as umrs-infer reads it)."""
    return data_class_for_project(pool if pool is not None else _default_pool(), project_id)


__all__ = [
    "SOURCE_CLASS_REFUSED",
    "is_public_row",
    "project_data_class",
    "row_admitted",
]
