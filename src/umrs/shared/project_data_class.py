"""ADR-062 model-locality seam: the per-project data_class lookup + the
public-provider kind vocabulary (D-51).

Both enforcement points share this module so ONE definition of
"public kind" and ONE never-invented lookup serve every seam:

* the venue-start gate (``venues/frontend_service.py``),
* the inference-proxy lease gate (``infer_proxy/app.py``).

The lookup never blocks its caller: pool-less, absent row, or a lookup
failure all answer ``""`` (no gate) — a class is never invented.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ADR-062 model locality: every user-pushable provider kind is PUBLIC; a
# sensitive project's sessions/leases must not reach any of them
# (self-deployed models need no pushed source).
PUBLIC_PROVIDER_KINDS = frozenset(
    {"openai", "openai_compat", "anthropic", "google", "azure_openai"}
)


def data_class_for_project(pool: Any, project: str) -> str:
    """The project row's ``data_class`` (migration 0107).

    Returns ``""`` when *project* is empty, *pool* is None, the row is
    absent, or the lookup fails — the caller's gate never blocks on a
    lookup failure and never invents a class.
    """
    if pool is None or not project:
        return ""
    try:
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT data_class FROM projects WHERE project_id = %s",
                (project,),
            )
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else ""
        finally:
            pool.putconn(conn)
    except Exception:  # noqa: BLE001 — the gate never blocks on lookup failure
        logger.warning("data_class lookup failed for %s", project, exc_info=True)
        return ""
