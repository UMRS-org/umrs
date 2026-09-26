"""The site's transition switch for uploaded step code (ADR-074 Amendment 1).

A foundation module so both the server's step policy and the inference loader
read one switch. Anything but an explicit ``legacy_in_process`` refuses.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

LEGACY_IN_PROCESS = "legacy_in_process"
UPLOADED_CODE_REFUSE = "refuse"


def load_site_env() -> None:
    """Export the site.yaml step settings into the environment (never overriding)."""
    try:
        from umrs.orchestrator.site_config import get_site_config

        get_site_config().inject_env(overwrite=False)
    except Exception:
        logger.debug("site config unavailable for the step policy", exc_info=True)


def uploaded_code_policy() -> str:
    """:data:`LEGACY_IN_PROCESS` only when the site chose it; otherwise refuse."""
    load_site_env()
    raw = os.environ.get("UMRS_STEP_UPLOADED_CODE", "").strip().lower()
    if raw == LEGACY_IN_PROCESS:
        return LEGACY_IN_PROCESS
    if raw not in ("", UPLOADED_CODE_REFUSE):
        logger.warning(
            "unknown UMRS_STEP_UPLOADED_CODE=%r: uploaded code is refused", raw
        )
    return UPLOADED_CODE_REFUSE
