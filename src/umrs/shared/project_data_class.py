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
import os
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# ADR-062 model locality: every user-pushable provider kind is PUBLIC; a
# sensitive project's sessions/leases must not reach any of them
# (self-deployed models need no pushed source).
PUBLIC_PROVIDER_KINDS = frozenset(
    {"openai", "openai_compat", "anthropic", "google", "azure_openai"}
)

# ADR-062 amendment (2026-10-15): the third provider class, INSTITUTIONAL —
# an inference service run by the user's own institution outside UMRS
# (Rikyu first). A sensitive project may use it; commercial and other
# public providers stay refused. Membership comes from this allowlist the
# SITE keeps, never from a user label: UMRS_INSTITUTIONAL_PROVIDER_HOSTS
# (comma/space separated) replaces the packaged default; empty = none.
DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS = frozenset({"api.rikyu.r-ccs.riken.jp"})


def institutional_provider_hosts() -> frozenset[str]:
    """The site's institutional-provider allowlist (host names only)."""
    raw = os.environ.get("UMRS_INSTITUTIONAL_PROVIDER_HOSTS")
    if raw is None:
        return DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS
    return frozenset(h.strip().lower() for h in raw.replace(",", " ").split() if h.strip())


def url_host(url: str) -> str:
    """The lowercased host of an (http(s)) URL — no port, no brackets.

    ``""`` when unparseable. A scheme-less ``host[:port]`` is accepted
    (sealed sources store allow_host with a scheme, but config rows vary).
    """
    text = str(url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return ""
    return (host or "").strip().lower().strip("[]")


def is_institutional_url(url: Any, hosts: frozenset[str] | None = None) -> bool:
    """True when *url*'s host is on the institutional allowlist.

    Exact host match only — no suffix games (``not-rikyu.r-ccs.riken.jp``
    is not ``api.rikyu.r-ccs.riken.jp``). Loopback is NOT institutional:
    the amendment's local class is a workstation-side notion; a remote
    gateway cannot see the client's machine (ADR-046).
    """
    host = url_host(str(url or ""))
    if not host:
        return False
    return host in (hosts if hosts is not None else institutional_provider_hosts())


def default_project_data_class() -> str:
    """The class a NEW project row takes (UMRS_PROJECT_DEFAULT_DATA_CLASS).

    ``"sensitive"`` unless the deployment opts out — the stand-alone
    single-host profile sets ``"non_sensitive"`` (ADR-062 amendment D-4).
    An invalid value falls back to ``"sensitive"``: a typo never widens.
    Existing rows are never rewritten by this (INSERT-time only).
    """
    raw = os.environ.get("UMRS_PROJECT_DEFAULT_DATA_CLASS", "").strip()
    if not raw:
        return "sensitive"
    if raw not in ("sensitive", "non_sensitive"):
        logger.warning(
            "invalid UMRS_PROJECT_DEFAULT_DATA_CLASS %r — using 'sensitive'", raw
        )
        return "sensitive"
    return raw


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
    except Exception:  # the gate never blocks on lookup failure
        logger.warning("data_class lookup failed for %s", project, exc_info=True)
        return ""
