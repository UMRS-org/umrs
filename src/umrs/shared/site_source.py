"""The site's own models as a venue source (ADR-092 slice 8b, D-63 venue leg).

A frontend-venue session normally reaches only the user's pushed sealed
sources through ``umrs-infer``. The reserved source ``site`` (a fixed UUID,
because a lease's ``source_ids`` column is ``UUID[]``) stands for the
site's own ``EndpointRegistry`` rows of the session's project: the D-63
model-backend seam, behind the same lease and the same proxy. No sealed
row backs it and no credential is pushed for it; the proxy injects the
site's own service key, which never reaches the sandbox.

Foundation module (stdlib only): the venue start and the proxy read it.
"""

from __future__ import annotations

import uuid

SITE_SOURCE_NAME = "site"
#: Deterministic: the same on every server and every restart.
SITE_SOURCE_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "umrs:venue-source:site-models"))
#: A sandbox selects the site source by this path prefix on its sidecar
#: (``http://egress:8080/site/v1/...``), so no header plumbing is needed.
SITE_PATH_PREFIX = "site/"


def is_site_source(value: str | None) -> bool:
    return value in (SITE_SOURCE_NAME, SITE_SOURCE_ID)


__all__ = ["SITE_PATH_PREFIX", "SITE_SOURCE_ID", "SITE_SOURCE_NAME", "is_site_source"]
