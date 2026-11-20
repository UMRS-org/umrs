"""Venue names: the ``cluster:<site>`` form (U3 S1 item 2, ADR-117).

A job spec may bind work to one site with ``venue: cluster:<site-name>``.
Claim scoping must be exact (``cluster:rccs-front`` is site.name
``rccs-front``), so one parser serves the job store, the loop start route and
the HPC user-work check instead of three hand-written prefix tests.
"""

from __future__ import annotations

CLUSTER_VENUE_PREFIX = "cluster:"


def is_cluster_venue(venue: str) -> bool:
    """True for a ``cluster:<site>`` venue (surrounding blanks ignored)."""
    return str(venue or "").strip().startswith(CLUSTER_VENUE_PREFIX)


def split_cluster_venue_name(venue: str) -> str:
    """``cluster:<site-name>`` → the site.name part (byte-exact with site.yaml).

    The name after the prefix must be non-empty and free of separators that
    would make reconciliation ambiguous.
    """
    name = venue[len(CLUSTER_VENUE_PREFIX) :].strip()
    if not name:
        raise ValueError("cluster venue requires a site name after 'cluster:'")
    if "/" in name or name != name.lower() or name.startswith((" ", "-")):
        raise ValueError(f"cluster site name is invalid: {name!r}")
    return name


__all__ = ["CLUSTER_VENUE_PREFIX", "is_cluster_venue", "split_cluster_venue_name"]
