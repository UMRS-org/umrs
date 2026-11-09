"""Usage reports over the ledger (ADR-112 wave D; plan §6 C/D.0 D4, K15).

- :func:`my_usage`: the caller's own current rows at one venue and their
  totals (``GET /umrs/sites/me/hpc/usage``);
- :func:`project_usage`: one project's budget (lifetime, across venues),
  totals, per-member sums and rows (``GET /umrs/projects/{id}/hpc-usage``).
  Members see their own rows and the totals; owners and admins see every
  row and ``members``; an owner who is not an admin sees other members'
  rows without the columns that identify their HPC account and audit trail
  (K15). ``since`` and ``venue`` filter rows, totals and members, never
  the budget.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .usage_ledger import budget_view, row_view, totals

LIMIT_MAX = 1000
LIMIT_DEFAULT = 200
ROLES = ("admin", "owner", "member")


def parse_since(text: str | None) -> datetime | None:
    """An ISO date or time (UTC when it names no zone); ValueError otherwise."""
    if not text:
        return None
    value = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def clamp_limit(limit: int | None) -> int:
    return max(1, min(LIMIT_MAX, int(limit or LIMIT_DEFAULT)))


def public_totals(sums: dict[str, Any]) -> dict[str, Any]:
    return {
        k: sums[k]
        for k in (
            "jobs",
            "gpu_hours",
            "core_hours",
            "reserved_gpu_hours",
            "reserved_core_hours",
        )
    }


def report_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return public_totals(totals(rows))


def my_usage(
    ledger: Any,
    venue: str,
    person: str,
    *,
    project: str | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    project = project or None
    rows = ledger.current(
        venue, person=person, project=project, since=since, limit=clamp_limit(limit)
    )
    sums = ledger.current_totals(venue, person=person, project=project, since=since)
    return {
        "venue": venue,
        "rows": [row_view(r) for r in rows],
        "totals": public_totals(sums),
    }


def _after(
    rows: list[dict[str, Any]], since: datetime | None, venue: str | None
) -> list[dict[str, Any]]:
    if venue:
        rows = [r for r in rows if r.get("venue") == venue]
    if since is None:
        return rows
    return [
        r
        for r in rows
        if not isinstance(r.get("recorded_at"), datetime) or r["recorded_at"] >= since
    ]


def _members(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    people = sorted({str(r.get("person") or "") for r in rows})
    out = []
    for person in people:
        sums = totals([r for r in rows if r.get("person") == person])
        out.append(
            {
                "person": person,
                "jobs": sums["jobs"],
                "gpu_hours": sums["gpu_hours"],
                "core_hours": sums["core_hours"],
                "reserved_gpu_hours": sums["reserved_gpu_hours"],
            }
        )
    return out


def _visible_rows(
    rows: list[dict[str, Any]], role: str, caller: str
) -> list[dict[str, Any]]:
    if role == "admin":
        return [row_view(r) for r in rows]
    if role == "owner":
        return [row_view(r, private=r.get("person") == caller) for r in rows]
    return [row_view(r) for r in rows if r.get("person") == caller]


def project_usage(
    ledger: Any,
    project_id: str,
    *,
    caller: str,
    role: str,
    since: datetime | None = None,
    venue: str | None = None,
) -> dict[str, Any]:
    """The project report for a caller of *role* (``admin``, ``owner``,
    ``member``; the route decides it and refuses anyone else)."""
    if role not in ROLES:
        raise ValueError(f"unknown report role {role!r}")
    all_rows = ledger.project_rows(project_id)
    rows = _after(all_rows, since, venue)
    report: dict[str, Any] = {
        "project_id": project_id,
        "budget": budget_view(ledger.max_gpu_hours(project_id), totals(all_rows)),
        "totals": report_totals(rows),
        "rows": _visible_rows(rows, role, caller),
    }
    if role != "member":
        report["members"] = _members(rows)
    return report


__all__ = [
    "LIMIT_MAX",
    "ROLES",
    "clamp_limit",
    "my_usage",
    "parse_since",
    "project_usage",
    "public_totals",
    "report_totals",
]
