"""Project-scoped MCP server resolution (ADR-098 D-11, slice SH-M1).

Migration 0140 made MCP server names unique per project: the registry key
is ``(COALESCE(TRIM(project_id), ''), name)``. A server row is visible to a
project when it is that project's own row or a global row (``project_id``
NULL, empty or blank). Another project's row is never visible, whether the
caller names it or passes its id.

Before 2026-11-24 the ``MCP_SERVER`` job kind dropped ``project_id`` from
its spec and resolved by name alone, so a job could bind to another
project's server of the same name. Every resolution for a project goes
through :func:`resolve_server_for_project`; it re-checks visibility after
the registry lookup so a registry that does not filter (an older store, a
test fake) still cannot hand out another project's row.

Both hosted fronts and a self-hosted stand-alone UMRS use this path; a
single-project stand-alone install sees its own rows plus the global ones.
"""

from __future__ import annotations

from typing import Any


def project_key(value: Any) -> str:
    """The registry's project key: NULL, '' and padding all mean global."""
    return str(value or "").strip()


def server_visible_to(server: Any, project_id: str) -> bool:
    """True when ``server`` is ``project_id``'s own row or a global row."""
    owner = project_key(getattr(server, "project_id", None))
    return owner in ("", project_key(project_id))


def _lookup(registry: Any, project: str, server_id: str, server_name: str):
    if server_id:
        return registry.get_server(server_id)
    return registry.get_server_by_name(server_name, project_id=project)


def resolve_server_for_project(
    registry: Any,
    *,
    project_id: str,
    server_id: str = "",
    server_name: str = "",
) -> Any | None:
    """The server row ``project_id`` may use, or ``None``.

    ``None`` covers both "absent" and "another project's row": the caller
    refuses the same way, so the answer is no existence oracle across
    projects. An empty ``project_id`` resolves nothing (fail closed).
    """
    project = project_key(project_id)
    if not project:
        return None
    try:
        server = _lookup(registry, project, server_id, server_name)
    except Exception:  # noqa: BLE001 — the caller names the refusal
        return None
    if server is None or not server_visible_to(server, project):
        return None
    return server
