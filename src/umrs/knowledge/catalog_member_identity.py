"""Catalog member refs by content identity, export side (DEC-D39, S-D39a).

A catalog row lists its members by ids that only mean something on this
installation (``member_graph_ids``, ``skill_ids``, ``mcp_server_ids``). A
bundle carries them by content identity instead, so another installation can
map them onto its own rows (``catalog_member_map.map_members``):

- graph:     ``{"default": True, "name": ""}`` for the project's default
             graph, else ``{"default": False, "name": <graph name>}``
- skill:     ``{"scope": ..., "name": ..., "version": ...}``
- MCP:       ``{"name": ..., "version": ...}`` (the trust class is NOT
             exported; the importer reads it from its own registry)
- embedding: the model id string itself (already a content identity)

Pure module: the caller has already fetched the registry rows.
"""

from __future__ import annotations

from typing import Any


def member_identities(
    row: dict[str, Any],
    graph_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    mcp_rows: list[dict[str, Any]],
    default_graph_id: str,
) -> dict[str, Any]:
    """The manifest ``members`` block for one catalog row.

    Inputs:

    - ``row``: the catalog row; reads ``member_graph_ids``, ``skill_ids``,
      ``mcp_server_ids``, ``embedding_model_ids``. Each is a list of str,
      or missing/None (read as []). Non-str items are ignored.
    - ``graph_rows``: dicts with ``graph_id`` and ``name``.
    - ``skill_rows``: dicts with ``skill_id``, ``scope``, ``name``, ``version``.
    - ``mcp_rows``: dicts with ``server_id``, ``name``, ``version``.
    - ``default_graph_id``: the exporting project's default graph id.

    Returns exactly these keys::

        {
          "graphs": [{"default": bool, "name": str}, ...],
          "skills": [{"scope": str, "name": str, "version": str}, ...],
          "mcp_servers": [{"name": str, "version": str}, ...],
          "embedding_models": [str, ...],
          "unresolved": int,
        }

    Rules:

    - Lists keep the order of the ids in ``row``; a repeated id appears once.
    - A graph id equal to ``default_graph_id`` gives
      ``{"default": True, "name": ""}`` without a ``graph_rows`` lookup.
    - Any other id is looked up in its rows list by its id key. Found:
      the entry above, values converted with ``str(value or "")``.
      Not found: no entry; ``unresolved`` counts it (once per distinct id).
    - ``embedding_models`` keeps every distinct non-empty str id verbatim.
    """
    graphs, graph_missing = _graph_identities(row, graph_rows, default_graph_id)
    skills, skill_missing = _skill_identities(row, skill_rows)
    mcp_servers, mcp_missing = _mcp_identities(row, mcp_rows)
    return {
        "graphs": graphs,
        "skills": skills,
        "mcp_servers": mcp_servers,
        "embedding_models": _member_ids(row, "embedding_model_ids"),
        "unresolved": graph_missing + skill_missing + mcp_missing,
    }


def _member_ids(row: dict[str, Any], key: str) -> list[str]:
    """Distinct non-empty str ids from ``row[key]``, first occurrence kept."""
    ids: list[str] = []
    items = row.get(key)
    if not isinstance(items, list):
        return ids
    for item in items:
        if isinstance(item, str) and item and item not in ids:
            ids.append(item)
    return ids


def _index_rows(rows: list[dict[str, Any]], id_key: str) -> dict[str, dict[str, Any]]:
    """Registry rows indexed by their id key."""
    return {str(r.get(id_key) or ""): r for r in rows}


def _graph_identities(
    row: dict[str, Any],
    graph_rows: list[dict[str, Any]],
    default_graph_id: str,
) -> tuple[list[dict[str, Any]], int]:
    """Graph identities plus the count of ids no registry row resolved."""
    by_id = _index_rows(graph_rows, "graph_id")
    entries: list[dict[str, Any]] = []
    unresolved = 0
    for graph_id in _member_ids(row, "member_graph_ids"):
        if graph_id == default_graph_id:
            entries.append({"default": True, "name": ""})
        elif graph_id in by_id:
            name = by_id[graph_id].get("name")
            entries.append({"default": False, "name": str(name or "")})
        else:
            unresolved += 1
    return entries, unresolved


def _skill_identities(
    row: dict[str, Any], skill_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Skill identities plus the count of ids no registry row resolved."""
    by_id = _index_rows(skill_rows, "skill_id")
    entries: list[dict[str, Any]] = []
    unresolved = 0
    for skill_id in _member_ids(row, "skill_ids"):
        found = by_id.get(skill_id)
        if found is None:
            unresolved += 1
        else:
            entries.append(
                {
                    "scope": str(found.get("scope") or ""),
                    "name": str(found.get("name") or ""),
                    "version": str(found.get("version") or ""),
                }
            )
    return entries, unresolved


def _mcp_identities(
    row: dict[str, Any], mcp_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """MCP identities (no trust class) plus the unresolved-id count."""
    by_id = _index_rows(mcp_rows, "server_id")
    entries: list[dict[str, Any]] = []
    unresolved = 0
    for server_id in _member_ids(row, "mcp_server_ids"):
        found = by_id.get(server_id)
        if found is None:
            unresolved += 1
        else:
            entries.append(
                {
                    "name": str(found.get("name") or ""),
                    "version": str(found.get("version") or ""),
                }
            )
    return entries, unresolved
