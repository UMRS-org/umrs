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
    raise NotImplementedError("s-d39a-02")
