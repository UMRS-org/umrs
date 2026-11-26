"""Catalog member refs mapped onto this installation, import side (DEC-D39).

The bundle's ``catalog.members`` block (``catalog_member_identity``) names
members by content identity. ``map_members`` turns it into the local id
lists a ``CatalogSpec`` needs, through a ``MemberResolver`` that the caller
has already scoped to the TARGET project (the resolver never sees another
project). The bundle's own ids are never used.

Trust classes are never taken from the bundle: they come from the
resolver (the local MCP registry), as for a catalog created by the route.

Pure module: all lookups go through the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class MemberResolver(Protocol):
    """Local lookups for one target project (implemented in
    ``catalog_member_resolver.SqlMemberResolver``)."""

    def default_graph_id(self) -> str:
        """The target project's default graph id."""
        ...

    def graph_id_by_name(self, name: str) -> str | None:
        """An active graph of the target project with this name, else None."""
        ...

    def skill_id(self, scope: str, name: str, version: str) -> str | None:
        """A skill visible to the target project, else None."""
        ...

    def mcp_server(self, name: str, version: str) -> tuple[str, str] | None:
        """(server_id, trust_class) visible to the target project, else None."""
        ...


@dataclass
class MemberMapping:
    """Local member ids for the imported catalog row."""

    member_graph_ids: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    mcp_server_ids: list[str] = field(default_factory=list)
    embedding_model_ids: list[str] = field(default_factory=list)
    mcp_trust_classes: dict[str, str] = field(default_factory=dict)
    # Members with no local match, as labels (see map_members).
    unmapped: list[str] = field(default_factory=list)


def map_members(members: Any, resolver: MemberResolver) -> MemberMapping:
    """Map a manifest ``members`` block onto local ids.

    ``members`` is the block written by ``member_identities`` (keys
    ``graphs``, ``skills``, ``mcp_servers``, ``embedding_models``). When it
    is not a dict, return an empty ``MemberMapping()``. A missing or
    non-list key reads as [].

    Kinds are processed in this order: graphs, skills, mcp_servers,
    embedding_models; entries of one kind in list order:

    - graphs: an entry whose ``default`` is exactly ``True`` maps to
      ``resolver.default_graph_id()``. Otherwise a non-empty str ``name``
      maps to ``resolver.graph_id_by_name(name)``; None adds the label
      ``"graph:<name>"`` to ``unmapped``.
    - skills: non-empty str ``name``; ``scope`` and ``version`` read with
      ``str(entry.get(k) or "")``. ``resolver.skill_id(scope, name,
      version)``; None adds ``"skill:<scope>:<name>@<version>"``.
    - mcp_servers: non-empty str ``name``; ``version`` as above.
      ``resolver.mcp_server(name, version)`` gives ``(server_id,
      trust_class)``: append server_id and set
      ``mcp_trust_classes[server_id] = trust_class``; None adds
      ``"mcp:<name>@<version>"``. Any ``trust_class`` key in the entry is
      ignored.
    - embedding_models: every non-empty str kept verbatim; an empty str
      is skipped with no label.
    - An entry that is not a dict, or has no non-empty str ``name`` (except
      a default graph), adds ``"malformed:<kind>"`` where kind is
      ``graph``, ``skill`` or ``mcp`` (a non-str embedding entry adds
      ``"malformed:embedding"``).

    Each id list keeps first-seen order with no repeats; ``unmapped``
    keeps every label in order (repeats allowed).
    """
    raise NotImplementedError("s-d39a-03")
