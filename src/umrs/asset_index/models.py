"""Asset index models (Review F4).

An AssetRow is a pointer INTO a subsystem store, not a copy:

    kind       subsystem family (artifact | model | mcp_server |
               research_session | loop | job)
    asset_id   the subsystem's own id — the row in its own table stays
               authoritative; the index never duplicates mutable state
    scope      canonical scope string (system / project:<id> /
               group:<id> / user:<id>) derived per subsystem
    status     lifecycle string as reported by the subsystem, verbatim
    lineage    edges derived from existing relations (artifact→parent,
               job→artifacts, session→loops) — see fanout.py
    portable   whether the asset may leave its scope via .kb export /
               bundle restore (INV-13); set per kind, not per row
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Kinds in v1 — each maps to exactly one subsystem store (fanout.py).
KINDS: tuple[str, ...] = (
    "artifact",
    "model",
    "mcp_server",
    "research_session",
    "loop",
    "job",
)


@dataclass(frozen=True)
class AssetKind:
    """Vocabulary entry for one indexed subsystem."""

    name: str
    store_hint: str  # doc-only: which subsystem backs rows of this kind

    def __post_init__(self) -> None:
        if self.name not in KINDS:
            raise ValueError(f"unknown asset kind: {self.name!r}")


@dataclass(frozen=True)
class AssetLineageEdge:
    """One derived lineage edge between two indexed assets."""

    from_kind: str
    from_id: str
    to_kind: str
    to_id: str
    relation: str  # derived_from | executed_by | contains | registered_by

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_kind": self.from_kind,
            "from_id": self.from_id,
            "to_kind": self.to_kind,
            "to_id": self.to_id,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class AssetRow:
    """One (kind, id) tuple from one subsystem, index-shaped."""

    kind: str
    asset_id: str
    scope: str
    status: str
    title: str = ""
    owner: str = ""
    lineage: tuple[AssetLineageEdge, ...] = field(default_factory=tuple)
    portable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown asset kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "asset_id": self.asset_id,
            "scope": self.scope,
            "status": self.status,
            "title": self.title,
            "owner": self.owner,
            "portable": self.portable,
            "metadata": dict(self.metadata),
            "lineage": [edge.to_dict() for edge in self.lineage],
        }
