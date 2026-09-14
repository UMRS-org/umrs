"""Asset index — cross-subsystem read-model (Review F4, ADR-035 plan track 1).

The platform's durable truth stays with each subsystem's own store (jobs,
artifacts, models, MCP servers, research sessions, loops). This module does
NOT persist anything: it fans out over those stores and composes one
``AssetRow`` per (kind, id) with scope, status, lineage edges and
portability flags — the server model behind the console's /catalog view.

Foundation-module rules (mutation-safety): NO imports from umrs.api or
umrs.server in this package; stores arrive as constructor arguments.
"""

from .fanout import IndexWarning, build_asset_index
from .models import KINDS, AssetKind, AssetLineageEdge, AssetRow

__all__ = [
    "KINDS",
    "AssetKind",
    "AssetLineageEdge",
    "AssetRow",
    "IndexWarning",
    "build_asset_index",
]
