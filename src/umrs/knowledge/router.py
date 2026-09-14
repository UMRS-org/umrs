"""Knowledge compilation router — Phase 1 of the knowledge persistence layer.

The router turns the canonical :class:`~umrs.knowledge.ir.KnowledgeIR` records
(Phase 0) into *derived indexes*. Each derived index is a :class:`KnowledgeSink`:
it declares which IRs it accepts and knows how to (re)build itself from a set of
IRs.

Design invariants:
  - **Canonical, not derived.** Truth lives in ``KnowledgeIR``. Every sink is a
    pure function of the canonical IRs, so any index can be regenerated at will
    (e.g. after the extractor or the ``KnowledgeLinker`` improves) without
    re-reading source bytes.
  - **Batch compile.** Some indexes — the knowledge graph in particular —
    require cross-artifact context, so a sink compiles over a *set* of IRs
    rather than one at a time.
  - **Sink isolation.** A failing sink is logged and skipped; it must not abort
    the compilation of the other sinks.
  - **Boundary-safe.** Foundation layer only; no ``umrs.api`` / ``umrs.server``
    imports.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .ir import KnowledgeIR
from .link import KnowledgeLinker
from .persistence import (
    knowledge_ir_dir,
    read_all_local_knowledge_ir,
    write_local_knowledge_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class CompileResult:
    """Outcome of compiling one sink over a set of IRs."""

    sink: str
    accepted: int
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink": self.sink,
            "accepted": self.accepted,
            "detail": self.detail,
            "error": self.error,
        }


@runtime_checkable
class KnowledgeSink(Protocol):
    """A derived index that can be rebuilt from canonical ``KnowledgeIR`` records."""

    name: str

    def accepts(self, ir: KnowledgeIR) -> bool:
        """Return True if this sink indexes *ir*."""
        ...

    def compile(self, irs: Sequence[KnowledgeIR]) -> CompileResult:
        """(Re)build the index from the accepted IRs."""
        ...


class KnowledgeRouter:
    """Dispatches canonical IRs to the registered derived-index sinks."""

    def __init__(self, sinks: Sequence[KnowledgeSink] | None = None) -> None:
        self._sinks: list[KnowledgeSink] = list(sinks or [])

    def register(self, sink: KnowledgeSink) -> KnowledgeRouter:
        """Register a sink; returns self for chaining."""
        self._sinks.append(sink)
        return self

    @property
    def sinks(self) -> tuple[KnowledgeSink, ...]:
        return tuple(self._sinks)

    def regenerate(self, irs: Sequence[KnowledgeIR]) -> list[CompileResult]:
        """Rebuild every registered index from the canonical IRs.

        Each sink receives only the IRs it accepts. A sink that raises is logged
        and recorded as a failed :class:`CompileResult`; other sinks still run.
        """
        materialized = list(irs)
        results: list[CompileResult] = []
        for sink in self._sinks:
            accepted = [ir for ir in materialized if sink.accepts(ir)]
            name = getattr(sink, "name", sink.__class__.__name__)
            try:
                results.append(sink.compile(accepted))
            except Exception as exc:  # noqa: BLE001 — sink isolation is intentional
                logger.warning("Knowledge sink %s failed to compile: %s", name, exc)
                results.append(
                    CompileResult(sink=name, accepted=len(accepted), error=str(exc))
                )
        return results


# ── Sinks ────────────────────────────────────────────────────────────────────


def ir_to_artifact_dict(ir: KnowledgeIR) -> dict[str, Any]:
    """Reconstruct the ``KnowledgeLinker`` artifact record from a canonical IR.

    The linker consumes ``artifact_id`` / ``artifact_type`` / ``filename`` /
    ``extracted_knowledge`` — all recoverable from the immutable IR, which is
    what makes the graph a *regenerable* derived index.
    """
    extracted = ir.extracted or {}
    return {
        "artifact_id": ir.artifact_id,
        "artifact_type": ir.artifact_type,
        "filename": extracted.get("filename", ""),
        "extracted_knowledge": extracted,
        "metadata": {
            "ir_id": ir.ir_id,
            "content_hash": ir.content_hash,
            "extractor_version": ir.extractor_version,
        },
    }


class LocalGraphSink:
    """Rebuilds the local knowledge-graph index (``knowledge_graph.json``).

    Demonstrates the core Phase 1 property: the knowledge graph — the least
    reliable, heuristic output — is a *derived index* fully regenerable from the
    canonical IRs. Rewritten (overwrite-latest) on every compile.
    """

    name = "local_graph"

    def __init__(
        self,
        storage_path: str,
        linker: KnowledgeLinker | None = None,
    ) -> None:
        self._storage_path = storage_path
        self._linker = linker or KnowledgeLinker()

    def accepts(self, ir: KnowledgeIR) -> bool:
        return True

    def compile(self, irs: Sequence[KnowledgeIR]) -> CompileResult:
        artifacts = [ir_to_artifact_dict(ir) for ir in irs]
        graph = self._linker.link(artifacts)
        graph_dict = graph.to_dict()
        path = write_local_knowledge_graph(self._storage_path, graph_dict)
        return CompileResult(
            sink=self.name,
            accepted=len(artifacts),
            detail={
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "path": str(path),
            },
        )


# ── Convenience: local regeneration ──────────────────────────────────────────


def default_local_router(storage_path: str) -> KnowledgeRouter:
    """Router with the default local sinks for a project storage directory."""
    return KnowledgeRouter([LocalGraphSink(storage_path)])


def regenerate_local_indexes(storage_path: str) -> list[CompileResult]:
    """Rebuild all local derived indexes from the canonical IRs on disk.

    Reads every ``knowledge_ir/<ir_id>.json`` and recompiles the derived
    indexes — proving the indexes are disposable and regenerable from the
    canonical records alone.
    """
    irs = read_all_local_knowledge_ir(knowledge_ir_dir(storage_path))
    return default_local_router(storage_path).regenerate(irs)
