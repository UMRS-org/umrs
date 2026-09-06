"""Loop findings → knowledge assets with provenance (ADR-036 D4, ADR-027 P4).

Before WP-05 a research-loop finding lived only inside its loop record
(``research_loop_iterations.findings`` JSONB): workflow artifacts entered the
knowledge store with PRODUCED_BY provenance, findings never did (G-P1-1).
This module mints each *real* finding of a completed iteration as one
knowledge artifact plus the graph nodes/edges that carry its lineage:

    finding  --produced_by-->  loop:<loop_id>          (provenance node)
    finding  --derived_from--> <evidence artifact id>  (one edge per evidence ref)

and records the minted artifact ids on the iteration (``artifact_ids``) so
the asset index derives ``loop → artifact`` (fanout.py) without a new model.

Honesty rules:
- Simulated findings are NEVER minted (a fabricated finding in the KG is
  exactly the failure the engine's evidence guard exists to prevent).
- Minting is idempotent: artifact id = finding id; already-recorded ids and
  hash-deduped rows are skipped, never duplicated.
- The sink is injected. ``KnowledgeFindingSink`` writes through the asyncpg
  ``KnowledgeStore`` from the sync driver thread (own event loop, own pool,
  like the ingest chain's provenance writer); ``InMemoryFindingSink`` backs
  tests and DB-less processes. Absence of a sink is reported on the loop
  (``metadata.finding_ingest.sink == "none"``), not swallowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .loop import LoopIteration, ResearchLoop

logger = logging.getLogger(__name__)

FINDING_ARTIFACT_KIND = "document"  # knowledge_artifacts.kind CHECK vocabulary
FINDING_MIME = "application/vnd.r4.finding+json"
RELATION_PRODUCED_BY = "produced_by"
RELATION_DERIVED_FROM = "derived_from"


class FindingSink(Protocol):
    """Where minted findings land (knowledge store or an in-memory double)."""

    @property
    def durability(self) -> str: ...

    def mint_artifact(
        self,
        *,
        artifact_id: str,
        content_hash: str,
        project_id: str,
        source_ref: str,
        size_bytes: int,
        metadata: dict[str, Any],
    ) -> bool: ...

    def link(
        self,
        *,
        project_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None: ...


@dataclass
class IngestOutcome:
    """What one ``mint_iteration`` call did (also folded into loop metadata)."""

    minted: list[str] = field(default_factory=list)
    skipped_simulated: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minted": list(self.minted),
            "skipped_simulated": self.skipped_simulated,
            "skipped_existing": self.skipped_existing,
            "errors": list(self.errors),
        }


# ── sinks ───────────────────────────────────────────────────────────────────


class InMemoryFindingSink:
    """Test / DB-less double: keeps artifacts, nodes and edges in dicts."""

    durability = "process"

    def __init__(self) -> None:
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def mint_artifact(self, **fields: Any) -> bool:
        artifact_id = fields["artifact_id"]
        if artifact_id in self.artifacts:
            return False
        self.artifacts[artifact_id] = dict(fields)
        return True

    def link(
        self,
        *,
        project_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        for node in nodes:
            self.nodes[node["node_id"]] = {**node, "project_id": project_id}
        for edge in edges:
            self.edges[edge["edge_id"]] = {**edge, "project_id": project_id}


class KnowledgeFindingSink:
    """Production sink over the asyncpg ``KnowledgeStore`` (migration 0055).

    Called from the sync driver thread: each call runs its coroutine on a
    helper thread with its own event loop and a fresh codec-aware pool —
    the same discipline the ingest chain's provenance writer uses
    (asyncpg pools are not loop-portable).
    """

    durability = "postgres"

    def mint_artifact(self, **fields: Any) -> bool:
        return bool(self._run(self._mint(fields), "finding-mint"))

    def link(
        self,
        *,
        project_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        self._run(self._link(project_id, nodes, edges), "finding-link")

    @staticmethod
    def _run(coro: Any, label: str) -> Any:
        from umrs.jobs.knowledge_adapters import _run_coroutine_sync

        return _run_coroutine_sync(coro, label)

    async def _mint(self, fields: dict[str, Any]) -> bool:
        from umrs.jobs.knowledge_adapters import _open_pool

        from .store import KnowledgeStore

        async with _open_pool() as pool:
            if pool is None:
                raise RuntimeError("knowledge pool unavailable")
            return await KnowledgeStore(pool).insert_artifact(
                artifact_id=fields["artifact_id"],
                content_hash=fields["content_hash"],
                kind=FINDING_ARTIFACT_KIND,
                project_id=fields["project_id"],
                source_ref=fields["source_ref"],
                size_bytes=fields["size_bytes"],
                mime_type=FINDING_MIME,
                metadata=fields["metadata"],
            )

    async def _link(
        self, project_id: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        from umrs.jobs.knowledge_adapters import _open_pool

        from .store import KnowledgeEdge, KnowledgeNode, KnowledgeStore

        async with _open_pool() as pool:
            if pool is None:
                raise RuntimeError("knowledge pool unavailable")
            store = KnowledgeStore(pool)
            for node in nodes:
                await store.upsert_node(KnowledgeNode(project_id=project_id, **node))
            for edge in edges:
                await store.upsert_edge(KnowledgeEdge(project_id=project_id, **edge))


# ── minting ─────────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finding_dict(finding: Any) -> dict[str, Any]:
    """Findings arrive as ``Finding`` dataclasses (engine) or dicts (store)."""
    if isinstance(finding, dict):
        return dict(finding)
    if hasattr(finding, "to_dict"):
        return dict(finding.to_dict())
    return {"finding_id": str(finding), "statement": str(finding)}


def _evidence_artifact_ids(finding: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for ref in finding.get("evidence_refs") or []:
        aid = (
            ref.get("artifact_id")
            if isinstance(ref, dict)
            else getattr(ref, "artifact_id", None)
        )
        if aid and str(aid) not in ids:
            ids.append(str(aid))
    return ids


def workflow_artifact_ids(iteration: LoopIteration) -> list[str]:
    """Artifact ids the iteration's workflow run reported (``results.artifacts``)."""
    ids: list[str] = []
    for item in (iteration.results or {}).get("artifacts") or []:
        aid = (
            item.get("artifact_id") or item.get("id")
            if isinstance(item, dict)
            else item
        )
        if aid and str(aid) not in ids:
            ids.append(str(aid))
    return ids


def _finding_payload(
    loop: ResearchLoop, iteration: LoopIteration, finding: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    content = {
        "finding_id": finding.get("finding_id", ""),
        "statement": finding.get("statement", ""),
        "hypothesis_id": finding.get("hypothesis_id", ""),
        "hypothesis": iteration.hypothesis,
        "confidence": finding.get("confidence", 0.0),
        "evidence_refs": finding.get("evidence_refs") or [],
        "workflow_run_id": finding.get("workflow_run_id", ""),
        "durable_job_id": finding.get("durable_job_id"),
        "source_loop": loop.loop_id,
        "iteration": iteration.iteration_number,
        "iteration_id": iteration.iteration_id,
        "analysis": iteration.analysis,
        "evaluation": iteration.evaluation,
    }
    raw = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    metadata = {
        "asset_kind": "finding",
        "source_loop": loop.loop_id,
        "iteration": iteration.iteration_number,
        "iteration_id": iteration.iteration_id,
        "hypothesis_id": finding.get("hypothesis_id", ""),
        "workflow_run_id": finding.get("workflow_run_id", ""),
        "durable_job_id": finding.get("durable_job_id") or "",
        "hosting_job_id": str(loop.metadata.get("durable_job_id") or ""),
        "confidence": finding.get("confidence", 0.0),
        "evidence_artifact_ids": _evidence_artifact_ids(finding),
        "finding": content,
    }
    return raw, metadata


def _edge_id(source: str, relation: str, target: str) -> str:
    return "edge-" + _sha256(f"{source}|{relation}|{target}".encode())[:24]


def _lineage(
    loop: ResearchLoop, artifact_id: str, finding: dict[str, Any], content_hash: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Nodes + edges anchoring one finding to its loop and its evidence.

    Node ids follow the ingest chain: an artifact's node id IS its artifact
    id (edge endpoints are FKs into ``knowledge_nodes``); ``node_type`` stays
    inside the DB CHECK vocabulary (semantic role lives in the label).
    """
    loop_node = f"loop:{loop.loop_id}"
    nodes: list[dict[str, Any]] = [
        {
            "node_id": artifact_id,
            "label": f"finding:{artifact_id}",
            "node_type": "concept",
            "content": {
                "statement": finding.get("statement", ""),
                "hypothesis_id": finding.get("hypothesis_id", ""),
                "confidence": finding.get("confidence", 0.0),
                "source_loop": loop.loop_id,
            },
            "content_hash": content_hash,
            "source_ir": None,
        },
        {
            "node_id": loop_node,
            "label": loop_node,
            "node_type": "entity",
            "content": {
                "loop_id": loop.loop_id,
                "project_id": loop.project_id,
                "research_goal": loop.research_goal,
                "hosting_job_id": str(loop.metadata.get("durable_job_id") or ""),
            },
            "content_hash": _sha256(f"loop-node-{loop.loop_id}".encode()),
            "source_ir": None,
        },
    ]
    edges: list[dict[str, Any]] = [
        {
            "edge_id": _edge_id(artifact_id, RELATION_PRODUCED_BY, loop_node),
            "source_id": artifact_id,
            "target_id": loop_node,
            "relation_type": RELATION_PRODUCED_BY,
            "content": {"iteration": finding.get("iteration", 0)},
        }
    ]
    for evidence_id in _evidence_artifact_ids(finding):
        nodes.append(
            {
                "node_id": evidence_id,
                "label": f"artifact:{evidence_id}",
                "node_type": "entity",
                "content": {"role": "evidence"},
                "content_hash": _sha256(f"artifact-node-{evidence_id}".encode()),
                "source_ir": None,
            }
        )
        edges.append(
            {
                "edge_id": _edge_id(artifact_id, RELATION_DERIVED_FROM, evidence_id),
                "source_id": artifact_id,
                "target_id": evidence_id,
                "relation_type": RELATION_DERIVED_FROM,
                "content": {},
            }
        )
    return nodes, edges


def _mint_one(
    loop: ResearchLoop,
    iteration: LoopIteration,
    finding: dict[str, Any],
    sink: FindingSink,
    outcome: IngestOutcome,
) -> None:
    artifact_id = str(finding.get("finding_id") or "")
    if not artifact_id:
        outcome.errors.append("finding without finding_id skipped")
        return
    if finding.get("simulated"):
        outcome.skipped_simulated += 1
        return
    if artifact_id in iteration.artifact_ids:
        outcome.skipped_existing += 1
        return
    raw, metadata = _finding_payload(loop, iteration, finding)
    content_hash = _sha256(raw)
    sink.mint_artifact(
        artifact_id=artifact_id,
        content_hash=content_hash,
        project_id=loop.project_id,
        source_ref=f"loop:{loop.loop_id}#iteration-{iteration.iteration_number}",
        size_bytes=len(raw),
        metadata=metadata,
    )
    nodes, edges = _lineage(loop, artifact_id, finding, content_hash)
    sink.link(project_id=loop.project_id, nodes=nodes, edges=edges)
    iteration.artifact_ids.append(artifact_id)
    outcome.minted.append(artifact_id)


def mint_iteration(
    loop: ResearchLoop, iteration: LoopIteration, *, sink: FindingSink | None
) -> IngestOutcome:
    """Mint a completed iteration's real findings; record artifact ids.

    Best-effort per finding (one failure never blocks the others); the
    outcome is returned AND folded into ``loop.metadata["finding_ingest"]``
    so the loop record says what happened (sink kind, counts, errors).
    """
    outcome = IngestOutcome()
    if iteration.status != "completed":
        return outcome
    for aid in workflow_artifact_ids(iteration):
        if aid not in iteration.artifact_ids:
            iteration.artifact_ids.append(aid)
    if sink is None:
        _fold(loop, outcome, sink_kind="none")
        return outcome
    for finding in iteration.findings:
        try:
            _mint_one(loop, iteration, _finding_dict(finding), sink, outcome)
        except Exception as exc:  # noqa: BLE001 — per-finding degradation, reported
            logger.warning("finding mint failed for loop %s: %s", loop.loop_id, exc)
            outcome.errors.append(str(exc)[:200])
    _fold(loop, outcome, sink_kind=sink.durability)
    return outcome


def _fold(loop: ResearchLoop, outcome: IngestOutcome, *, sink_kind: str) -> None:
    prior = loop.metadata.get("finding_ingest") or {}
    loop.metadata["finding_ingest"] = {
        "sink": sink_kind,
        "minted": int(prior.get("minted", 0)) + len(outcome.minted),
        "skipped_simulated": int(prior.get("skipped_simulated", 0))
        + outcome.skipped_simulated,
        "errors": int(prior.get("errors", 0)) + len(outcome.errors),
        "last_error": outcome.errors[-1]
        if outcome.errors
        else prior.get("last_error", ""),
    }


__all__ = [
    "FINDING_ARTIFACT_KIND",
    "FindingSink",
    "InMemoryFindingSink",
    "IngestOutcome",
    "KnowledgeFindingSink",
    "mint_iteration",
    "workflow_artifact_ids",
]
