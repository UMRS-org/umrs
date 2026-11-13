"""Research primitives — R4's native domain model.

R4 organizes scientific research the way Windows organizes applications,
Kubernetes organizes clusters, GitHub organizes software development.
These primitives are the native objects of the Research Studio.

Every primitive carries ``provenance`` as a first-class field — a directed
graph of ancestors. Nothing appears without showing where it came from.

All types are stdlib-only ``@dataclass`` — no Pydantic, no third-party deps.
This package mirrors the lightweight DTO portion as the client-side contract;
UMRS keeps its own copies as the implementation owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .provenance import ArtifactRef

__all__ = [
    # Status enums
    "PrimitiveKind",
    "HypothesisStatus",
    "ExperimentStatus",
    "WorkflowStatus",
    "PublicationStatus",
    "AgentStatus",
    "ConversationStatus",
    "SimulationStatus",
    "ComputeResourceStatus",
    "KnowledgeStatus",
    "DatasetStatus",
    "NotebookStatus",
    "ResearchGoalStatus",
    "ResearchProjectStatus",
    # Provenance
    "ProvenanceNode",
    "ProvenanceGraph",
    # Primitives
    "ResearchProject",
    "ResearchGoal",
    "Hypothesis",
    "Experiment",
    "Knowledge",
    "Dataset",
    "Publication",
    "Notebook",
    "Simulation",
    "ComputeResource",
    "Agent",
    "Conversation",
    # Research-loop provenance (RC1 Phase 0.5)
    "EvidenceSource",
    "EvidenceRef",
    "Finding",
    "LoopIterationRecord",
    "ResearchRunStatus",
    "ResearchRunHandle",
    # Research sessions (ADR-025)
    "ResearchSessionStatus",
    "ResearchSession",
    "ResearchSessionLoopLink",
]


# ---------------------------------------------------------------------------
# Primitive kinds — every primitive has one
# ---------------------------------------------------------------------------


class PrimitiveKind(str, Enum):
    """Kind discriminator for research primitives."""

    RESEARCH_PROJECT = "research_project"
    RESEARCH_GOAL = "research_goal"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    WORKFLOW = "workflow"
    KNOWLEDGE = "knowledge"
    DATASET = "dataset"
    PUBLICATION = "publication"
    NOTEBOOK = "notebook"
    SIMULATION = "simulation"
    COMPUTE_RESOURCE = "compute_resource"
    AGENT = "agent"
    CONVERSATION = "conversation"
    CATALOG = "catalog"


# ---------------------------------------------------------------------------
# Status enums per primitive
# ---------------------------------------------------------------------------


class ResearchProjectStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ResearchGoalStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PublicationStatus(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ComputeResourceStatus(str, Enum):
    REQUESTED = "requested"
    ALLOCATED = "allocated"
    RUNNING = "running"
    RELEASED = "released"
    FAILED = "failed"


class KnowledgeStatus(str, Enum):
    EMPTY = "empty"
    INDEXING = "indexing"
    READY = "ready"
    STALE = "stale"
    ERROR = "error"


class DatasetStatus(str, Enum):
    REGISTERED = "registered"
    UPLOADING = "uploading"
    READY = "ready"
    PROCESSING = "processing"
    ARCHIVED = "archived"


class NotebookStatus(str, Enum):
    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Provenance — first-class on every primitive
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceNode:
    """A single node in a provenance graph.

    References another primitive by id and kind, with the relationship
    that links them.
    """

    primitive_id: str
    primitive_kind: PrimitiveKind
    relation: str  # "produced_by" | "consumes" | "derived_from" | "cites" | ...
    label: str = ""  # human-readable label for this edge

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "primitive_kind": self.primitive_kind.value,
            "relation": self.relation,
            "label": self.label,
        }


@dataclass
class ProvenanceGraph:
    """A provenance graph attached to a primitive.

    This is the directed graph of ancestors — what produced this artifact,
    what it consumed, what it derives from. It is the lineage that makes
    research reproducible.
    """

    nodes: list[ProvenanceNode] = field(default_factory=list)

    def add(self, node: ProvenanceNode) -> None:
        self.nodes.append(node)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProvenanceGraph:
        if not data:
            return cls()
        nodes = []
        for n in data.get("nodes", []):
            kind_str = n.get("primitive_kind", "")
            try:
                kind = PrimitiveKind(kind_str)
            except ValueError:
                kind = PrimitiveKind.RESEARCH_PROJECT  # fallback
            nodes.append(
                ProvenanceNode(
                    primitive_id=n.get("primitive_id", ""),
                    primitive_kind=kind,
                    relation=n.get("relation", ""),
                    label=n.get("label", ""),
                )
            )
        return cls(nodes=nodes)


# ---------------------------------------------------------------------------
# Base primitive — every primitive has id, kind, status, provenance, timestamps
# ---------------------------------------------------------------------------


def _empty_provenance() -> ProvenanceGraph:
    return ProvenanceGraph()


# ---------------------------------------------------------------------------
# ResearchProject — the root workspace
# ---------------------------------------------------------------------------


@dataclass
class ResearchProject:
    """The root workspace in R4.

    A Research Project contains research goals, hypotheses, experiments,
    workflows, datasets, publications, notebooks, simulations, compute
    resources, agents, and conversations.
    """

    id: str
    name: str
    status: ResearchProjectStatus = ResearchProjectStatus.PLANNING
    description: str = ""
    owner_id: str = ""
    # Child primitive ids (all children of the root)
    goal_ids: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    workflow_ids: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    publication_ids: list[str] = field(default_factory=list)
    notebook_ids: list[str] = field(default_factory=list)
    simulation_ids: list[str] = field(default_factory=list)
    compute_resource_ids: list[str] = field(default_factory=list)
    agent_ids: list[str] = field(default_factory=list)
    conversation_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# ResearchGoal — what we're trying to achieve
# ---------------------------------------------------------------------------


@dataclass
class ResearchGoal:
    """What we're trying to achieve in a Research Project."""

    id: str
    project_id: str
    statement: str
    status: ResearchGoalStatus = ResearchGoalStatus.PROPOSED
    # Linked primitives
    hypothesis_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    publication_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    priority: str = "medium"  # low | medium | high
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Hypothesis — testable claim
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A testable claim within a Research Project."""

    id: str
    project_id: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    # Evidence
    confidence: float = 0.0  # 0.0–1.0
    evidence_for: list[str] = field(default_factory=list)  # experiment ids
    evidence_against: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Experiment — procedure to test hypothesis
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """A procedure to test a Hypothesis.

    An Experiment runs one or more Workflows on Datasets using ComputeResources,
    producing results that become evidence for or against the Hypothesis.
    """

    id: str
    project_id: str
    hypothesis_id: str
    protocol: dict[str, Any] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PENDING
    # Linked primitives
    workflow_ids: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    compute_resource_ids: list[str] = field(default_factory=list)
    simulation_ids: list[str] = field(default_factory=list)
    agent_ids: list[str] = field(default_factory=list)
    # Results
    results: dict[str, Any] | None = None
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    submitted_at: str | None = None
    completed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Knowledge — accumulated understanding (RAG index)
# ---------------------------------------------------------------------------


@dataclass
class Knowledge:
    """Accumulated understanding as a RAG index.

    Knowledge grows with every Literature survey, Experiment result, and
    Publication. Today's experiment is tomorrow's reference.
    """

    id: str
    project_id: str
    name: str
    status: KnowledgeStatus = KnowledgeStatus.EMPTY
    # RAG index
    index_uri: str = ""  # s3://…/indexes/<knowledge_id>/
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str = ""
    # Linked primitives (what contributes to this knowledge)
    publication_ids: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Dataset — input data
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """Input data for experiments and simulations."""

    id: str
    project_id: str
    name: str
    status: DatasetStatus = DatasetStatus.REGISTERED
    # Storage
    s3_uri: str = ""  # s3://…/datasets/<dataset_id>/
    size_bytes: int | None = None
    sha256: str | None = None
    media_type: str = ""  # MIME type
    # Schema
    schema: dict[str, Any] | None = None
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Publication — output artifact
# ---------------------------------------------------------------------------


@dataclass
class Publication:
    """Output artifact — a publication with provenance."""

    id: str
    project_id: str
    title: str
    status: PublicationStatus = PublicationStatus.DRAFT
    # Content
    content_uri: str = ""  # s3://…/publications/<publication_id>/
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    # Linked primitives (the provenance of this publication)
    experiment_ids: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)
    # Scientific catalogs whose versions pinned this draft (D-45 join)
    catalog_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    doi: str | None = None
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Notebook — interactive computation
# ---------------------------------------------------------------------------


@dataclass
class Notebook:
    """Interactive computation notebook.

    Notebooks are artifacts within R4, not the workspace itself.
    """

    id: str
    project_id: str
    name: str
    status: NotebookStatus = NotebookStatus.DRAFT
    # Content
    content_uri: str = ""  # s3://…/notebooks/<notebook_id>/
    cell_count: int = 0
    # Linked primitives
    dataset_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Simulation — computational experiment
# ---------------------------------------------------------------------------


@dataclass
class Simulation:
    """A computational experiment (runs on HPC)."""

    id: str
    project_id: str
    name: str
    status: SimulationStatus = SimulationStatus.PENDING
    # Execution
    workflow_id: str | None = None
    compute_resource_id: str | None = None
    # Inputs
    dataset_ids: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    # Outputs
    results: dict[str, Any] | None = None
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    submitted_at: str | None = None
    completed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# ComputeResource — HPC allocation (Slurm partition, GPUs)
# ---------------------------------------------------------------------------


@dataclass
class ComputeResource:
    """HPC allocation — Slurm partition, GPUs, memory.

    ComputeResources are first-class citizens in R4, not hidden behind
    a "run" button.
    """

    id: str
    project_id: str
    name: str
    status: ComputeResourceStatus = ComputeResourceStatus.REQUESTED
    # Slurm
    slurm_partition: str = ""
    slurm_job_id: str | None = None
    # Resources
    gpus: int = 0
    cpu_cores: int = 0
    memory_mb: int = 0
    wall_time_hours: float = 0.0
    # Linked primitives
    workflow_ids: list[str] = field(default_factory=list)
    simulation_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Agent — AI collaborator (TeamupSession member)
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """AI collaborator — a TeamupSession member.

    Agents are visible collaborators in R4, not hidden implementation details.
    Their work, reasoning, and outputs are visible.
    """

    id: str
    project_id: str
    name: str
    role: str = ""  # "supervisor" | "planning" | "literature" | "hpc" | ...
    status: AgentStatus = AgentStatus.IDLE
    # Current task
    current_task: str | None = None
    reasoning: str | None = None  # latest reasoning trace
    # Tools and memory
    tools: list[str] = field(default_factory=list)
    memory_size: int = 0
    # Linked primitives
    conversation_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    model: str = ""
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Conversation — one interaction modality (UMRS Session)
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    """One interaction modality — a UMRS Session.

    Conversation is a child artifact used for human-agent interaction.
    It is one interaction method among many in R4.
    """

    id: str
    project_id: str
    agent_id: str | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    # Messages (lightweight refs; full content in UMRS)
    message_count: int = 0
    last_message_at: str | None = None
    # Provenance
    provenance: ProvenanceGraph = field(default_factory=_empty_provenance)
    # Metadata
    title: str = ""
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Research-loop provenance DTOs (RC1 Phase 0.5)
# ---------------------------------------------------------------------------


class EvidenceSource(str, Enum):
    """Where an evidence excerpt was retrieved from."""

    GRAPH = "graph"
    CATALOG = "catalog"
    WORKFLOW_OUTPUT = "workflow_output"


class ResearchRunStatus(str, Enum):
    """Lifecycle status of an autonomous research run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvidenceRef:
    """A single piece of evidence backing a finding.

    ``artifact`` carries URI/hash via ``ArtifactRef``. ``artifact_id`` and
    ``s3_key`` are source-conditional: required for catalog / workflow_output,
    absent for pure graph relations.
    """

    source: EvidenceSource
    excerpt: str = ""
    artifact: ArtifactRef | None = None
    artifact_id: str | None = None
    s3_key: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, str):
            self.source = EvidenceSource(self.source)
        if (
            self.source in (EvidenceSource.CATALOG, EvidenceSource.WORKFLOW_OUTPUT)
            and not self.artifact_id
            and not (self.artifact and self.artifact.uri)
        ):
            raise ValueError(
                f"EvidenceRef source={self.source.value} requires artifact_id or artifact.uri"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value if isinstance(self.source, EvidenceSource) else self.source,
            "excerpt": self.excerpt,
            "artifact": (
                {
                    "uri": self.artifact.uri,
                    "media_type": self.artifact.media_type,
                    "size_bytes": self.artifact.size_bytes,
                    "sha256": self.artifact.sha256,
                }
                if self.artifact is not None
                else None
            ),
            "artifact_id": self.artifact_id,
            "s3_key": self.s3_key,
        }


@dataclass
class Finding:
    """A research finding with provenance back to evidence and execution."""

    finding_id: str
    statement: str
    hypothesis_id: str = ""
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    workflow_run_id: str = ""
    durable_job_id: str | None = None
    confidence: float = 0.0
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "workflow_run_id": self.workflow_run_id,
            "durable_job_id": self.durable_job_id,
            "confidence": self.confidence,
            "simulated": self.simulated,
        }


@dataclass
class LoopIterationRecord:
    """Persisted record of one research-loop iteration.

    Matches the engine's ``LoopIteration`` public shape (including analysis
    fields that live in the SQLite ``payload_json`` for run-store rows).
    """

    iteration_id: str
    loop_id: str = ""
    iteration_number: int = 0
    phase: str = ""
    hypothesis_id: str = ""
    hypothesis: str = ""
    experiment_id: str = ""
    workflow_run_id: str = ""
    durable_job_id: str | None = None
    retrieval_record_ids: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    # Artifact ids produced by this iteration (finding artifacts minted into
    # the knowledge store, workflow outputs) — ADR-036 D4 lineage anchor.
    artifact_ids: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    evaluation: str = ""
    status: str = "pending"
    simulated: bool = False
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "loop_id": self.loop_id,
            "iteration_number": self.iteration_number,
            "phase": self.phase,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis": self.hypothesis,
            "experiment_id": self.experiment_id,
            "workflow_run_id": self.workflow_run_id,
            "durable_job_id": self.durable_job_id,
            "retrieval_record_ids": list(self.retrieval_record_ids),
            "findings": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.findings],
            "artifact_ids": list(self.artifact_ids),
            "results": dict(self.results),
            "analysis": self.analysis,
            "evaluation": self.evaluation,
            "status": self.status,
            "simulated": self.simulated,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ResearchRunHandle:
    """Async identity for a research run (public contract will expose this)."""

    run_id: str
    project_id: str
    status: ResearchRunStatus = ResearchRunStatus.PENDING
    simulated: bool = False
    loop_id: str = ""
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ResearchRunStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "status": self.status.value
            if isinstance(self.status, ResearchRunStatus)
            else self.status,
            "simulated": self.simulated,
            "loop_id": self.loop_id,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Research sessions (ADR-025) — durable investigation identity owned by UMRS
# ---------------------------------------------------------------------------


class ResearchSessionStatus(str, Enum):
    """Lifecycle of a durable research session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ResearchSessionLoopLink:
    """One loop attached to a session."""

    session_id: str
    loop_id: str
    venue: str = "umrs"
    loop_role: str = "auto"
    attached_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "loop_id": self.loop_id,
            "venue": self.venue,
            "loop_role": self.loop_role,
            "attached_at": self.attached_at,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ResearchSessionLoopLink:
        data = data or {}
        return cls(
            session_id=str(data.get("session_id", "")),
            loop_id=str(data.get("loop_id", "")),
            venue=str(data.get("venue", "umrs")),
            loop_role=str(data.get("loop_role", "auto")),
            attached_at=str(data.get("attached_at", "")),
        )


@dataclass
class ResearchSession:
    """Durable umbrella over one thread of investigation (ADR-025 D1)."""

    session_id: str
    project_id: str
    title: str = ""
    research_goal: str = ""
    status: ResearchSessionStatus = ResearchSessionStatus.ACTIVE
    authoritative_runtime: str = "umrs"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    archived_at: str | None = None
    loops: list[ResearchSessionLoopLink] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ResearchSessionStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "title": self.title,
            "research_goal": self.research_goal,
            "status": (
                self.status.value if isinstance(self.status, ResearchSessionStatus) else self.status
            ),
            "authoritative_runtime": self.authoritative_runtime,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
            "loops": [link.to_dict() for link in self.loops],
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ResearchSession:
        data = data or {}
        return cls(
            session_id=str(data.get("session_id", "")),
            project_id=str(data.get("project_id", "")),
            title=str(data.get("title", "")),
            research_goal=str(data.get("research_goal", "")),
            status=data.get("status", ResearchSessionStatus.ACTIVE),
            authoritative_runtime=str(data.get("authoritative_runtime", "umrs")),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            archived_at=data.get("archived_at"),
            loops=[ResearchSessionLoopLink.from_mapping(row) for row in data.get("loops", [])],
        )
