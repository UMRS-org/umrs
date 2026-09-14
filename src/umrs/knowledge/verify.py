"""Scientific Knowledge Verification Chat.

Phase 6 of the Scientific Knowledge Build-up roadmap.

This module implements a dedicated "Knowledge Verification" mode where
the user can ask detailed questions about the project:

  - Explain my project.
  - What algorithms are implemented?
  - Which papers support this implementation?
  - How was Figure 5 generated?
  - Which notebook produced this dataset?
  - Which assumptions remain unverified?

Every answer must:
  1. Cite supporting evidence (uploaded files, notebook cells, code,
     experiment results, external papers, web search)
  2. Report confidence (0.0–1.0)
  3. Identify missing information
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .persistence import (
    read_local_extracted_knowledge,
    write_local_extracted_knowledge,
)

logger = logging.getLogger(__name__)


# ── Evidence Models ────────────────────────────────────────────────────────


class EvidenceType(str, Enum):
    """Types of evidence that can support an answer."""

    UPLOADED_FILE = "uploaded_file"
    NOTEBOOK_CELL = "notebook_cell"
    SOURCE_CODE = "source_code"
    EXPERIMENT_RESULT = "experiment_result"
    EXTERNAL_PAPER = "external_paper"
    WEB_SEARCH = "web_search"
    DATASET = "dataset"
    OBSERVATION = "observation"
    CATALOG_ENTRY = "catalog_entry"


@dataclass
class Evidence:
    """A piece of evidence supporting an answer."""

    evidence_type: EvidenceType
    source_id: str = ""  # artifact_id, paper_id, etc.
    source_name: str = ""  # filename, title, etc.
    content: str = ""  # the evidence text
    confidence: float = 0.0  # 0.0–1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type.value,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class VerificationAnswer:
    """An answer to a knowledge verification question.

    Every answer includes:
      1. The answer text
      2. Supporting evidence (citations)
      3. Confidence score (0.0–1.0)
      4. Missing information (gaps)
    """

    question: str
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0  # 0.0–1.0
    missing_info: list[str] = field(default_factory=list)
    answered: bool = True  # False if we couldn't answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "missing_info": self.missing_info,
            "answered": self.answered,
        }


# ── Knowledge Store Protocol ──────────────────────────────────────────────


class KnowledgeStore(Protocol):
    """Protocol for a knowledge store that the verifier can query."""

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """List all artifacts, optionally filtered by type."""

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get an artifact by id."""

    def get_extracted_knowledge(self, artifact_id: str) -> dict[str, Any] | None:
        """Get extracted knowledge for an artifact."""

    def search_artifacts(
        self, query: str, artifact_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Search artifacts by query."""

    def get_knowledge_graph(self) -> dict[str, Any]:
        """Get the knowledge graph."""


# ── In-Memory Knowledge Store ─────────────────────────────────────────────


@dataclass
class InMemoryKnowledgeStore:
    """Simple in-memory knowledge store for testing and small projects."""

    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    extracted_knowledge: dict[str, dict[str, Any]] = field(default_factory=dict)
    knowledge_graph: dict[str, Any] = field(default_factory=dict)
    persistence_dir: Path | None = None

    def add_artifact(self, artifact: dict[str, Any]) -> None:
        artifact_id = artifact["artifact_id"]
        self.artifacts[artifact_id] = artifact

    def add_extracted_knowledge(
        self,
        artifact_id: str,
        knowledge: dict[str, Any],
    ) -> None:
        self.extracted_knowledge[artifact_id] = knowledge
        if self.persistence_dir is not None:
            write_local_extracted_knowledge(
                self.persistence_dir,
                artifact_id,
                knowledge,
            )

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        artifacts = list(self.artifacts.values())
        if artifact_type:
            artifacts = [
                a for a in artifacts if a.get("artifact_type") == artifact_type
            ]
        return artifacts

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self.artifacts.get(artifact_id)

    def get_extracted_knowledge(self, artifact_id: str) -> dict[str, Any] | None:
        cached = self.extracted_knowledge.get(artifact_id)
        if cached is not None:
            return cached
        if self.persistence_dir is None:
            return None
        loaded = read_local_extracted_knowledge(self.persistence_dir, artifact_id)
        if loaded is not None:
            self.extracted_knowledge[artifact_id] = loaded
        return loaded

    def search_artifacts(
        self,
        query: str,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for artifact in self.list_artifacts(artifact_type):
            # Search in filename, summary, tags, keywords
            searchable = " ".join(
                [
                    artifact.get("filename", ""),
                    artifact.get("summary", ""),
                    artifact.get("title", ""),
                    " ".join(artifact.get("tags", [])),
                    " ".join(artifact.get("keywords", [])),
                ]
            ).lower()
            if query_lower in searchable:
                results.append(artifact)
        return results

    def get_knowledge_graph(self) -> dict[str, Any]:
        return self.knowledge_graph


# ── Knowledge Verifier ─────────────────────────────────────────────────────


class KnowledgeVerifier:
    """Answer knowledge verification questions with evidence.

    Usage:
        store = InMemoryKnowledgeStore()
        verifier = KnowledgeVerifier(store)
        answer = verifier.verify("What algorithms are implemented?")
    """

    # Question patterns mapped to handler methods
    QUESTION_PATTERNS = {
        "explain_project": [
            r"(?:explain|describe|overview)\s+(?:my\s+)?(?:project|research)",
            r"(?:what|tell me about)\s+(?:my\s+)?project",
            r"(?:summarize|summary of)\s+(?:my\s+)?(?:project|research)",
        ],
        "list_algorithms": [
            r"(?:what|which)\s+algorithms?\s+(?:are\s+)?(?:implemented|used|available)",
            r"(?:list|show)\s+algorithms?",
            r"algorithms?\s+in\s+(?:my\s+)?(?:project|code)",
        ],
        "list_papers": [
            r"(?:what|which)\s+papers?\s+(?:are|do\s+we\s+have)",
            r"(?:list|show)\s+papers?",
            r"(?:papers|publications)\s+in\s+(?:my\s+)?project",
        ],
        "list_code": [
            r"(?:what|which)\s+(?:code|implementations?)\s+(?:are|do\s+we\s+have)",
            r"(?:list|show)\s+(?:code|implementations?)",
            r"(?:what|which)\s+(?:code\s+)?files?\s+(?:do\s+we\s+have|are\s+there)",
            r"(?:code|source files?)\s+in\s+(?:my\s+)?project",
        ],
        "list_notebooks": [
            r"(?:what|which)\s+notebooks?\s+(?:are|do\s+we\s+have)",
            r"(?:list|show)\s+notebooks?",
            r"notebooks?\s+in\s+(?:my\s+)?project",
        ],
        "list_datasets": [
            r"(?:what|which)\s+datasets?\s+(?:are|do\s+we\s+have)",
            r"(?:list|show)\s+datasets?",
            r"datasets?\s+in\s+(?:my\s+)?project",
        ],
        "list_experiments": [
            r"(?:what|which)\s+experiments?\s+(?:are|have been|do\s+we\s+have)",
            r"(?:list|show)\s+experiments?",
            r"experiments?\s+in\s+(?:my\s+)?project",
        ],
        "list_assumptions": [
            r"(?:what|which)\s+assumptions?\s+(?:are|remain|do\s+we\s+have)",
            r"(?:list|show)\s+assumptions?",
            r"unverified\s+assumptions?",
        ],
        "list_conclusions": [
            r"(?:what|which)\s+conclusions?\s+(?:are|have been|do\s+we\s+have)",
            r"(?:list|show)\s+conclusions?",
            r"conclusions?\s+in\s+(?:my\s+)?project",
        ],
        "how_figure_generated": [
            r"(?:how|where)\s+(?:was|did)\s+(?:figure\s+)?fig(?:ure)?\s*(?:\d+)?\s+(?:generated|produced|created)",
            r"(?:how|where)\s+(?:was|did)\s+(?:figure\s+)?fig(?:ure)?\s*(?:\d+)?",
        ],
        "which_notebook_produced": [
            r"(?:which|what)\s+notebook\s+(?:produced|generated|created)\s+(?:this\s+)?(?:dataset|figure|result)",
            r"(?:which|what)\s+notebook\s+(?:produced|generated|created)",
        ],
        "which_papers_support": [
            r"(?:which|what)\s+papers?\s+(?:support|describe|back up|justify)\s+(?:this\s+)?(?:implementation|code|method|approach)",
            r"(?:which|what)\s+papers?\s+(?:support|describe)\s+",
        ],
    }

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def verify(self, question: str) -> VerificationAnswer:
        """Answer a knowledge verification question.

        Args:
            question: The user's question

        Returns:
            A ``VerificationAnswer`` with answer, evidence, confidence, and gaps.
        """
        question_lower = question.lower().strip()

        # Route to the appropriate handler
        handler = self._route_question(question_lower)

        if handler:
            return handler(question)
        else:
            return self._generic_answer(question)

    # ── Question Routing ──────────────────────────────────────────────────

    def _route_question(self, question: str) -> Any:
        """Route a question to the appropriate handler."""
        import re

        for handler_name, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return getattr(self, f"_handle_{handler_name}")

        return None

    # ── Answer Handlers ───────────────────────────────────────────────────

    def _handle_explain_project(self, question: str) -> VerificationAnswer:
        """Handle: 'Explain my project'."""
        artifacts = self.store.list_artifacts()
        evidence = []
        missing = []

        # Count artifacts by type
        by_type: dict[str, int] = {}
        for artifact in artifacts:
            atype = artifact.get("artifact_type", "unknown")
            by_type[atype] = by_type.get(atype, 0) + 1

        # Collect evidence from artifacts
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                summary = knowledge.get("summary", knowledge.get("objective", ""))
                if summary:
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.SOURCE_CODE
                            if artifact.get("artifact_type") == "code"
                            else EvidenceType.UPLOADED_FILE,
                            source_id=artifact_id,
                            source_name=artifact.get("filename", ""),
                            content=summary[:500],
                            confidence=0.7,
                        )
                    )

        # Build answer
        parts = []
        parts.append(f"Your project has {len(artifacts)} artifacts:")
        for atype, count in sorted(by_type.items()):
            parts.append(f"  - {count} {atype}(s)")

        if not artifacts:
            parts.append("No artifacts have been ingested yet.")
            missing.append("No artifacts ingested")

        answer_text = "\n".join(parts)

        # Calculate confidence
        confidence = min(1.0, len(evidence) / 10.0) if evidence else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence[:10],  # Limit to 10 evidence items
            confidence=confidence,
            missing_info=missing,
        )

    def _handle_list_algorithms(self, question: str) -> VerificationAnswer:
        """Handle: 'What algorithms are implemented?'"""
        code_artifacts = self.store.list_artifacts("code")
        evidence = []
        algorithms_found: list[str] = []

        for artifact in code_artifacts:
            artifact_id = artifact.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                algorithms = knowledge.get("algorithms", [])
                if algorithms:
                    algorithms_found.extend(algorithms)
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.SOURCE_CODE,
                            source_id=artifact_id,
                            source_name=artifact.get("filename", ""),
                            content=f"Algorithms: {', '.join(algorithms)}",
                            confidence=0.8,
                        )
                    )

        # Deduplicate
        algorithms_found = list(set(algorithms_found))

        if algorithms_found:
            answer_text = f"Found {len(algorithms_found)} algorithms:\n"
            answer_text += "\n".join(f"  - {alg}" for alg in sorted(algorithms_found))
        else:
            answer_text = "No algorithms found in the project."
            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.UPLOADED_FILE,
                    content="No code artifacts with algorithm annotations found",
                    confidence=0.0,
                )
            )

        confidence = min(1.0, len(algorithms_found) / 5.0) if algorithms_found else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[]
            if algorithms_found
            else ["No algorithms extracted from code"],
        )

    def _handle_list_papers(self, question: str) -> VerificationAnswer:
        """Handle: 'Which papers do we have?'"""
        paper_artifacts = self.store.list_artifacts("paper")
        evidence = []

        if paper_artifacts:
            answer_text = f"Found {len(paper_artifacts)} papers:\n"
            for paper in paper_artifacts:
                title = paper.get("title", paper.get("filename", ""))
                answer_text += f"  - {title}\n"
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.EXTERNAL_PAPER,
                        source_id=paper.get("artifact_id", ""),
                        source_name=title,
                        confidence=0.9,
                    )
                )
        else:
            answer_text = "No papers found in the project."

        confidence = min(1.0, len(paper_artifacts) / 5.0) if paper_artifacts else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if paper_artifacts else ["No papers ingested"],
        )

    def _handle_list_code(self, question: str) -> VerificationAnswer:
        """Handle: 'Which code files do we have?'"""
        code_artifacts = self.store.list_artifacts("code")
        evidence = []

        if code_artifacts:
            answer_text = f"Found {len(code_artifacts)} code files:\n"
            for code in code_artifacts:
                filename = code.get("filename", "")
                answer_text += f"  - {filename}\n"
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.SOURCE_CODE,
                        source_id=code.get("artifact_id", ""),
                        source_name=filename,
                        confidence=0.9,
                    )
                )
        else:
            answer_text = "No code files found in the project."

        confidence = min(1.0, len(code_artifacts) / 5.0) if code_artifacts else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if code_artifacts else ["No code files ingested"],
        )

    def _handle_list_notebooks(self, question: str) -> VerificationAnswer:
        """Handle: 'Which notebooks do we have?'"""
        notebook_artifacts = self.store.list_artifacts("notebook")
        evidence = []

        if notebook_artifacts:
            answer_text = f"Found {len(notebook_artifacts)} notebooks:\n"
            for nb in notebook_artifacts:
                filename = nb.get("filename", "")
                answer_text += f"  - {filename}\n"
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.NOTEBOOK_CELL,
                        source_id=nb.get("artifact_id", ""),
                        source_name=filename,
                        confidence=0.9,
                    )
                )
        else:
            answer_text = "No notebooks found in the project."

        confidence = (
            min(1.0, len(notebook_artifacts) / 5.0) if notebook_artifacts else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if notebook_artifacts else ["No notebooks ingested"],
        )

    def _handle_list_datasets(self, question: str) -> VerificationAnswer:
        """Handle: 'Which datasets do we have?'"""
        dataset_artifacts = self.store.list_artifacts("dataset")
        evidence = []

        if dataset_artifacts:
            answer_text = f"Found {len(dataset_artifacts)} datasets:\n"
            for ds in dataset_artifacts:
                name = ds.get("filename", ds.get("name", ""))
                answer_text += f"  - {name}\n"
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.DATASET,
                        source_id=ds.get("artifact_id", ""),
                        source_name=name,
                        confidence=0.9,
                    )
                )
        else:
            answer_text = "No datasets found in the project."

        confidence = (
            min(1.0, len(dataset_artifacts) / 5.0) if dataset_artifacts else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if dataset_artifacts else ["No datasets ingested"],
        )

    def _handle_list_experiments(self, question: str) -> VerificationAnswer:
        """Handle: 'Which experiments do we have?'"""
        experiment_artifacts = self.store.list_artifacts("experiment")
        evidence = []

        if experiment_artifacts:
            answer_text = f"Found {len(experiment_artifacts)} experiments:\n"
            for exp in experiment_artifacts:
                name = exp.get("filename", exp.get("name", ""))
                answer_text += f"  - {name}\n"
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.EXPERIMENT_RESULT,
                        source_id=exp.get("artifact_id", ""),
                        source_name=name,
                        confidence=0.9,
                    )
                )
        else:
            answer_text = "No experiments found in the project."

        confidence = (
            min(1.0, len(experiment_artifacts) / 5.0) if experiment_artifacts else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if experiment_artifacts else ["No experiments ingested"],
        )

    def _handle_list_assumptions(self, question: str) -> VerificationAnswer:
        """Handle: 'Which assumptions remain unverified?'"""
        all_artifacts = self.store.list_artifacts()
        evidence = []
        assumptions_found: list[str] = []

        for artifact in all_artifacts:
            artifact_id = artifact.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                assumptions = knowledge.get("assumptions", [])
                if assumptions:
                    assumptions_found.extend(assumptions)
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.UPLOADED_FILE,
                            source_id=artifact_id,
                            source_name=artifact.get("filename", ""),
                            content=f"Assumptions: {'; '.join(assumptions)}",
                            confidence=0.7,
                        )
                    )

        # Deduplicate
        assumptions_found = list(set(assumptions_found))

        if assumptions_found:
            answer_text = f"Found {len(assumptions_found)} assumptions:\n"
            answer_text += "\n".join(f"  - {a}" for a in sorted(assumptions_found))
        else:
            answer_text = "No assumptions found in the project."

        confidence = (
            min(1.0, len(assumptions_found) / 5.0) if assumptions_found else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if assumptions_found else ["No assumptions extracted"],
        )

    def _handle_list_conclusions(self, question: str) -> VerificationAnswer:
        """Handle: 'Which conclusions do we have?'"""
        all_artifacts = self.store.list_artifacts()
        evidence = []
        conclusions_found: list[str] = []

        for artifact in all_artifacts:
            artifact_id = artifact.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                conclusions = knowledge.get("conclusions", [])
                if conclusions:
                    conclusions_found.extend(conclusions)
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.UPLOADED_FILE,
                            source_id=artifact_id,
                            source_name=artifact.get("filename", ""),
                            content=f"Conclusions: {'; '.join(conclusions)}",
                            confidence=0.7,
                        )
                    )

        # Deduplicate
        conclusions_found = list(set(conclusions_found))

        if conclusions_found:
            answer_text = f"Found {len(conclusions_found)} conclusions:\n"
            answer_text += "\n".join(f"  - {c}" for c in sorted(conclusions_found))
        else:
            answer_text = "No conclusions found in the project."

        confidence = (
            min(1.0, len(conclusions_found) / 5.0) if conclusions_found else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if conclusions_found else ["No conclusions extracted"],
        )

    def _handle_how_figure_generated(self, question: str) -> VerificationAnswer:
        """Handle: 'How was Figure 5 generated?'"""
        notebook_artifacts = self.store.list_artifacts("notebook")
        evidence = []

        # Search notebooks for figures
        figures_found = []
        for nb in notebook_artifacts:
            artifact_id = nb.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                figures = knowledge.get("figures", [])
                if figures:
                    for fig in figures:
                        figures_found.append(
                            {
                                "notebook": nb.get("filename", ""),
                                "cell_index": fig.get("cell_index", 0),
                                "output_index": fig.get("output_index", 0),
                                "mime_type": fig.get("mime_type", ""),
                            }
                        )
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.NOTEBOOK_CELL,
                            source_id=artifact_id,
                            source_name=nb.get("filename", ""),
                            content=f"Figures: {len(figures)} found",
                            confidence=0.8,
                        )
                    )

        if figures_found:
            answer_text = f"Found {len(figures_found)} figures:\n"
            for fig in figures_found:
                answer_text += f"  - {fig['notebook']} (cell {fig['cell_index']}, output {fig['output_index']})\n"
        else:
            answer_text = "No figures found in notebooks."
            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.NOTEBOOK_CELL,
                    content="No notebook artifacts with figures found",
                    confidence=0.0,
                )
            )

        confidence = min(1.0, len(figures_found) / 5.0) if figures_found else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if figures_found else ["No figures found in notebooks"],
        )

    def _handle_which_notebook_produced(self, question: str) -> VerificationAnswer:
        """Handle: 'Which notebook produced this dataset?'"""
        notebook_artifacts = self.store.list_artifacts("notebook")
        evidence = []

        # Search notebooks for outputs
        outputs_found = []
        for nb in notebook_artifacts:
            artifact_id = nb.get("artifact_id", "")
            knowledge = self.store.get_extracted_knowledge(artifact_id)
            if knowledge:
                generated_outputs = knowledge.get("generated_outputs", [])
                if generated_outputs:
                    outputs_found.append(
                        {
                            "notebook": nb.get("filename", ""),
                            "outputs_count": len(generated_outputs),
                            "sample_output": generated_outputs[0][:200]
                            if generated_outputs
                            else "",
                        }
                    )
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.NOTEBOOK_CELL,
                            source_id=artifact_id,
                            source_name=nb.get("filename", ""),
                            content=f"Generated outputs: {len(generated_outputs)}",
                            confidence=0.8,
                        )
                    )

        if outputs_found:
            answer_text = f"Found {len(outputs_found)} notebooks with outputs:\n"
            for out in outputs_found:
                answer_text += (
                    f"  - {out['notebook']} ({out['outputs_count']} outputs)\n"
                )
        else:
            answer_text = "No notebooks with generated outputs found."

        confidence = min(1.0, len(outputs_found) / 5.0) if outputs_found else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if outputs_found else ["No notebooks with outputs found"],
        )

    def _handle_which_papers_support(self, question: str) -> VerificationAnswer:
        """Handle: 'Which papers support this implementation?'"""
        code_artifacts = self.store.list_artifacts("code")
        paper_artifacts = self.store.list_artifacts("paper")
        evidence = []

        # Find papers that cite or reference code
        supporting_papers = []
        for code in code_artifacts:
            code_id = code.get("artifact_id", "")
            code_knowledge = self.store.get_extracted_knowledge(code_id)
            if code_knowledge:
                # Look for papers that describe similar algorithms
                code_algorithms = code_knowledge.get("algorithms", [])
                for paper in paper_artifacts:
                    paper_id = paper.get("artifact_id", "")
                    paper_knowledge = self.store.get_extracted_knowledge(paper_id)
                    if paper_knowledge:
                        paper_keywords = paper_knowledge.get("keywords", [])
                        # Simple matching: if any algorithm name appears in paper keywords
                        for algo in code_algorithms:
                            if any(algo.lower() in kw.lower() for kw in paper_keywords):
                                supporting_papers.append(
                                    {
                                        "paper": paper.get("filename", ""),
                                        "code": code.get("filename", ""),
                                        "algorithm": algo,
                                    }
                                )
                                evidence.append(
                                    Evidence(
                                        evidence_type=EvidenceType.EXTERNAL_PAPER,
                                        source_id=paper_id,
                                        source_name=paper.get("filename", ""),
                                        content=f"Paper describes algorithm: {algo}",
                                        confidence=0.7,
                                    )
                                )
                                break

        if supporting_papers:
            answer_text = f"Found {len(supporting_papers)} supporting paper(s):\n"
            for sp in supporting_papers:
                answer_text += (
                    f"  - {sp['paper']} (supports {sp['code']} via {sp['algorithm']})\n"
                )
        else:
            answer_text = "No supporting papers found."

        confidence = (
            min(1.0, len(supporting_papers) / 5.0) if supporting_papers else 0.0
        )

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if supporting_papers else ["No supporting papers found"],
        )

    def _generic_answer(self, question: str) -> VerificationAnswer:
        """Handle unrecognized questions."""
        artifacts = self.store.list_artifacts()

        # Try to search artifacts for the query
        search_results = self.store.search_artifacts(question)
        evidence = []

        if search_results:
            for result in search_results[:5]:
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.CATALOG_ENTRY,
                        source_id=result.get("artifact_id", ""),
                        source_name=result.get("filename", ""),
                        content=result.get("summary", ""),
                        confidence=0.5,
                    )
                )

        answer_text = f"I found {len(search_results)} potentially relevant artifacts."
        if not search_results:
            answer_text = "I couldn't find any relevant information for your question."
            if not artifacts:
                answer_text += " No artifacts have been ingested yet."

        confidence = 0.3 if search_results else 0.0

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            missing_info=[] if search_results else ["No matching artifacts found"],
        )


# ── Convenience Function ──────────────────────────────────────────────────


def verify_knowledge(
    question: str,
    store: KnowledgeStore,
) -> VerificationAnswer:
    """Answer a knowledge verification question (convenience function).

    Args:
        question: The user's question
        store: The knowledge store to query

    Returns:
        A ``VerificationAnswer`` with answer, evidence, confidence, and gaps.
    """
    verifier = KnowledgeVerifier(store)
    return verifier.verify(question)
