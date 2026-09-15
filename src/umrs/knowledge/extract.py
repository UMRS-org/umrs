"""Knowledge extraction — extract structured scientific information from artifacts.

Phase 4 of the Scientific Knowledge Build-up roadmap.

Instead of only chunking documents, this module extracts structured
scientific information:

  From papers:
    - objective
    - methodology
    - assumptions
    - equations
    - datasets
    - experiments
    - conclusions
    - limitations

  From code:
    - modules
    - APIs
    - algorithms
    - dependencies
    - parameters
    - kernels
    - execution flow

  From notebooks:
    - markdown explanations
    - figures
    - generated outputs
    - execution order
    - experimental settings

The extracted knowledge is stored as structured ``ExtractedKnowledge`` objects
that can be used for RAG indexing, knowledge graph construction, and
knowledge verification chat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .parse_code import CodeParseResult, parse_code
from .parse_notebook import NotebookParseResult, parse_notebook

logger = logging.getLogger(__name__)


# Type-aware field sets for quality metrics (see knowledge_extraction_audit2.md PR1).
_PAPER_TEXT_TYPES = frozenset({"paper", "text", "document", "structured"})

_TYPE_COVERAGE_FIELDS: dict[str, list[str]] = {
    "paper": [
        "objective", "methodology", "conclusions", "limitations",
        "assumptions", "equations", "datasets", "experiments",
    ],
    "code": [
        "modules", "apis", "algorithms", "dependencies",
        "parameters", "kernels", "execution_flow",
    ],
    "notebook": [
        "markdown_explanations", "figures", "generated_outputs",
        "execution_order", "experimental_settings",
    ],
}

_TYPE_CONFIDENCE_STRING_FIELDS: dict[str, list[str]] = {
    "paper": ["objective", "methodology", "summary"],
    "code": ["execution_flow", "summary"],
    "notebook": ["summary"],
}

_TYPE_CONFIDENCE_LIST_FIELDS: dict[str, list[str]] = {
    "paper": [
        "assumptions", "equations", "datasets", "experiments",
        "conclusions", "limitations", "keywords", "tags",
    ],
    "code": [
        "modules", "apis", "algorithms", "dependencies", "parameters",
        "kernels", "experimental_settings", "tags",
    ],
    "notebook": [
        "markdown_explanations", "figures", "generated_outputs",
        "execution_order", "experimental_settings", "tags",
    ],
}


def coverage_fields_for_type(artifact_type: str) -> list[str]:
    """Return coverage key fields relevant to *artifact_type*."""
    if artifact_type in _TYPE_COVERAGE_FIELDS:
        return _TYPE_COVERAGE_FIELDS[artifact_type]
    if artifact_type in _PAPER_TEXT_TYPES:
        return _TYPE_COVERAGE_FIELDS["paper"]
    return _TYPE_COVERAGE_FIELDS["paper"]


def confidence_fields_for_type(
    artifact_type: str,
) -> tuple[list[str], list[str]]:
    """Return (string_fields, list_fields) for confidence scoring."""
    if artifact_type in _TYPE_CONFIDENCE_STRING_FIELDS:
        return (
            _TYPE_CONFIDENCE_STRING_FIELDS[artifact_type],
            _TYPE_CONFIDENCE_LIST_FIELDS[artifact_type],
        )
    if artifact_type in _PAPER_TEXT_TYPES:
        return (
            _TYPE_CONFIDENCE_STRING_FIELDS["paper"],
            _TYPE_CONFIDENCE_LIST_FIELDS["paper"],
        )
    return (
        _TYPE_CONFIDENCE_STRING_FIELDS["paper"],
        _TYPE_CONFIDENCE_LIST_FIELDS["paper"],
    )


def _is_binary_document_bytes(filename: str, content: bytes) -> bool:
    """Return True when *content* is a binary document needing Phase 3 text extraction."""
    if content.startswith(b"%PDF"):
        return True
    lower = filename.lower()
    return lower.endswith((".pptx", ".ppt")) and content.startswith(b"PK\x03\x04")


def resolve_extraction_content(filename: str, content: bytes) -> bytes:
    """Normalize binary document bytes to UTF-8 text for regex extraction.

    Reuses ``data_catalog.extract.extract_content`` (pypdf / python-pptx) so
    Phase 4 does not duplicate document parsers. Plain UTF-8 payloads — including
    catalog ``extracted_text_ref`` text — pass through unchanged.
    """
    if not content or not _is_binary_document_bytes(filename, content):
        return content

    from ..data_catalog.extract import extract_content

    _summary, extracted_text, _tags = extract_content(filename, content)
    if extracted_text.strip():
        return extracted_text.encode("utf-8")

    logger.debug(
        "No catalog text extracted for %s; falling back to raw bytes",
        filename,
    )
    return content


# ── Extracted Knowledge Models ──────────────────────────────────────────────


@dataclass
class ExtractedKnowledge:
    """Structured knowledge extracted from a scientific artifact.

    This is the output of the knowledge extraction pipeline. It contains
    structured scientific information that can be used for RAG indexing,
    knowledge graph construction, and knowledge verification chat.
    """

    artifact_id: str  # The id of the source artifact
    artifact_type: str  # "paper" | "code" | "notebook" | "document" | "text"
    filename: str

    # Extracted structured knowledge
    objective: str = ""
    methodology: str = ""
    assumptions: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # Code-specific
    modules: list[str] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    execution_flow: str = ""

    # Notebook-specific
    markdown_explanations: list[str] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    generated_outputs: list[str] = field(default_factory=list)
    execution_order: list[int] = field(default_factory=list)
    experimental_settings: list[str] = field(default_factory=list)

    # General metadata
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Quality metrics
    extraction_confidence: float = 0.0  # 0.0–1.0
    coverage_score: float = 0.0  # 0.0–1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "filename": self.filename,
            "objective": self.objective,
            "methodology": self.methodology,
            "assumptions": self.assumptions,
            "equations": self.equations,
            "datasets": self.datasets,
            "experiments": self.experiments,
            "conclusions": self.conclusions,
            "limitations": self.limitations,
            "modules": self.modules,
            "apis": self.apis,
            "algorithms": self.algorithms,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "kernels": self.kernels,
            "execution_flow": self.execution_flow,
            "markdown_explanations": self.markdown_explanations,
            "figures": self.figures,
            "generated_outputs": self.generated_outputs,
            "execution_order": self.execution_order,
            "experimental_settings": self.experimental_settings,
            "summary": self.summary,
            "tags": self.tags,
            "keywords": self.keywords,
            "extraction_confidence": self.extraction_confidence,
            "coverage_score": self.coverage_score,
        }


# ── Knowledge Extractor ─────────────────────────────────────────────────────


class KnowledgeExtractor:
    """Extract structured scientific knowledge from artifacts.

    Usage:
        extractor = KnowledgeExtractor()
        knowledge = extractor.extract(artifact_id, artifact_type, filename, content)
    """

    def __init__(self) -> None:
        # Patterns for extracting structured information from text
        self._objective_patterns = [
            r"(?:objective|goal|aim|purpose)\s*(?:of this (?:paper|study|work))?\s*(?:is|to)\s+(.+?)(?:\.|$)",
            r"(?:we aim to|we attempt to|the goal of)\s+(.+?)(?:\.|$)",
            r"(?:in this paper|in this work|in this study)\s*,?\s*we\s+(.+?)(?:\.|$)",
        ]
        self._methodology_patterns = [
            r"(?:methodology|method|approach|algorithm)\s*[:is]+\s*(.+?)(?:\.|$)",
            r"(?:we use|we employ|we propose|we present|we introduce)\s+(.+?)(?:\.|$)",
            r"(?:our method|our approach|our algorithm)\s+(?:is|consists of)\s+(.+?)(?:\.|$)",
        ]
        self._conclusion_patterns = [
            r"(?:conclusion|in conclusion|to conclude)\s*[:,]?\s*(.+?)(?:\.|$)",
            r"(?:we conclude that|our results show that|our findings suggest that)\s+(.+?)(?:\.|$)",
            r"(?:in summary|overall)\s*,?\s*(.+?)(?:\.|$)",
        ]
        self._limitation_patterns = [
            r"(?:limitation|limitation of|limitation in)\s*[:,]?\s*(.+?)(?:\.|$)",
            r"(?:however|a limitation is|one limitation is)\s+,?\s*(.+?)(?:\.|$)",
        ]
        self._assumption_patterns = [
            r"(?:assumption|assuming|we assume that|assuming that)\s+[:,]?\s*(.+?)(?:\.|$)",
            r"(?:we assume|it is assumed that|under the assumption that)\s+(.+?)(?:\.|$)",
        ]
        self._equation_patterns = [
            r"(?:equation|equation\s*\d+|eq\.|eq\.\s*\d+)\s*[:,]?\s*(.+?)(?:\.|$)",
            r"```(?:latex|math)?\s*(.+?)\s*```",
            r"\$\$(.+?)\$\$",
            r"\$(.+?)\$",
        ]
        self._dataset_patterns = [
            r"(?:dataset|datasets|data set|data sets)\s*[:,]?\s*(.+?)(?:\.|$)",
            r"(?:we use|we use the|using the)\s+(.+?)\s+dataset",
            r"(?:we use|using)\s+(.+?)\s+(?:dataset|data)",
        ]
        self._experiment_patterns = [
            r"(?:experiment|experimental setup|experimental settings)\s*[:,]?\s*(.+?)(?:\.|$)",
            r"(?:we experiment|we evaluate|we test)\s+(.+?)(?:\.|$)",
        ]

    def extract(
        self,
        artifact_id: str,
        artifact_type: str,
        filename: str,
        content: bytes,
    ) -> ExtractedKnowledge:
        """Extract structured knowledge from an artifact.

        Args:
            artifact_id: The id of the source artifact
            artifact_type: The type of artifact ("paper" | "code" | "notebook" | "document" | "text")
            filename: The filename of the artifact
            content: The raw bytes of the artifact

        Returns:
            An ``ExtractedKnowledge`` object with structured scientific information.
        """
        knowledge = ExtractedKnowledge(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
        )

        content = resolve_extraction_content(filename, content)

        if artifact_type == "paper":
            self._extract_from_paper(content, knowledge)
        elif artifact_type == "code":
            self._extract_from_code(filename, content, knowledge)
        elif artifact_type == "notebook":
            self._extract_from_notebook(filename, content, knowledge)
        elif artifact_type in ("document", "text", "structured"):
            self._extract_from_text(content, knowledge)
        else:
            # Unknown type — just extract text
            self._extract_from_text(content, knowledge)

        # Calculate quality metrics
        knowledge.extraction_confidence = self._calculate_confidence(knowledge)
        knowledge.coverage_score = self._calculate_coverage(knowledge)

        return knowledge

    # ── Paper Extraction ─────────────────────────────────────────────────────

    def _extract_from_paper(
        self,
        content: bytes,
        knowledge: ExtractedKnowledge,
    ) -> None:
        """Extract structured knowledge from a paper."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError:
                return

        self._extract_scientific_fields_from_text(text, knowledge)
        # Summary
        knowledge.summary = text[:500] + "..." if len(text) > 500 else text
        # Tags
        knowledge.tags = ["paper"] + knowledge.keywords[:10]

    # ── Code Extraction ──────────────────────────────────────────────────────

    def _extract_from_code(
        self,
        filename: str,
        content: bytes,
        knowledge: ExtractedKnowledge,
    ) -> None:
        """Extract structured knowledge from code."""
        result = parse_code(filename, content)

        knowledge.modules = result.modules
        knowledge.apis = result.apis
        knowledge.algorithms = result.algorithms
        knowledge.dependencies = result.dependencies
        knowledge.parameters = result.parameters
        knowledge.kernels = [k.name for k in result.kernels]
        knowledge.execution_flow = result.execution_flow
        knowledge.summary = result.summary
        knowledge.tags = result.tags

        # Extract experimental settings from code
        try:
            text = content.decode("utf-8")
            knowledge.experimental_settings = self._extract_settings_from_code(text)
        except UnicodeDecodeError:
            pass

    # ── Notebook Extraction ─────────────────────────────────────────────────

    def _extract_from_notebook(
        self,
        filename: str,
        content: bytes,
        knowledge: ExtractedKnowledge,
    ) -> None:
        """Extract structured knowledge from a notebook."""
        result = parse_notebook(filename, content)

        # Extract markdown explanations
        knowledge.markdown_explanations = [
            cell.source for cell in result.markdown_cells if cell.source.strip()
        ]

        # Extract figures
        knowledge.figures = [
            {
                "cell_index": fig.cell_index,
                "output_index": fig.output_index,
                "mime_type": fig.mime_type,
            }
            for fig in result.figures
        ]

        # Extract generated outputs
        knowledge.generated_outputs = []
        for cell in result.code_cells:
            for output in cell.outputs:
                output_type = output.get("output_type", "")
                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if text.strip():
                        knowledge.generated_outputs.append(text)
                elif output_type in ("execute_result", "display_data"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        text = data["text/plain"]
                        if isinstance(text, list):
                            text = "".join(text)
                        if text.strip():
                            knowledge.generated_outputs.append(text)

        # Extract execution order
        knowledge.execution_order = [
            cell.execution_count
            for cell in result.code_cells
            if cell.execution_count is not None
        ]

        # Extract experimental settings from code cells
        knowledge.experimental_settings = []
        for cell in result.code_cells:
            settings = self._extract_settings_from_code(cell.source)
            knowledge.experimental_settings.extend(settings)

        # Summary and tags
        knowledge.summary = result.summary
        knowledge.tags = result.tags

    # ── Text Extraction ─────────────────────────────────────────────────────

    def _extract_from_text(
        self,
        content: bytes,
        knowledge: ExtractedKnowledge,
    ) -> None:
        """Extract structured knowledge from text."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError:
                return

        self._extract_scientific_fields_from_text(text, knowledge)
        # Summary
        knowledge.summary = text[:500] + "..." if len(text) > 500 else text
        # Tags
        knowledge.tags = ["text"] + knowledge.keywords[:10]

    # ── Helper Methods ──────────────────────────────────────────────────────

    def _extract_scientific_fields_from_text(
        self,
        text: str,
        knowledge: ExtractedKnowledge,
    ) -> None:
        """Extract paper-like scientific fields from plain text."""
        knowledge.objective = self._extract_pattern(text, self._objective_patterns)
        knowledge.methodology = self._extract_pattern(text, self._methodology_patterns)
        knowledge.conclusions = self._extract_all_patterns(text, self._conclusion_patterns)
        knowledge.limitations = self._extract_all_patterns(text, self._limitation_patterns)
        knowledge.assumptions = self._extract_all_patterns(text, self._assumption_patterns)
        knowledge.equations = self._extract_all_patterns(text, self._equation_patterns)
        knowledge.datasets = self._extract_all_patterns(text, self._dataset_patterns)
        knowledge.experiments = self._extract_all_patterns(text, self._experiment_patterns)
        knowledge.keywords = self._extract_keywords(text)

    def _extract_pattern(self, text: str, patterns: list[str]) -> str:
        """Extract the first match from a list of patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_all_patterns(self, text: str, patterns: list[str]) -> list[str]:
        """Extract all matches from a list of patterns."""
        results: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                result = match.group(1).strip()
                if result and result not in results:
                    results.append(result)
        return results

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Look for explicit keyword sections
        keyword_patterns = [
            r"(?:keywords|key words|keyword)\s*[:,]?\s*(.+?)(?:\n|$)",
            r"(?:keywords|key words|keyword)\s*[:\-]?\s*(.+?)(?:\n\n|\.\n)",
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_str = match.group(1).strip()
                # Split by comma, semicolon, or pipe
                keywords = re.split(r"[,;|]", keywords_str)
                return [k.strip() for k in keywords if k.strip()]
        return []

    def _extract_settings_from_code(self, code: str) -> list[str]:
        """Extract experimental settings from code."""
        settings: list[str] = []
        # Look for common patterns like: learning_rate = 0.001, batch_size = 32, etc.
        setting_patterns = [
            r"(learning_rate|lr)\s*=\s*([\d.eE\-]+)",
            r"(batch_size|batch)\s*=\s*(\d+)",
            r"(epochs|num_epochs)\s*=\s*(\d+)",
            r"(hidden_size|hidden_dim)\s*=\s*(\d+)",
            r"(num_layers|layers)\s*=\s*(\d+)",
            r"(dropout)\s*=\s*([\d.]+)",
            r"(optimizer)\s*=\s*[\"']?(\w+)[\"']?",
            r"(model_name|model)\s*=\s*[\"']?([\w\-\.]+)[\"']?",
            r"(seed|random_seed)\s*=\s*(\d+)",
            r"(temperature)\s*=\s*([\d.]+)",
            r"(max_length|max_len)\s*=\s*(\d+)",
        ]
        for pattern in setting_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                setting = f"{match.group(1)}={match.group(2)}"
                if setting not in settings:
                    settings.append(setting)
        return settings

    def _calculate_confidence(self, knowledge: ExtractedKnowledge) -> float:
        """Calculate extraction confidence (0.0–1.0).

        Confidence is based on how many type-relevant fields were extracted.
        """
        string_fields, list_fields = confidence_fields_for_type(
            knowledge.artifact_type,
        )
        total_fields = len(string_fields) + len(list_fields)
        if total_fields == 0:
            return 0.0

        filled_fields = 0
        for field_name in string_fields:
            if getattr(knowledge, field_name, "").strip():
                filled_fields += 1
        for field_name in list_fields:
            if getattr(knowledge, field_name, []):
                filled_fields += 1

        return filled_fields / total_fields

    def _calculate_coverage(self, knowledge: ExtractedKnowledge) -> float:
        """Calculate coverage score (0.0–1.0).

        Coverage is based on how many type-relevant key fields were extracted.
        """
        key_fields = coverage_fields_for_type(knowledge.artifact_type)
        if not key_fields:
            return 0.0

        filled = 0
        for field_name in key_fields:
            value = getattr(knowledge, field_name, "")
            if isinstance(value, list) and value:
                filled += 1
            elif isinstance(value, str) and value.strip():
                filled += 1
        return filled / len(key_fields)


# ── Convenience Function ────────────────────────────────────────────────────


def extract_knowledge(
    artifact_id: str,
    artifact_type: str,
    filename: str,
    content: bytes,
) -> ExtractedKnowledge:
    """Extract structured knowledge from an artifact (convenience function).

    Args:
        artifact_id: The id of the source artifact
        artifact_type: The type of artifact ("paper" | "code" | "notebook" | "document" | "text")
        filename: The filename of the artifact
        content: The raw bytes of the artifact

    Returns:
        An ``ExtractedKnowledge`` object with structured scientific information.
    """
    extractor = KnowledgeExtractor()
    return extractor.extract(artifact_id, artifact_type, filename, content)
