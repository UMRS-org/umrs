"""Jupyter notebook (.ipynb) parser — extracts markdown, code, outputs, figures.

Notebooks are first-class scientific artifacts. This parser extracts:
  - Markdown explanations (cell source)
  - Code cells (cell source)
  - Generated outputs (cell outputs: text, images, HTML)
  - Execution order (execution_count)
  - Experimental settings (from code cells)
  - Figures (from image outputs)

The parser returns a structured ``NotebookParseResult`` that can be used
for RAG indexing, knowledge extraction, and catalog metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NotebookCell:
    """A single notebook cell."""

    cell_type: str  # "markdown" | "code" | "raw"
    source: str
    execution_count: int | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    cell_index: int = 0

    def is_markdown(self) -> bool:
        return self.cell_type == "markdown"

    def is_code(self) -> bool:
        return self.cell_type == "code"

    def is_raw(self) -> bool:
        return self.cell_type == "raw"


@dataclass
class NotebookFigure:
    """A figure extracted from a notebook cell output."""

    cell_index: int
    output_index: int
    mime_type: str  # "image/png", "image/jpeg", ...
    data_base64: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NotebookParseResult:
    """Structured result of parsing a Jupyter notebook."""

    filename: str
    cells: list[NotebookCell] = field(default_factory=list)
    figures: list[NotebookFigure] = field(default_factory=list)
    markdown_cells: list[NotebookCell] = field(default_factory=list)
    code_cells: list[NotebookCell] = field(default_factory=list)
    raw_cells: list[NotebookCell] = field(default_factory=list)
    # Extracted text (concatenation of markdown + code + outputs)
    extracted_text: str = ""
    # Summary (first few markdown cells or truncated text)
    summary: str = ""
    # Tags (extracted from cell metadata, keywords, etc.)
    tags: list[str] = field(default_factory=list)
    # Metadata
    language: str = "python"
    kernel: str = ""
    nbformat: int = 4
    nbformat_minor: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Error
    error: str | None = None


def parse_notebook(filename: str, content: bytes) -> NotebookParseResult:
    """Parse a Jupyter notebook (.ipynb) file.

    Args:
        filename: The filename (must end with .ipynb)
        content: The raw bytes of the notebook file

    Returns:
        A ``NotebookParseResult`` with structured notebook content.
    """
    result = NotebookParseResult(filename=filename)

    try:
        nb = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result.error = f"Failed to parse notebook JSON: {exc}"
        return result

    # Extract notebook-level metadata
    result.nbformat = nb.get("nbformat", 4)
    result.nbformat_minor = nb.get("nbformat_minor", 0)
    result.metadata = nb.get("metadata", {})

    # Extract kernel info
    kernel_info = result.metadata.get("kernelspec", {})
    result.kernel = kernel_info.get("name", "")
    result.language = kernel_info.get("language", "python")

    # Parse cells
    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        parsed_cell = _parse_cell(cell, idx)
        result.cells.append(parsed_cell)
        if parsed_cell.is_markdown():
            result.markdown_cells.append(parsed_cell)
        elif parsed_cell.is_code():
            result.code_cells.append(parsed_cell)
        else:
            result.raw_cells.append(parsed_cell)

        # Extract figures from code cell outputs
        if parsed_cell.is_code():
            for out_idx, output in enumerate(parsed_cell.outputs):
                figures = _extract_figures_from_output(
                    output, idx, out_idx
                )
                result.figures.extend(figures)

    # Build extracted text
    result.extracted_text = _build_extracted_text(result)
    result.summary = _build_summary(result)
    result.tags = _extract_tags(result)

    return result


def _parse_cell(cell: dict[str, Any], idx: int) -> NotebookCell:
    """Parse a single notebook cell."""
    cell_type = cell.get("cell_type", "raw")
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)

    execution_count = cell.get("execution_count")
    outputs = cell.get("outputs", [])
    metadata = cell.get("metadata", {})

    return NotebookCell(
        cell_type=cell_type,
        source=source,
        execution_count=execution_count,
        outputs=outputs,
        metadata=metadata,
        cell_index=idx,
    )


def _extract_figures_from_output(
    output: dict[str, Any],
    cell_idx: int,
    out_idx: int,
) -> list[NotebookFigure]:
    """Extract figures (image outputs) from a cell output."""
    figures: list[NotebookFigure] = []
    data = output.get("data", {})
    for mime_type, mime_data in data.items():
        if not mime_type.startswith("image/"):
            continue
        if isinstance(mime_data, list):
            mime_data = "".join(mime_data)
        figures.append(
            NotebookFigure(
                cell_index=cell_idx,
                output_index=out_idx,
                mime_type=mime_type,
                data_base64=mime_data,
                metadata=output.get("metadata", {}),
            )
        )
    return figures


def _build_extracted_text(result: NotebookParseResult) -> str:
    """Build extracted text from markdown + code + outputs."""
    parts: list[str] = []
    for cell in result.cells:
        if cell.is_markdown():
            parts.append(f"### Markdown Cell {cell.cell_index}\n{cell.source}")
        elif cell.is_code():
            parts.append(
                f"### Code Cell {cell.cell_index}"
                f" (execution_count={cell.execution_count})\n```python\n{cell.source}\n```"
            )
            # Add text outputs
            for output in cell.outputs:
                output_type = output.get("output_type", "")
                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    parts.append(f"#### Output (stream)\n{text}")
                elif output_type in ("execute_result", "display_data"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        text = data["text/plain"]
                        if isinstance(text, list):
                            text = "".join(text)
                        parts.append(f"#### Output (text/plain)\n{text}")
                    if "text/html" in data:
                        text = data["text/html"]
                        if isinstance(text, list):
                            text = "".join(text)
                        parts.append(f"#### Output (text/html)\n{text}")
                elif output_type == "error":
                    ename = output.get("ename", "")
                    evalue = output.get("evalue", "")
                    parts.append(f"#### Output (error)\n{ename}: {evalue}")
        elif cell.is_raw():
            parts.append(f"### Raw Cell {cell.cell_index}\n{cell.source}")

    text = "\n\n".join(parts)
    # Limit to 100,000 chars to avoid huge payloads
    if len(text) > 100_000:
        text = text[:100_000] + "\n... (truncated)"
    return text


def _build_summary(result: NotebookParseResult) -> str:
    """Build a summary from the first few markdown cells or truncated text."""
    # Try to use first few markdown cells as summary
    markdown_parts: list[str] = []
    for cell in result.markdown_cells[:5]:
        # Strip markdown formatting for a cleaner summary
        text = cell.source.strip()
        if text:
            markdown_parts.append(text)

    if markdown_parts:
        summary = "\n\n".join(markdown_parts)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return summary

    # Fallback: use truncated extracted text
    if result.extracted_text:
        summary = result.extracted_text[:500]
        if len(result.extracted_text) > 500:
            summary += "..."
        return summary

    return f"Notebook with {len(result.cells)} cells"


def _extract_tags(result: NotebookParseResult) -> list[str]:
    """Extract tags from notebook metadata, cell metadata, and keywords."""
    tags: set[str] = set()

    # From notebook metadata
    nb_meta = result.metadata
    if "tags" in nb_meta:
        tags.update(nb_meta["tags"])

    # From cell metadata
    for cell in result.cells:
        cell_tags = cell.metadata.get("tags", [])
        tags.update(cell_tags)

    # From language
    tags.add(result.language)

    # From kernel
    if result.kernel:
        tags.add(result.kernel)

    # From cell count
    tags.add(f"cells:{len(result.cells)}")

    # From figure count
    if result.figures:
        tags.add(f"figures:{len(result.figures)}")

    return sorted(tags)
