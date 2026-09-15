"""``NotebookStep`` — built-in step that parses Jupyter notebooks.

The step reads a ``.ipynb`` file from S3, extracts code and markdown
cells, and writes a JSON document containing the cell contents with
metadata (cell type, execution count, source lines).

The output is suitable for downstream embedding by an ``LLMStep`` or
a custom embedding step.

Modes
-----
- ``cells``: extract all cells (code + markdown) as separate chunks.
- ``code``: extract only code cells.
- ``markdown``: extract only markdown cells.
- ``concat``: concatenate all cells into a single text document.

Example ``workflow.json`` snippet::

    {
      "name": "notebook-parse",
      "entrypoint": "umrs.pipeline_sdk.templates.notebook_step:NotebookStep",
      "config": {"mode": "cells"}
    }
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class NotebookStep(Step):
    """Parse a Jupyter notebook (``.ipynb``) into cell chunks."""

    name = "notebook"

    def setup(self, config: dict[str, Any]) -> None:
        self.mode: str = str(config.get("mode", "cells"))
        if self.mode not in ("cells", "code", "markdown", "concat"):
            raise ValueError(
                f"NotebookStep mode must be one of: cells, code, markdown, concat. "
                f"Got: {self.mode}"
            )
        self.max_cell_lines: int = int(config.get("max_cell_lines", 200))
        self.include_outputs: bool = bool(config.get("include_outputs", False))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        out_keys: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        for key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(key)
                notebook = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                logger.warning("NotebookStep: failed to parse %s: %s", key, exc)
                continue

            chunks = self._extract_chunks(notebook, key)
            if not chunks:
                logger.warning("NotebookStep: no chunks extracted from %s", key)
                continue

            out_key = self._write_chunks(chunks, key, ctx)
            out_keys.append(out_key)
            metadata_list.append({
                "source_key": key,
                "num_chunks": len(chunks),
                "mode": self.mode,
            })

        return StepResult(
            output_keys=out_keys,
            metadata=metadata_list,
            extra={"step": "notebook", "mode": self.mode},
        )

    # ------------------------------------------------------------------
    # Chunk extraction
    # ------------------------------------------------------------------

    def _extract_chunks(
        self,
        notebook: dict[str, Any],
        source_key: str,
    ) -> list[dict[str, Any]]:
        """Extract cell chunks from a notebook dict."""
        cells = notebook.get("cells", [])
        chunks: list[dict[str, Any]] = []

        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", [])
            if isinstance(source, str):
                source_lines = source.split("\n")
            else:
                source_lines = source

            if self.mode == "code" and cell_type != "code":
                continue
            if self.mode == "markdown" and cell_type != "markdown":
                continue

            text = "\n".join(source_lines[: self.max_cell_lines])

            chunk: dict[str, Any] = {
                "cell_index": idx,
                "cell_type": cell_type,
                "text": text,
                "source_key": source_key,
                "execution_count": cell.get("execution_count"),
            }

            if self.include_outputs and cell_type == "code":
                outputs = cell.get("outputs", [])
                chunk["outputs"] = self._extract_outputs(outputs)

            chunks.append(chunk)

        if self.mode == "concat":
            combined = "\n\n---\n\n".join(c["text"] for c in chunks)
            chunks = [{
                "cell_index": 0,
                "cell_type": "concat",
                "text": combined,
                "source_key": source_key,
            }]

        return chunks

    def _extract_outputs(self, outputs: list[dict[str, Any]]) -> list[str]:
        """Extract text from notebook cell outputs."""
        text_outputs: list[str] = []
        for output in outputs:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                text_outputs.append(text)
            elif output.get("output_type") in ("execute_result", "display_data"):
                data = output.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    if isinstance(text, list):
                        text = "".join(text)
                    text_outputs.append(text)
        return text_outputs

    def _write_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_key: str,
        ctx: StepContext,
    ) -> str:
        """Write chunks as JSON to S3."""
        base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key
        out_key = f"notebook/{base}.json"
        data = json.dumps({
            "source": source_key,
            "mode": self.mode,
            "chunks": chunks,
        }, indent=2).encode("utf-8")
        ctx.upload_bytes(out_key, data, content_type="application/json")
        return out_key
