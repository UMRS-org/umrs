#!/usr/bin/env python3
"""
Demo 12 — Text Analysis Pipeline
=================================

A concrete Pipeline subclass that demonstrates the 6-phase lifecycle:
  configure → ingest → compute → collect → verify → present

This demo uses ComputeBackend.LOCAL so it runs without external services.
It reads text files, computes word-frequency statistics, and outputs
a summary table — showing how the Pipeline SDK structures any
multi-step AI workflow.

For real workloads, swap LOCAL with:
  - UMRS_EMBEDDING  → ZMQ via UMRSClientConnector (embeddings on Slurm)
  - BENTOML_TASK    → HTTP via BentoMLTaskClient (vision inference)

Usage:
    python3 text_analysis_pipeline.py [--input-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pipeline SDK imports
from umrs.pipeline_sdk import (
    ComputeBackend,
    DataItem,
    Pipeline,
    PipelineConfig,
    Result,
    TaskHandle,
)


class TextAnalysisPipeline(Pipeline):
    """Analyse text files: word counts, top words, and readability stats.

    Demonstrates all 6 pipeline phases using the LOCAL compute backend.
    """

    def __init__(self, input_dir: str, top_n: int = 10):
        super().__init__()
        self._input_dir = input_dir
        self._top_n = top_n

    # ── Phase 1: CONFIGURE ───────────────────────────────────────────────

    def configure(self) -> PipelineConfig:
        """Set up a LOCAL-mode pipeline — no external services needed."""
        return PipelineConfig(
            name="text-analysis-demo",
            compute_backend=ComputeBackend.LOCAL,
            extra={"top_n": self._top_n},
        )

    # ── Phase 2: INGEST ──────────────────────────────────────────────────

    def ingest(self, config: PipelineConfig) -> List[DataItem]:
        """Scan input directory for .txt files and create DataItems."""
        input_path = Path(self._input_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self._input_dir}")

        items: List[DataItem] = []
        for path in sorted(input_path.rglob("*.txt")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            items.append(
                DataItem(
                    key=path.name,
                    source=str(path),
                    metadata={
                        "size_bytes": path.stat().st_size,
                        "content": text,
                    },
                )
            )

        if not items:
            self._log.warning("No .txt files found in %s", self._input_dir)

        return items

    # ── Phase 3: COMPUTE ─────────────────────────────────────────────────

    def compute(self, config: PipelineConfig, items: List[DataItem]) -> List[TaskHandle]:
        """Compute word-frequency statistics for each text file.

        In LOCAL mode this runs in-process.  With UMRS_EMBEDDING or
        BENTOML_TASK the computation would be submitted to a remote
        backend and the handle would reference a pending task.
        """
        handles: List[TaskHandle] = []
        for item in items:
            text = item.metadata.get("content", "")
            stats = self._analyse_text(text, config.extra.get("top_n", 10))

            handle = TaskHandle(
                task_id=uuid.uuid4().hex[:12],
                endpoint="local/text_analysis",
                data_item=item,
                metadata={"result": stats},  # LOCAL: result is immediate
            )
            handles.append(handle)

        return handles

    # ── Phase 4: COLLECT ─────────────────────────────────────────────────

    def collect(self, config: PipelineConfig, handles: List[TaskHandle]) -> List[Result]:
        """Collect results from completed handles.

        For LOCAL mode the result is already in handle.metadata.
        For remote backends this would poll until tasks complete.
        """
        results: List[Result] = []
        for handle in handles:
            stats = handle.metadata.get("result")
            if stats is None:
                results.append(
                    Result(
                        task_handle=handle,
                        status="failure",
                        error="No result in handle metadata",
                    )
                )
            else:
                results.append(
                    Result(
                        task_handle=handle,
                        status="success",
                        data=stats,
                    )
                )
        return results

    # ── Phase 5: VERIFY (optional override) ──────────────────────────────

    def verify(self, config: PipelineConfig, results: List[Result]) -> bool:
        """Check that every file produced valid statistics."""
        ok = super().verify(config, results)
        if ok:
            # Additional check: each result must have word_count > 0
            for r in results:
                if r.data.get("word_count", 0) == 0:
                    self._log.warning(
                        "File %s has 0 words",
                        r.task_handle.data_item.key if r.task_handle.data_item else "?",
                    )
                    ok = False
        return ok

    # ── Phase 6: PRESENT (optional override) ─────────────────────────────

    def present(self, config: PipelineConfig, results: List[Result]) -> None:
        """Print a detailed summary including top words per file."""
        # Default table
        super().present(config, results)

        # Extra: show top words for each file
        print("\n── Top Words per File ──")
        for r in results:
            if r.status != "success":
                continue
            name = r.task_handle.data_item.key if r.task_handle.data_item else "?"
            top = r.data.get("top_words", [])
            top_str = ", ".join(f"{w}({c})" for w, c in top)
            print(f"  {name}: {top_str}")

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _analyse_text(text: str, top_n: int = 10) -> Dict[str, Any]:
        """Return word-frequency statistics for *text*."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        counter = Counter(words)
        total_chars = len(text)
        unique_words = len(counter)
        sentences = len(re.findall(r"[.!?]+", text)) or 1

        return {
            "mode": "local",
            "num_objects": len(words),
            "word_count": len(words),
            "unique_words": unique_words,
            "char_count": total_chars,
            "sentence_count": sentences,
            "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
            "avg_words_per_sentence": round(len(words) / sentences, 1),
            "top_words": counter.most_common(top_n),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        }


# ── CLI Entry Point ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline SDK Demo — Text Analysis Pipeline",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing .txt files to analyse (default: ../../data/file_of_texts/)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top words to show per file (default: 10)",
    )
    args = parser.parse_args()

    # Default to the project's sample text data
    input_dir = args.input_dir
    if input_dir is None:
        script_dir = Path(__file__).resolve().parent
        input_dir = str(script_dir / "../../data/file_of_texts")

    pipeline = TextAnalysisPipeline(input_dir=input_dir, top_n=args.top_n)
    results = pipeline.run()

    # Exit code: 0 if all succeeded, 1 otherwise
    failures = sum(1 for r in results if r.status != "success")
    raise SystemExit(failures)


if __name__ == "__main__":
    main()
