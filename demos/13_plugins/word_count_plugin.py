#!/usr/bin/env python3
"""
Sample plugin script — word frequency counter.

This is a self-contained plugin that can be executed by a PluginRunner
(Slurm via Singularity, or K8s Job). It reads input files from the
workspace, computes word frequencies, and writes results to the output
directory.

When submitted via UMRS, the runner sets these environment variables:
  PLUGIN_ENTRYPOINT   — "word_count_plugin:main"
  PLUGIN_SOURCE_TYPE  — "local_path"
  PLUGIN_SOURCE_URI   — path to this file
  PLUGIN_PROJECT_ID   — project identifier
  PLUGIN_USER_ID      — submitting user
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def main():
    """Plugin entrypoint — reads /workspace/input/, writes to /workspace/output/."""
    workspace = os.environ.get("PLUGIN_WORKSPACE", "/workspace")
    input_dir = Path(workspace) / "input"
    output_dir = Path(workspace) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}
    total_words = 0

    for txt_file in sorted(input_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        words = re.findall(r"[a-zA-Z]+", text.lower())
        counter = Counter(words)

        file_result = {
            "word_count": len(words),
            "unique_words": len(counter),
            "top_10": counter.most_common(10),
        }
        results[txt_file.name] = file_result
        total_words += len(words)
        print(f"Processed: {txt_file.name} ({len(words)} words)")

    # Write summary
    summary = {
        "files_processed": len(results),
        "total_words": total_words,
        "per_file": results,
    }

    output_file = output_dir / "word_frequencies.json"
    output_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Results written to: {output_file}")

    # Write metrics (read by the runner)
    metrics_file = Path(workspace) / "metrics.txt"
    metrics_file.write_text(f"files_processed={len(results)}\ntotal_words={total_words}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
