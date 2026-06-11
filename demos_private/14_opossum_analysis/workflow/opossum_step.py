"""Opossum motion analysis step for UMRS Pipeline SDK.

Wraps the opossum frame-difference analysis engine as a UDM Step.
Processes consecutive video pairs and outputs TSV files with motion events.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from umrs.pipeline_sdk import Step, StepContext, StepResult


class OpossumAnalyzeStep(Step):
    """Detect motion events in consecutive video files."""

    name = "opossum_analyze"

    def setup(self, config: dict) -> None:
        self.distance_threshold = config.get("distance_threshold", 0.05)
        self.time_threshold = config.get("time_threshold", 15)

    def run(self, ctx: StepContext) -> StepResult:
        # Ensure the opossum engine is importable
        repo_root = os.environ.get("REPO_ROOT", "")
        opossum_src = os.path.join(repo_root, "tmp", "opossum", "src")
        if os.path.isdir(opossum_src) and opossum_src not in sys.path:
            sys.path.insert(0, opossum_src)

        from opossum.analyze import get_duration

        # Sort video keys for consecutive pairing
        video_keys = sorted(k for k in ctx.input_keys if k.lower().endswith((".mp4", ".avi")))

        output_keys = []
        all_events = []

        # Download videos to a temp directory for OpenCV processing
        with tempfile.TemporaryDirectory() as tmpdir:
            local_paths = []
            for key in video_keys:
                data = ctx.download_bytes(key)
                local_path = Path(tmpdir) / os.path.basename(key)
                with open(local_path, "wb") as f:
                    f.write(data)
                local_paths.append(local_path)

            # Process consecutive pairs
            for i in range(len(local_paths) - 1):
                v1, v2 = local_paths[i], local_paths[i + 1]
                pair_name = f"{os.path.basename(v1)}_{os.path.basename(v2)}"
                try:
                    events = get_duration(
                        v1,
                        v2,
                        distance_threshold=self.distance_threshold,
                        time_threshold=self.time_threshold,
                    )
                    # First element is header ["start_time", "end_time"], skip it
                    data_events = events[1:] if events and len(events) > 1 else []
                    for ev in data_events:
                        all_events.append([pair_name] + list(ev))
                except Exception as exc:
                    ctx.metadata.setdefault("errors", []).append(
                        {"pair": pair_name, "error": str(exc)}
                    )

        # Write TSV output
        tsv_lines = ["video_pair\tstart_time\tend_time\n"]
        for ev in all_events:
            tsv_lines.append("\t".join(str(x) for x in ev) + "\n")
        tsv_bytes = "".join(tsv_lines).encode("utf-8")

        out_key = "results/motion_events.tsv"
        ctx.upload_bytes(out_key, tsv_bytes, content_type="text/tab-separated-values")
        output_keys.append(out_key)

        # Also write a JSON summary
        summary = {
            "total_videos": len(video_keys),
            "pairs_analyzed": max(0, len(video_keys) - 1),
            "total_events": len(all_events),
            "distance_threshold": self.distance_threshold,
            "time_threshold": self.time_threshold,
            "events": [
                {"pair": ev[0], "start": str(ev[1]), "end": str(ev[2])} for ev in all_events
            ],
        }
        summary_key = "results/analysis_summary.json"
        ctx.upload_bytes(
            summary_key,
            json.dumps(summary, indent=2).encode("utf-8"),
            content_type="application/json",
        )
        output_keys.append(summary_key)

        return StepResult(output_keys=output_keys)

    def teardown(self) -> None:
        pass
