"""
Example user-defined step: text statistics.

Reads text files from S3, computes character/word/line counts, writes
a JSON summary per file.  Demonstrates a text-processing step with
no ML dependencies.
"""

from umrs.pipeline_sdk import Step, StepContext, StepResult
import json


class TextStatsStep(Step):
    """Compute basic statistics for each input text file."""

    name = "text_stats"

    def setup(self, config: dict) -> None:
        self.output_prefix = config.get("output_prefix", "stats/")

    def run(self, ctx: StepContext) -> StepResult:
        output_keys = []
        checksums = []
        metadata_list = []

        for key in ctx.input_keys:
            raw = ctx.download_bytes(key)
            text = raw.decode("utf-8", errors="replace")

            stats = {
                "source_key": key,
                "characters": len(text),
                "words": len(text.split()),
                "lines": text.count("\n") + 1,
            }

            out_key = f"{self.output_prefix}{key}.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(stats, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)

            head = ctx.head_object(key)
            checksums.append(head["ETag"].strip('"'))
            metadata_list.append(head.get("Metadata", {}))

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
        )
