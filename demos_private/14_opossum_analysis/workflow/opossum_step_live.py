"""Live wrapper for opossum workflow uploads.

Uses a fresh module name so re-uploaded workflows pick up the updated
entrypoint code even when the long-lived server process still has the
older ``opossum_step`` module cached.
"""

import tempfile
from pathlib import Path
from typing import Any

from umrs.pipeline_sdk import StepContext, StepResult

from opossum_step import OpossumAnalyzeStep as _BaseOpossumAnalyzeStep
from opossum_step import _ensure_opossum_importable, _have_cuda


class OpossumAnalyzeStep(_BaseOpossumAnalyzeStep):
    """Demo upload-safe wrapper around the base opossum step."""

    def run(self, ctx: StepContext) -> StepResult:
        _ensure_opossum_importable()
        use_cuda = _have_cuda()
        ctx.metadata["use_cuda"] = use_cuda

        input_keys = ctx.input_keys or self._discover_input_keys(ctx)
        cameras = self._group_by_camera(input_keys)

        output_keys: list[str] = []
        all_events: list[dict[str, Any]] = []
        total_pairs = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            for camera_name, video_keys in cameras.items():
                local_paths = self._download_camera(camera_name, video_keys, tmp, ctx)
                if local_paths is None:
                    continue

                camera_events, camera_keys = self._process_camera_pairs(
                    ctx, tmp, camera_name, local_paths, use_cuda
                )
                all_events.extend(camera_events)
                output_keys.extend(camera_keys)
                total_pairs += max(0, len(local_paths) - 1)

        self._upload_summary(ctx, all_events, output_keys, total_pairs, cameras, use_cuda)
        return StepResult(output_keys=output_keys)

    @staticmethod
    def _discover_input_keys(ctx: StepContext) -> list[str]:
        paginator = ctx.s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=ctx.input_bucket):
            keys.extend(
                obj["Key"]
                for obj in page.get("Contents", [])
                if obj["Key"].lower().endswith(".mp4")
            )
        return keys
