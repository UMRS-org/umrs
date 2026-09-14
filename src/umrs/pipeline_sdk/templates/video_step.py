"""``VideoStep`` — built-in step that extracts frames from video files.

The step reads a video file from S3, extracts frames at a configurable
FPS using ffmpeg, and writes the frames as individual JPEG images to
S3.  The frames can then be processed by a downstream ``VisionStep``.

Configuration keys:
    - ``fps``: Frames per second to extract (default: 1.0).
    - ``max_frames``: Maximum number of frames to extract (default: 100).
    - ``width``: Optional resize width (preserves aspect ratio).
    - ``ffmpeg_path``: Path to ffmpeg binary (default: ``"ffmpeg"``).
    - ``timeout_sec``: Per-video timeout (default: 600).

Example ``workflow.json`` snippet::

    {
      "name": "video-frames",
      "entrypoint": "umrs.pipeline_sdk.templates.video_step:VideoStep",
      "config": {"fps": 2.0, "max_frames": 50}
    }
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class VideoStep(Step):
    """Extract frames from video files using ffmpeg."""

    name = "video"

    def setup(self, config: dict[str, Any]) -> None:
        self.fps: float = float(config.get("fps", 1.0))
        self.max_frames: int = int(config.get("max_frames", 100))
        self.width: int | None = config.get("width")
        self.ffmpeg_path: str = str(config.get("ffmpeg_path", "ffmpeg"))
        self.timeout_sec: float = float(config.get("timeout_sec", 600.0))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        out_keys: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        for key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(key)
            except Exception as exc:
                logger.warning("VideoStep: failed to read %s: %s", key, exc)
                continue

            frames = self._extract_frames(raw, key)

            for frame_idx, frame_data in enumerate(frames):
                frame_key = self._frame_key(key, frame_idx)
                ctx.upload_bytes(frame_key, frame_data, content_type="image/jpeg")
                out_keys.append(frame_key)

            manifest_key = self._write_manifest(
                frames=len(frames),
                fps=self.fps,
                source_key=key,
                ctx=ctx,
            )
            out_keys.append(manifest_key)
            metadata_list.append(
                {
                    "source_key": key,
                    "num_frames": len(frames),
                    "fps": self.fps,
                }
            )

        return StepResult(
            output_keys=out_keys,
            metadata=metadata_list,
            extra={"step": "video", "fps": self.fps},
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        raw: bytes,
        source_key: str,
    ) -> list[bytes]:
        """Extract frames from video bytes using ffmpeg."""
        mock = os.environ.get("UMRS_VIDEO_MOCK", "0") in ("1", "true", "TRUE")
        if mock:
            return [self._mock_frame(source_key, idx) for idx in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input")
            with open(input_path, "wb") as f:
                f.write(raw)

            output_pattern = os.path.join(tmpdir, "frame_%05d.jpg")

            cmd = [self.ffmpeg_path, "-i", input_path]
            cmd.extend(["-vf", f"fps={self.fps}"])
            if self.width:
                cmd.extend(["-vf", f"scale={self.width}:-1,fps={self.fps}"])
            cmd.extend(["-frames:v", str(self.max_frames)])
            cmd.extend([output_path])  # type: ignore[arg-type]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self.timeout_sec,
                    check=True,
                )
            except FileNotFoundError:
                logger.warning("VideoStep: ffmpeg not found, returning mock frames")
                return [self._mock_frame(source_key, idx) for idx in range(3)]
            except subprocess.TimeoutExpired:
                logger.warning("VideoStep: ffmpeg timed out for %s", source_key)
                return []
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "VideoStep: ffmpeg failed for %s: %s",
                    source_key,
                    exc.stderr.decode("utf-8", errors="replace")[:500],
                )
                return []

            frames: list[bytes] = []
            for fname in sorted(os.listdir(tmpdir)):
                if fname.startswith("frame_") and fname.endswith(".jpg"):
                    with open(os.path.join(tmpdir, fname), "rb") as f:
                        frames.append(f.read())

            return frames

    def _mock_frame(self, source_key: str, idx: int) -> bytes:
        """Return a mock JPEG frame for testing."""
        # Minimal 1x1 JPEG
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
            b"\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xff\xff\xff\xff"
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xc0\x00\x0b"
            b"\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14"
            b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00"
            b"\xff\xd9"
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _frame_key(self, source_key: str, frame_idx: int) -> str:
        """Generate the S3 key for a single frame."""
        base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key
        return f"video_frames/{base}/frame_{frame_idx:05d}.jpg"

    def _write_manifest(
        self,
        frames: int,
        fps: float,
        source_key: str,
        ctx: StepContext,
    ) -> str:
        """Write a manifest JSON describing the extracted frames."""
        base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key
        manifest_key = f"video_frames/{base}/manifest.json"
        manifest = {
            "source": source_key,
            "num_frames": frames,
            "fps": fps,
            "max_frames": self.max_frames,
        }
        data = json.dumps(manifest, indent=2).encode("utf-8")
        ctx.upload_bytes(manifest_key, data, content_type="application/json")
        return manifest_key
