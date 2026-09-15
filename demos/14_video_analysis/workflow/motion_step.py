"""
Motion Detection Step — analyse video files for motion events.

Uses OpenCV frame differencing (no external model/GPU required).
Compares consecutive frames, thresholds changes, and reports
contiguous motion events with timestamps.

Input keys:  ``*.mp4`` video files in the input bucket
Output keys: ``results/<basename>_motion.json`` per video
             ``results/summary.json``  aggregate report

Config:
  - ``diff_threshold``:    pixel intensity change threshold (default 30)
  - ``area_threshold``:    minimum contour area in pixels (default 500)
  - ``min_event_frames``:  min consecutive frames to count as event (default 3)
  - ``sample_fps``:        frame sampling rate (default 5, 0 = all frames)
  - ``output_prefix``:     prefix for output keys (default "results/")
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class MotionDetectStep(Step):
    """Detect motion events in video files via frame differencing."""

    name = "motion_detect"

    def setup(self, config: Dict[str, Any]) -> None:
        self.diff_threshold = config.get("diff_threshold", 30)
        self.area_threshold = config.get("area_threshold", 500)
        self.min_event_frames = config.get("min_event_frames", 3)
        self.sample_fps = config.get("sample_fps", 5)
        self.output_prefix = config.get("output_prefix", "results/")

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        all_summaries: List[Dict[str, Any]] = []

        for idx, key in enumerate(ctx.input_keys):
            if not key.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                logger.info("Skipping non-video file: %s", key)
                continue

            logger.info("[%d/%d] Analysing: %s", idx + 1, len(ctx.input_keys), key)
            video_bytes = ctx.download_bytes(key)

            events, frame_count, fps_actual = self._analyse_video(video_bytes)

            base = key.rsplit(".", 1)[0] if "." in key else key
            base = base.rsplit("/", 1)[-1]  # strip path prefix

            result = {
                "source_key": key,
                "total_frames": frame_count,
                "video_fps": fps_actual,
                "duration_seconds": round(frame_count / max(fps_actual, 1), 2),
                "motion_events": len(events),
                "events": events,
            }
            all_summaries.append(result)

            out_key = f"{self.output_prefix}{base}_motion.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(result, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            metadata_list.append(result)
            checksums.append("")

            logger.info(
                "  → %d motion events (%d frames, %.1fs)",
                len(events),
                frame_count,
                frame_count / max(fps_actual, 1),
            )

        # Aggregate summary
        summary = {
            "total_videos": len(all_summaries),
            "total_events": sum(s["motion_events"] for s in all_summaries),
            "videos_with_motion": sum(1 for s in all_summaries if s["motion_events"] > 0),
            "per_video": all_summaries,
        }
        summary_key = f"{self.output_prefix}summary.json"
        ctx.upload_bytes(
            summary_key,
            json.dumps(summary, indent=2).encode(),
            content_type="application/json",
        )
        output_keys.append(summary_key)

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={"total_events": summary["total_events"]},
        )

    # ------------------------------------------------------------------

    def _analyse_video(self, video_bytes: bytes) -> tuple[list[dict], int, float]:
        """Frame-difference analysis returning (events, frame_count, fps)."""
        # Write to tmpfile for OpenCV (it can't read from memory)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)

        if not cap.isOpened():
            logger.warning("Could not open video")
            return [], 0, 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(fps / self.sample_fps)) if self.sample_fps > 0 else 1

        prev_gray = None
        frame_idx = 0
        motion_frames: List[int] = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                has_motion = any(cv2.contourArea(c) >= self.area_threshold for c in contours)
                if has_motion:
                    motion_frames.append(frame_idx)

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # Cluster consecutive motion frames into events
        events = self._cluster_events(motion_frames, fps)
        return events, frame_count, fps

    def _cluster_events(self, motion_frames: List[int], fps: float) -> List[Dict[str, Any]]:
        """Group consecutive motion frames into discrete events."""
        if not motion_frames:
            return []

        events: List[Dict[str, Any]] = []
        start = motion_frames[0]
        prev = start

        for f in motion_frames[1:]:
            if f - prev > 3:  # gap of >3 sampled frames → new event
                if prev - start + 1 >= self.min_event_frames:
                    events.append(self._make_event(start, prev, fps))
                start = f
            prev = f

        # Close last event
        if prev - start + 1 >= self.min_event_frames:
            events.append(self._make_event(start, prev, fps))

        return events

    @staticmethod
    def _make_event(start: int, end: int, fps: float) -> Dict[str, Any]:
        return {
            "start_frame": start,
            "end_frame": end,
            "start_sec": round(start / max(fps, 1), 2),
            "end_sec": round(end / max(fps, 1), 2),
            "duration_sec": round((end - start) / max(fps, 1), 2),
        }
