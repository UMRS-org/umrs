"""Opossum motion analysis step for UMRS Pipeline SDK.

Wraps the real opossum frame-difference motion detection pipeline as a UDM Step.
Processes per-camera video directories, comparing consecutive video pairs with
GPU-accelerated OpenCV (cv2.cuda) where available.

Real pipeline stages implemented:
  1. VFR→CFR conversion (ffmpeg) for real camera footage
  2. Frame-difference motion detection (binarize → absdiff → morphology → merge)
  3. Per-camera incremental tracking via checkpoint files
  4. Per-pair TSV output + aggregated motion_events.tsv
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from umrs.pipeline_sdk import Step, StepContext, StepResult

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── GPU utility ─────────────────────────────────────────────────────────────

def _have_cuda() -> bool:
    """Check if OpenCV CUDA module is available."""
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


# ── VFR→CFR conversion (from real pipeline) ────────────────────────────────

def vfr2cfr(file_path: Path, frame_rate: int = 5) -> Path:
    """Convert variable-framerate video to constant framerate via ffmpeg.

    Real trail-camera footage is often VFR. This is a required pre-processing
    step before OpenCV frame-difference analysis.
    """
    cmd_ffprobe = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    r_frame_rate = subprocess.check_output(cmd_ffprobe, text=True).strip()

    if r_frame_rate == "0/0":
        temp_path = Path("/tmp") / f"{file_path.stem}_cfr.mp4"
        cmd_ffmpeg = [
            "ffmpeg", "-i", str(file_path),
            "-vf", f"fps={frame_rate}",
            "-c:v", "libx264", "-c:a", "copy",
            "-y", str(temp_path),
        ]
        subprocess.run(cmd_ffmpeg, check=True, capture_output=True)
        return temp_path
    return file_path


# ── Datetime extraction from filename (from real pipeline) ──────────────────

def get_datetime_from_filename(filename: str) -> datetime | None:
    """Extract recording timestamp from trail-camera filename.

    Supports:
      - New cameras (TS-WPTCAM2): 20230916_121705.mp4
      - Old cameras (QWATCH TS-NS310W): Record-2023-04-16_00-00-02.avi

    Returns None for filenames outside the recognised production patterns
    (foreign keys in the shared bucket must not crash camera discovery).
    """
    date_str = filename.replace("Record-", "")
    date_str = date_str.replace(".mp4", "").replace(".avi", "")
    date_str = date_str.replace("-", "")
    try:
        return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


# ── Checkpoint utils (incremental tracking) ────────────────────────────────

def _read_checkpoint(path: Path) -> datetime | None:
    """Read last-processed datetime from checkpoint file."""
    if path.exists():
        return datetime.strptime(path.read_text().strip(), DATETIME_FORMAT)
    return None


def _write_checkpoint(path: Path, dt: datetime) -> None:
    """Write last-processed datetime to checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dt.strftime(DATETIME_FORMAT))


# ── Motion analysis (frame-difference) ─────────────────────────────────────

def _frame_diff_gpu(frame, prev_frame) -> "np.ndarray":  # noqa: F821
    """GPU-accelerated frame difference: binarize → absdiff."""
    import cv2
    gpu_f = cv2.cuda_GpuMat()
    gpu_p = cv2.cuda_GpuMat()
    gpu_f.upload(frame)
    gpu_p.upload(prev_frame)
    gpu_bin = cv2.cuda.cvtColor(gpu_f, cv2.COLOR_BGR2GRAY)
    cv2.cuda.threshold(gpu_bin, 128, 255, cv2.THRESH_BINARY, dst=gpu_bin)
    gpu_pbin = cv2.cuda.cvtColor(gpu_p, cv2.COLOR_BGR2GRAY)
    cv2.cuda.threshold(gpu_pbin, 128, 255, cv2.THRESH_BINARY, dst=gpu_pbin)
    return cv2.cuda.absdiff(gpu_bin, gpu_pbin).download()


def _frame_diff_cpu(frame, prev_frame) -> "np.ndarray":  # noqa: F821
    """CPU frame difference: binarize → absdiff."""
    import cv2
    from opossum.utils import binarize_image
    return cv2.absdiff(binarize_image(frame), binarize_image(prev_frame))


def _process_frames(
    cap, frame_rate, merge_threshold, width, height, time_limit, use_cuda
) -> list[float]:
    """Read frames from *cap* and compute pixel-rate list, stopping at *time_limit* sec."""
    import cv2
    from opossum.utils import remove_white_noise, merge_adjacent_boxes, calculate_mask_percentage_in_rectangle

    rates: list[float] = []
    ret, prev = cap.read()
    if not ret:
        return rates
    ret, cur = cap.read()

    diff_fn = _frame_diff_gpu if use_cuda else _frame_diff_cpu

    while ret:
        if time_limit is not None:
            sec = round(cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_rate)
            if sec > time_limit:
                break

        bdiff = diff_fn(cur, prev)
        denoised = remove_white_noise(bdiff, 5)
        merged = merge_adjacent_boxes(denoised, threshold=merge_threshold)
        rate = calculate_mask_percentage_in_rectangle(
            merged, {"x": 0, "y": 0, "width": width, "height": height}
        )
        rates.append(rate)
        prev = cur.copy()
        ret, cur = cap.read()

    return rates


def _analyze_pair(
    v1: Path,
    v2: Path,
    distance_threshold: float,
    time_threshold: float,
    use_cuda: bool,
) -> list[list[str]]:
    """Compare two videos with frame-difference motion detection."""
    import cv2

    path1 = vfr2cfr(v1)
    path2 = vfr2cfr(v2)
    cap1 = cv2.VideoCapture(str(path1))
    cap2 = cv2.VideoCapture(str(path2))

    try:
        frame_rate = cap1.get(cv2.CAP_PROP_FPS)
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        merge_threshold = width * distance_threshold
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video 1: full; Video 2: capped at time_threshold
        rates1 = _process_frames(cap1, frame_rate, merge_threshold, width, height, None, use_cuda)
        rates2 = _process_frames(cap2, frame_rate, merge_threshold, width, height, time_threshold, use_cuda)

        events = _find_continuous_regions(rates1 + rates2, frame_rate * time_threshold)
        return [[_seconds_to_hms(start), _seconds_to_hms(end)] for start, end in events]
    finally:
        cap1.release()
        cap2.release()
        if path1 != v1:
            path1.unlink(missing_ok=True)
        if path2 != v2:
            path2.unlink(missing_ok=True)


def _find_continuous_regions(
    rates: list[float], min_gap_frames: float
) -> list[tuple[int, int]]:
    """Group non-zero pixel-rate frames into continuous motion regions.

    Regions separated by < min_gap_frames are merged.
    """
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    n = len(rates)

    for i in range(n):
        if rates[i] > 0:
            if not in_region:
                start = i
                in_region = True
        else:
            if in_region:
                # Look ahead: if next non-zero within min_gap_frames, continue
                gap_end = min(i + int(min_gap_frames) + 1, n)
                if any(rates[j] > 0 for j in range(i + 1, gap_end)):
                    continue
                regions.append((start, i - 1))
                in_region = False

    if in_region:
        regions.append((start, n - 1))

    return regions


def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Step class ──────────────────────────────────────────────────────────────

class OpossumAnalyzeStep(Step):
    """Detect motion events in per-camera consecutive video pairs.

    Mirrors the production ``main.py`` pipeline:
      - Multi-camera aware (opossum_* dirs)
      - VFR→CFR conversion
      - Frame-difference motion detection
      - Per-pair TSV output
      - Incremental checkpoint tracking
      - GPU-accelerated via cv2.cuda when available
    """

    name = "opossum_analyze"

    def setup(self, config: dict) -> None:
        self.distance_threshold = config.get("distance_threshold", 0.05)
        self.time_threshold = config.get("time_threshold", 15.0)

    def run(self, ctx: StepContext) -> StepResult:
        _ensure_opossum_importable()
        use_cuda = _have_cuda()
        ctx.metadata["use_cuda"] = use_cuda

        input_keys = ctx.input_keys
        if not input_keys:
            input_keys = self._discover_input_keys(ctx)

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

    def teardown(self) -> None:
        pass

    # ── Per-camera helpers ───────────────────────────────────────────────

    @staticmethod
    def _discover_input_keys(ctx: StepContext) -> list[str]:
        """List input bucket for video keys when input_keys is empty.

        Entry-point steps (no ``depends_on``) receive an empty
        ``ctx.input_keys`` from the workflow runner.  This method
        lists the input bucket and returns all ``.mp4`` keys so the
        step can discover its own inputs.
        """
        resp = ctx.s3.list_objects_v2(Bucket=ctx.input_bucket)
        return [
            obj["Key"]
            for obj in resp.get("Contents", [])
            if obj["Key"].lower().endswith(".mp4")
        ]

    def _download_camera(
        self, camera_name: str, video_keys: list[str], tmp: Path, ctx: StepContext
    ) -> list[Path] | None:
        """Download new videos for *camera_name*, respecting checkpoints.

        Returns local paths, or None if camera should be skipped.
        """
        # Only keys matching recognised production patterns participate;
        # foreign keys in the shared bucket are ignored (no crash on sort).
        stamped = [
            (dt, k)
            for k in video_keys
            if (dt := get_datetime_from_filename(os.path.basename(k))) is not None
        ]
        stamped.sort()
        video_keys = [k for _, k in stamped]

        checkpoint_path = tmp / "checkpoints" / f"last_target_{camera_name}.txt"
        last_processed = _read_checkpoint(checkpoint_path)

        new_keys = self._filter_new_keys(video_keys, last_processed)
        if len(new_keys) < 2:
            ctx.metadata.setdefault("skipped_cameras", []).append(
                f"{camera_name}: only {len(new_keys)} new file(s)"
            )
            return None

        local_paths: list[Path] = []
        for i, key in enumerate(new_keys):
            data = ctx.download_bytes(key)
            local_path = tmp / camera_name / os.path.basename(key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            local_paths.append(local_path)
            ctx.report_progress(
                (i / len(new_keys)) * 50,
                f"Downloading {camera_name}: {os.path.basename(key)}",
            )

        return local_paths

    def _process_camera_pairs(
        self, ctx: StepContext, tmp: Path, camera_name: str,
        local_paths: list[Path], use_cuda: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Analyze consecutive pairs, write per-pair TSVs and checkpoint."""
        all_events: list[dict[str, Any]] = []
        output_keys: list[str] = []
        checkpoint_path = tmp / "checkpoints" / f"last_target_{camera_name}.txt"
        newest_dt: datetime | None = None
        n = len(local_paths)

        for i in range(n - 1):
            v1, v2 = local_paths[i], local_paths[i + 1]
            pair_name = f"{v1.stem}_{v2.stem}"
            ctx.report_progress(
                50 + (i / max(1, n - 1)) * 50,
                f"Analyzing {camera_name}: {pair_name}",
            )

            try:
                events = _analyze_pair(
                    v1, v2,
                    distance_threshold=self.distance_threshold,
                    time_threshold=self.time_threshold,
                    use_cuda=use_cuda,
                )
                tsv_lines = ["start_time\tend_time\n"]
                for ev in events:
                    tsv_lines.append(f"{ev[0]}\t{ev[1]}\n")
                    all_events.append({
                        "camera": camera_name, "pair": pair_name,
                        "start_time": ev[0], "end_time": ev[1],
                    })

                pair_key = f"results/{camera_name}/{v1.stem}.tsv"
                ctx.upload_bytes(pair_key, "".join(tsv_lines).encode("utf-8"),
                                content_type="text/tab-separated-values")
                output_keys.append(pair_key)

            except Exception as exc:
                ctx.metadata.setdefault("errors", []).append(
                    {"camera": camera_name, "pair": pair_name, "error": str(exc)}
                )

            try:
                file_dt = get_datetime_from_filename(v2.name)
                if newest_dt is None or file_dt > newest_dt:
                    newest_dt = file_dt
            except ValueError:
                pass

        if newest_dt:
            _write_checkpoint(checkpoint_path, newest_dt)
            ckpt_key = f"results/{camera_name}/last_target.txt"
            ctx.upload_bytes(ckpt_key, newest_dt.strftime(DATETIME_FORMAT).encode("utf-8"),
                            content_type="text/plain")
            output_keys.append(ckpt_key)

        return all_events, output_keys

    @staticmethod
    def _filter_new_keys(
        video_keys: list[str], last_processed: datetime | None
    ) -> list[str]:
        """Return keys newer than *last_processed* by filename datetime."""
        new_keys = []
        for k in video_keys:
            try:
                file_dt = get_datetime_from_filename(os.path.basename(k))
                if last_processed is None or file_dt > last_processed:
                    new_keys.append(k)
            except ValueError:
                new_keys.append(k)
        return new_keys

    def _upload_summary(
        self, ctx: StepContext, all_events: list[dict[str, Any]],
        output_keys: list[str], total_pairs: int,
        cameras: dict[str, list[str]], use_cuda: bool,
    ) -> None:
        """Write aggregated TSV and JSON summary."""
        tsv_lines = ["camera\tvideo_pair\tstart_time\tend_time\n"]
        for ev in all_events:
            tsv_lines.append(
                f"{ev['camera']}\t{ev['pair']}\t{ev['start_time']}\t{ev['end_time']}\n"
            )
        tsv_key = "results/motion_events.tsv"
        ctx.upload_bytes(tsv_key, "".join(tsv_lines).encode("utf-8"),
                         content_type="text/tab-separated-values")
        output_keys.append(tsv_key)

        summary = {
            "gpu_accelerated": use_cuda,
            "cameras_analyzed": list(cameras.keys()),
            "total_pairs": total_pairs,
            "total_events": len(all_events),
            "distance_threshold": self.distance_threshold,
            "time_threshold": self.time_threshold,
        }
        summary_key = "results/analysis_summary.json"
        ctx.upload_bytes(summary_key, json.dumps(summary, indent=2).encode("utf-8"),
                         content_type="application/json")
        output_keys.append(summary_key)

    # ── Camera grouping ──────────────────────────────────────────────────

    @staticmethod
    def _group_by_camera(input_keys: list[str]) -> dict[str, list[str]]:
        """Group input S3 keys by camera directory.

        Real data layout:
          opossum_215_A/opossumA/Schedule/20260330/00/20260330_000001.mp4
          opossum_215_B/opossumB/Schedule/20260330/00/20260330_000001.mp4

        Camera name extracted as first two path components: ``opossum_215_A/opossumA``
        """
        from collections import defaultdict
        cameras: dict[str, list[str]] = defaultdict(list)

        for key in input_keys:
            parts = key.split("/")
            # Match pattern: opossum_NNN_X/opossumX/...
            if len(parts) >= 2 and parts[0].startswith("opossum_"):
                camera = f"{parts[0]}/{parts[1]}"
            else:
                # Fallback: use first directory component
                camera = parts[0] if len(parts) > 1 else "default"
            cameras[camera].append(key)

        return dict(cameras)


# ── Import helper ───────────────────────────────────────────────────────────

_OPOSSUM_IMPORTED = False


def _ensure_opossum_importable() -> None:
    """Ensure the opossum engine package is on sys.path.

    The opossum engine is bundled with the workflow upload and lives
    alongside this step file in the workflow source tree.
    """
    global _OPOSSUM_IMPORTED
    if _OPOSSUM_IMPORTED:
        return

    # In server-side execution, the workflow source tree is the working dir.
    # The opossum package should be in the same directory as this step.
    step_dir = Path(__file__).resolve().parent
    opossum_dir = step_dir / "opossum"
    if opossum_dir.is_dir() and str(step_dir) not in sys.path:
        sys.path.insert(0, str(step_dir))

    _OPOSSUM_IMPORTED = True
