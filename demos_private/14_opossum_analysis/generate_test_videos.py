#!/usr/bin/env python3
"""Generate synthetic test videos for the opossum motion analysis demo.

Creates short MP4 videos with moving objects (white rectangles on black
background) to simulate motion events detectable by the frame-difference
algorithm.

Usage:
    python generate_test_videos.py --output-dir sample_videos/ --num-videos 4
"""
import argparse
import os
import math

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python-headless and numpy are required.")
    print("  pip install opencv-python-headless numpy")
    raise SystemExit(1)


def generate_video(
    output_path: str,
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    label: str = "",
    add_motion: bool = True,
):
    """Generate a synthetic video with optional moving objects."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    num_frames = duration_sec * fps

    for frame_num in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Timestamp overlay
        t = frame_num / fps
        ts = f"Frame {frame_num:04d}  t={t:.2f}s"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if label:
            cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Optional moving rectangle to simulate motion
        if add_motion:
            cx = int(width * 0.5 + width * 0.3 * math.sin(2.0 * math.pi * t / duration_sec))
            cy = int(height * 0.5 + height * 0.2 * math.cos(2.0 * math.pi * t / duration_sec))
            size = 40
            cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (255, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"  Generated: {output_path} ({duration_sec}s, {fps}fps, {width}x{height})")


def main():
    parser = argparse.ArgumentParser(description="Generate test videos for opossum demo")
    parser.add_argument("--output-dir", default="sample_videos", help="Output directory")
    parser.add_argument("--num-videos", type=int, default=4, help="Number of videos")
    parser.add_argument("--duration", type=int, default=10, help="Duration per video (seconds)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    for i in range(args.num_videos):
        label = labels[i % len(labels)]
        # Alternate: some videos have motion, some are static
        has_motion = (i % 3 != 2)  # every 3rd video is static
        output_path = os.path.join(args.output_dir, f"camera01_{i+1:04d}.mp4")
        generate_video(
            output_path,
            duration_sec=args.duration,
            fps=args.fps,
            label=f"Video {i+1}: {label}",
            add_motion=has_motion,
        )

    print(f"Generated {args.num_videos} test videos in {args.output_dir}/")


if __name__ == "__main__":
    main()
