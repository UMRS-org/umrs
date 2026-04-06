#!/usr/bin/env python3
"""
Generate synthetic test videos for the video analysis demo.

Creates short MP4 videos with a moving white rectangle on a black
background.  Every 3rd video is static (no motion) so the motion
detection step has both positive and negative cases.

Usage:
    python generate_test_videos.py                    # 4 videos, 10 s each
    python generate_test_videos.py --count 8 --fps 30
    python generate_test_videos.py -o /tmp/vids
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("ERROR: opencv-python is required.\n  pip install opencv-python-headless")


def generate_video(
    path: str,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    duration: int = 10,
    moving: bool = True,
) -> None:
    """Write a single MP4 video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    box_w, box_h = 80, 60
    total_frames = fps * duration

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if moving:
            # Move diagonally with wrapping
            x = int((i * 3) % (width - box_w))
            y = int((i * 2) % (height - box_h))
        else:
            # Static rectangle in the center
            x = (width - box_w) // 2
            y = (height - box_h) // 2

        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test videos")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("--count", type=int, default=4, help="Number of videos")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    out_dir = args.output or os.path.join(os.path.dirname(__file__), "sample_videos")
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(args.count):
        moving = (idx % 3) != 2  # every 3rd video is static
        fname = f"test_video_{idx:03d}_{'motion' if moving else 'static'}.mp4"
        path = os.path.join(out_dir, fname)
        generate_video(
            path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            moving=moving,
        )
        tag = "motion" if moving else "static"
        print(f"  [{idx + 1}/{args.count}] {fname}  ({tag}, {args.duration}s)")

    print(f"\nGenerated {args.count} videos in {out_dir}/")


if __name__ == "__main__":
    main()
