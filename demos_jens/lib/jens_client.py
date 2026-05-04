#!/usr/bin/env python3
"""
Lightweight client for Jens's VisionFrontend BentoML service.

Submits async tasks to /vision_task and polls for completion.
Supports SAM3 and HMAE models with image input from local files,
HTTP(S) URLs, or S3 URIs.

Usage:
    python jens_client.py --server http://localhost --port 3000 \
        --model sam3 --task "cat" --image /path/to/image.jpg

    python jens_client.py --server http://localhost --port 3000 \
        --model hmae --task reconstruct --image s3://bucket/slice.png \
        --out s3://bucket/result.pkl
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import sys
import time
from pathlib import Path

try:
    import bentoml
except ImportError:
    sys.exit("bentoml is required: pip install bentoml>=1.4")


async def submit_and_poll(
    server: str,
    port: int,
    token: str | None,
    image: str,
    model: str,
    task: str,
    out: str = "raw",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> dict | None:
    """Submit a vision task and poll until complete."""
    url = f"{server}:{port}"
    async with bentoml.AsyncHTTPClient(url, token=token) as client:
        print(f"[jens] Submitting: model={model} task={task} image={image} out={out}")
        bento_task = await client.vision_task.submit(
            image=image, model=model, task=task, out=out,
        )
        print(f"[jens] Task ID: {bento_task.id}")

        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                print(f"[jens] ERROR: Timeout after {timeout}s", file=sys.stderr)
                return None

            status = await bento_task.get_status()
            if status.value == "completed":
                raw = await bento_task.get()
                result = pickle.loads(raw) if isinstance(raw, bytes) else raw
                print(f"[jens] Task completed in {elapsed:.1f}s")
                return result
            elif status.value in ("failed", "failure"):
                print(f"[jens] Task failed after {elapsed:.1f}s", file=sys.stderr)
                return None
            elif status.value == "canceled":
                print("[jens] Task canceled", file=sys.stderr)
                return None

            if int(elapsed) % 30 == 0 and elapsed > 0:
                print(f"[jens]   waiting... {elapsed:.0f}s ({status.value})")
            await asyncio.sleep(poll_interval)


def run(
    server: str = "http://localhost",
    port: int = 3000,
    token: str | None = None,
    model: str = "sam3",
    task: str = "cat",
    image: str = "",
    out: str = "raw",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> dict | None:
    """Synchronous wrapper around submit_and_poll."""
    return asyncio.run(
        submit_and_poll(server, port, token, image, model, task, out,
                        poll_interval, timeout)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Jens VisionFrontend client")
    parser.add_argument("--server", default=os.getenv("JENS_SERVER", "http://localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("JENS_PORT", "3000")))
    parser.add_argument("--token", default=os.getenv("JENS_TOKEN"))
    parser.add_argument("--model", required=True, choices=["sam3", "hmae"])
    parser.add_argument("--task", required=True,
                        help="SAM3: object name (e.g. 'cat'). HMAE: reconstruct|mae|features")
    parser.add_argument("--image", required=True,
                        help="Local path, HTTP(S) URL, or s3:// URI")
    parser.add_argument("--out", default="raw",
                        help="'raw' for pickle result, or s3:// URI for direct upload")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", default=None,
                        help="Save raw pickle result to this directory")
    args = parser.parse_args()

    result = run(
        server=args.server,
        port=args.port,
        token=args.token,
        model=args.model,
        task=args.task,
        image=args.image,
        out=args.out,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    if result is None:
        return 1

    status = result.get("status", "unknown")
    print(f"[jens] Result status: {status}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(args.image).stem
        out_file = out_dir / f"{img_stem}_{args.model}_{args.task}.pkl"
        with open(out_file, "wb") as handle:
            pickle.dump(result, handle)
        print(f"[jens] Saved result to {out_file}")

    if status in ("ok", "success"):
        data_keys = [key for key in result if key not in ("status",)]
        print(f"[jens] Result keys: {data_keys}")
        return 0

    print(f"[jens] ERROR: {result}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())