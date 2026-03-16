#!/usr/bin/env python3
"""
SAM3 Segmentation Demo via BentoML Task Queue API.

Demonstrates:
1. Upload an image to MinIO S3 bucket
2. Submit SAM3 segmentation task via BentoML async task queue (POST /submit)
3. Poll task status (GET /status)
4. Retrieve segmentation results (GET /get)
5. Display results (masks, bounding boxes, scores)

The BentoML vision service exposes:
    POST /sam3_segment/submit   — submit task, returns task_id
    GET  /sam3_segment/status   — check task status
    GET  /sam3_segment/get      — retrieve result
    POST /sam3_segment/cancel   — cancel pending task
    PUT  /sam3_segment/retry    — retry failed task

Requires:
    - BentoML vision service running (via Rivault stack with vision-bentoml profile)
    - MinIO S3 bucket with test images
    - HF_TOKEN env var set for facebook/sam3 model access

Usage:
    python3 demo_sam3_segmentation.py \\
        --bentoml-url http://localhost:3000 \\
        --s3-endpoint http://localhost:9000 \\
        --image-path /path/to/test_image.jpg \\
        --prompt "crack"
"""

import argparse
import json
import os
import sys
import time

import requests


def upload_image_to_s3(s3_endpoint, access_key, secret_key, bucket, image_path):
    """Upload an image to MinIO S3 bucket."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    key = os.path.basename(image_path)
    s3.upload_file(image_path, bucket, key)
    print(f"  Uploaded: {image_path} -> s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


def submit_sam3_task(bentoml_url, image_url, prompt, threshold=0.5):
    """Submit a SAM3 segmentation task via BentoML task queue."""
    url = f"{bentoml_url}/sam3_segment/submit"
    payload = {
        "image_url": image_url,
        "prompt": prompt,
        "threshold": threshold,
        "mask_threshold": 0.5,
    }

    print(f"\n  POST {url}")
    print(f"  Payload: {json.dumps(payload, indent=2)}")

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("task_id") or data.get("id")
    print(f"  Task submitted! ID: {task_id}")
    return task_id


def poll_task_status(bentoml_url, task_id, poll_interval=2.0, timeout=600.0):
    """Poll task status until completion."""
    url = f"{bentoml_url}/sam3_segment/status"
    start = time.time()

    while True:
        elapsed = time.time() - start

        try:
            resp = requests.get(url, params={"task_id": task_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"\r\033[2K  [{elapsed:.1f}s] Poll error: {exc}")
            if elapsed > timeout:
                return "timeout"
            time.sleep(poll_interval)
            continue

        status = str(data.get("status", data.get("value", "unknown"))).strip()
        print(f"\r\033[2K  [{elapsed:.1f}s] Status: {status}", end="", flush=True)

        if status in ("success", "completed"):
            print(f"\n  Task completed in {elapsed:.1f}s")
            return "success"

        if status in ("failure", "failed", "cancelled"):
            print(f"\n  Task ended: {status}")
            return "failure"

        if elapsed > timeout:
            print(f"\n  TIMEOUT: Task did not complete within {timeout}s")
            return "timeout"

        time.sleep(poll_interval)


def get_task_result(bentoml_url, task_id):
    """Retrieve the result of a completed task."""
    url = f"{bentoml_url}/sam3_segment/get"
    resp = requests.get(url, params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="SAM3 Segmentation Demo")
    parser.add_argument("--bentoml-url", default="http://localhost:3000",
                        help="BentoML vision service URL")
    parser.add_argument("--s3-endpoint", default="http://localhost:9000",
                        help="MinIO/S3 endpoint URL")
    parser.add_argument("--s3-access-key", default="minioadmin")
    parser.add_argument("--s3-secret-key", default="minioadmin")
    parser.add_argument("--bucket", default="vision-demo",
                        help="S3 bucket name for demo images")
    parser.add_argument("--image-path", required=True,
                        help="Path to image file to segment")
    parser.add_argument("--prompt", default="object",
                        help="Text prompt for segmentation (e.g. 'crack', 'void')")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detection")
    parser.add_argument("--timeout", type=float, default=600,
                        help="Max seconds to wait for task completion")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip upload (image already in S3)")
    parser.add_argument("--image-url",
                        help="Direct S3 URL (skip upload entirely)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAM3 Segmentation Demo (BentoML Task Queue)")
    print("=" * 60)

    # --- Step 1: Check BentoML service health ---
    print("\n--- Step 1: Check BentoML service health ---")
    try:
        resp = requests.get(f"{args.bentoml_url}/healthz", timeout=5)
        if resp.status_code == 200:
            print(f"  BentoML service healthy at {args.bentoml_url}")
        else:
            print(f"  WARNING: BentoML returned status {resp.status_code}")
    except requests.ConnectionError:
        print(f"  ERROR: Cannot connect to BentoML at {args.bentoml_url}")
        sys.exit(1)

    # --- Step 2: Upload image to S3 ---
    if args.image_url:
        image_url = args.image_url
        print(f"\n--- Step 2: Using provided S3 URL ---")
        print(f"  {image_url}")
    elif args.skip_upload:
        image_url = f"s3://{args.bucket}/{os.path.basename(args.image_path)}"
        print(f"\n--- Step 2: Skipping upload, using ---")
        print(f"  {image_url}")
    else:
        print(f"\n--- Step 2: Upload image to S3 ---")
        image_url = upload_image_to_s3(
            args.s3_endpoint, args.s3_access_key, args.s3_secret_key,
            args.bucket, args.image_path,
        )

    # --- Step 3: Submit SAM3 segmentation task ---
    print(f"\n--- Step 3: Submit SAM3 task ---")
    print(f"  Image: {image_url}")
    print(f"  Prompt: '{args.prompt}'")
    task_id = submit_sam3_task(args.bentoml_url, image_url, args.prompt, args.threshold)

    # --- Step 4: Poll status ---
    print(f"\n--- Step 4: Waiting for task completion ---")
    status = poll_task_status(args.bentoml_url, task_id,
                               timeout=args.timeout)

    if status != "success":
        print(f"\n  Task ended with status: {status}")
        sys.exit(1)

    # --- Step 5: Retrieve results ---
    print(f"\n--- Step 5: Retrieve results ---")
    result = get_task_result(args.bentoml_url, task_id)

    print(f"\n  Results:")
    print(f"  - Objects found: {result.get('num_objects', 0)}")
    print(f"  - Prompt: '{result.get('prompt', args.prompt)}'")

    if result.get("boxes"):
        print(f"  - Bounding boxes:")
        for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            print(f"    [{i}] box={box}, score={score:.3f}")

    if result.get("overlay_s3_key"):
        print(f"  - Overlay image saved to: {result['overlay_s3_key']}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
