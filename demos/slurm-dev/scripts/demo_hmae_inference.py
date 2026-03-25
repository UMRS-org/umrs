#!/usr/bin/env python3
"""
HMAE Inference Demo via BentoML Task Queue API.

Demonstrates three inference modes for HMAE (Hierarchical Masked Auto-Encoder)
on X-ray CT images of concrete specimens:

  1. MAE Reconstruction (with masking)
     - Randomly masks 50% of patches, reconstructs masked regions
     - Output: side-by-side Original / Quadtree Grid + Mask / Reconstruction

  2. Full Reconstruction (no masking)
     - All patches fed to encoder then decoder
     - Output: side-by-side Original / Quadtree Grid / Full Reconstruction

  3. Feature Extraction
     - Encoder only, outputs CLS token + patch features
     - Output: .pt file with cls, patches, coords tensors

The BentoML vision service exposes:
    POST /infer/submit  — submit task, returns task_id
    GET  /infer/status  — check task status
    GET  /infer/get     — retrieve result
    POST /hmae_reconstruct/cancel  — cancel pending task
    PUT  /hmae_reconstruct/retry   — retry failed task

Requires:
    - BentoML vision service running (via Rivault stack with vision-bentoml profile)
    - MinIO S3 bucket with test images
    - HMAE checkpoint loaded in the service

Usage:
    # Run all 3 modes on Spring8 concrete specimen images
    python3 demo_hmae_inference.py \\
        --bentoml-url http://localhost:3000 \\
        --s3-endpoint http://localhost:9000 \\
        --image-dir /home/mnagaso/hmae-sample/spring8concrete/ \\
        --all

    # Run a specific mode
    python3 demo_hmae_inference.py \\
        --bentoml-url http://localhost:3000 \\
        --image-path /path/to/volume_170.png \\
        --mode mae
"""

import argparse
import json
import os
import sys
import time

import bentoml


def upload_images_to_s3(s3_endpoint, access_key, secret_key, bucket, image_paths):
    """Upload images to MinIO S3 bucket."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    s3_urls = []
    for path in image_paths:
        key = os.path.basename(path)
        s3.upload_file(path, bucket, key)
        url = f"s3://{bucket}/{key}"
        print(f"  Uploaded: {path} -> {url}")
        s3_urls.append(url)

    return s3_urls


def submit_and_wait_hmae(client, image_url, mode="reconstruct",
                         poll_interval=2.0, timeout=600.0):
    """Submit an HMAE task and wait for the result."""
    print(f"  Submitting HMAE task: image={image_url}, mode={mode}")

    task = client.infer.submit(
        model_type="hmae",
        inference_config={"mode": mode},
        image_url=image_url,
    )
    print(f"  Task submitted! ID: {task.id}")

    start = time.time()
    while True:
        elapsed = time.time() - start
        status = task.get_status()
        print(f"\r\033[2K  [{elapsed:.1f}s] Status: {status.value}", end="", flush=True)

        if status.value == "success":
            print(f"\n  Task completed in {elapsed:.1f}s")
            return task.get()

        if status.value in ("failure", "cancelled"):
            print(f"\n  Task ended: {status.value}")
            return {"error": f"Task {task.id} {status.value}"}

        if elapsed > timeout:
            print(f"\n  TIMEOUT: Task did not complete within {timeout}s")
            return {"error": f"Task timed out after {timeout}s"}

        time.sleep(poll_interval)


def run_single_mode(client, image_url, mode, timeout=300):
    """Run one HMAE inference mode and display results."""
    mode_descriptions = {
        "mae": "MAE Reconstruction (with masking)",
        "reconstruct": "Full Reconstruction (no masking)",
        "features": "Feature Extraction (encoder only)",
    }

    print(f"\n{'─' * 50}")
    print(f"  Mode: {mode_descriptions.get(mode, mode)}")
    print(f"  Image: {image_url}")
    print(f"{'─' * 50}")

    # Submit
    result = submit_and_wait_hmae(client, image_url, mode, timeout=timeout)

    if "error" in result:
        print(f"  Task failed: {result['error']}")
        return None

    print(f"\n  Result:")
    print(f"  - Mode: {result.get('mode', mode)}")
    print(f"  - Source: {result.get('source_image', image_url)}")

    if mode == "mae" and "mae_loss" in result:
        print(f"  - MAE Loss: {result['mae_loss']:.6f}")

    if mode == "features":
        if "cls_shape" in result:
            print(f"  - CLS token shape: {result['cls_shape']}")
        if "patch_shape" in result:
            print(f"  - Patch tokens shape: {result['patch_shape']}")

    if "output_s3_key" in result:
        print(f"  - Output saved to: {result['output_s3_key']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="HMAE Inference Demo (3 modes)")
    parser.add_argument("--bentoml-url", default="http://localhost:3000",
                        help="BentoML vision service URL")
    parser.add_argument("--s3-endpoint", default="http://localhost:9000",
                        help="MinIO/S3 endpoint URL")
    parser.add_argument("--s3-access-key", default="minioadmin")
    parser.add_argument("--s3-secret-key", default="minioadmin")
    parser.add_argument("--bucket", default="hmae-demo",
                        help="S3 bucket name for demo images")
    parser.add_argument("--image-path",
                        help="Path to a single image file")
    parser.add_argument("--image-dir",
                        help="Directory of images (batch mode)")
    parser.add_argument("--image-url",
                        help="Direct S3 URL (skip upload)")
    parser.add_argument("--mode", choices=["reconstruct", "mae", "features"],
                        default="reconstruct",
                        help="Inference mode (ignored with --all)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 3 modes on each image")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Max seconds to wait per task")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip upload (images already in S3)")
    args = parser.parse_args()

    print("=" * 60)
    print("  HMAE Inference Demo (BentoML Task Queue)")
    print("  Hierarchical Masked Auto-Encoder for X-ray CT")
    print("=" * 60)

    # --- Step 1: Check BentoML service health ---
    print("\n--- Step 1: Check BentoML service health ---")
    try:
        client = bentoml.SyncHTTPClient(args.bentoml_url,
                                        server_ready_timeout=30)
        print(f"  BentoML service healthy at {args.bentoml_url}")
    except Exception as exc:
        print(f"  ERROR: Cannot connect to BentoML at {args.bentoml_url}: {exc}")
        sys.exit(1)

    # --- Step 2: Collect image URLs ---
    image_urls = []

    if args.image_url:
        image_urls = [args.image_url]
        print(f"\n--- Step 2: Using provided S3 URL ---")
        print(f"  {args.image_url}")
    else:
        # Collect local image paths
        image_paths = []
        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        if args.image_path:
            image_paths = [args.image_path]
        elif args.image_dir:
            image_paths = sorted(
                os.path.join(args.image_dir, f)
                for f in os.listdir(args.image_dir)
                if f.lower().endswith(extensions)
            )

        if not image_paths:
            print("  ERROR: No images specified. Use --image-path, --image-dir, or --image-url")
            sys.exit(1)

        print(f"\n--- Step 2: {'Skipping upload' if args.skip_upload else 'Upload images to S3'} ---")
        print(f"  Found {len(image_paths)} image(s)")

        if args.skip_upload:
            image_urls = [f"s3://{args.bucket}/{os.path.basename(p)}" for p in image_paths]
        else:
            image_urls = upload_images_to_s3(
                args.s3_endpoint, args.s3_access_key, args.s3_secret_key,
                args.bucket, image_paths,
            )

    # --- Step 3: Run inference ---
    modes = ["mae", "reconstruct", "features"] if args.all else [args.mode]

    print(f"\n--- Step 3: Run HMAE inference ---")
    print(f"  Images: {len(image_urls)}")
    print(f"  Modes: {', '.join(modes)}")

    all_results = []
    for image_url in image_urls:
        for mode in modes:
            result = run_single_mode(client, image_url, mode, args.timeout)
            if result:
                all_results.append(result)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Demo Summary")
    print(f"{'=' * 60}")
    print(f"  Total tasks: {len(all_results)} completed successfully")

    for r in all_results:
        mode = r.get("mode", "unknown")
        output = r.get("output_s3_key", "N/A")
        extra = ""
        if mode == "mae" and "mae_loss" in r:
            extra = f" (loss={r['mae_loss']:.6f})"
        elif mode == "features" and "cls_shape" in r:
            extra = f" (CLS={r['cls_shape']})"
        print(f"  [{mode}] -> {output}{extra}")

    print(f"\n  Done!")
    client.close()


if __name__ == "__main__":
    main()
