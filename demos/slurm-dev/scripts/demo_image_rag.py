#!/usr/bin/env python3
"""
Image RAG Demo: Upload images, compute CLIP embeddings, and search.

This script demonstrates the UMRS image RAG pipeline:
1. Upload images to MinIO via the UMRS client
2. UMRS computes CLIP embeddings (SentenceTransformer, runs locally)
3. Embeddings stored in pgvector
4. Cross-modal text-to-image search
5. SAM3 segmentation via MCP tool call (if always-on LLM is running)

Run inside umrs-client or ai-login container:
    python3 /opt/scripts/demo_image_rag.py

Prerequisites:
    - MinIO must have the rivault-image-bucket (run setup_demo_data.sh first)
    - Images must be uploaded to the bucket
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, "/opt/umrs")


def upload_images(server_ip, conf_file, bucket_alias, image_dir):
    """Upload image files to the UMRS server."""
    print("\n" + "=" * 60)
    print("  Step 1: Upload Images")
    print("=" * 60)

    extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
    )

    if not image_files:
        print(f"  No image files found in {image_dir}")
        return []

    uploaded = []
    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        print(f"\n  Uploading: {fname}")
        cmd = [
            "python3", "/opt/umrs/client_umrs.py",
            "-c", conf_file,
            "--server-ip", server_ip,
            "--no-check-server",
            "upload_file", fpath, bucket_alias,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    OK: {result.stdout.strip()[:200]}")
            uploaded.append(fname)
        else:
            print(f"    ERROR: {result.stderr.strip()[:200]}")

    print(f"\n  Uploaded {len(uploaded)}/{len(image_files)} images.")
    return uploaded


def wait_for_embeddings(server_ip, conf_file, timeout=300):
    """Wait for the server to finish computing CLIP embeddings."""
    print("\n" + "=" * 60)
    print("  Step 2: Wait for CLIP Embedding Computation")
    print("=" * 60)

    print("  Waiting for UMRS server to compute CLIP embeddings...")
    print("  (CLIP runs locally via SentenceTransformer, no Slurm needed)")

    start = time.time()
    while time.time() - start < timeout:
        cmd = [
            "python3", "/opt/umrs/client_umrs.py",
            "-c", conf_file,
            "--server-ip", server_ip,
            "status", "all",
            "--output", "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                status = json.loads(result.stdout)
                print(f"  Status: {json.dumps(status, indent=2)[:300]}")
                if status.get("status") == "ok":
                    summary = status.get("summary", {})
                    if isinstance(summary, dict):
                        task_status = summary.get("task_status", "")
                        if task_status in ("COMPLETED", "completed"):
                            print("  CLIP embeddings computed!")
                            return True
            except json.JSONDecodeError:
                pass

        time.sleep(5)

    print(f"  WARNING: Timed out after {timeout}s.")
    return False


def image_search(rag_url, query, top_k=5):
    """Cross-modal text-to-image search."""
    print(f"\n  Query: \"{query}\"")

    payload = {
        "query": query,
        "top_k": top_k,
        "modality": "image",
        "bucket_alias": "rivault-image-bucket",
        "xp_name": "Image_CLIP",
    }

    try:
        resp = requests.post(f"{rag_url}/rag/search", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            print("    No results found.")
            return results

        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            meta = r.get("metadata", {})
            s3_obj = meta.get("s3_object", "unknown")
            print(f"    [{i}] score={score:.4f} image={s3_obj}")

        return results

    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return []


def sam3_segment(llm_url, llm_key, image_url, prompt):
    """Request SAM3 segmentation via the always-on LLM's MCP tool calling."""
    print(f"\n  SAM3 Segmentation Request:")
    print(f"    Image: {image_url}")
    print(f"    Prompt: {prompt}")

    headers = {
        "Authorization": f"Bearer {llm_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to image analysis tools. "
                    "Use the vision/segment tool to analyze images when asked."
                ),
            },
            {
                "role": "user",
                "content": f"Please segment this image and describe what you see: {image_url}. {prompt}",
            },
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            f"{llm_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        print(f"\n    Response:\n    {answer[:500]}")
        return data

    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        print("    (SAM3 requires the always-on LLM profile to be active)")
        return {}


def main():
    parser = argparse.ArgumentParser(description="UMRS Image RAG Demo")
    parser.add_argument(
        "--server-ip", default="ai-login",
        help="UMRS server IP/hostname (default: ai-login)",
    )
    parser.add_argument(
        "--rag-url", default="http://ai-login:8081",
        help="RAG service base URL",
    )
    parser.add_argument(
        "--llm-url", default="http://llm-litellm:4000",
        help="Always-on LLM API URL",
    )
    parser.add_argument(
        "--llm-key", default="sk-llm-demo",
        help="LiteLLM master key for always-on LLM",
    )
    parser.add_argument(
        "--conf", default="/opt/umrs/conf_umrs.json",
        help="Client config file path",
    )
    parser.add_argument(
        "--image-dir", default="/opt/test_data/images",
        help="Directory containing image test files",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip upload step",
    )
    parser.add_argument(
        "--skip-wait", action="store_true",
        help="Skip waiting for embedding computation",
    )
    parser.add_argument(
        "--skip-sam3", action="store_true",
        help="Skip SAM3 segmentation demo",
    )
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("#  UMRS Image RAG Demo (CLIP + SAM3)")
    print("#" * 60)

    bucket_alias = "rivault-image-bucket"

    # Step 1: Upload
    if not args.skip_upload:
        upload_images(args.server_ip, args.conf, bucket_alias, args.image_dir)
    else:
        print("\n  Skipping upload (--skip-upload)")

    # Step 2: Wait for CLIP embeddings
    if not args.skip_wait:
        wait_for_embeddings(args.server_ip, args.conf)
    else:
        print("\n  Skipping wait (--skip-wait)")

    # Step 3: Cross-modal search
    print("\n" + "=" * 60)
    print("  Step 3: Cross-Modal Text-to-Image Search")
    print("=" * 60)

    queries = [
        "a photo of a cat",
        "a bird flying in the sky",
        "a dog playing outdoors",
    ]

    for q in queries:
        image_search(args.rag_url, q)

    # Step 4: SAM3 segmentation (requires always-on LLM)
    if not args.skip_sam3:
        print("\n" + "=" * 60)
        print("  Step 4: SAM3 Image Segmentation via MCP")
        print("=" * 60)

        sam3_segment(
            args.llm_url,
            args.llm_key,
            "s3://rivault-image-bucket/Cat_1.jpeg",
            "Identify and segment the main subject in the image.",
        )
    else:
        print("\n  Skipping SAM3 (--skip-sam3)")

    print("\n" + "#" * 60)
    print("#  Image RAG Demo Complete")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
