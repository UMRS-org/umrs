#!/usr/bin/env python3
"""
Text RAG Demo: Upload text documents, compute embeddings, and search.

This script demonstrates the full UMRS text RAG pipeline:
1. Upload text files to MinIO via the UMRS client
2. UMRS triggers Slurm job -> Rivault -> vLLM embedding computation
3. Embeddings stored in pgvector
4. Semantic search via /rag/search
5. QA with LLM via /rag/qa (if always-on LLM is running)

Run inside umrs-client or ai-login container:
    python3 /opt/scripts/demo_text_rag.py

Or with custom server IP:
    python3 /opt/scripts/demo_text_rag.py --server-ip ai-login
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

# Add UMRS source to path
sys.path.insert(0, "/opt/umrs")


def upload_text_files(server_ip, conf_file, bucket_alias, text_dir):
    """Upload all .txt files from text_dir to the UMRS server."""
    print("\n" + "=" * 60)
    print("  Step 1: Upload Text Files")
    print("=" * 60)

    txt_files = sorted(
        f for f in os.listdir(text_dir) if f.endswith(".txt")
    )

    if not txt_files:
        print(f"  No .txt files found in {text_dir}")
        return []

    uploaded = []
    for fname in txt_files:
        fpath = os.path.join(text_dir, fname)
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

    print(f"\n  Uploaded {len(uploaded)}/{len(txt_files)} files.")
    return uploaded


def wait_for_embeddings(server_ip, conf_file, timeout=300):
    """Wait for the server to finish computing embeddings."""
    print("\n" + "=" * 60)
    print("  Step 2: Wait for Embedding Computation")
    print("=" * 60)

    print("  Waiting for UMRS server to process uploads...")
    print("  (This triggers Slurm job -> Rivault -> vLLM -> embeddings)")

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
                # Check if all tasks are completed
                if status.get("status") == "ok":
                    summary = status.get("summary", {})
                    if isinstance(summary, dict):
                        task_status = summary.get("task_status", "")
                        if task_status in ("COMPLETED", "completed"):
                            print("  Embeddings computed successfully!")
                            return True
            except json.JSONDecodeError:
                pass

        time.sleep(10)

    print(f"  WARNING: Timed out after {timeout}s waiting for embeddings.")
    return False


def rag_search(rag_url, query, top_k=5):
    """Perform semantic search via the RAG service."""
    print(f"\n  Query: \"{query}\"")

    payload = {
        "query": query,
        "top_k": top_k,
        "modality": "text",
        "bucket_alias": "rivault-demo-bucket",
        "xp_name": "Text_Rivault",
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
            content = r.get("content", "")[:150]
            print(f"    [{i}] score={score:.4f}: {content}...")

        return results

    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return []


def rag_qa(rag_url, query, top_k=5):
    """Ask a question via the RAG QA endpoint."""
    print(f"\n  Question: \"{query}\"")

    payload = {
        "query": query,
        "top_k": top_k,
        "modality": "text",
        "bucket_alias": "rivault-demo-bucket",
        "xp_name": "Text_Rivault",
        "max_context_chunks": 5,
        "enable_query_rewrite": True,
    }

    try:
        resp = requests.post(f"{rag_url}/rag/qa", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "(no answer)")
        rewritten = data.get("rewritten_queries", [])
        contexts = data.get("contexts", [])

        if rewritten:
            print(f"    Rewritten queries: {rewritten}")
        print(f"    Context chunks used: {len(contexts)}")
        print(f"\n    Answer:\n    {answer}")

        return data

    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="UMRS Text RAG Demo")
    parser.add_argument(
        "--server-ip", default="ai-login",
        help="UMRS server IP/hostname for ZeroMQ (default: ai-login)",
    )
    parser.add_argument(
        "--rag-url", default="http://ai-login:8081",
        help="RAG service base URL (default: http://ai-login:8081)",
    )
    parser.add_argument(
        "--conf", default="/opt/umrs/conf_umrs.json",
        help="Client config file path",
    )
    parser.add_argument(
        "--text-dir", default="/opt/test_data/text",
        help="Directory containing .txt test files",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip upload step (data already in MinIO)",
    )
    parser.add_argument(
        "--skip-wait", action="store_true",
        help="Skip waiting for embedding computation",
    )
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("#  UMRS Text RAG Demo")
    print("#" * 60)

    bucket_alias = "rivault-demo-bucket"

    # Step 1: Upload
    if not args.skip_upload:
        upload_text_files(args.server_ip, args.conf, bucket_alias, args.text_dir)
    else:
        print("\n  Skipping upload (--skip-upload)")

    # Step 2: Wait for embeddings
    if not args.skip_wait:
        wait_for_embeddings(args.server_ip, args.conf)
    else:
        print("\n  Skipping wait (--skip-wait)")

    # Step 3: Search
    print("\n" + "=" * 60)
    print("  Step 3: Semantic Search")
    print("=" * 60)

    queries = [
        "How do seismic waves propagate through Earth's crust?",
        "What are the benefits of GPU computing for scientific simulations?",
        "How does the RiVault platform serve AI models on HPC?",
    ]

    for q in queries:
        rag_search(args.rag_url, q)

    # Step 4: QA (requires always-on LLM)
    print("\n" + "=" * 60)
    print("  Step 4: RAG Question Answering")
    print("=" * 60)

    qa_queries = [
        "Explain the difference between P-waves and S-waves and their significance.",
        "How can machine learning improve seismic interpretation?",
    ]

    for q in qa_queries:
        rag_qa(args.rag_url, q)

    print("\n" + "#" * 60)
    print("#  Text RAG Demo Complete")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
