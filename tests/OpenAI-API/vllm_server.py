#!/usr/bin/env python3
"""
vllm_server.py

Launch a vLLM OpenAI-compatible API server with automatic CPU/GPU detection
and safe default configuration.
"""

import subprocess
import sys

import torch

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """Detect whether CUDA GPU is available or fallback to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Automatic vLLM configuration
# ---------------------------------------------------------------------------


def auto_vllm_config(
    model_max_len: int | None = None,
) -> dict:
    """
    Build a safe vLLM configuration depending on CPU vs GPU.

    Guarantees:
        max_num_batched_tokens >= max_model_len
    """
    device = detect_device()

    if device == "cuda":
        # GPU configuration
        max_len = model_max_len or 32768
        max_len = min(max_len, 32768)

        return {
            "device": "cuda",
            "dtype": "bfloat16",
            "max_model_len": max_len,
            "max_num_batched_tokens": max_len,
        }

    # CPU configuration (conservative & stable)
    max_len = min(model_max_len or 4096, 4096)

    return {
        "device": "cpu",
        "dtype": "float16",
        "max_model_len": max_len,
        "max_num_batched_tokens": max_len,
    }


# ---------------------------------------------------------------------------
# Launch vLLM OpenAI server
# ---------------------------------------------------------------------------


def launch_vllm_server(
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    convert: str = "auto",
    runner: str = "auto",
    model_max_len: int | None = None,
):
    """
    Launch the vLLM OpenAI-compatible API server.
    """
    cfg = auto_vllm_config(model_max_len)

    # Safety check
    assert cfg["max_num_batched_tokens"] >= cfg["max_model_len"], (
        "Invalid vLLM configuration: max_num_batched_tokens must be >= max_model_len"
    )

    print("=== vLLM configuration ===")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("==========================")
    # /hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/Users/ejeannot/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "--runner",
        runner,
        "--convert",
        convert,
        "--host",
        host,
        "--port",
        str(port),
        "--distributed-executor-backend",
        "None",  # désactiver TCPStore / multi-node
    ]

    cmd += [
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
    ]

    # Pour le mode embedding / pooling, pas besoin de max-model-len ni max-num-batched-tokens
    is_embedding = (convert == "embed") or (runner == "pooling")
    if not is_embedding:
        cmd += [
            "--max-model-len",
            str(cfg["max_model_len"]),
            "--max-num-batched-tokens",
            str(cfg["max_num_batched_tokens"]),
        ]

    print("\nLaunching vLLM server:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Change the model if needed
    MODEL_NAME = "intfloat/e5-base-v2"

    launch_vllm_server(
        model_name=MODEL_NAME,
        host="0.0.0.0",
        port=8001,
        convert="embed",
        runner="pooling",
        model_max_len=None,  # Let auto config decide
    )
