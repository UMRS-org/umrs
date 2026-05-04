"""Shared output artifact naming for built-in vision adapters.

This module is intentionally lightweight so it can be imported by both the
UMRS server process and the BentoML runtime without pulling in heavy ML
dependencies.
"""

from __future__ import annotations

from typing import Dict


_HMAE_MODE_TO_TASK = {
    "reconstruct": "reconstruction",
    "mae": "mae_reconstruction",
    "features": "feature_extraction",
}

# HuggingFace task → primary output suffix mapping
_HF_TASK_SUFFIXES: Dict[str, str] = {
    "image-classification": "_results.pkl",
    "zero-shot-image-classification": "_results.pkl",
    "object-detection": "_results.pkl",
    "zero-shot-object-detection": "_results.pkl",
    "image-segmentation": "_seg.png",
    "semantic-segmentation": "_seg.png",
    "instance-segmentation": "_seg.png",
    "panoptic-segmentation": "_seg.png",
    "image-feature-extraction": "_features.pt",
    "depth-estimation": "_depth.png",
    "image-to-image": "_output.png",
}


def resolve_hmae_task(config: dict) -> str:
    """Resolve the effective HMAE task from adapter config."""
    task = config.get("task")
    if task:
        return task
    mode = config.get("mode", "")
    return _HMAE_MODE_TO_TASK.get(mode, "reconstruction")


def describe_output_artifacts(model_type: str, key: str, config: dict) -> Dict[str, str]:
    """Return deterministic output keys and content types for built-in adapters."""
    base = key.rsplit(".", 1)[0] if "." in key else key

    if model_type == "sam3":
        return {
            f"{base}_sam3_seg.png": "image/png",
            f"{base}_sam3_results.pkl": "application/octet-stream",
        }

    if model_type == "hmae":
        task = resolve_hmae_task(config)
        if task == "reconstruction":
            return {
                f"{base}_recon.png": "image/png",
                f"{base}_recon_results.pkl": "application/octet-stream",
            }
        if task == "mae_reconstruction":
            return {
                f"{base}_mae.png": "image/png",
                f"{base}_mae_results.pkl": "application/octet-stream",
            }
        if task == "feature_extraction":
            return {
                f"{base}_features.pt": "application/octet-stream",
                f"{base}_features_results.pkl": "application/octet-stream",
            }

    if model_type == "huggingface":
        hf_task = config.get("task", "image-classification")
        suffix = _HF_TASK_SUFFIXES.get(hf_task, "_results.pkl")
        outputs: Dict[str, str] = {}
        if suffix.endswith(".png"):
            outputs[f"{base}{suffix}"] = "image/png"
        elif suffix.endswith(".pt"):
            outputs[f"{base}{suffix}"] = "application/octet-stream"
        outputs[f"{base}_hf_results.pkl"] = "application/octet-stream"
        return outputs

    return {}
