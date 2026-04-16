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

    return {}