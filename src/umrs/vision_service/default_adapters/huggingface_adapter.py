"""Generic HuggingFace Vision Adapter.

Serves ANY HuggingFace vision model through the ``transformers`` library
without requiring model-specific adapter code.

Supported tasks (auto-detected from model config or specified via
``inference_config.task``):
  - image-classification
  - object-detection
  - image-segmentation
  - image-feature-extraction
  - zero-shot-image-classification
  - zero-shot-object-detection
  - depth-estimation

Usage::

    # In /infer/submit payload:
    {
        "model_type": "huggingface",
        "inference_config": {
            "model_id": "google/vit-base-patch16-224",
            "task": "image-classification"   # optional if auto-detectable
        },
        "image_url": "s3://bucket/image.png"
    }

    # For segmentation with SAM-like models:
    {
        "model_type": "huggingface",
        "inference_config": {
            "model_id": "facebook/sam-vit-base",
            "task": "image-segmentation",
            "prompt": "cat"
        },
        "image_url": "s3://bucket/image.png"
    }

Environment variables:
    HF_HOME             - HuggingFace cache directory (model weights)
    TRANSFORMERS_OFFLINE - Set "1" to disable network access (use pre-staged)
    HF_DEFAULT_DEVICE   - Default device ("cuda", "cpu", "auto")
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter

logger = logging.getLogger(__name__)

# Task types that produce different output formats
_CLASSIFICATION_TASKS = {
    "image-classification",
    "zero-shot-image-classification",
}
_DETECTION_TASKS = {
    "object-detection",
    "zero-shot-object-detection",
}
_SEGMENTATION_TASKS = {
    "image-segmentation",
    "semantic-segmentation",
    "instance-segmentation",
    "panoptic-segmentation",
}
_FEATURE_TASKS = {
    "image-feature-extraction",
}
_DEPTH_TASKS = {
    "depth-estimation",
}

_ALL_SUPPORTED_TASKS = (
    _CLASSIFICATION_TASKS
    | _DETECTION_TASKS
    | _SEGMENTATION_TASKS
    | _FEATURE_TASKS
    | _DEPTH_TASKS
    | {"image-to-image"}
)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_device() -> str:
    return os.environ.get("HF_DEFAULT_DEVICE", "auto")


def _upload_result(s3, dst_bucket, out_key, data, content_type, presigned_output_urls):
    """Upload result via presigned URL or direct S3."""
    presigned_url = (presigned_output_urls or {}).get(out_key)
    if presigned_url:
        req = urllib.request.Request(
            presigned_url,
            data=data,
            method="PUT",
            headers={"Content-Type": content_type},
        )
        urllib.request.urlopen(req, timeout=120)
    else:
        if _require_presigned_urls() or s3 is None:
            raise RuntimeError(f"Missing presigned PUT URL for {out_key}")
        s3.put_object(
            Bucket=dst_bucket, Key=out_key, Body=data, ContentType=content_type
        )


def _resolve_task(config: dict[str, Any]) -> str:
    """Resolve the HuggingFace pipeline task from config."""
    task = config.get("task", "")
    if task and task in _ALL_SUPPORTED_TASKS:
        return task
    if task:
        raise ValueError(
            f"Unsupported HuggingFace vision task '{task}'. "
            f"Supported tasks: {sorted(_ALL_SUPPORTED_TASKS)}"
        )
    return "image-classification"


def _output_suffix_for_task(task: str) -> str:
    """Return the primary output file suffix based on task type."""
    if task in _SEGMENTATION_TASKS:
        return "_seg.png"
    if task in _DEPTH_TASKS:
        return "_depth.png"
    if task in _FEATURE_TASKS:
        return "_features.pt"
    if task == "image-to-image":
        return "_output.png"
    # Classification and detection produce results pickle only.
    return "_results.pkl"


class HuggingFaceVisionAdapter(VisionModelAdapter):
    """Generic adapter for any HuggingFace vision model.

    Loads models via ``transformers.pipeline()`` which auto-detects the
    correct model class, processor, and inference logic.
    """

    def capabilities(self) -> dict[str, Any] | None:
        return {
            "name": "huggingface",
            "description": "Generic HuggingFace vision model adapter",
            "supported_tasks": sorted(_ALL_SUPPORTED_TASKS),
            "supported_inputs": ["single_image", "image_with_text"],
            "gpu_memory_mb": 4096,
            "supports_batch": False,
            "max_batch_size": 1,
            "model_source": "huggingface",
            "framework": "pytorch",
        }

    def describe_outputs(self, key: str, config: dict[str, Any]) -> dict[str, str]:
        """Describe output artifacts based on the task type."""
        task = _resolve_task(config)
        base = key.rsplit(".", 1)[0] if "." in key else key
        suffix = _output_suffix_for_task(task)

        outputs: dict[str, str] = {}

        # Primary output (image or tensor)
        if suffix.endswith(".png"):
            outputs[f"{base}{suffix}"] = "image/png"
        elif suffix.endswith(".pt"):
            outputs[f"{base}{suffix}"] = "application/octet-stream"

        # Always produce a results pickle with full metadata
        outputs[f"{base}_hf_results.pkl"] = "application/octet-stream"

        return outputs

    def cache_key(self, config: dict[str, Any]) -> str:
        model_id = config.get("model_id", "")
        task = _resolve_task(config)
        revision = config.get("revision", "main")
        return f"hf:{model_id}:{task}:{revision}"

    def load(self, config: dict[str, Any]) -> Any:
        """Load model via transformers.pipeline()."""
        import torch
        from transformers import pipeline as hf_pipeline

        model_id = config.get("model_id")
        if not model_id:
            raise ValueError(
                "inference_config must include 'model_id' for the huggingface adapter"
            )

        task = _resolve_task(config)
        revision = config.get("revision", "main")
        device = config.get("device", _default_device())
        torch_dtype = config.get("torch_dtype", "auto")

        # Resolve torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        logger.info(
            "Loading HuggingFace model: %s (task=%s, device=%s, dtype=%s)",
            model_id,
            task,
            device,
            torch_dtype,
        )

        pipe_kwargs: dict[str, Any] = {
            "model": model_id,
            "revision": revision,
        }

        if device != "auto":
            pipe_kwargs["device"] = device
        else:
            pipe_kwargs["device_map"] = "auto"

        if resolved_dtype != "auto":
            pipe_kwargs["torch_dtype"] = resolved_dtype

        # Create pipeline. It handles model and processor loading.
        if task:
            pipe_kwargs["task"] = task

        pipe = hf_pipeline(**pipe_kwargs)

        return {
            "pipeline": pipe,
            "model_id": model_id,
            "task": task,
            "device": str(pipe.device) if hasattr(pipe, "device") else device,
        }

    def warmup(self, model: Any, _config: dict[str, Any]) -> bool:
        """Run a quick inference on a tiny synthetic image to verify the model."""
        try:
            pipe = model["pipeline"]
            task = model["task"]
            # Create a tiny test image
            test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))

            if (
                task in _CLASSIFICATION_TASKS
                or task in _FEATURE_TASKS
                or task in _DEPTH_TASKS
            ):
                pipe(test_img)
            # Skip warmup for detection/segmentation (may need valid prompts)
            else:
                logger.info("Skipping warmup inference for task=%s", task)

            logger.info("Warmup succeeded for %s", model["model_id"])
            return True
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", model.get("model_id"), exc)
            return False

    def estimate_gpu_memory_mb(self, config: dict[str, Any]) -> int:
        """Estimate GPU memory based on model size hints."""
        # Rough heuristics based on model name
        model_id = config.get("model_id", "").lower()
        if "large" in model_id or "xl" in model_id:
            return 8192
        if "base" in model_id:
            return 4096
        if "small" in model_id or "tiny" in model_id:
            return 2048
        return 4096

    def infer(
        self,
        model: Any,
        image: Image.Image,
        bucket: str,
        key: str,
        config: dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls: dict[str, str] | None = None,
        raw_image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Run inference using the HuggingFace pipeline."""
        del raw_image_bytes

        pipe = model["pipeline"]
        task = model["task"]
        dst_bucket = output_bucket or bucket
        described = self.describe_outputs(key, config)

        logger.info(
            "[HF] infer: model=%s task=%s bucket=%s key=%s",
            model["model_id"],
            task,
            bucket,
            key,
        )

        # Build pipeline kwargs based on task
        pipe_kwargs: dict[str, Any] = {}
        if task in {"zero-shot-image-classification", "zero-shot-object-detection"}:
            candidate_labels = config.get("candidate_labels", config.get("labels", []))
            if isinstance(candidate_labels, str):
                candidate_labels = [
                    label.strip() for label in candidate_labels.split(",")
                ]
            pipe_kwargs["candidate_labels"] = candidate_labels

        # Run inference
        result = pipe(image, **pipe_kwargs)

        response: dict[str, Any] = {
            "source_image": f"s3://{bucket}/{key}",
            "model_id": model["model_id"],
            "task": task,
        }

        # Process results based on task type
        if task in _CLASSIFICATION_TASKS:
            response.update(self._handle_classification(result))
        elif task in _DETECTION_TASKS:
            response.update(self._handle_detection(result))
        elif task in _SEGMENTATION_TASKS:
            response.update(
                self._handle_segmentation(
                    result, image, dst_bucket, described, s3, presigned_output_urls
                )
            )
        elif task in _FEATURE_TASKS:
            response.update(
                self._handle_features(
                    result, dst_bucket, described, s3, presigned_output_urls
                )
            )
        elif task in _DEPTH_TASKS:
            response.update(
                self._handle_depth(
                    result, dst_bucket, described, s3, presigned_output_urls
                )
            )
        elif task == "image-to-image":
            response.update(
                self._handle_image_to_image(
                    result, dst_bucket, described, s3, presigned_output_urls
                )
            )
        else:
            # Generic fallback: pickle the raw result.
            response["raw_result"] = result

        # Always upload results pickle
        pkl_key = next((k for k in described if k.endswith(".pkl")), None)
        if pkl_key:
            # Build a serializable result (strip non-picklable objects)
            pkl_payload = {
                k: v for k, v in response.items() if k not in ("raw_result",)
            }
            pkl_data = pickle.dumps(pkl_payload, protocol=pickle.HIGHEST_PROTOCOL)
            _upload_result(
                s3,
                dst_bucket,
                pkl_key,
                pkl_data,
                "application/octet-stream",
                presigned_output_urls,
            )
            response["results_pkl_s3_key"] = f"s3://{dst_bucket}/{pkl_key}"

        return response

    def infer_batch(
        self,
        model: Any,
        images: list[Image.Image],
        keys: list[str],
        bucket: str,
        config: dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls_list: list[dict[str, str]] | None = None,
        raw_image_bytes_list: list[bytes] | None = None,
    ) -> list[dict[str, Any]]:
        """Batch inference fallback that preserves per-object artifact uploads."""
        return super().infer_batch(
            model,
            images,
            keys,
            bucket,
            config,
            s3,
            output_bucket,
            presigned_output_urls_list,
            raw_image_bytes_list,
        )

    # ------------------------------------------------------------------
    # Task-specific result handlers
    # ------------------------------------------------------------------

    def _handle_classification(self, result: Any) -> dict[str, Any]:
        """Process classification results."""
        if isinstance(result, list):
            predictions = [
                {"label": r.get("label", ""), "score": float(r.get("score", 0))}
                for r in result
            ]
        else:
            predictions = [{"label": str(result), "score": 1.0}]
        return {
            "predictions": predictions,
            "top_label": predictions[0]["label"] if predictions else "",
            "top_score": predictions[0]["score"] if predictions else 0.0,
        }

    def _handle_detection(self, result: Any) -> dict[str, Any]:
        """Process object detection results."""
        detections = []
        if isinstance(result, list):
            for det in result:
                detections.append(
                    {
                        "label": det.get("label", ""),
                        "score": float(det.get("score", 0)),
                        "box": det.get("box", {}),
                    }
                )
        return {
            "detections": detections,
            "num_detections": len(detections),
        }

    def _handle_segmentation(
        self,
        result: Any,
        image: Image.Image,
        dst_bucket: str,
        described: dict[str, str],
        s3,
        presigned_output_urls: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Process segmentation results and upload mask overlay."""
        response: dict[str, Any] = {}

        # Find the PNG output key
        png_key = next((k for k in described if k.endswith(".png")), None)

        if isinstance(result, list) and result:
            # Instance/panoptic segmentation: list of segments.
            segments = []
            combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            for i, seg in enumerate(result):
                mask = seg.get("mask")
                if mask is not None:
                    if hasattr(mask, "numpy"):
                        mask_np = mask.numpy()
                    elif isinstance(mask, Image.Image):
                        mask_np = np.array(mask)
                    else:
                        mask_np = np.array(mask)
                    combined_mask[mask_np > 0] = i + 1
                segments.append(
                    {
                        "label": seg.get("label", f"segment_{i}"),
                        "score": float(seg.get("score", 0)),
                    }
                )
            response["segments"] = segments
            response["num_segments"] = len(segments)

            # Create colored overlay
            if png_key:
                overlay = self._create_segmentation_overlay(image, combined_mask)
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                png_data = buf.getvalue()
                _upload_result(
                    s3,
                    dst_bucket,
                    png_key,
                    png_data,
                    "image/png",
                    presigned_output_urls,
                )
                response["output_s3_key"] = f"s3://{dst_bucket}/{png_key}"

        elif isinstance(result, dict) and "mask" in result:
            # Semantic segmentation: single mask.
            mask = result["mask"]
            mask_np = np.array(mask)
            response["num_segments"] = int(mask_np.max())

            if png_key:
                overlay = self._create_segmentation_overlay(image, mask_np)
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                png_data = buf.getvalue()
                _upload_result(
                    s3,
                    dst_bucket,
                    png_key,
                    png_data,
                    "image/png",
                    presigned_output_urls,
                )
                response["output_s3_key"] = f"s3://{dst_bucket}/{png_key}"

        return response

    def _handle_features(
        self,
        result: Any,
        dst_bucket: str,
        described: dict[str, str],
        s3,
        presigned_output_urls: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Process feature extraction results and upload tensor."""
        import torch

        pt_key = next((k for k in described if k.endswith(".pt")), None)

        # Convert result to tensor
        if isinstance(result, np.ndarray):
            features = torch.from_numpy(result)
        elif isinstance(result, list):
            features = torch.tensor(result)
        elif hasattr(result, "last_hidden_state"):
            features = result.last_hidden_state
        else:
            features = torch.tensor(np.array(result))

        response: dict[str, Any] = {
            "feature_shape": list(features.shape),
        }

        if pt_key:
            buf = io.BytesIO()
            torch.save({"features": features}, buf)
            buf.seek(0)
            _upload_result(
                s3,
                dst_bucket,
                pt_key,
                buf.getvalue(),
                "application/octet-stream",
                presigned_output_urls,
            )
            response["output_s3_key"] = f"s3://{dst_bucket}/{pt_key}"

        return response

    def _handle_depth(
        self,
        result: Any,
        dst_bucket: str,
        described: dict[str, str],
        s3,
        presigned_output_urls: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Process depth estimation results and upload depth map."""
        png_key = next((k for k in described if k.endswith(".png")), None)
        response: dict[str, Any] = {}

        depth_map = result.get("depth") if isinstance(result, dict) else result
        if isinstance(depth_map, Image.Image):
            depth_img = depth_map
        elif hasattr(depth_map, "numpy"):
            depth_np = depth_map.numpy()
            # Normalize to 0-255 for visualization
            depth_norm = (
                (depth_np - depth_np.min())
                / (depth_np.max() - depth_np.min() + 1e-8)
                * 255
            ).astype(np.uint8)
            depth_img = Image.fromarray(depth_norm)
        else:
            depth_np = np.array(depth_map)
            depth_norm = (
                (depth_np - depth_np.min())
                / (depth_np.max() - depth_np.min() + 1e-8)
                * 255
            ).astype(np.uint8)
            depth_img = Image.fromarray(depth_norm)

        if png_key:
            buf = io.BytesIO()
            depth_img.save(buf, format="PNG")
            png_data = buf.getvalue()
            _upload_result(
                s3, dst_bucket, png_key, png_data, "image/png", presigned_output_urls
            )
            response["output_s3_key"] = f"s3://{dst_bucket}/{png_key}"

        return response

    def _handle_image_to_image(
        self,
        result: Any,
        dst_bucket: str,
        described: dict[str, str],
        s3,
        presigned_output_urls: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Process image-to-image results."""
        png_key = next((k for k in described if k.endswith(".png")), None)
        response: dict[str, Any] = {}

        if isinstance(result, Image.Image):
            output_img = result
        elif isinstance(result, list) and result:
            output_img = result[0] if isinstance(result[0], Image.Image) else None
        else:
            output_img = None

        if output_img and png_key:
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            png_data = buf.getvalue()
            _upload_result(
                s3, dst_bucket, png_key, png_data, "image/png", presigned_output_urls
            )
            response["output_s3_key"] = f"s3://{dst_bucket}/{png_key}"

        return response

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_segmentation_overlay(
        image: Image.Image,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Create a colored overlay of segmentation mask on the original image."""
        # Generate distinct colors for each segment
        num_segments = int(mask.max())
        colors = _generate_colors(num_segments)

        # Create RGBA overlay
        overlay = np.array(image.convert("RGBA"))
        color_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)

        for seg_id in range(1, num_segments + 1):
            seg_pixels = mask == seg_id
            color = colors[(seg_id - 1) % len(colors)]
            color_mask[seg_pixels] = [*color, int(255 * alpha)]

        # Composite
        color_img = Image.fromarray(color_mask, mode="RGBA")
        base_img = Image.fromarray(overlay, mode="RGBA")
        composite = Image.alpha_composite(base_img, color_img)
        return composite.convert("RGB")


def _generate_colors(n: int) -> list[tuple]:
    """Generate n visually distinct RGB colors."""
    if n == 0:
        return [(255, 0, 0)]
    colors = []
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
        # HSV to RGB (simplified)
        h = hue * 6
        c = 200  # saturation * value
        x = int(c * (1 - abs(h % 2 - 1)))
        c = int(c)
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        colors.append(rgb)
    return colors
