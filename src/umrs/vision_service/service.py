"""BentoML Vision Model Service for UMRS.

Provides async task queue endpoints for SAM3 segmentation and HMAE inference.
Models are loaded sequentially (one GPU, one model at a time) and share the
same BentoML service. Auto-generates task queue API:
    POST /submit — submit a task, returns task_id immediately
    GET  /status — check task status by task_id
    GET  /get    — retrieve task result by task_id
    POST /cancel — cancel a pending task
    PUT  /retry  — retry a failed task

Environment variables:
    HF_TOKEN           — HuggingFace token for downloading facebook/sam3
    SAM3_MODEL_PATH    — path to SAM3 model weights (or HF repo id)
    HMAE_CONFIG_PATH   — path to HMAE inference YAML config
    HMAE_CHECKPOINT    — path to HMAE checkpoint (.pth)
    S3_ENDPOINT_URL    — MinIO/S3 endpoint
    S3_ACCESS_KEY      — S3 access key
    S3_SECRET_KEY      — S3 secret key
"""

import os
import io
import base64
import logging
from typing import Optional

import bentoml
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _get_s3_client():
    """Create a boto3 S3 client from environment variables."""
    import boto3

    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("S3_SECRET_KEY", "minioadmin")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _download_image_from_s3(s3, image_url):
    """Parse an S3 URL (s3://bucket/key) and download as PIL Image."""
    if image_url.startswith("s3://"):
        path = image_url[5:]
    else:
        path = image_url

    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 image URL: {image_url}. Expected format: s3://bucket/key")
    bucket, key = parts

    obj = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = obj["Body"].read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), bucket, key


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 600},
)
class VisionModelService:
    """Unified vision model service supporting SAM3 segmentation and HMAE inference.

    Models are loaded lazily on first use and cached. Only one model is active
    at a time to conserve GPU memory (sequential execution).
    """

    def __init__(self):
        self._sam3_model = None
        self._sam3_processor = None
        self._hmae_model = None
        self._hmae_config = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._s3 = None

    def _ensure_s3(self):
        if self._s3 is None:
            self._s3 = _get_s3_client()
        return self._s3

    def _ensure_sam3(self):
        if self._sam3_model is None:
            from umrs.models.sam3_segmentation import load_sam3_model
            model_path = os.environ.get("SAM3_MODEL_PATH", "facebook/sam3")
            self._sam3_model, self._sam3_processor, self._device = load_sam3_model(model_path)
        return self._sam3_model, self._sam3_processor

    def _ensure_hmae(self):
        if self._hmae_model is None:
            from umrs.models.hmae_inference import load_hmae_model
            cfg_path = os.environ.get("HMAE_CONFIG_PATH",
                                       "umrs/models/hmae/configs/infer_hmae_vit_b.yaml")
            ckpt_path = os.environ.get("HMAE_CHECKPOINT",
                                        "./checkpoints/latest_checkpoint.pth")
            self._hmae_model, self._device, self._hmae_config = load_hmae_model(cfg_path, ckpt_path)
        return self._hmae_model, self._hmae_config

    # -------------------------------------------------------------------
    # SAM3 Segmentation
    # -------------------------------------------------------------------

    @bentoml.task
    def sam3_segment(self, image_url: str, prompt: str,
                     threshold: float = 0.5,
                     mask_threshold: float = 0.5) -> dict:
        """Segment objects in an image using SAM3 with a text prompt.

        Args:
            image_url: S3 URL to the image (s3://bucket/key).
            prompt: text prompt describing what to segment (e.g. "crack", "void").
            threshold: confidence threshold.
            mask_threshold: mask binarization threshold.

        Returns:
            dict with: num_objects, boxes, scores, overlay_s3_key
        """
        try:
            from umrs.models.sam3_segmentation import segment_image_with_text, overlay_masks_on_image

            model, processor = self._ensure_sam3()
            s3 = self._ensure_s3()

            image, bucket, key = _download_image_from_s3(s3, image_url)
            result = segment_image_with_text(model, processor, image, prompt, self._device,
                                              threshold, mask_threshold)
        except Exception as exc:
            logger.exception("sam3_segment failed")
            return {"error": str(exc), "num_objects": 0, "boxes": [], "scores": []}

        response = {
            "num_objects": result["num_objects"],
            "boxes": result["boxes"],
            "scores": result["scores"],
            "source_image": image_url,
            "prompt": prompt,
        }

        # Upload overlay visualization
        if result["num_objects"] > 0:
            overlay = overlay_masks_on_image(image, result["masks"])
            base = key.rsplit(".", 1)[0] if "." in key else key
            overlay_key = f"{base}_sam3_seg.png"
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            s3.put_object(Bucket=bucket, Key=overlay_key, Body=buf.getvalue(),
                          ContentType="image/png")
            response["overlay_s3_key"] = f"s3://{bucket}/{overlay_key}"

        logger.info("SAM3 segmentation: found %d objects for prompt '%s'",
                     result["num_objects"], prompt)
        return response

    # -------------------------------------------------------------------
    # HMAE Inference (3 modes)
    # -------------------------------------------------------------------

    @bentoml.task
    def hmae_reconstruct(self, image_url: str,
                         mode: str = "reconstruct") -> dict:
        """Run HMAE inference on an image.

        Args:
            image_url: S3 URL to the image (s3://bucket/key).
            mode: One of "reconstruct", "mae", "features".

        Returns:
            dict with mode-specific results and output S3 keys.
        """
        try:
            from umrs.models.hmae_inference import (
                preprocess_image_from_bytes, infer_reconstruct, infer_mae,
                infer_features, visualize_reconstruction,
            )

            model, config = self._ensure_hmae()
            s3 = self._ensure_s3()
            model_cfg = config["model"]
            use_bf16 = config["inference"].get("use_bf16", True)
            if self._device == "cpu":
                use_bf16 = False

            # Download image
            image, bucket, key = _download_image_from_s3(s3, image_url)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # Preprocess
            data = preprocess_image_from_bytes(
                image_bytes,
                img_size=model_cfg["img_size"],
                in_channels=model_cfg.get("in_channels", 1),
                fixed_length=model_cfg.get("fixed_length", 1024),
                patch_size=model_cfg["patch_size"],
            )

            device = torch.device(self._device)
            base = key.rsplit(".", 1)[0] if "." in key else key
            response = {"source_image": image_url, "mode": mode}

            if mode == "reconstruct":
                pred = infer_reconstruct(model, data, device, config, use_bf16)
                png_bytes = visualize_reconstruction(data, pred, config)
                out_key = f"{base}_recon.png"
                s3.put_object(Bucket=bucket, Key=out_key, Body=png_bytes, ContentType="image/png")
                response["output_s3_key"] = f"s3://{bucket}/{out_key}"

            elif mode == "mae":
                pred, mask, loss = infer_mae(model, data, device, config, use_bf16)
                png_bytes = visualize_reconstruction(data, pred, config, mask=mask, loss=loss)
                out_key = f"{base}_mae.png"
                s3.put_object(Bucket=bucket, Key=out_key, Body=png_bytes, ContentType="image/png")
                response["output_s3_key"] = f"s3://{bucket}/{out_key}"
                response["mae_loss"] = loss

            elif mode == "features":
                features = infer_features(model, data, device, config, use_bf16)
                cls_token = features[:, 0, :]
                patch_tokens = features[:, 1:, :]

                out_key = f"{base}_features.pt"
                feat_buf = io.BytesIO()
                torch.save({"cls": cls_token, "patches": patch_tokens, "coords": data["coords"]}, feat_buf)
                feat_buf.seek(0)
                s3.put_object(Bucket=bucket, Key=out_key, Body=feat_buf.getvalue())
                response["output_s3_key"] = f"s3://{bucket}/{out_key}"
                response["cls_shape"] = list(cls_token.shape)
                response["patch_shape"] = list(patch_tokens.shape)

            else:
                raise ValueError(f"Unknown HMAE mode: {mode}. Use 'reconstruct', 'mae', or 'features'.")

        except Exception as exc:
            logger.exception("hmae_reconstruct failed")
            return {"error": str(exc), "source_image": image_url, "mode": mode}

        logger.info("HMAE %s completed for %s", mode, image_url)
        return response
