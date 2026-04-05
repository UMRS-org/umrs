"""HMAE inference adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from typing import Any, Dict

import torch
from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter

logger = logging.getLogger(__name__)


def _upload_result(s3, dst_bucket, out_key, data, content_type,
                   presigned_output_urls):
    """Upload result via presigned URL or direct S3.

    Helper to avoid repeating the presigned/direct branching in each task.
    """
    presigned_url = (presigned_output_urls or {}).get(out_key)
    if presigned_url:
        req = urllib.request.Request(
            presigned_url,
            data=data,
            method="PUT",
            headers={"Content-Type": content_type},
        )
        urllib.request.urlopen(req, timeout=120)  # noqa: S310
    else:
        s3.put_object(Bucket=dst_bucket, Key=out_key,
                      Body=data, ContentType=content_type)


class HMAEAdapter(VisionModelAdapter):
    """Load and run HMAE inference (reconstruction / mae_reconstruction / feature_extraction)."""

    def cache_key(self, config: Dict[str, Any]) -> str:
        cfg_path = config.get(
            "config_path",
            os.environ.get("HMAE_CONFIG_PATH",
                           "umrs/models/hmae/configs/infer_hmae_vit_b.yaml"),
        )
        checkpoint = config.get(
            "checkpoint",
            os.environ.get("HMAE_CHECKPOINT",
                           "./checkpoints/latest_checkpoint.pth"),
        )
        return f"hmae:{cfg_path}:{checkpoint}"

    def load(self, config: Dict[str, Any]) -> Any:
        from umrs.models.hmae_inference import load_hmae_model

        cfg_path = config.get(
            "config_path",
            os.environ.get("HMAE_CONFIG_PATH",
                           "umrs/models/hmae/configs/infer_hmae_vit_b.yaml"),
        )
        checkpoint = config.get(
            "checkpoint",
            os.environ.get("HMAE_CHECKPOINT",
                           "./checkpoints/latest_checkpoint.pth"),
        )
        model, device, hmae_config = load_hmae_model(cfg_path, checkpoint)
        return {"model": model, "device": device, "config": hmae_config}

    def infer(
        self,
        model: Any,
        image: Image.Image,
        bucket: str,
        key: str,
        config: Dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        from umrs.models.hmae_inference import (
            preprocess_image_from_bytes,
            infer_reconstruct,
            infer_mae,
            infer_features,
            visualize_reconstruction,
        )

        # Accept both "task" and "mode" keys for backward compatibility.
        # The demo scripts historically used "mode" while the adapter expected "task".
        _MODE_TO_TASK = {
            "reconstruct": "reconstruction",
            "mae": "mae_reconstruction",
            "features": "feature_extraction",
        }
        task = config.get("task") or _MODE_TO_TASK.get(config.get("mode", ""), "") or "reconstruction"
        logger.info("[HMAE] infer: bucket=%s key=%s task=%s "
                    "output_bucket=%s s3=%s presigned_output_urls=%s",
                    bucket, key, task, output_bucket, s3 is not None,
                    list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)")
        hmae_config = model["config"]
        device = torch.device(model["device"])
        model_cfg = hmae_config["model"]

        use_bf16 = hmae_config["inference"].get("use_bf16", True)
        if str(model["device"]) == "cpu":
            use_bf16 = False

        # Serialise image to PNG bytes for preprocessing
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        data = preprocess_image_from_bytes(
            image_bytes,
            img_size=model_cfg["img_size"],
            in_channels=model_cfg.get("in_channels", 1),
            fixed_length=model_cfg.get("fixed_length", 1024),
            patch_size=model_cfg["patch_size"],
        )

        base = key.rsplit(".", 1)[0] if "." in key else key
        dst_bucket = output_bucket or bucket
        response: Dict[str, Any] = {
            "source_image": f"s3://{bucket}/{key}",
            "task": task,
        }

        if task == "reconstruction":
            pred = infer_reconstruct(model["model"], data, device,
                                     hmae_config, use_bf16)
            png_bytes = visualize_reconstruction(data, pred, hmae_config)
            out_key = f"{base}_recon.png"
            _upload_result(s3, dst_bucket, out_key, png_bytes, "image/png",
                           presigned_output_urls)
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"

        elif task == "mae_reconstruction":
            pred, mask, loss = infer_mae(model["model"], data, device,
                                         hmae_config, use_bf16)
            png_bytes = visualize_reconstruction(data, pred, hmae_config,
                                                 mask=mask, loss=loss)
            out_key = f"{base}_mae.png"
            _upload_result(s3, dst_bucket, out_key, png_bytes, "image/png",
                           presigned_output_urls)
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
            response["mae_loss"] = loss

        elif task == "feature_extraction":
            features = infer_features(model["model"], data, device,
                                      hmae_config, use_bf16)
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]

            out_key = f"{base}_features.pt"
            feat_buf = io.BytesIO()
            torch.save({"cls": cls_token, "patches": patch_tokens,
                        "coords": data["coords"]}, feat_buf)
            feat_buf.seek(0)
            feat_data = feat_buf.getvalue()
            _upload_result(s3, dst_bucket, out_key, feat_data,
                           "application/octet-stream", presigned_output_urls)
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
            response["cls_shape"] = list(cls_token.shape)
            response["patch_shape"] = list(patch_tokens.shape)

        else:
            raise ValueError(
                f"Unknown HMAE task: {task}. "
                "Use 'reconstruction', 'mae_reconstruction', or 'feature_extraction'."
            )

        logger.info("HMAE %s completed for s3://%s/%s", task, bucket, key)
        return response
