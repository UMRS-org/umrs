"""HMAE inference adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict

import torch
from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter

logger = logging.getLogger(__name__)


class HMAEAdapter(VisionModelAdapter):
    """Load and run HMAE inference (reconstruct / mae / features)."""

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
    ) -> Dict[str, Any]:
        from umrs.models.hmae_inference import (
            preprocess_image_from_bytes,
            infer_reconstruct,
            infer_mae,
            infer_features,
            visualize_reconstruction,
        )

        mode = config.get("mode", "reconstruct")
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
            "mode": mode,
        }

        if mode == "reconstruct":
            pred = infer_reconstruct(model["model"], data, device,
                                     hmae_config, use_bf16)
            png_bytes = visualize_reconstruction(data, pred, hmae_config)
            out_key = f"{base}_recon.png"
            s3.put_object(Bucket=dst_bucket, Key=out_key,
                          Body=png_bytes, ContentType="image/png")
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"

        elif mode == "mae":
            pred, mask, loss = infer_mae(model["model"], data, device,
                                         hmae_config, use_bf16)
            png_bytes = visualize_reconstruction(data, pred, hmae_config,
                                                 mask=mask, loss=loss)
            out_key = f"{base}_mae.png"
            s3.put_object(Bucket=dst_bucket, Key=out_key,
                          Body=png_bytes, ContentType="image/png")
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
            response["mae_loss"] = loss

        elif mode == "features":
            features = infer_features(model["model"], data, device,
                                      hmae_config, use_bf16)
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]

            out_key = f"{base}_features.pt"
            feat_buf = io.BytesIO()
            torch.save({"cls": cls_token, "patches": patch_tokens,
                        "coords": data["coords"]}, feat_buf)
            feat_buf.seek(0)
            s3.put_object(Bucket=dst_bucket, Key=out_key,
                          Body=feat_buf.getvalue())
            response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
            response["cls_shape"] = list(cls_token.shape)
            response["patch_shape"] = list(patch_tokens.shape)

        else:
            raise ValueError(
                f"Unknown HMAE mode: {mode}. "
                "Use 'reconstruct', 'mae', or 'features'."
            )

        logger.info("HMAE %s completed for s3://%s/%s", mode, bucket, key)
        return response
