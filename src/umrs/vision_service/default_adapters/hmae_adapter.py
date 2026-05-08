"""HMAE inference adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
import pickle
import urllib.request
from typing import Any

import torch
from PIL import Image
from umrs.vision_service.adapters.base import VisionModelAdapter
from umrs.vision_service.output_specs import (
    describe_output_artifacts,
    resolve_hmae_task,
)

logger = logging.getLogger(__name__)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1", "true", "yes", "on",
    }


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
        if _require_presigned_urls() or s3 is None:
            raise RuntimeError(f"Missing presigned PUT URL for {out_key}")
        s3.put_object(Bucket=dst_bucket, Key=out_key,
                      Body=data, ContentType=content_type)


class HMAEAdapter(VisionModelAdapter):
    """Load and run HMAE inference (reconstruction / mae_reconstruction / feature_extraction)."""

    def capabilities(self):
        return {
            "name": "hmae",
            "description": "Hierarchical Masked Autoencoder for CT/volume analysis",
            "supported_tasks": [
                "custom:reconstruction",
                "custom:mae-reconstruction",
                "custom:feature-extraction",
            ],
            "supported_inputs": ["single_image", "volume_slices"],
            "output_types": ["reconstructed_image", "feature_tensors", "embeddings"],
            "gpu_memory_mb": 8000,
            "supports_batch": False,
            "max_batch_size": 1,
            "typical_inference_seconds": 10.0,
            "model_source": "custom",
            "framework": "pytorch",
        }

    def estimate_gpu_memory_mb(self, config: dict[str, Any]) -> int:
        # HMAE with ViT-Base needs ~6-8GB depending on image size
        img_size = config.get("img_size", 128)
        if img_size >= 256:
            return 12000
        return 8000

    def warmup(self, model: Any, _config: dict[str, Any]) -> bool:
        """Verify HMAE model loaded correctly by checking structure."""
        try:
            if model is None or "model" not in model:
                return False
            if model["device"] is None:
                return False
            if model["config"] is None:
                return False
            logger.info("HMAE warmup: model loaded on device=%s", model["device"])
            return True
        except Exception as exc:
            logger.warning("HMAE warmup failed: %s", exc)
            return False

    def describe_outputs(self, key: str, config: dict[str, Any]) -> dict[str, str]:
        return describe_output_artifacts("hmae", key, config)

    def cache_key(self, config: dict[str, Any]) -> str:
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

    def load(self, config: dict[str, Any]) -> Any:
        from umrs.inference.hmae_inference import load_hmae_model

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
        config: dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls: dict[str, str] | None = None,
        raw_image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        from umrs.inference.hmae_inference import (
            infer_features,
            infer_mae,
            infer_reconstruct,
            preprocess_image_from_bytes,
            visualize_reconstruction,
        )

        # Accept both "task" and "mode" keys for backward compatibility.
        # The demo scripts historically used "mode" while the adapter expected "task".
        task = resolve_hmae_task(config)
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

        # Use raw S3 bytes when available (matches reference cv2.imread path).
        # Fall back to PIL re-serialisation only when raw bytes are missing.
        if raw_image_bytes is not None:
            image_bytes = raw_image_bytes
        else:
            in_ch = model_cfg.get("in_channels", 1)
            image_conv = image.convert("L") if in_ch == 1 else image.convert("RGB")
            buf = io.BytesIO()
            image_conv.save(buf, format="PNG")
            image_bytes = buf.getvalue()

        try:
            data = preprocess_image_from_bytes(
                image_bytes,
                img_size=model_cfg["img_size"],
                in_channels=model_cfg.get("in_channels", 1),
                fixed_length=model_cfg.get("fixed_length", 1024),
                patch_size=model_cfg["patch_size"],
            )
        except Exception as exc:
            logger.error("[HMAE] Preprocessing failed for %s: %s", key, exc)
            return {
                "error": f"Preprocessing failed: {exc}",
                "source_image": f"s3://{bucket}/{key}",
                "task": task,
            }

        described_outputs = self.describe_outputs(key, config)
        dst_bucket = output_bucket or bucket
        response: dict[str, Any] = {
            "source_image": f"s3://{bucket}/{key}",
            "task": task,
        }

        try:
            if task == "reconstruction":
                pred = infer_reconstruct(model["model"], data, device,
                                         hmae_config, use_bf16)
                png_bytes = visualize_reconstruction(data, pred, hmae_config)
                out_key = next(iter(described_outputs))
                _upload_result(s3, dst_bucket, out_key, png_bytes, "image/png",
                               presigned_output_urls)
                response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
                response["prediction"] = pred

                # Upload pickled result arrays
                pkl_key = next(k for k in described_outputs if k.endswith(".pkl"))
                pkl_data = pickle.dumps(
                    {
                        "prediction": pred,
                        "coords": data["coords"],
                        "patches": data["patches"],
                        "depths": data["depths"],
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                _upload_result(s3, dst_bucket, pkl_key, pkl_data,
                               "application/octet-stream", presigned_output_urls)
                response["results_pkl_s3_key"] = f"s3://{dst_bucket}/{pkl_key}"

            elif task == "mae_reconstruction":
                pred, mask, loss = infer_mae(model["model"], data, device,
                                             hmae_config, use_bf16)
                png_bytes = visualize_reconstruction(data, pred, hmae_config,
                                                     mask=mask, loss=loss)
                out_key = next(iter(described_outputs))
                _upload_result(s3, dst_bucket, out_key, png_bytes, "image/png",
                               presigned_output_urls)
                response["output_s3_key"] = f"s3://{dst_bucket}/{out_key}"
                response["mae_loss"] = loss
                response["prediction"] = pred
                response["mae_mask"] = mask

                # Upload pickled result arrays
                pkl_key = next(k for k in described_outputs if k.endswith(".pkl"))
                pkl_data = pickle.dumps(
                    {
                        "prediction": pred,
                        "mask": mask,
                        "loss": loss,
                        "coords": data["coords"],
                        "patches": data["patches"],
                        "depths": data["depths"],
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                _upload_result(s3, dst_bucket, pkl_key, pkl_data,
                               "application/octet-stream", presigned_output_urls)
                response["results_pkl_s3_key"] = f"s3://{dst_bucket}/{pkl_key}"

            elif task == "feature_extraction":
                features = infer_features(model["model"], data, device,
                                          hmae_config, use_bf16)
                cls_token = features[:, 0, :]
                patch_tokens = features[:, 1:, :]

                out_key = next(iter(described_outputs))
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
                response["cls_token"] = cls_token
                response["patch_tokens"] = patch_tokens
                response["coords"] = data["coords"]

                # Upload pickled result arrays
                pkl_key = next(k for k in described_outputs if k.endswith(".pkl"))
                pkl_data = pickle.dumps(
                    {
                        "cls_token": cls_token,
                        "patch_tokens": patch_tokens,
                        "coords": data["coords"],
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                _upload_result(s3, dst_bucket, pkl_key, pkl_data,
                               "application/octet-stream", presigned_output_urls)
                response["results_pkl_s3_key"] = f"s3://{dst_bucket}/{pkl_key}"

            else:
                raise ValueError(
                    f"Unknown HMAE task: {task}. "
                    "Use 'reconstruction', 'mae_reconstruction', or 'feature_extraction'."
                )

        except torch.cuda.OutOfMemoryError as oom_exc:
            logger.error("[HMAE] CUDA OOM during %s for %s: %s", task, key, oom_exc)
            # Attempt to free GPU memory for subsequent requests
            torch.cuda.empty_cache()
            return {
                "error": f"CUDA out of memory during {task}: {oom_exc}",
                "source_image": f"s3://{bucket}/{key}",
                "task": task,
                "recoverable": True,
            }
        except RuntimeError as rt_exc:
            # Catch CUDA-related RuntimeErrors (device-side asserts, etc.)
            if "CUDA" in str(rt_exc) or "device-side" in str(rt_exc):
                logger.error("[HMAE] CUDA error during %s for %s: %s", task, key, rt_exc)
                torch.cuda.empty_cache()
                return {
                    "error": f"CUDA error during {task}: {rt_exc}",
                    "source_image": f"s3://{bucket}/{key}",
                    "task": task,
                    "recoverable": True,
                }
            raise

        logger.info("HMAE %s completed for s3://%s/%s", task, bucket, key)
        return response
