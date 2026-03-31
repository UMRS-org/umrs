"""
HMAE inference step — user_script version.

Loads the HMAE (Hierarchical Masked Autoencoder) model from a local
checkpoint, runs one of three tasks (reconstruction, mae_reconstruction,
feature_extraction), and uploads visualization / feature results to the
output bucket.

This step wraps the existing ``umrs.models.hmae_inference`` module so that
HMAE can be used via the ``user_script://`` path without BentoML/Rivault.

Requirements: torch, opencv-python, numpy, matplotlib, PyYAML
  (plus the HMAE model code already bundled with UMRS)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEInferenceStep(Step):
    """Run HMAE inference (reconstruction / mae_reconstruction / feature_extraction) on images."""

    name = "hmae_inference"

    def setup(self, config: dict) -> None:
        self.output_prefix = config.get("output_prefix", "hmae_out/")
        self.task = config.get("task", "reconstruction")  # reconstruction | mae_reconstruction | feature_extraction
        self.cfg_file = config.get("cfg_file", "")
        self.checkpoint_path = config.get("checkpoint_path", "")
        self._model: Any = None
        self._device: Any = None
        self._config: Dict[str, Any] = {}

    def _ensure_model(self) -> None:
        """Lazy-load the HMAE model on first call."""
        if self._model is not None:
            return

        from umrs.models.hmae_inference import load_hmae_model

        if not self.cfg_file or not self.checkpoint_path:
            raise ValueError(
                "hmae_inference step requires 'cfg_file' and 'checkpoint_path' in config"
            )

        logger.info(
            "Loading HMAE model: cfg=%s ckpt=%s", self.cfg_file, self.checkpoint_path
        )
        self._model, self._device, self._config = load_hmae_model(
            self.cfg_file, self.checkpoint_path
        )
        logger.info("HMAE model loaded (task=%s)", self.task)

    def run(self, ctx: StepContext) -> StepResult:
        from umrs.models.hmae_inference import (
            preprocess_image_from_bytes,
            infer_reconstruct,
            infer_mae,
            infer_features,
            visualize_reconstruction,
        )

        self._ensure_model()

        model_cfg = self._config["model"]
        img_size = model_cfg["img_size"]
        patch_size = model_cfg["patch_size"]
        in_channels = model_cfg.get("in_channels", 1)
        fixed_length = model_cfg.get("fixed_length", 1024)

        output_keys = []
        checksums = []
        metadata_list = []

        for key in ctx.input_keys:
            raw = ctx.download_bytes(key)

            # Preprocess: quadtree decomposition
            data = preprocess_image_from_bytes(
                raw,
                img_size=img_size,
                in_channels=in_channels,
                fixed_length=fixed_length,
                patch_size=patch_size,
            )

            if self.task == "reconstruction":
                pred = infer_reconstruct(
                    self._model, data, self._device, self._config
                )
                png_bytes = visualize_reconstruction(data, pred, self._config)
                out_key = f"{self.output_prefix}{key}_reconstruct.png"
                ctx.upload_bytes(out_key, png_bytes, content_type="image/png")
                output_keys.append(out_key)

            elif self.task == "mae_reconstruction":
                pred, mask, loss = infer_mae(
                    self._model, data, self._device, self._config
                )
                png_bytes = visualize_reconstruction(
                    data, pred, self._config, mask=mask, loss=loss
                )
                out_key = f"{self.output_prefix}{key}_mae.png"
                ctx.upload_bytes(out_key, png_bytes, content_type="image/png")
                output_keys.append(out_key)

                # Also upload loss info
                loss_key = f"{self.output_prefix}{key}_mae_loss.json"
                ctx.upload_bytes(
                    loss_key,
                    json.dumps({"loss": loss, "source_key": key}).encode(),
                    content_type="application/json",
                )

            elif self.task == "feature_extraction":
                features = infer_features(
                    self._model, data, self._device, self._config
                )
                # CLS token is first; rest are patch tokens
                feat_np = features[0].numpy()  # [L+1, D]
                cls_token = feat_np[0].tolist()
                out_key = f"{self.output_prefix}{key}_features.json"
                ctx.upload_bytes(
                    out_key,
                    json.dumps(
                        {
                            "source_key": key,
                            "cls_embedding": cls_token,
                            "num_patches": feat_np.shape[0] - 1,
                            "feature_dim": feat_np.shape[1],
                        },
                        indent=2,
                    ).encode(),
                    content_type="application/json",
                )
                output_keys.append(out_key)

            else:
                raise ValueError(f"Unknown HMAE task: {self.task}")

            head = ctx.head_object(key)
            checksums.append(head["ETag"].strip('"'))
            metadata_list.append(head.get("Metadata", {}))

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
        )
