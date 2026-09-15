"""HMAE inference integration for UMRS.

Merges the clean hmae-infer implementation (quadtree + HMAEVIT) with the
UMRS S3 pipeline from the HMAE_inference branch. Supports three tasks:
  - reconstruction: full reconstruction (encode all patches, decode all)
  - mae_reconstruction: MAE-style masked reconstruction (random masking + decode)
  - feature_extraction: encoder-only feature extraction (CLS + patch tokens)
"""

import io
import logging
import os

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from torch.amp import autocast

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hmae import create_model
from .hmae.engine import HierarchicalMaskedAutoEncoder

logger = logging.getLogger(__name__)

_model_cache = {}

_HMAE_TASK_ALIASES = {
    "reconstruction": "reconstruction",
    "reconstruct": "reconstruction",
    "full_reconstruction": "reconstruction",
    "custom:reconstruction": "reconstruction",
    "mae_reconstruction": "mae_reconstruction",
    "mae": "mae_reconstruction",
    "masked_reconstruction": "mae_reconstruction",
    "custom:mae_reconstruction": "mae_reconstruction",
    "feature_extraction": "feature_extraction",
    "features": "feature_extraction",
    "embedding": "feature_extraction",
    "custom:feature_extraction": "feature_extraction",
}


def normalize_hmae_task(task=None):
    """Normalize reference hmae-infer mode names to UMRS task names."""
    if task is None:
        return "reconstruction"
    key = str(task).strip().lower().replace("-", "_")
    return _HMAE_TASK_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_hmae_from_config(config, device):
    """Instantiate HMAEVIT and load checkpoint weights."""
    model_cfg = config["model"]
    model = create_model(
        model_type=model_cfg["model_type"],
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_channels=model_cfg.get("in_channels", 1),
        mask_ratio=model_cfg.get("mask_ratio", 0.5),
        norm_pix_loss=model_cfg.get("norm_pix_loss", True),
    )

    ckpt_path = config["inference"]["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HMAE checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Handle torch.compile keys (_orig_mod. prefix)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    epoch = checkpoint.get("epoch", "unknown")
    logger.info("Loaded HMAE checkpoint from %s (epoch %s)", ckpt_path, epoch)

    return model.to(device).eval()


def load_hmae_model(cfg_file, checkpoint_path):
    """Load HMAE model with caching.  Returns (model, device, config)."""
    cache_key = f"{cfg_file}:{checkpoint_path}"
    if cache_key in _model_cache:
        logger.info("HMAE model already cached, reusing")
        return _model_cache[cache_key]

    with open(cfg_file) as f:
        config = yaml.safe_load(f)

    config["inference"]["checkpoint_path"] = checkpoint_path

    device_str = config["inference"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    model = _load_hmae_from_config(config, device)
    result = (model, device, config)
    _model_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Image preprocessing (S3 → quadtree patches)
# ---------------------------------------------------------------------------


def preprocess_image_from_s3(
    s3, bucket, s3_key, img_size=1024, in_channels=1, fixed_length=1024, patch_size=32
):
    """Download image from S3, run quadtree decomposition, return patch tensors."""
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    checksum = obj["ETag"].strip('"')
    metadata = obj.get("Metadata", {})

    image_bytes = obj["Body"].read()
    image_array = np.frombuffer(image_bytes, np.uint8)

    if in_channels == 1:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if full_image is None:
        raise FileNotFoundError(f"Cannot decode image from S3: {s3_key}")

    if full_image.shape[0] != img_size or full_image.shape[1] != img_size:
        full_image = cv2.resize(full_image, (img_size, img_size))

    # Edge detection for quadtree
    gray = (
        full_image if in_channels == 1 else cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    )
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)

    # Quadtree decomposition
    engine = HierarchicalMaskedAutoEncoder(
        fixed_length=fixed_length, patch_size=patch_size
    )
    img_np = full_image[..., np.newaxis] if in_channels == 1 else full_image
    patches, coords, depths = engine.process_single(img_np, edges)

    # Original image tensor for visualization
    if in_channels == 1:
        original = torch.from_numpy(full_image).float().unsqueeze(0) / 255.0
    else:
        original = (
            torch.from_numpy(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
            .float()
            .permute(2, 0, 1)
            / 255.0
        )

    data = {
        "patches": patches.unsqueeze(0),  # [1, L, C, P, P]
        "coords": coords.unsqueeze(0),  # [1, L, 4]
        "depths": depths.unsqueeze(0),  # [1, L]
        "original_image": original.unsqueeze(0),  # [1, C, H, W]
    }
    return data, checksum, metadata


def preprocess_image_from_bytes(
    image_bytes, img_size=1024, in_channels=1, fixed_length=1024, patch_size=32
):
    """Preprocess raw image bytes (for BentoML service path)."""
    image_array = np.frombuffer(image_bytes, np.uint8)

    if in_channels == 1:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if full_image is None:
        raise ValueError("Cannot decode image from provided bytes")

    if full_image.shape[0] != img_size or full_image.shape[1] != img_size:
        full_image = cv2.resize(full_image, (img_size, img_size))

    gray = (
        full_image if in_channels == 1 else cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    )
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)

    engine = HierarchicalMaskedAutoEncoder(
        fixed_length=fixed_length, patch_size=patch_size
    )
    img_np = full_image[..., np.newaxis] if in_channels == 1 else full_image
    patches, coords, depths = engine.process_single(img_np, edges)

    if in_channels == 1:
        original = torch.from_numpy(full_image).float().unsqueeze(0) / 255.0
    else:
        original = (
            torch.from_numpy(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
            .float()
            .permute(2, 0, 1)
            / 255.0
        )

    return {
        "patches": patches.unsqueeze(0),
        "coords": coords.unsqueeze(0),
        "depths": depths.unsqueeze(0),
        "original_image": original.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Inference functions (3 modes)
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_reconstruct(model, data, device, config, use_bf16=True):
    """Full reconstruction: encode all patches → decode all."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type != "cpu" or not use_bf16),
    ):
        pred = model.reconstruct(patches, coords, depths)
    return pred.float().cpu()


@torch.no_grad()
def infer_mae(model, data, device, config, use_bf16=True):
    """MAE-style inference: random masking → encode visible → decode all."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type != "cpu" or not use_bf16),
    ):
        loss, pred, mask = model(patches, coords, depths)
    return pred.float().cpu(), mask.float().cpu(), loss.item()


@torch.no_grad()
def infer_features(model, data, device, config, use_bf16=True):
    """Extract encoder features: CLS token + patch tokens."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type != "cpu" or not use_bf16),
    ):
        features = model.encode(patches, coords, depths)
    return features.float().cpu()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_reconstruction(data, pred, config, mask=None, loss=None):
    """Render side-by-side: Original | Quadtree Grid | Reconstruction → PNG bytes."""
    img_size = config["model"]["img_size"]
    norm_pix_loss = config["model"].get("norm_pix_loss", True)
    in_channels = config["model"].get("in_channels", 1)
    patch_size = config["model"]["patch_size"]

    target_patches = data["patches"][0].cpu().float()
    coords_np = data["coords"][0].cpu().numpy().astype(int)
    recon_flat = pred[0].cpu().float()

    if in_channels == 1:
        canvas_orig = np.zeros((img_size, img_size), dtype=np.float32)
        canvas_recon = np.zeros((img_size, img_size), dtype=np.float32)
        canvas_qdt = np.zeros((img_size, img_size), dtype=np.float32)
    else:
        canvas_orig = np.zeros((img_size, img_size, 3), dtype=np.float32)
        canvas_recon = np.zeros((img_size, img_size, 3), dtype=np.float32)
        canvas_qdt = np.zeros((img_size, img_size, 3), dtype=np.float32)

    mask_np = mask[0].cpu().numpy() if mask is not None else None
    L = target_patches.shape[0]

    for i in range(L):
        x1, x2, y1, y2 = coords_np[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        t_patch = target_patches[i].permute(1, 2, 0).numpy()
        if in_channels == 1:
            t_patch = t_patch.squeeze()
        resized_t = cv2.resize(t_patch, (w, h), interpolation=cv2.INTER_NEAREST)
        is_masked = mask_np is not None and mask_np[i] == 1

        # Original canvas
        if in_channels == 1:
            canvas_orig[y1:y2, x1:x2] = resized_t
            canvas_qdt[y1:y2, x1:x2] = 0.3 if is_masked else resized_t
            cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), 1.0, 1)
        else:
            canvas_orig[y1:y2, x1:x2] = resized_t
            canvas_qdt[y1:y2, x1:x2] = 0.3 if is_masked else resized_t
            cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1)

        # Reconstruction canvas
        if mask_np is not None and mask_np[i] == 0:
            # Visible patch → show original
            if in_channels == 1:
                canvas_recon[y1:y2, x1:x2] = resized_t
            else:
                canvas_recon[y1:y2, x1:x2] = resized_t
        else:
            r_patch = (
                recon_flat[i]
                .view(in_channels, patch_size, patch_size)
                .permute(1, 2, 0)
                .numpy()
            )
            if in_channels == 1:
                r_patch = r_patch.squeeze()
            if norm_pix_loss:
                if in_channels == 1:
                    mean, std = t_patch.mean(), t_patch.std()
                else:
                    mean = t_patch.mean(axis=(0, 1))
                    std = t_patch.std(axis=(0, 1))
                r_patch = r_patch * (std + 1e-6) + mean
            r_patch = np.clip(r_patch, 0, 1)
            resized_r = cv2.resize(r_patch, (w, h), interpolation=cv2.INTER_NEAREST)
            if in_channels == 1:
                canvas_recon[y1:y2, x1:x2] = resized_r
            else:
                canvas_recon[y1:y2, x1:x2] = resized_r

    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    title_str = "HMAE Inference"
    if loss is not None:
        title_str += f" | Loss: {loss:.6f}"
    fig.suptitle(title_str, fontsize=14)

    for ax, canvas, title in zip(
        axes,
        [canvas_orig, canvas_qdt, canvas_recon],
        ["Original", "Quadtree Grid", "Reconstruction"],
        strict=False,
    ):
        if in_channels == 1:
            ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(canvas)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _upload_png_to_s3(s3, bucket, key, png_bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=png_bytes, ContentType="image/png")


# ---------------------------------------------------------------------------
# UMRS compute pipeline (for embedding worker / model_loading integration)
# ---------------------------------------------------------------------------


def compute_hmae_inference(
    self,
    s3,
    s3_keys,
    path_to_bucket,
    output_bucket,
    task="reconstruction",
    skip_existing=True,
):
    """Main UMRS pipeline: process images from S3 through HMAE inference.

    Args:
        self: EmbeddingModel instance (has .model, .device attributes).
        s3: boto3 S3 client.
        s3_keys: list of S3 object keys.
        path_to_bucket: source bucket name.
        output_bucket: destination bucket for results.
        task: "reconstruction", "mae_reconstruction", or "feature_extraction".
        skip_existing: If True, skip processing when output already exists
            in output_bucket.

    Returns:
        (cls_embeddings, checksums, metadata_list, saved_keys)
    """
    task = normalize_hmae_task(task)
    config = self._hmae_config
    model = self.model
    device = self.device
    model_cfg = config["model"]
    use_bf16 = config["inference"].get("use_bf16", True)
    if device.type == "cpu":
        use_bf16 = False

    exts = config["input"].get(
        "img_extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    )
    filtered_keys = [k for k in s3_keys if any(k.lower().endswith(ext) for ext in exts)]

    saved_keys = []
    cls_embeddings = []
    checksums = []
    metadata_list = []

    for idx, s3_key in enumerate(filtered_keys):
        logger.info(
            "[%d/%d] Processing %s (task=%s)", idx + 1, len(filtered_keys), s3_key, task
        )

        base = os.path.splitext(s3_key)[0]

        # Determine expected output key for this task
        if task == "reconstruction":
            expected_out_key = f"{base}_recon.png"
        elif task == "mae_reconstruction":
            expected_out_key = f"{base}_mae.png"
        elif task == "feature_extraction":
            expected_out_key = f"{base}_features.pt"
        else:
            expected_out_key = None

        # Skip if output already exists in the output bucket
        if skip_existing and output_bucket and expected_out_key:
            try:
                s3.head_object(Bucket=output_bucket, Key=expected_out_key)
                logger.info(
                    "[%d/%d] Skipping %s — output already exists in %s",
                    idx + 1,
                    len(filtered_keys),
                    s3_key,
                    output_bucket,
                )
                obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
                checksums.append(obj["ETag"].strip('"'))
                metadata_list.append(obj.get("Metadata", {}))
                saved_keys.append(f"{output_bucket}/{expected_out_key}")
                cls_embeddings.append(None)
                continue
            except Exception:
                pass  # Output does not exist, proceed with computation

        data, checksum, metadata = preprocess_image_from_s3(
            s3,
            path_to_bucket,
            s3_key,
            img_size=model_cfg["img_size"],
            in_channels=model_cfg.get("in_channels", 1),
            fixed_length=model_cfg.get("fixed_length", 1024),
            patch_size=model_cfg["patch_size"],
        )
        checksums.append(checksum)
        metadata_list.append(metadata)

        if task == "reconstruction":
            pred = infer_reconstruct(model, data, device, config, use_bf16)
            saved_key = f"{base}_recon.png"
            png_bytes = visualize_reconstruction(data, pred, config)
            _upload_png_to_s3(s3, output_bucket, saved_key, png_bytes)
            embedding = None

        elif task == "mae_reconstruction":
            pred, mask, loss = infer_mae(model, data, device, config, use_bf16)
            logger.info("  MAE loss: %.6f", loss)
            saved_key = f"{base}_mae.png"
            png_bytes = visualize_reconstruction(
                data, pred, config, mask=mask, loss=loss
            )
            _upload_png_to_s3(s3, output_bucket, saved_key, png_bytes)
            embedding = None

        elif task == "feature_extraction":
            features = infer_features(model, data, device, config, use_bf16)
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]
            logger.info(
                "  CLS token: %s, Patch tokens: %s", cls_token.shape, patch_tokens.shape
            )
            saved_key = f"{base}_features.pt"
            buffer = io.BytesIO()
            torch.save(
                {"cls": cls_token, "patches": patch_tokens, "coords": data["coords"]},
                buffer,
            )
            buffer.seek(0)
            s3.put_object(Bucket=output_bucket, Key=saved_key, Body=buffer.getvalue())
            embedding = cls_token
        else:
            raise ValueError(f"Unknown HMAE task: {task}")

        saved_keys.append(f"{output_bucket}/{saved_key}")
        cls_embeddings.append(embedding)

    return cls_embeddings, checksums, metadata_list, saved_keys


def make_hmae_compute_func(modality, cfg_file, task=None):
    """Create a bound compute function for a specific HMAE task.

    Args:
        modality: Modality name (unused, kept for API compatibility).
        cfg_file: Path to HMAE YAML config.
        task: UMRS task name — ``"reconstruction"``, ``"mae_reconstruction"``,
            ``"feature_extraction"``, or ``"embedding"``.
            Defaults to ``"reconstruction"`` when ``None``.
    """
    with open(cfg_file) as f:
        yaml.safe_load(f)

    resolved_task = normalize_hmae_task(task)

    def func(self, s3, s3_keys, path_to_bucket, output_bucket, skip_existing=True):
        return compute_hmae_inference(
            self,
            s3,
            s3_keys,
            path_to_bucket,
            output_bucket,
            resolved_task,
            skip_existing=skip_existing,
        )

    return func
