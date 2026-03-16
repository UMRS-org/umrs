"""SAM segmentation integration for UMRS.

Supports both SAM3 (text-prompted segmentation) and SAM2 (automatic mask
generation) models.  Auto-detects the model type from config.json and adjusts
the inference pipeline accordingly.

Input: image (from S3 URL or raw bytes) + text prompt
Output: segmentation masks, bounding boxes, confidence scores
"""

import io
import json
import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_sam3_cache = {}

# Sentinel to distinguish Sam2 from Sam3 in the cache
_MODEL_TYPE_KEY = "__model_type__"


def _detect_model_type(model_name_or_path):
    """Return 'sam3' or 'sam2' by inspecting config.json."""
    cfg_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if any("Sam2" in a for a in archs):
            return "sam2"
    return "sam3"


def load_sam3_model(model_name_or_path="facebook/sam3"):
    """Load SAM model and processor from HuggingFace.

    Auto-detects whether the weights are SAM3 or SAM2 and loads the
    appropriate classes.

    Returns:
        (model, processor, device)
    """
    if model_name_or_path in _sam3_cache:
        logger.info("SAM model already cached, reusing")
        return _sam3_cache[model_name_or_path]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = _detect_model_type(model_name_or_path)
    logger.info("Detected model type: %s at '%s'", model_type, model_name_or_path)

    if model_type == "sam2":
        from transformers import Sam2Model, Sam2Processor

        logger.info("Loading SAM2 model '%s' on %s ...", model_name_or_path, device)
        model = Sam2Model.from_pretrained(model_name_or_path).to(device)
        processor = Sam2Processor.from_pretrained(model_name_or_path)
    else:
        from transformers import Sam3Model, Sam3Processor

        logger.info("Loading SAM3 model '%s' on %s ...", model_name_or_path, device)
        model = Sam3Model.from_pretrained(model_name_or_path).to(device)
        processor = Sam3Processor.from_pretrained(model_name_or_path)

    model.eval()
    result = (model, processor, device)
    _sam3_cache[model_name_or_path] = result
    logger.info("SAM model loaded successfully (%s)", model_type)
    return result


def _mask_to_box(mask):
    """Compute bounding box [x1, y1, x2, y2] from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def _segment_sam2_auto(model, processor, image, device, mask_threshold=0.5,
                       points_per_side=8, max_dim=1024):
    """Automatic mask generation with SAM2 using a grid of point prompts.

    Generates a grid of points across the image, runs SAM2 for each point,
    and returns the best mask per point (deduplication via IoU filtering).
    Large images are resized to *max_dim* before inference for speed; masks
    are upscaled back to the original resolution.
    """
    orig_w, orig_h = image.size

    # Down-scale large images to keep inference tractable on CPU
    scale = 1.0
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resized image from %dx%d to %dx%d for inference",
                     orig_w, orig_h, new_w, new_h)

    w, h = image.size

    # Generate sparse point grid
    xs = np.linspace(0, w - 1, points_per_side, dtype=np.float32)
    ys = np.linspace(0, h - 1, points_per_side, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    all_masks = []
    all_scores = []

    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch_pts = points[i:i + batch_size]
        # Sam2 expects: input_points shape [batch, num_objects, num_points, 2]
        # For single image: we pass each point as a separate object
        input_points = [[[pt.tolist()] for pt in batch_pts]]
        input_labels = [[[1] for _ in batch_pts]]

        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = outputs.pred_masks  # [1, num_objects, 1, H, W]
        iou_scores = outputs.iou_scores  # [1, num_objects, 1]

        masks = masks.squeeze(0).squeeze(1)  # [num_objects, H, W]
        iou_scores = iou_scores.squeeze(0).squeeze(-1)  # [num_objects]

        # Resize masks to original image size
        for j in range(masks.shape[0]):
            score = iou_scores[j].item()
            if score < 0.5:
                continue
            mask_t = torch.nn.functional.interpolate(
                masks[j:j+1].unsqueeze(0).float(),
                size=(orig_h, orig_w), mode="bilinear", align_corners=False,
            ).squeeze()
            mask_np = (mask_t.cpu().numpy() > mask_threshold).astype(np.uint8)
            if mask_np.sum() == 0:
                continue
            all_masks.append(mask_np)
            all_scores.append(score)

    if not all_masks:
        return {"masks": [], "boxes": [], "scores": [], "num_objects": 0}

    # Simple IoU-based NMS to remove duplicate masks
    keep = _nms_masks(all_masks, all_scores, iou_threshold=0.5)
    masks_out = [all_masks[i] for i in keep]
    scores_out = [all_scores[i] for i in keep]
    boxes_out = [_mask_to_box(m) for m in masks_out]

    return {
        "masks": masks_out,
        "boxes": boxes_out,
        "scores": scores_out,
        "num_objects": len(masks_out),
    }


def _nms_masks(masks, scores, iou_threshold=0.5):
    """Non-maximum suppression on masks by IoU."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx in suppressed or jdx == idx:
                continue
            inter = np.logical_and(masks[idx], masks[jdx]).sum()
            union = np.logical_or(masks[idx], masks[jdx]).sum()
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(jdx)
    return keep


def segment_image_with_text(model, processor, image, text_prompt, device="cuda",
                             threshold=0.5, mask_threshold=0.5):
    """Run segmentation with a text prompt.

    For SAM3: uses text-prompted instance segmentation.
    For SAM2: falls back to automatic mask generation (text prompt is logged
    but not used, since SAM2 does not support text prompts).

    Returns:
        dict with keys: masks (list of np arrays), boxes (list), scores (list)
    """
    from transformers import Sam2Model

    if isinstance(model, Sam2Model):
        logger.info("SAM2 detected — using automatic mask generation "
                     "(text prompt '%s' ignored)", text_prompt)
        return _segment_sam2_auto(model, processor, image, device,
                                  mask_threshold=mask_threshold)

    # Sam3 path — text-prompted instance segmentation
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs.get("original_sizes")
    if original_sizes is not None:
        target_sizes = original_sizes.tolist()
    else:
        target_sizes = [list(image.size[::-1])]  # [height, width]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    masks_np = [m.cpu().numpy() for m in results["masks"]] if len(results["masks"]) > 0 else []
    boxes_list = results["boxes"].cpu().tolist() if len(results["boxes"]) > 0 else []
    scores_list = results["scores"].cpu().tolist() if len(results["scores"]) > 0 else []

    return {
        "masks": masks_np,
        "boxes": boxes_list,
        "scores": scores_list,
        "num_objects": len(masks_np),
    }


def segment_image_with_boxes(model, processor, image, boxes, labels, device="cuda",
                              threshold=0.5, mask_threshold=0.5):
    """Run segmentation with bounding box prompts.

    Works with both SAM2 and SAM3 (both support box prompts).
    """
    from transformers import Sam2Model

    if isinstance(model, Sam2Model):
        return _segment_sam2_boxes(model, processor, image, boxes, labels,
                                   device, mask_threshold)

    # Sam3 path
    input_boxes = [boxes]
    input_boxes_labels = [labels]

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs.get("original_sizes")
    if original_sizes is not None:
        target_sizes = original_sizes.tolist()
    else:
        target_sizes = [list(image.size[::-1])]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    masks_np = [m.cpu().numpy() for m in results["masks"]] if len(results["masks"]) > 0 else []
    boxes_list = results["boxes"].cpu().tolist() if len(results["boxes"]) > 0 else []
    scores_list = results["scores"].cpu().tolist() if len(results["scores"]) > 0 else []

    return {
        "masks": masks_np,
        "boxes": boxes_list,
        "scores": scores_list,
        "num_objects": len(masks_np),
    }


def _segment_sam2_boxes(model, processor, image, boxes, labels, device, mask_threshold):
    """SAM2 box-prompted segmentation."""
    w, h = image.size
    input_boxes = [[box] for box in boxes]  # [num_objects, 1, 4]

    inputs = processor(
        images=image,
        input_boxes=[input_boxes],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = outputs.pred_masks.squeeze(0).squeeze(1)  # [num_objects, H, W]
    iou_scores = outputs.iou_scores.squeeze(0).squeeze(-1)  # [num_objects]

    masks_np = []
    scores_list = []
    boxes_list = []
    for j in range(masks.shape[0]):
        mask_t = torch.nn.functional.interpolate(
            masks[j:j+1].unsqueeze(0).float(),
            size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze()
        mask_np = (mask_t.cpu().numpy() > mask_threshold).astype(np.uint8)
        masks_np.append(mask_np)
        scores_list.append(iou_scores[j].item())
        boxes_list.append(_mask_to_box(mask_np))

    return {
        "masks": masks_np,
        "boxes": boxes_list,
        "scores": scores_list,
        "num_objects": len(masks_np),
    }


def overlay_masks_on_image(image, masks, alpha=0.5):
    """Overlay segmentation masks on an image with rainbow colors.

    Args:
        image: PIL.Image (RGB).
        masks: list of 2D numpy arrays (binary masks).
        alpha: mask overlay transparency.

    Returns:
        PIL.Image with masks overlaid.
    """
    import matplotlib

    image = image.convert("RGBA")
    n_masks = len(masks)
    if n_masks == 0:
        return image

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask_uint8 = (255 * mask.astype(np.uint8))
        mask_img = Image.fromarray(mask_uint8)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_mask = mask_img.point(lambda v: int(v * alpha))
        overlay.putalpha(alpha_mask)
        image = Image.alpha_composite(image, overlay)

    return image


def image_from_s3(s3, bucket, key):
    """Download an image from S3 and return as PIL Image."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = obj["Body"].read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def compute_sam3_segmentation(self, s3, s3_keys, path_to_bucket, output_bucket,
                               text_prompt="object"):
    """UMRS pipeline: download images from S3, segment with SAM3, upload results.

    Returns:
        (results_list, checksums, metadata_list, saved_keys)
    """
    model, processor, device = self.model, self._sam3_processor, self.device
    saved_keys = []
    results_list = []
    checksums = []
    metadata_list = []

    for idx, s3_key in enumerate(s3_keys):
        logger.info("[%d/%d] SAM3 segmenting: %s (prompt='%s')",
                     idx + 1, len(s3_keys), s3_key, text_prompt)

        obj = s3.get_object(Bucket=path_to_bucket, Key=s3_key)
        checksum = obj["ETag"].strip('"')
        metadata = obj.get("Metadata", {})
        checksums.append(checksum)
        metadata_list.append(metadata)

        image_bytes = obj["Body"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = segment_image_with_text(model, processor, image, text_prompt, device)
        results_list.append(result)

        # Save overlay visualization
        if result["num_objects"] > 0:
            overlay = overlay_masks_on_image(image, result["masks"])
            base = s3_key.rsplit(".", 1)[0] if "." in s3_key else s3_key
            saved_key = f"{base}_sam3_seg.png"
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            s3.put_object(Bucket=output_bucket, Key=saved_key, Body=buf.getvalue(),
                          ContentType="image/png")
            saved_keys.append(f"{output_bucket}/{saved_key}")
        else:
            saved_keys.append(None)

    return results_list, checksums, metadata_list, saved_keys
