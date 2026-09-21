#!/usr/bin/env python3
"""
Standalone embedding extractor for JPEG images.

Models:
  resnet  — torchvision ResNet-50 (ImageNet weights), 1000-dim output
  clip    — SentenceTransformer clip-ViT-B-32, 512-dim output

Each input image foo/bar.jpeg produces foo/bar.emb (one float per line).

Usage:
  ./embed_images.py --model resnet dir1 dir2 ...
  ./embed_images.py --model clip --batch-size 64 --timing dir1 dir2 ...
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_resnet():
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    device = _device()
    model.to(device)
    return model, device


def load_clip():
    from sentence_transformers import SentenceTransformer

    device = _device()
    model = SentenceTransformer("clip-ViT-B-32")
    model.to(device)
    return model, device


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_resnet_tensor(pil_images):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(img) for img in pil_images])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_resnet(model, tensor, device):
    with torch.no_grad():
        embs = model(tensor.to(device))
    return embs.cpu().numpy()


def infer_clip(model, pil_images, device):
    with torch.no_grad():
        embs = model.encode(pil_images, convert_to_tensor=(device == "cuda"))
    if hasattr(embs, "cpu"):
        embs = embs.cpu().numpy()
    return np.array(embs)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(paths, model_name, model, device, timing):
    # --- IO ---
    t0 = time.perf_counter()
    images = [Image.open(p).convert("RGB") for p in paths]
    t_io = time.perf_counter() - t0

    # --- Tensor construction ---
    t1 = time.perf_counter()
    if model_name == "resnet":
        tensor = build_resnet_tensor(images)
    else:
        tensor = None   # CLIP preprocessing is internal to encode()
    t_tensor = time.perf_counter() - t1

    # --- Inference ---
    t2 = time.perf_counter()
    if model_name == "resnet":
        embeddings = infer_resnet(model, tensor, device)
    else:
        embeddings = infer_clip(model, images, device)
    t_infer = time.perf_counter() - t2

    # --- Save embeddings ---
    t3 = time.perf_counter()
    for path, emb in zip(paths, embeddings):
        np.savetxt(path.with_suffix(".emb"), emb.reshape(-1, 1))
    t_save = time.perf_counter() - t3

    if timing:
        n = len(paths)
        total = t_io + t_tensor + t_infer + t_save
        print(
            f"  n={n:4d}  "
            f"io={t_io:.3f}s  "
            f"tensor={t_tensor:.3f}s  "
            f"infer={t_infer:.3f}s  "
            f"save={t_save:.3f}s  "
            f"total={total:.3f}s  "
            f"({n/total:.1f} img/s)"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract image embeddings (ResNet-50 or CLIP) and save as .emb files."
    )
    parser.add_argument("--model", choices=["resnet", "clip"], required=True,
                        help="Embedding model to use")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Images per batch (default: all at once)")
    parser.add_argument("--timing", action="store_true",
                        help="Print IO / tensor / inference / save times per batch")
    parser.add_argument("dirs", nargs="+",
                        help="Directories containing JPEG images")
    args = parser.parse_args()

    # Collect images
    paths = []
    for d in args.dirs:
        paths.extend(sorted(p for p in Path(d).iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg")))
    if not paths:
        print("No JPEG files found.")
        return
    print(f"Found {len(paths)} images across {len(args.dirs)} director(y/ies)")

    # Load model
    print(f"Loading {args.model}…")
    t0 = time.perf_counter()
    if args.model == "resnet":
        model, device = load_resnet()
    else:
        model, device = load_clip()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s  device={device}")

    # Batch processing
    batch_size = args.batch_size or len(paths)
    n_batches = math.ceil(len(paths) / batch_size)
    t_start = time.perf_counter()

    for i in range(n_batches):
        batch = paths[i * batch_size:(i + 1) * batch_size]
        if args.timing:
            print(f"Batch {i + 1}/{n_batches}:")
        process_batch(batch, args.model, model, device, args.timing)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {len(paths)} embeddings in {elapsed:.2f}s "
          f"({len(paths) / elapsed:.1f} img/s)")


if __name__ == "__main__":
    main()
