import logging
import time
from io import BytesIO

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

_RESIZE_CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

################################ ResNet-50 ###########################

_model_cache = {}


def load_resnet50(model_path=None):
    """Load ResNet-50 model, handling DataParallel checkpoints if needed.
    Uses updated PyTorch conventions."""

    if model_path in _model_cache:
        print(f"\t\t\t--> Model '{model_path}' already loaded...")
        return _model_cache[model_path]

    print(f"\t\t\t--> Loading model '{model_path}'...")

    # Use weights if no model_path provided
    if model_path is None:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if model_path:
        # Use weights_only=True for safer deserialization
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )

        # Handle 'module.' prefix in state_dict keys from DataParallel
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        # Load state dict with relaxed strictness
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()  # Set to evaluation mode
    _model_cache[model_path] = model
    return model


def preprocess_images_batch(images):
    """Preprocess a list of PIL Images into a batch tensor for ResNet-50."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transform to each image and stack them (.convert ensures RGB for
    # grayscale or RGBA inputs — ResNet expects 3-channel tensors)
    image_tensors = [transform(image.convert("RGB")) for image in images]
    batch_tensor = torch.stack(image_tensors)  # Shape: (batch_size, 3, 224, 224)

    return batch_tensor


def compute_embeddings_batch(model, image_batch_tensor, device="cpu"):
    """Compute embeddings for a batch of images using ResNet-50."""
    image_batch_tensor = image_batch_tensor.to(device)

    with torch.no_grad():
        embeddings = model(image_batch_tensor)  # Batch forward pass

    return embeddings.cpu()  # Convert to NumPy array if needed


def compute_embeddings_resnet_50(self, images):
    t0 = time.perf_counter()
    image_batch_tensor = preprocess_images_batch(images)
    t_pil = time.perf_counter() - t0

    t0 = time.perf_counter()
    embeddings = compute_embeddings_batch(
        self.model, image_batch_tensor, device=self.device
    )
    t_gpu = time.perf_counter() - t0

    logger.info("[resnet50] n=%d  pil=%.3fs  forward=%.3fs", len(images), t_pil, t_gpu)
    return embeddings


def preprocess_bytes_uint8(raw):
    """Image bytes -> 224x224x3 uint8 (Resize + CenterCrop); runs in preprocessing pool workers."""
    img = Image.open(BytesIO(raw)).convert("RGB")
    return np.asarray(_RESIZE_CROP(img), dtype=np.uint8)


def uint8_batch_to_input(arrays, device):
    """HWC uint8 images -> normalized NCHW float batch on *device* (same as ToTensor + Normalize)."""
    x = torch.from_numpy(np.stack(arrays)).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous()


def compute_embeddings_resnet_50_from_uint8(self, arrays):
    t0 = time.perf_counter()
    embeddings = compute_embeddings_batch(
        self.model, uint8_batch_to_input(arrays, self.device), device=self.device
    )
    logger.info("[resnet50] n=%d  forward=%.3fs  (preproc pool)", len(arrays), time.perf_counter() - t0)
    return embeddings
