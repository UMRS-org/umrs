import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Dataset

from umrs.shared.storage.s3 import s3_key_exists, save_tiff_image_to_s3


################################ Masked Autoencoder(  ###########################
def load_masked_autoencoder(config_path, checkpoint_path):
    """
    1. Load the trained MAE model from checkpoint
    """

    sys.path.append("..")
    from external.MAE import MAE

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model architecture
    model = MAE(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=config["model"]["in_channels"],
        encoder_dim=config["model"]["encoder_embed_dim"],
        encoder_depth=config["model"]["encoder_depth"],
        encoder_heads=config["model"]["encoder_heads"],
        decoder_dim=config["model"]["decoder_embed_dim"],
        decoder_depth=config["model"]["decoder_depth"],
        decoder_heads=config["model"]["decoder_heads"],
        mask_ratio=config["model"]["mask_ratio"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Set to evaluation mode
    model.num_patches = (
        config["model"]["img_size"] // config["model"]["patch_size"]
    ) ** 2
    model.enc_embed_dim = config["model"]["encoder_embed_dim"]
    model.img_size = config["model"]["img_size"]
    return model


class Spring8DatasetFromS3(Dataset):
    """
    PyTorch Dataset for loading Spring8 .raw images from S3 keys.
    """

    def __init__(self, s3_client, resolution, s3_key_list, bucket_name, transform=None):
        self.s3 = s3_client
        self.bucket = bucket_name
        self.resolution = resolution
        self.s3_keys = s3_key_list

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.390480], std=[0.29985]),
                ]
            )

    def __len__(self):
        return len(self.s3_keys)

    def __getitem__(self, idx):
        key = self.s3_keys[idx]

        # Get object from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)

        # Extract metadata and checksum
        checksum = obj["ETag"].strip('"')  # ETag is often quoted
        metadata = obj.get("Metadata", {})  # User metadata dictionary

        # extract raw data
        raw_bytes = obj["Body"].read()

        # Convert raw bytes to numpy array
        image = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(
            self.resolution, self.resolution, 1
        )

        # Convert to uint8 range like before
        image = (image / 65535.0 * 255.0).astype(np.uint8)

        # Apply transforms
        image_tensor = self.transform(image)

        return image_tensor, checksum, metadata


def build_s8d_val_dataloader(
    s3, s3_keys, path_to_bucket, batch_size, num_workers, resolution
):
    """
    Builds and returns only the validation (test) DataLoader,
    and lists all file paths included in the validation set.
    """

    from torch.utils.data._utils.collate import default_collate

    def custom_collate(batch):
        images, checksums, metadatas = zip(*batch, strict=False)

        images = default_collate(images)
        checksums = default_collate(checksums)
        # Keep list of dicts as is (no collate)
        metadatas = list(metadatas)

        return images, checksums, metadatas

    val_dataset = Spring8DatasetFromS3(s3, resolution, s3_keys, path_to_bucket)

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )

    return val_loader


def extract_embeddings(model, data_loader, device):
    """
    3. & 4. Compute inference and extract embeddings
    """
    num_samples = len(data_loader.dataset)  # Total number of samples (not batches!)
    num_patches = model.num_patches
    visible_patches = int(num_patches * (1 - model.mask_ratio))
    received_patches = visible_patches + 1  # +1 pour le CLS
    patch_dim = model.patch_size**2 * model.in_chans
    embed_dim = model.enc_embed_dim

    # Preallocate array for all embeddings
    all_embeddings = torch.empty(
        (num_samples, received_patches, embed_dim), dtype=torch.float32, device=device
    )
    all_reconstructions = torch.empty(
        (num_samples, num_patches, patch_dim), dtype=torch.float32, device=device
    )

    model.eval()

    res_checksums = []
    res_metadata = []

    with torch.no_grad():
        sample_idx = 0  # keep track of current sample index
        for batch_idx, (images, checksums, metadatas) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

            # Accumulate metadata and checksums from current batch
            res_checksums.extend(checksums)
            res_metadata.extend(metadatas)

            images = images.to(device)
            # embeddings = model.encode(images, convert_to_tensor=False)
            _, recon_patches_flat, _, embeddings = model(images)
            batch_count = embeddings.shape[0]
            all_embeddings[sample_idx : sample_idx + batch_count] = embeddings
            all_reconstructions[sample_idx : sample_idx + batch_count] = (
                recon_patches_flat
            )
            sample_idx += batch_count

    if sample_idx > 0:
        # Concatenate all embeddings
        final_embeddings = all_embeddings.cpu().numpy()
        final_reconstructions = (
            all_reconstructions.cpu().numpy()
        )  # shape [num_samples, num_patches, patch_dim]
        # embedding_vector = final_embeddings[:, 0, :].tolist()  # shape [batch_size, embedding_dim]
    else:
        return None, None, None, None, None, None

    cls_embeddings = final_embeddings[:, 0, :]  # shape: [num_samples, embed_dim]

    patch_shape = (model.patch_size, model.patch_size, model.in_chans)

    return (
        final_embeddings,
        cls_embeddings,
        final_reconstructions,
        patch_shape,
        res_checksums,
        res_metadata,
    )


def save_one_reconstruction_in_one_image(
    patches,
    s3,
    s3_key,
    path_to_bucket,
    patch_shape,  # (patch, patch, C)
    resolution,
    overwrite=True,
):
    """
    Reconstruct a square image from flattened patches and save it as a TIFF.

    Parameters:
        patches: np.ndarray or torch.Tensor of shape [num_patches, patch_dim]
                 Contains all flattened patches of a single image.

        save_file: str
            Output filename (.tiff)

        patch_shape: tuple
            (patch_height, patch_width, C) — typically (P, P, channels)

        resolution: int
            Size of the final (square) output image: resolution x resolution.
            This must match the patch grid: resolution = P * sqrt(num_patches).

        overwrite: bool
            If False, do not overwrite an existing file.

    Returns:
        save_file: str
            Path to the saved TIFF image.
    """

    # Prevent overwriting existing file unless explicitly allowed
    if (not overwrite) and s3_key_exists(s3, path_to_bucket, s3_key):
        print(f"{s3_key} already exists. Skipping... Use overwrite=True.")
        return s3_key

    # Convert Torch → NumPy if needed
    if hasattr(patches, "cpu"):
        patches = patches.cpu().numpy()

    # Extract patch dimensions
    P, _, C = patch_shape  # patch height, patch width, channels
    num_patches, patch_dim = patches.shape

    # Sanity check: number of patches must form a perfect square
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"num_patches={num_patches} is not a perfect square; "
            f"cannot form a square image grid."
        )

    # Cross-check that patch geometry matches the requested image size
    if resolution != grid_size * P:
        raise ValueError(
            f"Inconsistent resolution: expected {grid_size * P}, got {resolution}."
        )

    # Reshape each flattened patch back to (C, P, P)
    patches = patches.reshape(num_patches, C, P, P)

    # Convert to (P, P, C)
    patches = patches.transpose(0, 2, 3, 1)

    # Arrange patches into a 2D grid (grid_size x grid_size)
    grid = patches.reshape(grid_size, grid_size, P, P, C)

    # Merge grid into a full (resolution, resolution, C) image
    full_img = grid.transpose(0, 2, 1, 3, 4).reshape(resolution, resolution, C)

    # Scale to [0,255] and convert to uint8
    full_img = np.clip(full_img * 255, 0, 255).astype(np.uint8)

    # If grayscale, remove the last channel dimension
    if full_img.shape[-1] == 1:
        full_img = full_img[:, :, 0]

    response = save_tiff_image_to_s3(s3, full_img, path_to_bucket, s3_key)

    status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status_code != 200:
        print(f"❌ Upload failed, status code: {status_code}")
        print(response)

    return path_to_bucket + "/" + s3_key


def save_all_reconstructions_in_one_image(
    all_reconstructions, patch_shape, s3, s3_keys, path_to_bucket, resolution=8192
):
    """
    Save each reconstruction (set of patches) as one large mosaic TIFF image.

    Parameters:
        all_reconstructions : Tensor/array [N, P, patch_dim]
        in_file_list        : list of input filenames
        patch_shape         : (H, W, C)
        resolution          : final resolution limit (max width/height)

    Returns:
        list of output TIFF image paths
    """
    saved_keys = []

    # convert tensor → numpy
    if hasattr(all_reconstructions, "cpu"):
        all_reconstructions = all_reconstructions.cpu().numpy()

    num_samples = all_reconstructions.shape[0]
    assert num_samples == len(s3_keys), (
        "Mismatch: number of reconstructions must match number of input files"
    )

    for idx in range(num_samples):
        patches = all_reconstructions[idx]  # shape [num_patches, patch_dim]

        base = os.path.splitext(s3_keys[idx])[0]
        save_key = f"{base}_patches.tiff"

        saved_path = save_one_reconstruction_in_one_image(
            patches,
            s3,
            save_key,
            path_to_bucket,
            patch_shape,
            resolution,
            overwrite=True,
        )
        # print(f"Output saved to: {saved_path}")

        saved_keys.append(saved_path)

    return saved_keys


def compute_embeddings_masked_autoencoder(
    self, s3, s3_keys, path_to_bucket, output_bucket
):
    # 2. Load data
    data_loader = build_s8d_val_dataloader(
        s3,
        s3_keys,
        path_to_bucket,
        batch_size=8,  # Adjust based on your GPU memory
        num_workers=0,  # this could be larger but requires a s3 endpoint per worker and hnce to pass the s3 configuration file and not the s3_connection
        resolution=self.model.img_size,
    )

    # 3 & 4. Extract embeddings

    _, cls_embeddings, all_reconstructions, patch_shape, checksums, metadata = (
        extract_embeddings(self.model, data_loader, self.device)
    )
    # _, cls_embeddings, checksums,  metadata = extract_embeddings(self.model, data_loader, self.device)

    saved_keys = None
    if all_reconstructions is not None:
        saved_keys = save_all_reconstructions_in_one_image(
            all_reconstructions,
            patch_shape,
            s3,
            s3_keys,
            output_bucket,
            resolution=self.model.img_size,
        )
        # saved_files = save_all_reconstructions(all_reconstructions, "./reconstruction", file_list, patch_shape

    return cls_embeddings, checksums, metadata, saved_keys
