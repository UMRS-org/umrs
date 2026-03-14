import sys, os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from umrs.s3 import s3_key_exists, save_tiff_image_to_s3
from transformers.utils import logging

from .hmae_model import create_model
import cv2
import io 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .engine.hmae import HierarchicalMaskedAutoEncoder

_model_cache = {}

def load_model(config, device):
    """Load HMAE model from checkpoint."""
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
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle compiled model keys (remove _orig_mod. prefix)
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            cleaned[new_key] = v
        model.load_state_dict(cleaned, strict=False)
        epoch = checkpoint.get("epoch", "unknown")
    else:
        raise ValueError (f"Error: {ckpt_path} does not exists")

    model = model.to(device).eval()
    return model


def load_masked_autoencoder_infer(cfg_file, checkpoint_path, model_name):
    model_key = cfg_file+checkpoint_path+model_name #config and weights and name
    if model_key in _model_cache:
        print(f"\t\t\t--> Model '{model_name}' already loaded...")
        return _model_cache[model_key]

    logging.set_verbosity_error()  # avoid verbose warning
    print(f"\t\t\t--> Loading model '{model_name}'...")

    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)


    config["inference"]["checkpoint_path"] = checkpoint_path
    device_str = config["inference"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
  
   
    model = load_model(config, device)
    _model_cache[model_key] = (model, device)
    return model, device



def preprocess_image(s3, bucket, s3_key, img_size=1024, in_channels=1, fixed_length=1024, patch_size=32):
    """Load and preprocess a single image through the HMAE quadtree engine."""

    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    
    # Extract metadata and checksum
    checksum = obj['ETag'].strip('"')  # ETag is often quoted
    metadata = obj.get("Metadata", {})   # User metadata dictionary
       

    image_bytes = obj["Body"].read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    if in_channels == 1:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        full_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if full_image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if full_image.shape[0] != img_size or full_image.shape[1] != img_size:
        full_image = cv2.resize(full_image, (img_size, img_size))

    # Edge detection for quadtree
    if in_channels == 1:
        gray = full_image
    else:
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)

    # Quadtree decomposition
    engine = HierarchicalMaskedAutoEncoder(fixed_length=fixed_length, patch_size=patch_size)
    if in_channels == 1:
        img_np = full_image[..., np.newaxis]
    else:
        img_np = full_image

    patches, coords, depths = engine.process_single(img_np, edges)

    # Store original for visualization
    if in_channels == 1:
        original = torch.from_numpy(full_image).float().unsqueeze(0) / 255.0
    else:
        original = torch.from_numpy(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0

    return {
        "patches": patches.unsqueeze(0),       # [1, L, C, P, P]
        "coords": coords.unsqueeze(0),         # [1, L, 4]
        "depths": depths.unsqueeze(0),          # [1, L]
        "original_image": original.unsqueeze(0),  # [1, C, H, W]
    }, checksum, metadata




@torch.no_grad()
def infer_reconstruct(model, data, device, config, use_bf16=True):
    """Run full reconstruction (encode all, decode all)."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(device_type=device.type, dtype=dtype, enabled=(device.type != "cpu" or not use_bf16)):
        pred = model.reconstruct(patches, coords, depths)

    return pred.float().cpu()


@torch.no_grad()
def infer_mae(model, data, device, config, use_bf16=True):
    """Run MAE-style inference with masking (for visualizing reconstruction quality)."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(device_type=device.type, dtype=dtype, enabled=(device.type != "cpu" or not use_bf16)):
        loss, pred, mask = model(patches, coords, depths)

    return pred.float().cpu(), mask.float().cpu(), loss.item()


@torch.no_grad()
def infer_features(model, data, device, config, use_bf16=True):
    """Extract encoder features (CLS + patch tokens)."""
    patches = data["patches"].to(device)
    coords = data["coords"].to(device)
    depths = data["depths"].to(device)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with autocast(device_type=device.type, dtype=dtype, enabled=(device.type != "cpu" or not use_bf16)):
        features = model.encode(patches, coords, depths)

    return features.float().cpu()


def visualize_reconstruction(data, pred, s3, output_bucket, s3_key, config, mask=None, loss=None):
    """Render original, quadtree grid, and reconstruction side by side."""
    img_size = config["model"]["img_size"]
    norm_pix_loss = config["model"].get("norm_pix_loss", True)
    in_channels = config["model"].get("in_channels", 1)
    patch_size = config["model"]["patch_size"]

    target_patches = data["patches"][0].cpu().float()   # [L, C, P, P]
    coords = data["coords"][0].cpu().numpy().astype(int)  # [L, 4]
    recon_flat = pred[0].cpu().float()                   # [L, patch_dim]

    if in_channels == 1:
        canvas_orig = np.zeros((img_size, img_size), dtype=np.float32)
        canvas_recon = np.zeros((img_size, img_size), dtype=np.float32)
        canvas_qdt = np.zeros((img_size, img_size), dtype=np.float32)
        cmap = "gray"
    else:
        canvas_orig = np.zeros((img_size, img_size, 3), dtype=np.float32)
        canvas_recon = np.zeros((img_size, img_size, 3), dtype=np.float32)
        canvas_qdt = np.zeros((img_size, img_size, 3), dtype=np.float32)
        cmap = None

    mask_np = mask[0].cpu().numpy() if mask is not None else None
    L = target_patches.shape[0]

    for i in range(L):
        x1, x2, y1, y2 = coords[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        # Original patch
        t_patch = target_patches[i].permute(1, 2, 0).numpy()
        if in_channels == 1:
            t_patch = t_patch.squeeze()
        resized_t = cv2.resize(t_patch, (w, h), interpolation=cv2.INTER_NEAREST)

        is_masked = mask_np is not None and mask_np[i] == 1

        if in_channels == 1:
            canvas_orig[y1:y2, x1:x2] = resized_t
            if is_masked:
                canvas_qdt[y1:y2, x1:x2] = 0.3  # masked region shown as dark gray
            else:
                canvas_qdt[y1:y2, x1:x2] = resized_t
            cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), 1.0, 1)
        else:
            canvas_orig[y1:y2, x1:x2] = resized_t
            if is_masked:
                canvas_qdt[y1:y2, x1:x2] = 0.3
            else:
                canvas_qdt[y1:y2, x1:x2] = resized_t
            cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1)

        # Reconstruction
        show_recon = True
        if mask_np is not None and mask_np[i] == 0:
            # Visible patch - show original
            if in_channels == 1:
                canvas_recon[y1:y2, x1:x2] = resized_t
            else:
                canvas_recon[y1:y2, x1:x2] = resized_t
            show_recon = False

        if show_recon:
            r_patch = recon_flat[i].view(in_channels, patch_size, patch_size).permute(1, 2, 0).numpy()
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

    # Plot
    ncols = 3
    titles = ["Original", "Quadtree Grid", "Reconstruction"]
    canvases = [canvas_orig, canvas_qdt, canvas_recon]

    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
    title_str = "HMAE Inference"
    if loss is not None:
        title_str += f" | Loss: {loss:.6f}"
    fig.suptitle(title_str, fontsize=14)

    for ax, canvas, title in zip(axes, canvases, titles):
        if in_channels == 1:
            ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(canvas)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    # Save figure to a BytesIO buffer instead of local disk
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Upload to S3
    s3.upload_fileobj(buf, output_bucket, s3_key)
    #logger.info(f"Saved visualization to S3: s3://{output_bucket}/{s3_key}")


def process_infer_masked_autoencode(self, s3, s3_keys, path_to_bucket, output_bucket, mode, config):
    model_cfg = config["model"]
    exts = config["input"].get("img_extensions", [".png", ".jpg"])
    model = self.model
    device = self.device
    use_bf16 = config["inference"].get("use_bf16", True)
    if device.type == "cpu":
        use_bf16 = False
 
    filtered_keys = [k for k in s3_keys if any(k.endswith(ext) for ext in exts)]
     
    saved_keys = []
    cls_embeddings = []
    checksums = []
    metadata_list = []
    for idx, s3_key in enumerate(filtered_keys):
        # logger.info(f"[{idx+1}/{len(image_paths)}] {s3_key}")

        data, checksum, metadata = preprocess_image(
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

        base = os.path.splitext(s3_keys[idx])[0]

        if mode == "reconstruct":
            pred = infer_reconstruct(model, data, device, config, use_bf16)
            saved_key= f"{base}_recon.png"
            visualize_reconstruction(data, pred, s3, output_bucket, saved_key, config)
            embedding = None

        elif mode == "mae":
            pred, mask, loss = infer_mae(model, data, device, config, use_bf16)
            # logger.info(f"  MAE loss: {loss:.6f}")
            saved_key= f"{base}_mae.png"
            visualize_reconstruction(data, pred, s3, output_bucket, saved_key, config, mask=mask, loss=loss)
            embedding = None

        elif mode == "features":
            features = infer_features(model, data, device, config, use_bf16)
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]
            # logger.info(f"  CLS token: {cls_token.shape}, Patch tokens: {patch_tokens.shape}")
            saved_key= f"{base}_features.pt"
            buffer = io.BytesIO()
            torch.save({"cls": cls_token, "patches": patch_tokens, "coords": data["coords"]}, buffer)
            buffer.seek(0)  # important: reset pointer to start
            s3.put_object(
                Bucket=output_bucket,
                Key=saved_key,
                Body=buffer.getvalue()
            )
            # logger.info(f"  Saved features: {feat_path}")
            embedding = cls_token.flatten()
        else:
            # logger.error(f"Unknown mode: {mode}")
            return
        
        saved_path = output_bucket+"/"+saved_key
        saved_keys.append(saved_path)

        
        cls_embeddings.append(embedding)

    return cls_embeddings, checksums, metadata_list, saved_keys


def infer_masked_autoencoder(modality, cfg_file):
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    mode_map = {
        "tomography_reconstruction": "reconstruct",
        "tomography_mae_reconstruction": "mae",
        "tomography_features_extraction": "features",
    }

    mode = mode_map[modality]

    def func(self, s3, s3_keys, path_to_bucket, output_bucket):
        return process_infer_masked_autoencode(self, s3, s3_keys, path_to_bucket, output_bucket, mode, config)
    return func

    












































############################### OLD Stuff ###################################
################################ Masked Autoencoder(  ###########################
def load_masked_autoencoder(config_path, checkpoint_path):
    """
    1. Load the trained MAE model from checkpoint
    """

    sys.path.append("..")
    from external.MAE import MAE

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model architecture
    model = MAE(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_channels'],
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads'],
        mask_ratio=config['model']['mask_ratio']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    model.num_patches = (config['model']['img_size'] // config['model']['patch_size']) ** 2
    model.enc_embed_dim = config['model']['encoder_embed_dim']
    model.img_size = config['model']['img_size']
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
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.390480], std=[0.29985])
            ])
    
    def __len__(self):
        return len(self.s3_keys)
    
    def __getitem__(self, idx):
        key = self.s3_keys[idx]
        
        # Get object from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)

        # Extract metadata and checksum
        checksum = obj['ETag'].strip('"')  # ETag is often quoted
        metadata = obj.get("Metadata", {})   # User metadata dictionary
       
        # extract raw data
        raw_bytes = obj['Body'].read()
        
        # Convert raw bytes to numpy array
        image = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(self.resolution, self.resolution, 1)
        
        # Convert to uint8 range like before
        image = (image / 65535.0 * 255.0).astype(np.uint8)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor, checksum, metadata


def build_s8d_val_dataloader(s3, s3_keys, path_to_bucket, batch_size, num_workers, resolution):
    """
    Builds and returns only the validation (test) DataLoader, 
    and lists all file paths included in the validation set.
    """

    from torch.utils.data._utils.collate import default_collate

    def custom_collate(batch):
        images, checksums, metadatas = zip(*batch)
    
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
    received_patches = visible_patches + 1   # +1 pour le CLS
    patch_dim = model.patch_size**2 * model.in_chans
    embed_dim = model.enc_embed_dim   
    
    # Preallocate array for all embeddings
    all_embeddings = torch.empty((num_samples, received_patches, embed_dim), dtype=torch.float32, device=device)
    all_reconstructions = torch.empty((num_samples, num_patches, patch_dim), dtype=torch.float32, device=device)

    model.eval()

    res_checksums = []
    res_metadata = []

    with torch.no_grad():
        sample_idx = 0  # keep track of current sample index
        for batch_idx, (images, checksums, metadatas) in enumerate(data_loader):
            print(f"Processing batch {batch_idx+1}/{len(data_loader)}")

            # Accumulate metadata and checksums from current batch
            res_checksums.extend(checksums)
            res_metadata.extend(metadatas)
    
            images = images.to(device)
            # embeddings = model.encode(images, convert_to_tensor=False)  
            _, recon_patches_flat, _, embeddings = model(images)
            batch_count = embeddings.shape[0]
            all_embeddings[sample_idx:sample_idx+batch_count] = embeddings
            all_reconstructions[sample_idx:sample_idx+batch_count] = recon_patches_flat
            sample_idx += batch_count
            
            
            
    if sample_idx > 0:
        # Concatenate all embeddings
        final_embeddings = all_embeddings.cpu().numpy()
        final_reconstructions = all_reconstructions.cpu().numpy() #shape [num_samples, num_patches, patch_dim]
        # embedding_vector = final_embeddings[:, 0, :].tolist()  # shape [batch_size, embedding_dim]
    else :
        return None, None, None, None, None, None


    cls_embeddings = final_embeddings[:, 0, :]  # shape: [num_samples, embed_dim]

    patch_shape = (model.patch_size, model.patch_size, model.in_chans)

    return final_embeddings, cls_embeddings, final_reconstructions, patch_shape, res_checksums, res_metadata


  
def save_one_reconstruction_in_one_image(patches,
                                         s3, s3_key, path_to_bucket,
                                         patch_shape,   # (patch, patch, C)
                                         resolution,
                                         overwrite=True):
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
    P, _, C = patch_shape     # patch height, patch width, channels
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
            f"Inconsistent resolution: expected {grid_size * P}, "
            f"got {resolution}."
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

    return path_to_bucket+"/"+s3_key


def save_all_reconstructions_in_one_image(all_reconstructions,
                                          patch_shape, s3, s3_keys, path_to_bucket,resolution=8192):
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
    assert num_samples == len(s3_keys), \
        "Mismatch: number of reconstructions must match number of input files"

    for idx in range(num_samples):
        patches = all_reconstructions[idx]     # shape [num_patches, patch_dim]

        base = os.path.splitext(s3_keys[idx])[0]
        save_key= f"{base}_patches.tiff"

        saved_path = save_one_reconstruction_in_one_image(
            patches, s3, save_key, path_to_bucket, patch_shape, resolution, overwrite=True
        )
        # print(f"Output saved to: {saved_path}")
        
        saved_keys.append(saved_path)

    return saved_keys    

def compute_embeddings_masked_autoencoder(self, s3, s3_keys, path_to_bucket, output_bucket):
     # 2. Load data
    data_loader = build_s8d_val_dataloader(
        s3,
        s3_keys,
        path_to_bucket,
        batch_size=8,  # Adjust based on your GPU memory
        num_workers=0, # this could be larger but requires a s3 endpoint per worker and hnce to pass the s3 configuration file and not the s3_connection
        resolution=self.model.img_size
    )
    
    # 3 & 4. Extract embeddings

    _, cls_embeddings, all_reconstructions, patch_shape, checksums,  metadata = extract_embeddings(self.model, data_loader, self.device)
    # _, cls_embeddings, checksums,  metadata = extract_embeddings(self.model, data_loader, self.device)
    
    saved_keys = None
    if all_reconstructions is not None:
        saved_keys = save_all_reconstructions_in_one_image(all_reconstructions, patch_shape, s3, s3_keys, output_bucket, resolution=self.model.img_size)
        # saved_files = save_all_reconstructions(all_reconstructions, "./reconstruction", file_list, patch_shape
        

    
    return cls_embeddings, checksums, metadata, saved_keys
    
