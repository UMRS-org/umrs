import sys
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
    num_patches = model.num_patches + 1 # +1 for CLS token 
    embed_dim = model.enc_embed_dim   
    
    # Preallocate array for all embeddings
    all_embeddings = np.empty((num_samples, num_patches, embed_dim), dtype=np.float32)

    res_checksums = []
    res_metadata = []

    model.eval()
    with torch.no_grad():
        sample_idx = 0  # keep track of current sample index
        for batch_idx, (images, checksums, metadatas) in enumerate(data_loader):

             # Accumulate metadata and checksums from current batch
            res_checksums.extend(checksums)
            res_metadata.extend(metadatas)

            images = images.to(device)
            embeddings = model.encode(images, convert_to_tensor=False)  
            batch_count = embeddings.shape[0]
            all_embeddings[sample_idx:sample_idx+batch_count] = embeddings
            sample_idx += batch_count
            
            print(f"Processed batch {batch_idx}/{len(data_loader)}")
    
    if(len(all_embeddings)):
        # Concatenate all embeddings
        final_embeddings= np.array(all_embeddings.cpu()) if device == "cuda" else all_embeddings
                
    cls_embeddings = final_embeddings[:, 0, :]  # shape: [num_samples, embed_dim]

    return final_embeddings, cls_embeddings, res_checksums, res_metadata

def compute_embeddings_masked_autoencoder(self, s3, s3_keys, path_to_bucket):
     # 2. Load data
    data_loader = build_s8d_val_dataloader(
        s3,
        s3_keys,
        path_to_bucket,
        batch_size=32,  # Adjust based on your GPU memory
        num_workers=0, # this could be larger but requires a s3 endpoint per worker and hnce to pass the s3 configuration file and not the s3_connection
        resolution=self.model.img_size
    )
    
    # 3 & 4. Extract embeddings
    _, cls_embeddings, checksums,  metadata = extract_embeddings(self.model, data_loader, self.device)
    return cls_embeddings, checksums, metadata
    
