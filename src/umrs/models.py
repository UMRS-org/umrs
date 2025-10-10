import torch
import spacy
from umrs.utils import flatten_text_chunks, add_embeddings_to_chunks_dict

from typing import Optional
from torch.utils.data import Dataset, DataLoader

import io
import numpy as np
        

class EmbeddingModel:
    def __init__(self, model_name: str, embedding_vector_size: int, max_tokens: Optional[int], model, input_type: str, compute_embeddings, modality, spacy_model, tokenizer):
        self.model_name = model_name
        self.embedding_vector_size = int(embedding_vector_size)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.input_type = input_type
        self.model = model
        self.modality = modality
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.compute_embeddings = compute_embeddings.__get__(self)  # Bind the method to the instance
        if spacy_model:
            self.nlp = spacy.load(spacy_model, disable=["ner", "tagger", "parser"])
            self.nlp.add_pipe("sentencizer")  # simple sentence seperator
        else:
            self.nlp = None


            
################################ Sentence Transformer ###########################
#  Sentence Transformer Loader
def load_sentence_transformers_model(model_name):
    from sentence_transformers import SentenceTransformer
    print(f"\t\t\t--> Loading model '{model_name}'...")
    try :
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print("\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face...")
        model = SentenceTransformer(model_name)
        print("\t\t\t--> Downloaded and cached model.")
    return model;


#  Compute Embeddings for Text
def compute_embeddings_text(self, texts):
    flat_texts, chunks_dict = flatten_text_chunks(texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
    embeddings = self.model.encode(flat_texts, convert_to_tensor=(self.device == "cuda"))
    embeddings = embeddings.cpu().tolist() if self.device == "cuda" else embeddings
    return add_embeddings_to_chunks_dict(embeddings, chunks_dict)

#  Compute Embeddings for Images
def compute_embeddings_img(self, images):
    # Ensure images are in RGB format
    images = [img.convert("RGB") for img in images]

    # Compute embeddings
    with torch.no_grad():
        embeddings = self.model.encode(images, convert_to_tensor=(self.device == "cuda"))

    return embeddings.cpu().tolist() if self.device == "cuda" else embeddings


def sentence_transformers(modality):
    if modality == "text":
        compute_embeddings_func = compute_embeddings_text
        input_type = "text"
    elif modality == "image":
        compute_embeddings_func = compute_embeddings_img
        input_type = "image"
    else:
        raise ValueError(f"Unsupported modality '{modality}' for SentenceTransformer model")
    return compute_embeddings_func, input_type

################################ ResNet-50 ###########################

def load_resnet50(model_path=None):
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    """Load ResNet-50 model, handling DataParallel checkpoints if needed.
    Uses updated PyTorch conventions."""
    # Use weights if no model_path provided
    if model_path is None:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if model_path:
        # Use weights_only=True for safer deserialization
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        # Handle 'module.' prefix in state_dict keys from DataParallel
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Load state dict with relaxed strictness
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()  # Set to evaluation mode
    return model


def preprocess_images_batch(images):
    from torchvision import transforms
    """Preprocess a list of PIL Images into a batch tensor for ResNet-50."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply transform to each image and stack them
    image_tensors = [transform(image) for image in images]
    batch_tensor = torch.stack(image_tensors)  # Shape: (batch_size, 3, 224, 224)

    return batch_tensor

def compute_embeddings_batch(model, image_batch_tensor, device='cpu'):
    """Compute embeddings for a batch of images using ResNet-50."""
    image_batch_tensor = image_batch_tensor.to(device)

    with torch.no_grad():
        embeddings = model(image_batch_tensor)  # Batch forward pass

    return embeddings.cpu()  # Convert to NumPy array if needed


def compute_embeddings_resnet_50(self, images):
 # Preprocess all images into a single batch tensor
    image_batch_tensor = preprocess_images_batch(images)

    # Compute embeddings for the entire batch
    embeddings = compute_embeddings_batch(self.model, image_batch_tensor, device=self.device)

    return embeddings


################################ TransformerEncoder  ###########################
def load_transformer_encoder(model_name):
    from transformers import AutoTokenizer, AutoModel
    from transformers.utils import logging

    logging.set_verbosity_error()  # avoid verbose warning
    print(f"\t\t\t--> Loading model '{model_name}'...")

    try:
        # 1. Try to load locally
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        print("\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("\t\t\t--> Downloaded and cached model.")

    return model, tokenizer


def compute_embeddings_transformer_enncoder(self, texts):
     # Step 1: Chunk the texts respecting token limits
    flat_texts, chunks_dict = flatten_text_chunks(
        texts,
        self.nlp,
        tokenizer=self.tokenizer,
        max_tokens=self.max_tokens
    )

    # Step 2: Tokenize the chunks
    inputs = self.tokenizer(flat_texts, return_tensors="pt", truncation=True, padding=True)

    # Step 3: Send inputs to correct device
    inputs = {key: value.to(self.device) for key, value in inputs.items()}

    # Step 4: Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    # Step 5: Bring embeddings back to CPU if needed
    embeddings = embeddings.cpu().tolist() if self.device == "cuda" else embeddings

    # Step 6: Reinsert embeddings into chunk metadata
    return add_embeddings_to_chunks_dict(embeddings, chunks_dict)




################################ Masked Autoencoder(  ###########################
def load_masked_autoencoder(config_path, checkpoint_path):
    """
    1. Load the trained MAE model from checkpoint
    """
    import sys
    import yaml
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

    from torch.utils.data import DataLoader

    def __init__(self, s3_client, resolution, s3_key_list, bucket_name, transform=None):
        import torchvision.transforms as transforms
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
    

################################ Models loading  ###########################
#  Load the Model Dynamically Based on Type
def load_model(model_cfg, modality):
    model_name = model_cfg["model_name"]
    vector_size = model_cfg["vector_size"]
    model_type = model_cfg["model_type"]
    
    tokenizer = None
    max_tokens = None
    spacy_model = model_cfg.get("spacy_model", None)
    cfg_file = model_cfg.get("model_cfg_file", None)
    checkpoint_path = model_cfg.get("checkpoint_path", None)


    if model_type == "SentenceTransformer":
        model = load_sentence_transformers_model(model_name)
        tokenizer = model._first_module().tokenizer
        max_tokens = getattr(tokenizer, "model_max_length", None)
        compute_embeddings_func, input_type = sentence_transformers(modality);
    elif model_type == "Resnet-50":
        model = load_resnet50(checkpoint_path)
        compute_embeddings_func = compute_embeddings_resnet_50
        input_type = "image"
    elif model_type == "TransformerEncoder":
        model, tokenizer = load_transformer_encoder(model_name)
        max_tokens = tokenizer.model_max_length
        compute_embeddings_func = compute_embeddings_transformer_enncoder
        input_type = "text"
    elif model_type == "MAE" :
        model = load_masked_autoencoder(cfg_file, checkpoint_path)
        compute_embeddings_func = compute_embeddings_masked_autoencoder
        input_type = "raw_image"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # Create and return the EmbeddingModel object
    return EmbeddingModel(model_name, vector_size, max_tokens, model, input_type, compute_embeddings_func, modality, spacy_model, tokenizer)

