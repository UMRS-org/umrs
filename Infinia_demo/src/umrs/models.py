import torch
import torchvision


class EmbeddingModel:
    def __init__(self, model_name: str, embedding_vector_size: int, model, input_type: str, compute_embeddings, tokenizer = None):
        self.model_name = model_name
        self.embedding_vector_size = int(embedding_vector_size)
        self.input_type = input_type
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.compute_embeddings = compute_embeddings.__get__(self)  # Bind the method to the instance


################################ Setence Transformer ###########################
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
    return self.model.encode(texts, convert_to_tensor=False).tolist()

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
        raise ValueError(f"Unsupported modality \"{model_type}\" for model type: {model_type}")
    return compute_embeddings_func, input_type

################################ ResNet-50 ###########################

def load_resnet50(model_path=None):
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    """
    Load ResNet-50 model, handling DataParallel checkpoints if needed.
    Uses updated PyTorch conventions.
    """
    # Use weights if no model_path provided
    if model_path is None:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if model_path:
        # Use weights_only=True for safer deserialization
        print(f"\t\t\t--> Loading ResNet-50 model from {model_path}...")
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
    import os

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
    # Tokenize the list of texts with appropriate padding
    inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(self.device) for key, value in inputs.items()}

    # Compute embedding
    with torch.no_grad():
        outputs = self.model(**inputs)

    # Mean pooling over token embeddings for each text
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Move embeddings back to CPU if needed
    return embeddings.cpu() if self.device == "cuda" else embeddings



################################ Models loading  ###########################
#  Load the Model Dynamically Based on Type
def load_model(model_cfg, modality):
    model_name = model_cfg["model_name"]
    vector_size = model_cfg["vector_size"]
    model_type = model_cfg["model_type"]
    tokenizer = None

    # dehbug
    print(f"\t\t\t--> Loading model '{model_name}' of type '{model_type}' with vector size {vector_size}...")
    # path to the model
    if "checkpoint_path" in model_cfg:
        print(f"\t\t\t--> Using checkpoint path: {model_cfg['checkpoint_path']}")

    if model_type == "SentenceTransformer":
        model = load_sentence_transformers_model(model_name)
        compute_embeddings_func, input_type = sentence_transformers(modality);
    elif model_type == "Resnet-50":
        checkpoint_path = model_cfg["checkpoint_path"] if model_cfg["checkpoint_path"] else None
        model = load_resnet50(checkpoint_path)
        compute_embeddings_func = compute_embeddings_resnet_50
        input_type = "image"
    elif model_type == "TransformerEncoder":
        model, tokenizer = load_transformer_encoder(model_name)
        compute_embeddings_func = compute_embeddings_transformer_enncoder
        input_type = "text"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # Create and return the EmbeddingModel object
    return EmbeddingModel(model_name, vector_size, model, input_type, compute_embeddings_func, tokenizer = tokenizer)

