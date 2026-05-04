import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights

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
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        # Handle 'module.' prefix in state_dict keys from DataParallel
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Load state dict with relaxed strictness
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()  # Set to evaluation mode
    _model_cache[model_path] = model
    return model


def preprocess_images_batch(images):
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
