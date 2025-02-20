#!/usr/bin/pythin 3

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_resnet50(model_path=None):
    """Load ResNet-50 model, either pretrained or from a checkpoint."""
    model = models.resnet50(pretrained=(model_path is None))
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image_path):
    """Preprocess image for ResNet-50."""
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def compute_embedding(model, image_tensor, device='cpu'):
    """Compute image embedding using ResNet-50."""
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.squeeze().cpu().numpy()

if __name__ == "__main__":
    # Load the model (set model_path to your downloaded checkpoint if needed)
    model = load_resnet50(model_path=nvidia_resnet50_200821.pth.tar)  # Or provide path: model_path="resnet50.pth"

    # Load and preprocess image
    image_path = "your_image.jpg"  # Replace with your image path
    image_tensor = preprocess_image(image_path)

    # Compute embeddings
    embedding = compute_embedding(model, image_tensor)
    
    print("Embedding shape:", embedding.shape)
    print("Embedding vector:", embedding)
