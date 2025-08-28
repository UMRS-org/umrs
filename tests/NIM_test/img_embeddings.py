#!/usr/bin/python3
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_resnet50(model_path=None):
    """Load ResNet-50 model, handling DataParallel checkpoints if needed."""
    model = models.resnet50(pretrained=(model_path is None))

    if model_path:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Handle 'module.' prefix in state_dict keys
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
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


If __name__ == "__main__":
    # Load the model (set model_path to your downloaded checkpoint if needed)
    model = load_resnet50(model_path="models/nvidia_resnet50_200821.pth.tar")  # Or provide path: model_path="resnet50.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")
    
    # Load and preprocess image
    image_path = "../data/img/Cat_1.jpeg"  # Replace with your image path
    image_tensor = preprocess_image(image_path)

    # Compute embeddings
    embedding = compute_embedding(model, image_tensor)
    
    print("Embedding len:", len(embedding))
    print("Embedding shape:", embedding.shape)
    # print("Embedding vector:", embedding)
