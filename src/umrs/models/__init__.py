"""
Embedding models for UMRS.

Provides model implementations for computing embeddings:
- SentenceTransformers (text)
- ResNet50 (image)
- TransformerEncoder (custom)
- MaskedAutoencoder (custom)
"""

from .sentence_transformers import load_sentence_transformers_model as load_sentence_transformer
from .resnet_50 import load_resnet50 as load_resnet_50

__all__ = [
    "load_sentence_transformer",
    "load_resnet_50",
]
