"""
Embedding models for UMRS.

Provides model implementations for computing embeddings:
- SentenceTransformers (text)
- ResNet50 (image)
- TransformerEncoder (custom)
- MaskedAutoencoder (custom)
"""

__all__ = []

try:
    from .sentence_transformers import (
        load_sentence_transformers_model as load_sentence_transformer,  # noqa: F401
    )
    __all__.append("load_sentence_transformer")
except ImportError:
    pass

try:
    from .resnet_50 import load_resnet50 as load_resnet_50  # noqa: F401
    __all__.append("load_resnet_50")
except ImportError:
    pass
