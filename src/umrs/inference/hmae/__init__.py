"""HMAE (Hierarchical Masked Auto-Encoder) model package.

Ported from https://github.com/zhangenzhi/hmae-infer
An MAE based on adaptive quadtree decomposition, supporting reconstruction,
feature extraction, and visualization. Uses RoPE positional encoding and
custom ViT architecture (HMAEVIT).
"""

from .hvit import hvit_b, hvit_l, hvit_xl

_model_factory = {
    "hvit_b": hvit_b,
    "hvit_l": hvit_l,
    "hvit_xl": hvit_xl,
}


def create_model(model_type, **kwargs):
    """Factory function to create HMAE model variants.

    Args:
        model_type: One of "hvit_b", "hvit_l", "hvit_xl".
        **kwargs: Passed through to the model constructor (img_size, patch_size,
                  in_channels, mask_ratio, norm_pix_loss, etc.).
    """
    if model_type not in _model_factory:
        available = ", ".join(_model_factory.keys())
        raise ValueError(f"Unsupported HMAE model type: {model_type}. Available: {available}")
    return _model_factory[model_type](**kwargs)
