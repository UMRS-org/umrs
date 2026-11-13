"""ResNet-50 preprocessing accepts any PIL mode (port of upstream cb0a5ca3; plan U1 A1).

The transform pipeline normalises with 3-channel mean/std, so a grayscale
("L") or RGBA image used to crash ``preprocess_images_batch``. No model
weights are loaded here: only the preprocessing is exercised.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from umrs.inference.resnet_50 import preprocess_images_batch


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB", "P", "LA"])
def test_preprocess_accepts_mode(mode):
    image = Image.new(mode, (300, 260))
    batch = preprocess_images_batch([image])
    assert tuple(batch.shape) == (1, 3, 224, 224)


def test_preprocess_mixed_modes_stack_into_one_batch():
    images = [
        Image.new("L", (64, 64)),
        Image.new("RGBA", (80, 40)),
        Image.new("RGB", (32, 32)),
    ]
    batch = preprocess_images_batch(images)
    assert tuple(batch.shape) == (3, 3, 224, 224)
    assert torch.isfinite(batch).all()
