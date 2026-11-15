"""ResNet-50 preprocessing accepts any PIL mode (port of upstream cb0a5ca3; plan U1 A1).

The transform pipeline normalises with 3-channel mean/std, so a grayscale
("L") or RGBA image used to crash ``preprocess_images_batch``. No model
weights are loaded here: only the preprocessing is exercised.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from umrs.inference import resnet_50  # noqa: E402
from umrs.inference.resnet_50 import preprocess_images_batch  # noqa: E402


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


class _Identity(torch.nn.Module):
    def forward(self, batch):
        return batch.flatten(1)[:, :4]


def test_resnet_logs_preprocess_and_forward_time_at_info(caplog):
    """Upstream 30ad54f4's ``[resnet50] n= pil= forward=`` line, at INFO
    so a server at its default level shows it (review U1)."""

    class _Self:
        model = _Identity()
        device = "cpu"

    with caplog.at_level(logging.INFO, logger=resnet_50.logger.name):
        out = resnet_50.compute_embeddings_resnet_50(_Self(), [Image.new("RGB", (32, 32))] * 2)
    assert tuple(out.shape) == (2, 4)
    lines = [r for r in caplog.records if r.getMessage().startswith("[resnet50]")]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert "n=2" in lines[0].getMessage()
    assert set(lines[0].umrs_resnet_timing) == {"n_images", "preprocess_s", "forward_s"}
