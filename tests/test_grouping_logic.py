"""Real grouping module (not the stub): edge cases, preprocessing, batching, device.

Needs torch, torchvision and scikit-learn (requirements-dev.txt); skipped otherwise.
No pretrained weights are downloaded: the embedding model is replaced by a tiny fake.
"""

import importlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")


class FakeEmbedder(torch.nn.Module):
    """Mimics ResNet18 without the classifier: (B, 3, H, W) -> (B, 512, 1, 1).

    Each embedding is the per-channel mean repeated, so crops with the same
    colour get identical embeddings. Records the batch sizes it was called with.
    """

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        means = x.mean(dim=(2, 3))  # (B, 3)
        return means.repeat(1, 171)[:, :512].reshape(-1, 512, 1, 1)


@pytest.fixture
def grouping(monkeypatch):
    """Import the real grouping module, bypassing the conftest stub, with a fake model."""
    monkeypatch.delitem(sys.modules, "grouping_service.grouping", raising=False)
    mod = importlib.import_module("grouping_service.grouping")
    fake = FakeEmbedder()
    monkeypatch.setattr(mod, "_model", fake)
    mod.fake = fake
    return mod


def crop(bgr=(0, 0, 255), size=(30, 20)):
    """Solid-colour BGR uint8 crop, as produced by OpenCV."""
    return np.full((size[1], size[0], 3), bgr, dtype=np.uint8)


# ---------------------------------------------------------------- edge cases
def test_zero_detections(grouping):
    assert grouping.group_products([], [], (100, 100, 3)) == []
    assert grouping.fake.batch_sizes == []  # model not called


def test_single_detection_does_not_crash(grouping):
    assert grouping.group_products([crop()], [[0, 0, 10, 10]], (100, 100, 3)) == [0]
    assert grouping.fake.batch_sizes == []  # clustering and model skipped


def test_two_detections(grouping):
    labels = grouping.group_products([crop(), crop()], [[0, 0, 10, 10], [12, 0, 22, 10]], (100, 100, 3))
    assert len(labels) == 2 and all(isinstance(x, int) for x in labels)
    assert labels[0] == labels[1]  # identical crops, neighbouring boxes


def test_mismatched_lengths_rejected(grouping):
    with pytest.raises(ValueError):
        grouping.group_products([crop()], [], (100, 100, 3))


# ---------------------------------------------------------------- preprocessing
def test_preprocess_shape_and_dtype(grouping):
    batch = grouping.preprocess([crop(size=(30, 20)), crop(size=(224, 224)), crop(size=(5, 90))])
    assert batch.shape == (3, 3, grouping.IMAGE_SIZE, grouping.IMAGE_SIZE)
    assert batch.dtype == torch.float32


def test_preprocess_applies_imagenet_normalisation(grouping):
    mean = torch.tensor(grouping.IMAGENET_MEAN)
    std = torch.tensor(grouping.IMAGENET_STD)
    white = grouping.preprocess([crop((255, 255, 255))])[0].mean(dim=(1, 2))
    black = grouping.preprocess([crop((0, 0, 0))])[0].mean(dim=(1, 2))
    assert torch.allclose(white, (1 - mean) / std, atol=1e-4)
    assert torch.allclose(black, -mean / std, atol=1e-4)


def test_preprocess_converts_bgr_to_rgb(grouping):
    # Pure blue in OpenCV's BGR order must end up in channel 2 (B) of the RGB tensor.
    channels = grouping.preprocess([crop((255, 0, 0))])[0].mean(dim=(1, 2))
    assert channels.argmax().item() == 2


# ---------------------------------------------------------------- batching and device
def test_embeddings_are_batched(grouping):
    emb = grouping.get_embeddings([crop()] * 70, batch_size=32)
    assert emb.shape == (70, 512)
    assert grouping.fake.batch_sizes == [32, 32, 6]


def test_empty_embeddings_shape(grouping):
    assert grouping.get_embeddings([]).shape == (0, 512)


def test_cuda_request_falls_back_to_cpu(grouping, monkeypatch):
    monkeypatch.setattr(grouping.torch.cuda, "is_available", lambda: False)
    assert grouping.resolve_device("cuda").type == "cpu"


def test_load_model_uses_weights_enum(grouping, monkeypatch):
    calls = {}
    real_resnet18 = grouping.models.resnet18

    def fake_resnet18(**kwargs):
        calls.update(kwargs)
        return real_resnet18(weights=None)  # same architecture, no download

    monkeypatch.setattr(grouping.models, "resnet18", fake_resnet18)
    monkeypatch.setattr(grouping, "_model", None)
    model = grouping.load_model()
    assert calls == {"weights": grouping.models.ResNet18_Weights.IMAGENET1K_V1}
    assert model(torch.zeros(2, 3, 224, 224)).flatten(1).shape == (2, 512)
    assert grouping.load_model() is model  # cached, built once
