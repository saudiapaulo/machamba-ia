"""Peças partilhadas pelos testes: labels falsos e um classificador falso."""

import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.vision.gate import ModelConfidenceGate
from app.vision.labels import LabelSet, load_labels
from app.vision.service import VisionService

# 3 culturas e 6 classes: chega para testar todos os ramos da decisão.
# "verticulium" está mal escrito de propósito, como no labels.json real.
FAKE_CLASSES = [
    ("Maize", "milho", "healthy"),
    ("Maize", "milho", "leaf blight"),
    ("Maize", "milho", "leaf spot"),
    ("Tomato", "tomate", "healthy"),
    ("Tomato", "tomate", "verticulium wilt"),
    ("Cassava", "mandioca", "mosaic"),
]


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    """Escreve um labels.json falso, com a mesma estrutura do gerado pelo notebook."""
    data = {
        "model": "efficientnet_b0",
        "num_classes": len(FAKE_CLASSES),
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "classes": {
            str(i): {"class": f"{crop} - {disease}", "crop": crop, "crop_pt": crop_pt,
                     "disease": disease, "folder": f"{crop}/{disease}"}
            for i, (crop, crop_pt, disease) in enumerate(FAKE_CLASSES)
        },
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def labels(labels_file: Path) -> LabelSet:
    return load_labels(labels_file)


class FakeClassifier:
    """Classificador falso: devolve sempre as probabilidades que lhe dermos."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.last_batch_shape: tuple[int, ...] | None = None

    def predict_proba(self, batch: torch.Tensor) -> list[float]:
        self.last_batch_shape = tuple(batch.shape)
        return self.probabilities


@pytest.fixture
def make_service(labels: LabelSet):
    """Fábrica: make_service([probabilidades]) → VisionService com o classificador falso."""
    def factory(probabilities: list[float]) -> VisionService:
        return VisionService(FakeClassifier(probabilities), labels, ModelConfidenceGate(0.50),
                             high_confidence=0.80, low_confidence=0.50)
    return factory


def image_bytes(size: tuple[int, int] = (400, 300), fmt: str = "PNG") -> bytes:
    """Gera uma imagem verde em memória."""
    buffer = BytesIO()
    Image.new("RGB", size, (40, 140, 40)).save(buffer, format=fmt)
    return buffer.getvalue()
