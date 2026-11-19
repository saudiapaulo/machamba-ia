"""Testes do serviço de visão: labels, pré-processamento, resumo e decisão."""

from io import BytesIO

import pytest
import torch
from PIL import Image

from app.vision.labels import LabelSet
from app.vision.prediction import summarize
from app.vision.preprocess import build_transform, prepare_image, preprocess
from app.vision.service import Level
from tests.conftest import FakeClassifier

# Ordem das classes falsas:
# 0 Maize healthy, 1 Maize leaf blight, 2 Maize leaf spot,
# 3 Tomato healthy, 4 Tomato verticillium wilt, 5 Cassava mosaic


# ---------- labels.json ----------

def test_name_fixes_change_display_name_only(labels: LabelSet) -> None:
    """'verticulium' é corrigido no nome, mas a pasta original mantém-se."""
    wilt = labels.classes[4]
    assert wilt.name == "Tomato - verticillium wilt"
    assert wilt.disease == "verticillium wilt"
    assert wilt.folder == "Tomato/verticulium wilt"


def test_crops_and_portuguese_names(labels: LabelSet) -> None:
    assert labels.crops == ["Maize", "Tomato", "Cassava"]
    assert labels.crop_pt("Maize") == "milho"


# ---------- pré-processamento ----------

def test_large_image_is_reduced_to_short_side_256() -> None:
    prepared = prepare_image(Image.new("RGB", (1000, 500)), resize_size=256)
    assert prepared.size == (512, 256)


def test_small_image_is_not_reduced() -> None:
    prepared = prepare_image(Image.new("RGB", (100, 80)), resize_size=256)
    assert prepared.size == (100, 80)


def test_exif_orientation_is_applied() -> None:
    """Uma foto de telemóvel 'deitada' (EXIF orientação 6) é rodada antes de tudo."""
    exif = Image.Exif()
    exif[0x0112] = 6  # 0x0112 = Orientation; 6 = rodar 90°
    buffer = BytesIO()
    Image.new("RGB", (300, 200)).save(buffer, format="JPEG", exif=exif)
    prepared = prepare_image(Image.open(BytesIO(buffer.getvalue())), resize_size=256)
    assert prepared.size == (200, 300)


def test_grayscale_png_with_alpha_becomes_rgb() -> None:
    prepared = prepare_image(Image.new("LA", (300, 300)), resize_size=256)
    assert prepared.mode == "RGB"


def test_preprocess_output_shape_and_normalization(labels: LabelSet) -> None:
    """Uma imagem branca dá (1 - média) / desvio em cada canal, com forma (1, 3, 224, 224)."""
    tensor = preprocess(Image.new("RGB", (640, 480), (255, 255, 255)), labels, build_transform(labels))
    assert tensor.shape == (1, 3, 224, 224)
    expected = torch.tensor([(1 - m) / s for m, s in zip(labels.mean, labels.std)])
    assert torch.allclose(tensor[0, :, 0, 0], expected, atol=1e-4)


# ---------- resumo das probabilidades ----------

def test_summarize_top3_and_crop_confidence(labels: LabelSet) -> None:
    prediction = summarize([0.05, 0.45, 0.35, 0.05, 0.05, 0.05], labels)
    assert prediction.top.info.disease == "leaf blight"
    assert [s.info.index for s in prediction.top3] == [1, 2, 0]
    assert prediction.crop == "Maize"
    assert prediction.crop_confidence == pytest.approx(0.85)
    assert sum(prediction.crop_confidences.values()) == pytest.approx(1.0)


def test_summarize_rejects_wrong_number_of_classes(labels: LabelSet) -> None:
    with pytest.raises(ValueError):
        summarize([0.5, 0.5], labels)


# ---------- decisão (limites de confiança) ----------

@pytest.mark.parametrize(
    ("probabilities", "level", "forward", "new_photo"),
    [
        # doença ≥ 0.80 → alta
        ([0.0, 0.90, 0.04, 0.02, 0.02, 0.02], Level.HIGH, False, False),
        ([0.0, 0.80, 0.05, 0.05, 0.05, 0.05], Level.HIGH, False, False),        # fronteira exacta
        # 0.50 ≤ doença < 0.80 → média
        ([0.0, 0.79, 0.06, 0.05, 0.05, 0.05], Level.MEDIUM, False, True),
        ([0.0, 0.50, 0.20, 0.10, 0.10, 0.10], Level.MEDIUM, False, True),       # fronteira exacta
        # doença < 0.50, mas cultura ≥ 0.80 → só cultura
        ([0.0, 0.40, 0.40, 0.10, 0.05, 0.05], Level.CROP_ONLY, True, True),     # milho = 0.80
        # doença < 0.50 e 0.50 ≤ cultura < 0.80 → baixa
        ([0.0, 0.30, 0.20, 0.20, 0.20, 0.10], Level.LOW, True, False),          # milho = 0.50
        # cultura < 0.50 → não suportado
        ([0.10, 0.20, 0.10, 0.20, 0.20, 0.20], Level.NOT_SUPPORTED, True, False),
    ],
)
def test_decision_levels(make_service, probabilities, level, forward, new_photo) -> None:
    diagnosis = make_service(probabilities).diagnose(Image.new("RGB", (300, 300)))
    assert diagnosis.level == level
    assert diagnosis.forward_to_technician is forward
    assert diagnosis.ask_new_photo is new_photo
    assert diagnosis.message


def test_high_confidence_message_names_crop_and_disease(make_service) -> None:
    diagnosis = make_service([0.0, 0.90, 0.04, 0.02, 0.02, 0.02]).diagnose(Image.new("RGB", (300, 300)))
    assert diagnosis.crop_pt == "milho"
    assert diagnosis.disease == "leaf blight"
    assert "milho" in diagnosis.message


def test_healthy_plant_message(make_service) -> None:
    diagnosis = make_service([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]).diagnose(Image.new("RGB", (300, 300)))
    assert diagnosis.is_healthy
    assert "saudável" in diagnosis.message


def test_crop_only_says_crop_but_not_disease(make_service) -> None:
    diagnosis = make_service([0.0, 0.40, 0.40, 0.10, 0.05, 0.05]).diagnose(Image.new("RGB", (300, 300)))
    assert diagnosis.crop_pt == "milho"
    assert diagnosis.disease is None
    assert "milho" in diagnosis.message


def test_not_supported_lists_known_crops(make_service) -> None:
    diagnosis = make_service([0.10, 0.20, 0.10, 0.20, 0.20, 0.20]).diagnose(Image.new("RGB", (300, 300)))
    assert diagnosis.crop is None
    assert "folha de milho, tomate ou mandioca" in diagnosis.message
    assert "planta" not in diagnosis.message


def test_classifier_receives_preprocessed_batch(labels: LabelSet) -> None:
    from app.vision.gate import ModelConfidenceGate
    from app.vision.service import VisionService

    classifier = FakeClassifier([1 / 6] * 6)
    VisionService(classifier, labels, ModelConfidenceGate(0.5), 0.8, 0.5).predict(Image.new("RGB", (800, 600)))
    assert classifier.last_batch_shape == (1, 3, 224, 224)
