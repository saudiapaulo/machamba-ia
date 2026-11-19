"""Serviço de diagnóstico: foto → previsão → decisão com os limites de confiança.

Não depende do FastAPI nem do Telegram: a API e o bot chamam este serviço.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from PIL import Image

from app.config import Settings
from app.vision import messages
from app.vision.gate import CropGate, ModelConfidenceGate
from app.vision.labels import LabelSet, load_labels
from app.vision.model import Classifier, TimmClassifier
from app.vision.prediction import Prediction, summarize
from app.vision.preprocess import build_transform, preprocess

logger = logging.getLogger(__name__)


class Level(str, Enum):
    """Nível de confiança do diagnóstico."""

    HIGH = "alta"                  # responde normalmente
    MEDIUM = "media"               # responde com cautela e pede outra foto
    CROP_ONLY = "so_cultura"       # sabe a cultura, não a doença: pede outra foto + técnico
    LOW = "baixa"                  # não diagnostica: técnico
    NOT_SUPPORTED = "nao_suportado"  # não parece uma das culturas suportadas: técnico


@dataclass(frozen=True)
class Diagnosis:
    """A decisão final, pronta a mostrar ao agricultor."""

    level: Level
    prediction: Prediction
    crop: str | None          # só preenchido quando a cultura é fiável
    crop_pt: str | None
    disease: str | None       # só preenchido quando a doença é fiável (alta ou média)
    disease_pt: str | None
    is_healthy: bool
    forward_to_technician: bool
    ask_new_photo: bool
    message: str


def decide(prediction: Prediction, supported: bool, high: float, low: float, crops_pt: list[str]) -> Diagnosis:
    """Aplica as regras do CLAUDE.md (alta ≥ high, média ≥ low, baixa < low)."""
    top = prediction.top
    crop_pt = prediction.crop_pt
    healthy = top.info.is_healthy

    if not supported:
        return Diagnosis(Level.NOT_SUPPORTED, prediction, None, None, None, None, False,
                         forward_to_technician=True, ask_new_photo=False,
                         message=messages.msg_not_supported(crops_pt))

    if prediction.confidence >= high:
        message = messages.msg_high_healthy(crop_pt) if healthy else messages.msg_high(crop_pt, top.disease_pt)
        return Diagnosis(Level.HIGH, prediction, prediction.crop, crop_pt, top.info.disease, top.disease_pt,
                         healthy, forward_to_technician=False, ask_new_photo=False, message=message)

    if prediction.confidence >= low:
        message = messages.msg_medium_healthy(crop_pt) if healthy else messages.msg_medium(crop_pt, top.disease_pt)
        return Diagnosis(Level.MEDIUM, prediction, prediction.crop, crop_pt, top.info.disease, top.disease_pt,
                         healthy, forward_to_technician=False, ask_new_photo=True, message=message)

    if prediction.crop_confidence >= high:
        return Diagnosis(Level.CROP_ONLY, prediction, prediction.crop, crop_pt, None, None, False,
                         forward_to_technician=True, ask_new_photo=True,
                         message=messages.msg_crop_only(crop_pt))

    return Diagnosis(Level.LOW, prediction, None, None, None, None, False,
                     forward_to_technician=True, ask_new_photo=False, message=messages.msg_low())


class VisionService:
    """Junta o pré-processamento, o modelo, o filtro de entrada e a decisão."""

    def __init__(self, classifier: Classifier, labels: LabelSet, gate: CropGate,
                 high_confidence: float, low_confidence: float) -> None:
        self.classifier = classifier
        self.labels = labels
        self.gate = gate
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.transform = build_transform(labels)
        self.crops_pt = [labels.crop_pt(crop) for crop in labels.crops]

    def predict(self, image: Image.Image) -> Prediction:
        """Foto → cultura, doença, confiança e top-3 (sem aplicar limites)."""
        batch = preprocess(image, self.labels, self.transform)
        return summarize(self.classifier.predict_proba(batch), self.labels)

    def diagnose(self, image: Image.Image) -> Diagnosis:
        """Foto → diagnóstico completo, com a mensagem para o agricultor."""
        prediction = self.predict(image)
        supported = self.gate.is_supported(image, prediction)
        return decide(prediction, supported, self.high_confidence, self.low_confidence, self.crops_pt)


def load_vision_service(settings: Settings) -> VisionService:
    """Carrega o modelo real (demora alguns segundos: fazer só uma vez, no arranque)."""
    labels = load_labels(settings.labels_path)
    classifier = TimmClassifier(settings.model_path, labels.num_classes, labels.architecture)
    gate = ModelConfidenceGate(min_crop_confidence=settings.vision_low_confidence)
    logger.info("Modelo de visão carregado: %d classes, %s", labels.num_classes, settings.model_path)
    return VisionService(classifier, labels, gate, settings.vision_high_confidence, settings.vision_low_confidence)


def try_load_vision_service(settings: Settings) -> VisionService | None:
    """Como load_vision_service, mas devolve None se os ficheiros do modelo não existirem."""
    missing = [p for p in (settings.model_path, settings.labels_path) if not p.exists()]
    if missing:
        logger.warning("Diagnóstico por foto desligado: faltam %s", ", ".join(str(p) for p in missing))
        return None
    return load_vision_service(settings)
