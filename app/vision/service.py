"""Serviço de diagnóstico: foto → previsão → filtro de entrada → decisão.

Não depende do FastAPI nem do Telegram: a API e o bot chamam este serviço.
"""

import logging
from dataclasses import dataclass, replace
from enum import Enum

from PIL import Image

from app.config import Settings
from app.vision import messages
from app.vision.clip_filter import OpenClipScorer
from app.vision.clip_prompts import ACCEPT_PROMPTS, REJECT_PROMPTS
from app.vision.gate import (ClipGate, CombinedGate, CropGate, GateCheck, GateResult,
                             ModelConfidenceGate, UnavailableGate)
from app.vision.labels import LabelSet, load_labels
from app.vision.model import Classifier, TimmClassifier
from app.vision.prediction import Prediction, summarize
from app.vision.preprocess import build_transform, preprocess

logger = logging.getLogger(__name__)


class Level(str, Enum):
    """Nível de confiança do diagnóstico."""

    HIGH = "alta"                    # responde normalmente
    MEDIUM = "media"                 # responde com cautela e pede outra foto
    CROP_ONLY = "so_cultura"         # sabe a cultura, não a doença: pede outra foto + técnico
    LOW = "baixa"                    # não diagnostica: técnico
    NOT_SUPPORTED = "nao_suportado"  # não parece uma das culturas suportadas: técnico
    UNVERIFIED = "nao_verificado"    # o filtro de entrada não funcionou: técnico (falhar com segurança)


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
    clip_score: float | None = None  # pontuação do filtro CLIP (para o técnico e para afinar o limite)


def _forward(level: Level, prediction: Prediction, message: str, ask_new_photo: bool = False,
             crop: str | None = None, crop_pt: str | None = None) -> Diagnosis:
    """Diagnóstico que vai para o técnico, sem doença mostrada ao agricultor."""
    return Diagnosis(level, prediction, crop, crop_pt, None, None, False,
                     forward_to_technician=True, ask_new_photo=ask_new_photo, message=message)


def decide(prediction: Prediction, gate: GateCheck, high: float, low: float, crops_pt: list[str]) -> Diagnosis:
    """Aplica o filtro de entrada e as regras do CLAUDE.md (alta ≥ high, média ≥ low, baixa < low)."""
    if gate.result is GateResult.UNAVAILABLE:
        diagnosis = _forward(Level.UNVERIFIED, prediction, messages.msg_unverified())
    elif gate.result is GateResult.NOT_SUPPORTED:
        diagnosis = _forward(Level.NOT_SUPPORTED, prediction, messages.msg_not_supported(crops_pt))
    else:
        diagnosis = _decide_supported(prediction, high, low)
    return replace(diagnosis, clip_score=gate.clip_score)


def _decide_supported(prediction: Prediction, high: float, low: float) -> Diagnosis:
    """Decisão para uma foto que passou o filtro de entrada."""
    top = prediction.top
    crop_pt = prediction.crop_pt
    healthy = top.info.is_healthy

    if prediction.confidence >= high:
        message = messages.msg_high_healthy(crop_pt) if healthy else messages.msg_high(crop_pt, top.disease_pt)
        return Diagnosis(Level.HIGH, prediction, prediction.crop, crop_pt, top.info.disease, top.disease_pt,
                         healthy, forward_to_technician=False, ask_new_photo=False, message=message)

    if prediction.confidence >= low:
        message = messages.msg_medium_healthy(crop_pt) if healthy else messages.msg_medium(crop_pt, top.disease_pt)
        return Diagnosis(Level.MEDIUM, prediction, prediction.crop, crop_pt, top.info.disease, top.disease_pt,
                         healthy, forward_to_technician=False, ask_new_photo=True, message=message)

    if prediction.crop_confidence >= high:
        return _forward(Level.CROP_ONLY, prediction, messages.msg_crop_only(crop_pt), ask_new_photo=True,
                        crop=prediction.crop, crop_pt=crop_pt)

    return _forward(Level.LOW, prediction, messages.msg_low())


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
        """Foto → cultura, doença, confiança e top-3 (sem filtro nem limites)."""
        batch = preprocess(image, self.labels, self.transform)
        return summarize(self.classifier.predict_proba(batch), self.labels)

    def diagnose(self, image: Image.Image) -> Diagnosis:
        """Foto → diagnóstico completo, com a mensagem para o agricultor."""
        prediction = self.predict(image)
        gate_check = self.gate.check(image, prediction)
        return decide(prediction, gate_check, self.high_confidence, self.low_confidence, self.crops_pt)


def build_gate(settings: Settings) -> CropGate:
    """Cria o filtro de entrada. Se o CLIP falhar, o sistema arranca na mesma em modo seguro."""
    model_gate = ModelConfidenceGate(min_crop_confidence=settings.vision_low_confidence)
    if not settings.clip_enabled:
        logger.warning("Filtro CLIP desligado (CLIP_ENABLED=false): todos os diagnósticos vão para o técnico.")
        return CombinedGate([UnavailableGate("CLIP_ENABLED=false"), model_gate])
    try:
        scorer = OpenClipScorer(settings.clip_model, settings.clip_pretrained, settings.clip_cache_dir,
                                ACCEPT_PROMPTS, REJECT_PROMPTS)
    except Exception as error:
        logger.exception("O filtro CLIP não carregou: todos os diagnósticos vão para o técnico.")
        return CombinedGate([UnavailableGate(f"CLIP não carregou: {type(error).__name__}"), model_gate])
    logger.info("Filtro CLIP carregado: %s / %s", settings.clip_model, settings.clip_pretrained)
    label = f"clip ({settings.clip_model}, limite {settings.clip_min_score:g})"
    return CombinedGate([ClipGate(scorer, settings.clip_min_score, label), model_gate])


def load_vision_service(settings: Settings) -> VisionService:
    """Carrega o modelo real e o filtro (demora alguns segundos: fazer só uma vez, no arranque)."""
    labels = load_labels(settings.labels_path)
    classifier = TimmClassifier(settings.model_path, labels.num_classes, labels.architecture)
    logger.info("Modelo de visão carregado: %d classes, %s", labels.num_classes, settings.model_path)
    return VisionService(classifier, labels, build_gate(settings),
                         settings.vision_high_confidence, settings.vision_low_confidence)


def try_load_vision_service(settings: Settings) -> VisionService | None:
    """Como load_vision_service, mas devolve None se os ficheiros do modelo não existirem."""
    missing = [p for p in (settings.model_path, settings.labels_path) if not p.exists()]
    if missing:
        logger.warning("Diagnóstico por foto desligado: faltam %s", ", ".join(str(p) for p in missing))
        return None
    return load_vision_service(settings)
