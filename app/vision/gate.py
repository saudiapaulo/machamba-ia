"""Filtro de entrada: a foto é de uma das culturas suportadas?

Há três respostas possíveis:
- SUPPORTED: sim, pode seguir para o diagnóstico;
- NOT_SUPPORTED: não parece milho, mandioca, caju nem tomate;
- UNAVAILABLE: o filtro não conseguiu verificar (ex.: CLIP não carregou).
  Nesse caso falhamos com segurança: o diagnóstico vai para o técnico.

Implementações:
- ModelConfidenceGate (opção A): confiança do próprio classificador na cultura.
  Sozinho NÃO chega: o classificador dá confiança alta a fotos do céu ou de papel.
- ClipGate (opção B): um modelo CLIP compara a foto com frases ("a photo of a
  maize leaf", "a photo of a person", ...). Ver app/vision/clip_prompts.py.
- CombinedGate: junta vários filtros; a foto só passa se todos concordarem.
Para a opção C (classe "outra coisa"), basta criar mais uma classe com check().
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from PIL import Image

from app.vision.clip_filter import ClipScorer
from app.vision.prediction import Prediction

logger = logging.getLogger(__name__)


class GateResult(str, Enum):
    SUPPORTED = "suportado"
    NOT_SUPPORTED = "nao_suportado"
    UNAVAILABLE = "indisponivel"


@dataclass(frozen=True)
class GateCheck:
    """Resposta de um filtro. 'clip_score' vem preenchido quando o CLIP correu."""

    result: GateResult
    clip_score: float | None = None


class CropGate(Protocol):
    """Decide se a foto é de uma cultura que o sistema conhece."""

    def check(self, image: Image.Image, prediction: Prediction) -> GateCheck:
        ...

    def describe(self) -> str:
        """Descrição curta para o /health (ex.: 'clip (ViT-B-32)')."""
        ...

    @property
    def available(self) -> bool:
        """False se este filtro nunca vai conseguir verificar fotos."""
        ...


class ModelConfidenceGate:
    """Opção A: aceita se a confiança na melhor cultura atingir o mínimo."""

    def __init__(self, min_crop_confidence: float) -> None:
        self.min_crop_confidence = min_crop_confidence

    def check(self, image: Image.Image, prediction: Prediction) -> GateCheck:
        ok = prediction.crop_confidence >= self.min_crop_confidence
        return GateCheck(GateResult.SUPPORTED if ok else GateResult.NOT_SUPPORTED)

    def describe(self) -> str:
        return "confianca_modelo"

    @property
    def available(self) -> bool:
        return True


class ClipGate:
    """Opção B: aceita se a pontuação CLIP ("é uma das 4 culturas") atingir o limite."""

    def __init__(self, scorer: ClipScorer, min_score: float, label: str = "clip") -> None:
        self.scorer = scorer
        self.min_score = min_score
        self.label = label

    def check(self, image: Image.Image, prediction: Prediction) -> GateCheck:
        try:
            score = self.scorer.accept_score(image)
        except Exception:
            # Um erro no filtro nunca pode deixar passar a foto sem verificação.
            logger.exception("Erro no filtro CLIP: o caso vai para o técnico.")
            return GateCheck(GateResult.UNAVAILABLE)
        result = GateResult.SUPPORTED if score >= self.min_score else GateResult.NOT_SUPPORTED
        return GateCheck(result, clip_score=score)

    def describe(self) -> str:
        return self.label

    @property
    def available(self) -> bool:
        return True


class UnavailableGate:
    """Usado quando o CLIP não carregou ou está desligado: nunca dá a foto por verificada."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def check(self, image: Image.Image, prediction: Prediction) -> GateCheck:
        return GateCheck(GateResult.UNAVAILABLE)

    def describe(self) -> str:
        return f"indisponivel ({self.reason})"

    @property
    def available(self) -> bool:
        return False


class CombinedGate:
    """Junta vários filtros. 'Indisponível' ganha a tudo; depois, basta um 'não' para rejeitar."""

    def __init__(self, gates: list[CropGate]) -> None:
        self.gates = gates

    def check(self, image: Image.Image, prediction: Prediction) -> GateCheck:
        checks = [gate.check(image, prediction) for gate in self.gates]
        clip_score = next((c.clip_score for c in checks if c.clip_score is not None), None)
        results = {c.result for c in checks}
        if GateResult.UNAVAILABLE in results:
            return GateCheck(GateResult.UNAVAILABLE, clip_score)
        if GateResult.NOT_SUPPORTED in results:
            return GateCheck(GateResult.NOT_SUPPORTED, clip_score)
        return GateCheck(GateResult.SUPPORTED, clip_score)

    def describe(self) -> str:
        return " + ".join(gate.describe() for gate in self.gates)

    @property
    def available(self) -> bool:
        return all(gate.available for gate in self.gates)
