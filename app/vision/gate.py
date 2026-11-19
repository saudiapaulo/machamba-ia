"""Filtro de entrada: a foto é de uma das culturas suportadas?

Para o MVP usamos a opção A: a confiança do próprio modelo na cultura.
Limitação: o modelo só conhece as 22 classes do treino, por isso uma foto de
outra cultura parecida (ex.: feijão) pode sair "mandioca" com confiança alta e
passar o filtro. A rede de segurança é o técnico.

Para trocar mais tarde por um modelo CLIP (opção B) ou por uma classe
"outra coisa" (opção C), basta criar outra classe com o método is_supported.
"""

from typing import Protocol

from PIL import Image

from app.vision.prediction import Prediction


class CropGate(Protocol):
    """Decide se a foto é de uma cultura que o sistema conhece."""

    def is_supported(self, image: Image.Image, prediction: Prediction) -> bool:
        ...


class ModelConfidenceGate:
    """Opção A: aceita a foto se a confiança na melhor cultura atingir o mínimo."""

    def __init__(self, min_crop_confidence: float) -> None:
        self.min_crop_confidence = min_crop_confidence

    def is_supported(self, image: Image.Image, prediction: Prediction) -> bool:
        return prediction.crop_confidence >= self.min_crop_confidence
