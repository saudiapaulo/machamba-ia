"""Resultado bruto do modelo: classes, culturas e confianças."""

from dataclasses import dataclass

from app.vision.labels import ClassInfo, LabelSet
from app.vision.messages import disease_pt


@dataclass(frozen=True)
class ClassScore:
    """Uma classe e a sua probabilidade."""

    info: ClassInfo
    confidence: float

    @property
    def disease_pt(self) -> str:
        return disease_pt(self.info.folder, self.info.disease)


@dataclass(frozen=True)
class Prediction:
    """O que o modelo viu, antes de aplicar os limites de confiança.

    - top: a classe mais provável e a sua probabilidade (confiança da doença);
    - crop / crop_confidence: a cultura com maior soma de probabilidades.
      Normalmente é a cultura de 'top', mas quando o modelo hesita entre
      várias doenças da mesma cultura, a soma revela a cultura com mais clareza.
    """

    top: ClassScore
    top3: tuple[ClassScore, ...]
    crop: str
    crop_pt: str
    crop_confidence: float
    crop_confidences: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.top.confidence


def summarize(probabilities: list[float], labels: LabelSet) -> Prediction:
    """Transforma as 22 probabilidades numa Prediction."""
    if len(probabilities) != labels.num_classes:
        raise ValueError(
            f"O modelo devolveu {len(probabilities)} probabilidades, mas há {labels.num_classes} classes."
        )

    scores = sorted(
        (ClassScore(info, float(p)) for info, p in zip(labels.classes, probabilities)),
        key=lambda score: score.confidence,
        reverse=True,
    )

    # Confiança por cultura = soma das probabilidades das suas classes
    crop_confidences = {crop: 0.0 for crop in labels.crops}
    for score in scores:
        crop_confidences[score.info.crop] += score.confidence
    best_crop = max(crop_confidences, key=crop_confidences.get)

    return Prediction(
        top=scores[0],
        top3=tuple(scores[:3]),
        crop=best_crop,
        crop_pt=labels.crop_pt(best_crop),
        crop_confidence=crop_confidences[best_crop],
        crop_confidences=crop_confidences,
    )
