"""Leitura do labels.json gerado pelo notebook de treino."""

import json
from dataclasses import dataclass
from pathlib import Path

# Correcções de ortografia — as mesmas do DISPLAY_FIXES do notebook.
# Só mudam o nome apresentado; o campo "folder" fica como no dataset.
NAME_FIXES: dict[str, str] = {
    "grasshoper": "grasshopper",
    "verticulium": "verticillium",
}


def fix_name(text: str) -> str:
    """Aplica as correcções de ortografia conhecidas."""
    for wrong, right in NAME_FIXES.items():
        text = text.replace(wrong, right)
    return text


@dataclass(frozen=True)
class ClassInfo:
    """Uma das classes do modelo (ex.: 'Maize - leaf blight')."""

    index: int
    name: str
    crop: str
    crop_pt: str
    disease: str
    folder: str

    @property
    def is_healthy(self) -> bool:
        """True se a classe for 'planta saudável'."""
        return self.disease == "healthy"


@dataclass(frozen=True)
class LabelSet:
    """As classes do modelo e o pré-processamento usado no treino."""

    classes: tuple[ClassInfo, ...]
    image_size: int
    resize_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    architecture: str = "efficientnet_b0"

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def crops(self) -> list[str]:
        """Culturas, pela ordem em que aparecem nas classes."""
        return list(dict.fromkeys(c.crop for c in self.classes))

    def crop_pt(self, crop: str) -> str:
        """Nome da cultura em português (ex.: 'Maize' → 'milho')."""
        return next(c.crop_pt for c in self.classes if c.crop == crop)


def load_labels(path: Path) -> LabelSet:
    """Lê o labels.json e devolve as classes com os nomes corrigidos."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_classes = data["classes"]

    expected = [str(i) for i in range(len(raw_classes))]
    if sorted(raw_classes, key=int) != expected:
        raise ValueError("labels.json: os índices das classes têm de ser 0, 1, 2, ... sem falhas.")

    classes = tuple(
        ClassInfo(
            index=int(key),
            name=fix_name(raw["class"]),
            crop=raw["crop"],
            crop_pt=raw["crop_pt"],
            disease=fix_name(raw["disease"]),
            folder=raw["folder"],
        )
        for key, raw in sorted(raw_classes.items(), key=lambda item: int(item[0]))
    )
    return LabelSet(
        classes=classes,
        image_size=int(data["image_size"]),
        resize_size=int(data["resize_size"]),
        mean=tuple(data["mean"]),
        std=tuple(data["std"]),
        architecture=data.get("model", "efficientnet_b0"),
    )
