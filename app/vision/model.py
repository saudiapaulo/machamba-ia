"""O classificador de imagens: interface e implementação com timm."""

from pathlib import Path
from typing import Protocol

import timm
import torch


class Classifier(Protocol):
    """Qualquer objecto que, dada uma imagem preparada, devolve uma probabilidade por classe.

    Os testes usam um classificador falso com esta mesma interface.
    """

    def predict_proba(self, batch: torch.Tensor) -> list[float]:
        """Recebe um tensor (1, 3, H, W) e devolve as probabilidades das classes (somam 1)."""
        ...


class TimmClassifier:
    """EfficientNet-B0 treinado no Colab, a correr em CPU."""

    def __init__(self, weights_path: Path, num_classes: int, architecture: str = "efficientnet_b0") -> None:
        # pretrained=False: a arquitectura vem do timm, mas os pesos são os nossos.
        self.model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
        # weights_only=True: só aceita tensores, não executa código escondido no ficheiro.
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)  # strict: falha se o ficheiro não bater com a arquitectura
        self.model.eval()  # modo de avaliação (desliga o dropout, etc.)

    @torch.inference_mode()
    def predict_proba(self, batch: torch.Tensor) -> list[float]:
        logits = self.model(batch)
        return torch.softmax(logits, dim=1)[0].tolist()
