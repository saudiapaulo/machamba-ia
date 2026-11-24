"""Pontuação CLIP: "esta foto é de milho, mandioca, caju ou tomate?" (0 = não, 1 = sim)."""

from pathlib import Path
from typing import Protocol

import torch
from PIL import Image, ImageOps


class ClipScorer(Protocol):
    """Qualquer objecto que dá à foto uma pontuação entre 0 e 1. Os testes usam um falso."""

    def accept_score(self, image: Image.Image) -> float:
        ...


class OpenClipScorer:
    """CLIP real (biblioteca open_clip), a correr em CPU.

    Os textos nunca mudam, por isso calculamos os seus vectores UMA vez, aqui,
    e depois deitamos fora a parte de texto do modelo. Fica em memória só o
    codificador de imagem — cerca de metade da RAM.
    """

    def __init__(self, model_name: str, pretrained: str, cache_dir: Path,
                 accept_prompts: list[str], reject_prompts: list[str]) -> None:
        import open_clip  # importado aqui: se a biblioteca faltar, só o filtro falha (com segurança)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=str(cache_dir))
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.inference_mode():
            text = model.encode_text(tokenizer(accept_prompts + reject_prompts))
            self.text_features = text / text.norm(dim=-1, keepdim=True)
        self.logit_scale = float(model.logit_scale.detach().exp())
        self.num_accept = len(accept_prompts)
        self.preprocess = preprocess
        self.visual = model.visual  # guardamos só a parte de imagem
        del model

    @torch.inference_mode()
    def accept_score(self, image: Image.Image) -> float:
        image = ImageOps.exif_transpose(image).convert("RGB")
        features = self.visual(self.preprocess(image).unsqueeze(0))
        features = features / features.norm(dim=-1, keepdim=True)
        # Probabilidade de cada frase; a pontuação é a soma das frases "aceitar"
        probabilities = (self.logit_scale * features @ self.text_features.T).softmax(dim=-1)[0]
        return float(probabilities[: self.num_accept].sum())
