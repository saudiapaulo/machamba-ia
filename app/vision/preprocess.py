"""Pré-processamento das fotos, igual ao usado no treino.

No notebook, cada foto passou por dois passos:
1. cache: orientação EXIF → RGB → reduzir o lado menor para 256 (bicubic, só se for maior);
2. transformação de validação: Resize(256) → recorte central 224 → tensor → normalizar.
Aqui repetimos os dois, pela mesma ordem.
"""

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from app.vision.labels import LabelSet


def prepare_image(image: Image.Image, resize_size: int) -> Image.Image:
    """Corrige a orientação, converte para RGB e reduz o lado menor (passo 'cache' do treino)."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    width, height = image.size
    scale = resize_size / min(width, height)
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
    return image


def build_transform(labels: LabelSet) -> transforms.Compose:
    """Transformação de validação/teste do notebook, com os valores do labels.json."""
    return transforms.Compose([
        transforms.Resize(labels.resize_size),
        transforms.CenterCrop(labels.image_size),
        transforms.ToTensor(),
        transforms.Normalize(labels.mean, labels.std),
    ])


def preprocess(image: Image.Image, labels: LabelSet, transform: transforms.Compose) -> torch.Tensor:
    """Foto → tensor com forma (1, 3, 224, 224), pronto para o modelo."""
    prepared = prepare_image(image, labels.resize_size)
    return transform(prepared).unsqueeze(0)
