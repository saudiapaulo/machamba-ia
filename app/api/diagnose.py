"""Endpoint POST /diagnose: diagnóstico por foto."""

from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.vision.prediction import ClassScore
from app.vision.service import Diagnosis, Level, VisionService

router = APIRouter(tags=["diagnóstico"])


class ClassScoreOut(BaseModel):
    """Uma das classes mais prováveis."""

    class_name: str
    crop: str
    crop_pt: str
    disease: str
    disease_pt: str
    confidence: float

    @classmethod
    def from_score(cls, score: ClassScore) -> "ClassScoreOut":
        info = score.info
        return cls(class_name=info.name, crop=info.crop, crop_pt=info.crop_pt, disease=info.disease,
                   disease_pt=score.disease_pt, confidence=round(score.confidence, 4))


class DiagnoseResponse(BaseModel):
    """Resposta do /diagnose. 'crop' e 'disease' só vêm preenchidos quando são fiáveis."""

    level: Level
    message: str
    crop: str | None
    crop_pt: str | None
    crop_confidence: float
    disease: str | None
    disease_pt: str | None
    confidence: float
    is_healthy: bool
    forward_to_technician: bool
    ask_new_photo: bool
    top3: list[ClassScoreOut]

    @classmethod
    def from_diagnosis(cls, d: Diagnosis) -> "DiagnoseResponse":
        p = d.prediction
        return cls(
            level=d.level, message=d.message,
            crop=d.crop, crop_pt=d.crop_pt, crop_confidence=round(p.crop_confidence, 4),
            disease=d.disease, disease_pt=d.disease_pt, confidence=round(p.confidence, 4),
            is_healthy=d.is_healthy, forward_to_technician=d.forward_to_technician,
            ask_new_photo=d.ask_new_photo, top3=[ClassScoreOut.from_score(s) for s in p.top3],
        )


def get_vision_service(request: Request) -> VisionService:
    """Devolve o serviço carregado no arranque, ou 503 se o modelo não estiver disponível."""
    service = getattr(request.app.state, "vision", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="O diagnóstico por foto não está disponível: o modelo não foi carregado.")
    return service


def read_image(data: bytes) -> Image.Image:
    """Abre os bytes como imagem, ou responde 400 se não for uma imagem válida."""
    try:
        image = Image.open(BytesIO(data))
        image.load()  # obriga a descomprimir já, para apanhar ficheiros estragados aqui
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400,
                            detail="O ficheiro enviado não é uma imagem válida. Envie uma foto (JPG ou PNG).")


# 'def' (e não 'async def'): o FastAPI corre a função numa thread à parte,
# para a previsão (pesada para o CPU) não bloquear os outros pedidos.
@router.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(
    file: UploadFile = File(..., description="Foto da planta"),
    service: VisionService = Depends(get_vision_service),
    settings: Settings = Depends(get_settings),
) -> DiagnoseResponse:
    """Recebe uma foto e devolve o diagnóstico com o nível de confiança."""
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    data = file.file.read(max_bytes + 1)  # lê no máximo 1 byte a mais do que o limite
    if len(data) > max_bytes:
        raise HTTPException(status_code=413,
                            detail=f"A foto é demasiado grande. O máximo é {settings.max_upload_mb:g} MB.")
    image = read_image(data)
    return DiagnoseResponse.from_diagnosis(service.diagnose(image))
