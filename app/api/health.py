"""Endpoint de verificação de saúde do serviço."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["saúde"])


class HealthResponse(BaseModel):
    """Resposta do /health.

    status é 'ok' quando tudo está carregado e 'degradado' quando a API funciona,
    mas o diagnóstico por foto está desligado ou a encaminhar tudo para o técnico.
    """

    status: str
    app: str
    environment: str
    vision_model_loaded: bool
    crop_filter: str | None
    crop_filter_available: bool


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Confirma que a API está a correr e mostra o estado do diagnóstico por foto."""
    settings = get_settings()
    vision = getattr(request.app.state, "vision", None)
    filter_ok = vision is not None and vision.gate.available
    return HealthResponse(
        status="ok" if filter_ok else "degradado",
        app=settings.app_name,
        environment=settings.environment,
        vision_model_loaded=vision is not None,
        crop_filter=vision.gate.describe() if vision is not None else None,
        crop_filter_available=filter_ok,
    )
