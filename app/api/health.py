"""Endpoint de verificação de saúde do serviço."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["saúde"])


class HealthResponse(BaseModel):
    """Resposta do /health."""

    status: str
    app: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirma que a API está a correr."""
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, environment=settings.environment)
