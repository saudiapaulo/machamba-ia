"""Ponto de entrada da API FastAPI."""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import get_settings


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI e regista os routers."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Assistente agrícola com IA para pequenos produtores de Moçambique.",
        version="0.1.0",
    )
    app.include_router(health_router)
    return app


app = create_app()
