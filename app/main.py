"""Ponto de entrada da API FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.diagnose import router as diagnose_router
from app.api.health import router as health_router
from app.config import get_settings
from app.vision.service import try_load_vision_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Corre uma vez no arranque: carrega o modelo de visão para a memória."""
    app.state.vision = try_load_vision_service(get_settings())
    yield


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI e regista os routers."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        description="Assistente agrícola com IA para pequenos produtores de Moçambique.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(diagnose_router)
    return app


app = create_app()
