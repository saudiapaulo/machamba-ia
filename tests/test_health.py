"""Testes do endpoint /health e da configuração."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """O /health responde 200 com status 'ok'."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Machamba IA"


def test_default_thresholds() -> None:
    """Os limites por omissão são os definidos no CLAUDE.md."""
    settings = Settings(_env_file=None)

    assert settings.vision_high_confidence == 0.80
    assert settings.vision_low_confidence == 0.50


def test_invalid_thresholds_are_rejected() -> None:
    """Um limite baixo maior que o alto é rejeitado no arranque."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vision_low_confidence=0.9, vision_high_confidence=0.8)
