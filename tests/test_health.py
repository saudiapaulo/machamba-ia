"""Testes do endpoint /health e da configuração."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import app
from app.vision.gate import CombinedGate, ModelConfidenceGate, UnavailableGate
from tests.conftest import FakeClassifier


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.state.vision = None


def test_health_without_model_is_degraded(client: TestClient) -> None:
    """Sem modelo carregado, a API responde, mas avisa que está 'degradada'."""
    app.state.vision = None
    body = client.get("/health").json()

    assert body["status"] == "degradado"
    assert body["app"] == "Machamba IA"
    assert body["vision_model_loaded"] is False
    assert body["crop_filter_available"] is False


def test_health_with_model_but_no_clip_is_degraded(client: TestClient, labels) -> None:
    from app.vision.service import VisionService

    gate = CombinedGate([UnavailableGate("teste"), ModelConfidenceGate(0.5)])
    app.state.vision = VisionService(FakeClassifier([1 / 6] * 6), labels, gate, 0.8, 0.5)
    body = client.get("/health").json()

    assert body["status"] == "degradado"
    assert body["vision_model_loaded"] is True
    assert "indisponivel" in body["crop_filter"]


def test_health_ok_when_everything_loaded(client: TestClient, make_service) -> None:
    app.state.vision = make_service([1 / 6] * 6)  # filtro do modelo: sempre disponível
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["crop_filter_available"] is True


def test_default_thresholds() -> None:
    """Os limites por omissão são os definidos no CLAUDE.md."""
    settings = Settings(_env_file=None)

    assert settings.vision_high_confidence == 0.80
    assert settings.vision_low_confidence == 0.50


def test_invalid_thresholds_are_rejected() -> None:
    """Um limite baixo maior que o alto é rejeitado no arranque."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vision_low_confidence=0.9, vision_high_confidence=0.8)


def test_invalid_clip_score_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, clip_min_score=1.5)
