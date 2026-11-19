"""Testes com o modelo real. Só correm se o ficheiro existir em models/.

Correr só estes: pytest -m real_model
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import app
from app.vision.service import Level, load_vision_service
from tests.conftest import image_bytes

settings = Settings()
pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not (settings.model_path.exists() and settings.labels_path.exists()),
        reason="modelo real não encontrado em models/",
    ),
]


def test_real_model_loads_and_predicts() -> None:
    service = load_vision_service(settings)
    assert service.labels.num_classes == 22

    prediction = service.predict(Image.new("RGB", (640, 480), (40, 140, 40)))
    assert sum(prediction.crop_confidences.values()) == pytest.approx(1.0, abs=1e-4)
    assert len(prediction.top3) == 3
    confidences = [s.confidence for s in prediction.top3]
    assert confidences == sorted(confidences, reverse=True)

    diagnosis = service.diagnose(Image.new("RGB", (640, 480), (40, 140, 40)))
    assert diagnosis.level in Level
    assert diagnosis.message


def test_real_model_through_api() -> None:
    """Com 'with', o arranque corre e o modelo é carregado como no servidor real."""
    with TestClient(app) as client:
        response = client.post("/diagnose", files={"file": ("f.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["level"] in {level.value for level in Level}
