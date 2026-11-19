"""Testes do endpoint POST /diagnose, com o classificador falso."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.diagnose import get_vision_service
from app.config import Settings, get_settings
from app.main import app
from tests.conftest import image_bytes

HIGH_MAIZE_BLIGHT = [0.0, 0.90, 0.04, 0.02, 0.02, 0.02]


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Sem "with TestClient(...)": o arranque (lifespan) não corre, e o modelo real não é carregado.
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_fake(make_service, probabilities: list[float]) -> None:
    app.dependency_overrides[get_vision_service] = lambda: make_service(probabilities)


def test_diagnose_returns_high_confidence_diagnosis(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    response = client.post("/diagnose", files={"file": ("folha.png", image_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "alta"
    assert body["crop_pt"] == "milho"
    assert body["disease"] == "leaf blight"
    assert body["confidence"] == pytest.approx(0.90)
    assert body["forward_to_technician"] is False
    assert len(body["top3"]) == 3
    assert body["message"]


def test_diagnose_accepts_jpeg(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    response = client.post("/diagnose", files={"file": ("folha.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200


def test_crop_only_hides_disease(client: TestClient, make_service) -> None:
    use_fake(make_service, [0.0, 0.40, 0.40, 0.10, 0.05, 0.05])
    body = client.post("/diagnose", files={"file": ("f.png", image_bytes(), "image/png")}).json()
    assert body["level"] == "so_cultura"
    assert body["crop_pt"] == "milho"
    assert body["disease"] is None
    assert body["forward_to_technician"] is True


def test_non_image_file_returns_400(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    response = client.post("/diagnose", files={"file": ("nota.txt", b"isto nao e uma foto", "text/plain")})
    assert response.status_code == 400


def test_truncated_image_returns_400(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    broken = image_bytes(fmt="JPEG")[:200]
    response = client.post("/diagnose", files={"file": ("f.jpg", broken, "image/jpeg")})
    assert response.status_code == 400


def test_too_large_file_returns_413(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, max_upload_mb=0.01)
    response = client.post("/diagnose", files={"file": ("f.png", b"0" * 20_000, "image/png")})
    assert response.status_code == 413


def test_missing_file_returns_422(client: TestClient, make_service) -> None:
    use_fake(make_service, HIGH_MAIZE_BLIGHT)
    assert client.post("/diagnose").status_code == 422


def test_model_not_loaded_returns_503(client: TestClient) -> None:
    app.state.vision = None
    response = client.post("/diagnose", files={"file": ("f.png", image_bytes(), "image/png")})
    assert response.status_code == 503
