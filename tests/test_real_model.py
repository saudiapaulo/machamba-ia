"""Testes com os modelos reais. Só correm se os ficheiros existirem.

Correr só estes: pytest -m "real_model or real_clip"
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.config import Settings
from app.main import app
from app.vision.clip_filter import OpenClipScorer
from app.vision.clip_prompts import ACCEPT_PROMPTS, REJECT_PROMPTS
from app.vision.service import Level, load_vision_service
from tests.conftest import image_bytes

settings = Settings()
# Nestes testes o CLIP fica desligado, para não descarregar ~600 MB sem aviso.
no_clip = Settings(clip_enabled=False)

model_missing = not (settings.model_path.exists() and settings.labels_path.exists())
clip_missing = not any(settings.clip_cache_dir.rglob("open_clip_*")) if settings.clip_cache_dir.exists() else True


@pytest.mark.real_model
@pytest.mark.skipif(model_missing, reason="modelo real não encontrado em models/")
def test_real_model_loads_and_predicts() -> None:
    service = load_vision_service(no_clip)
    assert service.labels.num_classes == 22

    prediction = service.predict(Image.new("RGB", (640, 480), (40, 140, 40)))
    assert sum(prediction.crop_confidences.values()) == pytest.approx(1.0, abs=1e-4)
    assert len(prediction.top3) == 3
    confidences = [s.confidence for s in prediction.top3]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.real_model
@pytest.mark.skipif(model_missing, reason="modelo real não encontrado em models/")
def test_real_model_through_api_without_clip_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modelo real, CLIP desligado: a API arranca e tudo vai para o técnico."""
    monkeypatch.setattr(main_module, "get_settings", lambda: no_clip)
    with TestClient(app) as client:
        response = client.post("/diagnose", files={"file": ("f.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
        health = client.get("/health").json()
    assert response.status_code == 200
    assert response.json()["level"] == Level.UNVERIFIED.value
    assert response.json()["forward_to_technician"] is True
    assert health["status"] == "degradado"


@pytest.mark.real_clip
@pytest.mark.skipif(clip_missing, reason="pesos do CLIP não encontrados em models/clip/")
def test_real_clip_scores_between_0_and_1() -> None:
    scorer = OpenClipScorer(settings.clip_model, settings.clip_pretrained, settings.clip_cache_dir,
                            ACCEPT_PROMPTS, REJECT_PROMPTS)
    score = scorer.accept_score(Image.new("RGB", (640, 480), (120, 170, 230)))
    assert 0.0 <= score <= 1.0
