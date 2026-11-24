"""Testes do filtro de entrada (CLIP falso), do modo seguro e do script de avaliação."""

import pytest
from PIL import Image

from app.config import Settings
from app.vision import service as service_module
from app.vision.gate import (ClipGate, CombinedGate, GateCheck, GateResult, ModelConfidenceGate,
                             UnavailableGate)
from app.vision.labels import LabelSet
from app.vision.prediction import summarize
from app.vision.service import Level, VisionService, build_gate
from scripts.avaliar_filtro import compute_rates, suggest_threshold, upper_bound_zero_errors
from tests.conftest import FakeClassifier

HIGH_MAIZE_BLIGHT = [0.0, 0.90, 0.04, 0.02, 0.02, 0.02]
IMAGE = Image.new("RGB", (300, 300))


class FakeScorer:
    """CLIP falso: devolve sempre a mesma pontuação, ou falha se lhe pedirmos."""

    def __init__(self, score: float = 0.9, fail: bool = False) -> None:
        self.score = score
        self.fail = fail

    def accept_score(self, image: Image.Image) -> float:
        if self.fail:
            raise RuntimeError("CLIP avariado")
        return self.score


def service_with(labels: LabelSet, gate, probabilities=HIGH_MAIZE_BLIGHT) -> VisionService:
    return VisionService(FakeClassifier(probabilities), labels, gate, high_confidence=0.80, low_confidence=0.50)


# ---------- ClipGate ----------

@pytest.mark.parametrize(("score", "expected"), [
    (0.60, GateResult.SUPPORTED),       # fronteira exacta: passa
    (0.95, GateResult.SUPPORTED),
    (0.59, GateResult.NOT_SUPPORTED),
    (0.0, GateResult.NOT_SUPPORTED),
])
def test_clip_gate_threshold(labels: LabelSet, score: float, expected: GateResult) -> None:
    prediction = summarize(HIGH_MAIZE_BLIGHT, labels)
    check = ClipGate(FakeScorer(score), min_score=0.60).check(IMAGE, prediction)
    assert check.result is expected
    assert check.clip_score == score


def test_clip_error_on_one_photo_is_unavailable(labels: LabelSet) -> None:
    check = ClipGate(FakeScorer(fail=True), min_score=0.5).check(IMAGE, summarize(HIGH_MAIZE_BLIGHT, labels))
    assert check.result is GateResult.UNAVAILABLE


# ---------- CombinedGate ----------

class FixedGate:
    def __init__(self, result: GateResult) -> None:
        self.result = result

    def check(self, image, prediction) -> GateCheck:
        return GateCheck(self.result)

    def describe(self) -> str:
        return self.result.value

    @property
    def available(self) -> bool:
        return self.result is not GateResult.UNAVAILABLE


@pytest.mark.parametrize(("results", "expected"), [
    ([GateResult.SUPPORTED, GateResult.SUPPORTED], GateResult.SUPPORTED),
    ([GateResult.SUPPORTED, GateResult.NOT_SUPPORTED], GateResult.NOT_SUPPORTED),
    ([GateResult.UNAVAILABLE, GateResult.SUPPORTED], GateResult.UNAVAILABLE),
    ([GateResult.UNAVAILABLE, GateResult.NOT_SUPPORTED], GateResult.UNAVAILABLE),  # indisponível ganha a tudo
])
def test_combined_gate(labels: LabelSet, results, expected) -> None:
    gate = CombinedGate([FixedGate(r) for r in results])
    assert gate.check(IMAGE, summarize(HIGH_MAIZE_BLIGHT, labels)).result is expected


# ---------- decisão com o filtro ----------

def test_clip_rejection_gives_not_supported(labels: LabelSet) -> None:
    gate = CombinedGate([ClipGate(FakeScorer(0.1), 0.5), ModelConfidenceGate(0.5)])
    diagnosis = service_with(labels, gate).diagnose(IMAGE)
    assert diagnosis.level is Level.NOT_SUPPORTED
    assert diagnosis.disease is None
    assert diagnosis.forward_to_technician
    assert diagnosis.clip_score == pytest.approx(0.1)


def test_clip_acceptance_keeps_normal_diagnosis(labels: LabelSet) -> None:
    gate = CombinedGate([ClipGate(FakeScorer(0.9), 0.5), ModelConfidenceGate(0.5)])
    diagnosis = service_with(labels, gate).diagnose(IMAGE)
    assert diagnosis.level is Level.HIGH
    assert diagnosis.clip_score == pytest.approx(0.9)


def test_unavailable_filter_sends_everything_to_technician(labels: LabelSet) -> None:
    """Mesmo com o modelo muito confiante, sem filtro o agricultor não recebe diagnóstico."""
    gate = CombinedGate([UnavailableGate("teste"), ModelConfidenceGate(0.5)])
    diagnosis = service_with(labels, gate).diagnose(IMAGE)
    assert diagnosis.level is Level.UNVERIFIED
    assert diagnosis.forward_to_technician
    assert diagnosis.crop is None and diagnosis.disease is None
    assert "técnico" in diagnosis.message
    assert diagnosis.prediction.top.info.disease == "leaf blight"  # o palpite fica para o técnico


def test_clip_error_during_diagnosis_is_unverified(labels: LabelSet) -> None:
    gate = CombinedGate([ClipGate(FakeScorer(fail=True), 0.5), ModelConfidenceGate(0.5)])
    assert service_with(labels, gate).diagnose(IMAGE).level is Level.UNVERIFIED


# ---------- build_gate: falhar com segurança no arranque ----------

def test_build_gate_with_clip_disabled() -> None:
    gate = build_gate(Settings(_env_file=None, clip_enabled=False))
    assert not gate.available
    assert "CLIP_ENABLED=false" in gate.describe()


def test_build_gate_when_clip_fails_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_scorer(*args, **kwargs):
        raise OSError("pesos não encontrados")

    monkeypatch.setattr(service_module, "OpenClipScorer", broken_scorer)
    gate = build_gate(Settings(_env_file=None, clip_enabled=True))
    assert not gate.available
    assert "OSError" in gate.describe()


def test_build_gate_with_working_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "OpenClipScorer", lambda *args, **kwargs: FakeScorer(0.9))
    gate = build_gate(Settings(_env_file=None, clip_enabled=True, clip_min_score=0.7))
    assert gate.available
    assert gate.describe().startswith("clip")


# ---------- script de avaliação ----------

def test_compute_rates() -> None:
    rates = compute_rates(accept_scores=[0.9, 0.8, 0.4, 0.2], reject_scores=[0.1, 0.3, 0.6], threshold=0.5)
    assert (rates.false_accepts, rates.false_rejects) == (1, 2)
    assert rates.false_accept_rate == pytest.approx(1 / 3)
    assert rates.false_reject_rate == pytest.approx(0.5)


def test_compute_rates_threshold_is_inclusive() -> None:
    """A API aceita pontuação ≥ limite; o script tem de contar da mesma forma."""
    rates = compute_rates(accept_scores=[0.5], reject_scores=[0.5], threshold=0.5)
    assert rates.false_accepts == 1 and rates.false_rejects == 0


def test_suggest_lowest_threshold_within_false_accept_limit() -> None:
    accept = [0.95, 0.9, 0.85, 0.7, 0.6, 0.3]
    reject = [0.05, 0.1, 0.2, 0.4, 0.65]
    table = [compute_rates(accept, reject, t) for t in (0.3, 0.5, 0.7, 0.9)]
    best = suggest_threshold(table, max_false_accept_rate=0.0)
    assert best.threshold == 0.7          # 0.5 ainda deixa passar o 0.65
    assert best.false_rejects == 2        # 0.6 e 0.3 vão para o técnico


def test_suggest_returns_none_when_impossible() -> None:
    table = [compute_rates([0.5], [0.99], t) for t in (0.5, 0.9)]
    assert suggest_threshold(table, max_false_accept_rate=0.05) is None


def test_rule_of_three_with_few_photos() -> None:
    assert upper_bound_zero_errors(50) == pytest.approx(0.06)
