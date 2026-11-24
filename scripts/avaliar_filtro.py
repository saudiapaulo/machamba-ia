"""Avalia o filtro de entrada CLIP com fotos tuas e sugere o CLIP_MIN_SCORE.

Pastas (incluindo subpastas):
    data/filtro/aceitar/   fotos de milho, mandioca, caju e tomate (o filtro deve deixar passar)
    data/filtro/rejeitar/  tudo o resto (o filtro deve barrar)

Uso (na pasta do projecto, com o ambiente virtual activo):
    python scripts/avaliar_filtro.py
    python scripts/avaliar_filtro.py --max-falsos-aceites 0.03
    python scripts/avaliar_filtro.py --modelo ViT-B-32 --pretrained laion2b_s34b_b79k

Os dois erros possíveis:
- FALSO ACEITE: uma foto de "rejeitar" passou o filtro. É o erro PERIGOSO:
  o agricultor recebe um diagnóstico errado. Por isso o limite sugerido é o
  que mantém os falsos aceites ≤ 5% (por omissão).
- FALSO REJEITADO: uma foto boa foi barrada. Só custa tempo ao técnico.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Permite correr "python scripts/avaliar_filtro.py" a partir da pasta do projecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95


@dataclass(frozen=True)
class Rates:
    """Erros do filtro para um limite."""

    threshold: float
    false_accepts: int     # fotos de "rejeitar" com pontuação ≥ limite
    num_reject: int
    false_rejects: int     # fotos de "aceitar" com pontuação < limite
    num_accept: int

    @property
    def false_accept_rate(self) -> float:
        return self.false_accepts / self.num_reject if self.num_reject else 0.0

    @property
    def false_reject_rate(self) -> float:
        return self.false_rejects / self.num_accept if self.num_accept else 0.0


def compute_rates(accept_scores: list[float], reject_scores: list[float], threshold: float) -> Rates:
    """Conta os erros para um limite (a foto passa se pontuação ≥ limite, como na API)."""
    return Rates(
        threshold=threshold,
        false_accepts=sum(s >= threshold for s in reject_scores),
        num_reject=len(reject_scores),
        false_rejects=sum(s < threshold for s in accept_scores),
        num_accept=len(accept_scores),
    )


def suggest_threshold(table: list[Rates], max_false_accept_rate: float) -> Rates | None:
    """O limite MAIS BAIXO com falsos aceites ≤ máximo.

    Subir o limite só pode baixar os falsos aceites e subir os falsos rejeitados.
    Por isso, o limite mais baixo que cumpre a regra é o que rejeita menos fotos boas.
    """
    ok = [r for r in sorted(table, key=lambda r: r.threshold) if r.false_accept_rate <= max_false_accept_rate]
    return ok[0] if ok else None


def upper_bound_zero_errors(n: int) -> float:
    """Com 0 erros em n fotos, a taxa real pode ainda chegar a ~3/n (regra do 3, 95% de confiança)."""
    return min(1.0, 3 / n) if n else 1.0


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def score_folder(scorer, folder: Path) -> tuple[list[tuple[Path, float]], list[Path], float]:
    """Pontua todas as fotos de uma pasta. Devolve (pontuações, ficheiros ilegíveis, segundos)."""
    from PIL import Image

    scores, unreadable = [], []
    start = time.time()
    for path in list_images(folder):
        try:
            with Image.open(path) as img:
                img.load()
                scores.append((path, scorer.accept_score(img)))
        except Exception:
            unreadable.append(path)
    return scores, unreadable, time.time() - start


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> None:
    from app.config import BASE_DIR, get_settings
    from app.vision.clip_filter import OpenClipScorer
    from app.vision.clip_prompts import ACCEPT_PROMPTS, REJECT_PROMPTS

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Avalia o filtro CLIP e sugere o CLIP_MIN_SCORE.")
    parser.add_argument("--aceitar", type=Path, default=BASE_DIR / "data" / "filtro" / "aceitar")
    parser.add_argument("--rejeitar", type=Path, default=BASE_DIR / "data" / "filtro" / "rejeitar")
    parser.add_argument("--modelo", default=settings.clip_model)
    parser.add_argument("--pretrained", default=settings.clip_pretrained)
    parser.add_argument("--max-falsos-aceites", type=float, default=0.05,
                        help="taxa máxima de falsos aceites aceitável (por omissão 0.05 = 5%%)")
    args = parser.parse_args()

    for folder in (args.aceitar, args.rejeitar):
        if not folder.is_dir() or not list_images(folder):
            sys.exit(f"❌ Não encontrei fotos em {folder}. Vê as instruções no início deste ficheiro.")

    print(f"A carregar o CLIP {args.modelo} / {args.pretrained}...")
    scorer = OpenClipScorer(args.modelo, args.pretrained, settings.clip_cache_dir, ACCEPT_PROMPTS, REJECT_PROMPTS)

    accept, bad_a, secs_a = score_folder(scorer, args.aceitar)
    reject, bad_r, secs_r = score_folder(scorer, args.rejeitar)
    for path in bad_a + bad_r:
        print(f"⚠️  Ignorada (não abre): {path}")
    n_photos = len(accept) + len(reject)
    print(f"\nFotos: {len(accept)} para aceitar, {len(reject)} para rejeitar "
          f"({1000 * (secs_a + secs_r) / max(n_photos, 1):.0f} ms por foto)\n")

    accept_scores = [s for _, s in accept]
    reject_scores = [s for _, s in reject]
    table = [compute_rates(accept_scores, reject_scores, t) for t in DEFAULT_THRESHOLDS]

    print(f"{'limite':>7} | {'falsos aceites (perigoso)':>27} | {'falsos rejeitados (técnico)':>28}")
    print("-" * 70)
    for r in table:
        print(f"{r.threshold:7.2f} | {r.false_accepts:4d}/{r.num_reject:<4d} {pct(r.false_accept_rate):>14} | "
              f"{r.false_rejects:4d}/{r.num_accept:<4d} {pct(r.false_reject_rate):>15}")

    best = suggest_threshold(table, args.max_falsos_aceites)
    print()
    if best is None:
        print(f"❌ Nenhum limite mantém os falsos aceites ≤ {pct(args.max_falsos_aceites).strip()}. "
              "O filtro não separa bem estas fotos: revê os prompts ou as fotos da pasta 'rejeitar'.")
    else:
        print(f"✅ Limite sugerido: CLIP_MIN_SCORE={best.threshold:g}")
        print(f"   Falsos aceites: {best.false_accepts} de {best.num_reject} ({pct(best.false_accept_rate).strip()})")
        print(f"   Custo: {best.false_rejects} de {best.num_accept} fotos boas "
              f"({pct(best.false_reject_rate).strip()}) iriam para o técnico sem necessidade.")
        if best.false_accepts == 0:
            print(f"   Atenção: 0 erros em {best.num_reject} fotos não garante 0% na realidade — a taxa real "
                  f"pode chegar a ~{pct(upper_bound_zero_errors(best.num_reject)).strip()}. Mais fotos = mais certeza.")
        print("   Para usar: põe esta linha no teu .env e reinicia o servidor.")

    print("\nFotos de 'rejeitar' com pontuação MAIS ALTA (as mais perigosas):")
    for path, score in sorted(reject, key=lambda x: -x[1])[:10]:
        print(f"  {score:.3f}  {path.relative_to(args.rejeitar)}")
    print("\nFotos de 'aceitar' com pontuação MAIS BAIXA:")
    for path, score in sorted(accept, key=lambda x: x[1])[:10]:
        print(f"  {score:.3f}  {path.relative_to(args.aceitar)}")


if __name__ == "__main__":
    main()
