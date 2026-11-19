"""Configuração da aplicação, lida do ficheiro .env e das variáveis de ambiente."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pasta raiz do projecto (machamba-ia/), para os caminhos não dependerem
# da pasta de onde se corre o servidor.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Todas as definições do sistema num só lugar.

    O pydantic-settings procura cada campo nas variáveis de ambiente
    (sem distinguir maiúsculas/minúsculas) e no ficheiro .env.
    Se não encontrar, usa o valor por omissão aqui definido.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Aplicação ---
    app_name: str = "Machamba IA"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Limites de confiança da visão (ver CLAUDE.md) ---
    vision_high_confidence: float = 0.80
    vision_low_confidence: float = 0.50

    # --- Modelo de visão (gerado pelo notebook de treino) ---
    model_path: Path = BASE_DIR / "models" / "machamba_efficientnet_b0.pt"
    labels_path: Path = BASE_DIR / "models" / "labels.json"
    max_upload_mb: float = 8.0

    # --- RAG ---
    rag_min_similarity: float = 0.80

    @field_validator("model_path", "labels_path")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Caminhos relativos no .env são relativos à pasta do projecto."""
        return value if value.is_absolute() else BASE_DIR / value

    @model_validator(mode="after")
    def check_thresholds(self) -> "Settings":
        """Garante que os limites fazem sentido: 0 <= baixo < alto <= 1."""
        if not 0.0 <= self.vision_low_confidence < self.vision_high_confidence <= 1.0:
            raise ValueError(
                "Os limites de confiança têm de cumprir 0 <= VISION_LOW_CONFIDENCE "
                "< VISION_HIGH_CONFIDENCE <= 1."
            )
        if not 0.0 <= self.rag_min_similarity <= 1.0:
            raise ValueError("RAG_MIN_SIMILARITY tem de estar entre 0 e 1.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Devolve as definições, lidas uma única vez e reutilizadas depois."""
    return Settings()
