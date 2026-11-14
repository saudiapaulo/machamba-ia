"""Configuração da aplicação, lida do ficheiro .env e das variáveis de ambiente."""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- RAG ---
    rag_min_similarity: float = 0.80

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
