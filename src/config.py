"""
PharmaPath AI — Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Все настройки приложения. Читаются из ENV или .env файла.
    """

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "output"
    model_dir: Path = Path(__file__).resolve().parent.parent / "models"

    # ── API ──────────────────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    app_title: str = "PharmaPath AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # ── Optimizer ────────────────────────────────────────────────────────
    max_visits_per_day: int = 14
    day_start_hour: int = 9
    day_end_hour: int = 18
    default_visit_duration_min: int = 20
    avg_speed_kmh: float = 18.0        # средняя скорость по Москве
    solver_time_limit_sec: int = 5
    max_waiting_min: int = 45           # макс. ожидание у кабинета
    top_n_candidates: int = 50          # топ врачей для оптимизатора

    # ── Scoring ──────────────────────────────────────────────────────────
    score_formula: str = "value * probability"  # Value × P(success)

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_backend: str = "mock"           # "mock" | "ollama" | "openai"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_prefix = "PHARMA_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()