"""
PharmaPath AI — FastAPI Application
=====================================
Точка входа. Инициализация всех сервисов.

Запуск:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.data_provider import DataProvider
from src.services.optimizer import RouteOptimizer
from src.services.scoring import ScoringService

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)
logger = structlog.get_logger()

# ── Глобальные сервисы (Singleton) ────────────────────────────────────────────
_data_provider: DataProvider | None = None
_scoring_service: ScoringService | None = None
_optimizer: RouteOptimizer | None = None


def get_data_provider() -> DataProvider:
    assert _data_provider is not None, "DataProvider not initialized"
    return _data_provider


def get_scoring_service() -> ScoringService:
    assert _scoring_service is not None, "ScoringService not initialized"
    return _scoring_service


def get_optimizer() -> RouteOptimizer:
    assert _optimizer is not None, "RouteOptimizer not initialized"
    return _optimizer


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown."""
    global _data_provider, _scoring_service, _optimizer

    logger.info("=" * 60)
    logger.info("  🚀 PharmaPath AI — Starting up...")
    logger.info("=" * 60)

    # Data
    _data_provider = DataProvider(settings.data_dir)
    logger.info(
        "Data loaded",
        doctors=_data_provider.doctors_count,
        visits=_data_provider.visits_count,
    )

    # ML Models
    _scoring_service = ScoringService(settings.model_dir)
    _scoring_service.load_models()

    # Optimizer
    _optimizer = RouteOptimizer(
        avg_speed_kmh=settings.avg_speed_kmh,
        solver_time_limit_sec=settings.solver_time_limit_sec,
        max_waiting_min=settings.max_waiting_min,
    )

    logger.info("  ✅ All services ready")
    logger.info("=" * 60)

    yield  # ← приложение работает

    logger.info("Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS (для Streamlit фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency Overrides (для роутеров) ───────────────────────────────────────

def _scoring_dep() -> ScoringService:
    return get_scoring_service()


def _optimizer_dep() -> RouteOptimizer:
    return get_optimizer()


# ── Роутеры ───────────────────────────────────────────────────────────────────

from src.routers import doctors, reports, routes  # noqa: E402

# Override dependencies в routes router
routes.router.dependencies = []
app.include_router(routes.router, prefix=settings.api_prefix)
app.include_router(doctors.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)


# Переопределяем Depends в routes
from src.routers.routes import generate_route  # noqa: E402
app.dependency_overrides[ScoringService] = _scoring_dep
app.dependency_overrides[RouteOptimizer] = _optimizer_dep


# ── Health Check ──────────────────────────────────────────────────────────────

from src.schemas import HealthResponse  # noqa: E402


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_loaded=_scoring_service.is_loaded if _scoring_service else False,
        doctors_count=_data_provider.doctors_count if _data_provider else 0,
        visits_count=_data_provider.visits_count if _data_provider else 0,
    )


@app.get("/")
async def root():
    return {
        "service": settings.app_title,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }