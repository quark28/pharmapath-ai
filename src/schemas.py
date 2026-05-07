"""
PharmaPath AI — Pydantic Schemas (API contracts)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class SpecialtyEnum(str, Enum):
    THERAPIST = "Therapist"
    CARDIOLOGIST = "Cardiologist"
    NEUROLOGIST = "Neurologist"
    ENDOCRINOLOGIST = "Endocrinologist"
    GASTROENTEROLOGIST = "Gastroenterologist"
    PULMONOLOGIST = "Pulmonologist"
    RHEUMATOLOGIST = "Rheumatologist"
    UROLOGIST = "Urologist"


class CategoryEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class VisitStatusEnum(str, Enum):
    SUCCESS = "Success"
    CANCELLED = "Cancelled"
    MOVED = "Moved"


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class GenerateRouteRequest(BaseModel):
    """Запрос на генерацию маршрута."""
    rep_id: str = Field(..., example="REP-001")
    latitude: float = Field(..., ge=54.0, le=57.0, example=55.7558)
    longitude: float = Field(..., ge=36.0, le=39.0, example=37.6173)
    target_date: date = Field(default_factory=date.today)
    max_visits: int = Field(default=14, ge=1, le=25)
    visited_ids: List[str] = Field(
        default_factory=list,
        description="Уже посещённые — исключить из маршрута",
    )


class SubmitReportRequest(BaseModel):
    """Отчёт медпреда после визита."""
    rep_id: str
    doctor_id: str
    visit_date: date = Field(default_factory=date.today)
    visit_time: str = Field(..., example="10:30")
    duration_minutes: int = Field(default=20, ge=0, le=120)
    status: VisitStatusEnum = VisitStatusEnum.SUCCESS
    report_text: str = Field(..., min_length=5)


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ExplanationFactor(BaseModel):
    """Один фактор объяснения выбора врача."""
    factor: str
    impact: str  # positive / negative / neutral


class RouteStop(BaseModel):
    """Одна остановка в маршруте."""
    order: int
    doctor_id: str
    doctor_name: str
    specialty: str
    category: str
    address: str
    latitude: float
    longitude: float
    estimated_arrival: str           # "09:15"
    estimated_departure: str         # "09:35"
    time_window_start: str           # "09:00"
    time_window_end: str             # "17:00"
    value_score: float
    probability_score: float
    combined_score: float
    explanation: List[ExplanationFactor]


class SkippedDoctor(BaseModel):
    """Врач, не вошедший в маршрут."""
    doctor_id: str
    doctor_name: str
    combined_score: float
    reason: str


class RouteResponse(BaseModel):
    """Полный ответ с маршрутом."""
    route_id: str
    rep_id: str
    target_date: str
    generated_at: str
    total_score: float
    total_distance_km: float
    total_duration_minutes: int
    num_visits: int
    stops: List[RouteStop]
    skipped: List[SkippedDoctor]
    optimizer_status: str            # "OPTIMAL" | "FEASIBLE" | "NO_SOLUTION"


class DoctorInfo(BaseModel):
    """Карточка врача (для API /doctors)."""
    id: str
    full_name: str
    specialty: str
    category: str
    work_address: str
    latitude: float
    longitude: float
    loyalty_score: float
    avg_sales_brick: float
    total_visits: int = 0
    success_rate: float = 0.0
    days_since_last_visit: int = 999


class ReportResult(BaseModel):
    """Результат обработки отчёта LLM."""
    visit_id: str
    sentiment: str
    competitors: List[str]
    objections: List[str]
    agreements: List[str]
    key_topics: List[str]
    raw_text: str
    llm_backend: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: bool
    doctors_count: int
    visits_count: int