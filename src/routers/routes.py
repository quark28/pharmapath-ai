"""
PharmaPath AI — Route Generation Router
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import List

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.config import settings
from src.schemas import (
    ExplanationFactor,
    GenerateRouteRequest,
    RouteResponse,
    RouteStop,
    SkippedDoctor,
)
from src.services.optimizer import CandidateDoctor, RouteOptimizer, minutes_to_time_str
from src.services.scoring import ScoringService

logger = structlog.get_logger()
router = APIRouter(prefix="/routes", tags=["Routes"])


@router.post("/generate", response_model=RouteResponse)
async def generate_route(
    req: GenerateRouteRequest,
):
    """
    🧠 Главный эндпоинт — генерация оптимального маршрута.

    Pipeline:
    1. Score all doctors (Value × Probability)
    2. Filter: working on target_date, not in visited_ids
    3. Top-N by combined score
    4. OR-Tools VRPTW → optimal route
    5. Enrich with explanations
    """
    from src.main import get_data_provider, get_scoring_service, get_optimizer
    
    data = get_data_provider()
    scoring = get_scoring_service()
    optimizer = get_optimizer()

    if not scoring.is_loaded:
        raise HTTPException(503, "ML models not loaded")

    # ... остальной код без изменений ...

    if not scoring.is_loaded:
        raise HTTPException(503, "ML models not loaded")

    # ── 1. Scoring ────────────────────────────────────────────────────────
    doctors_df = data.get_all_doctors()
    visit_stats = data.get_bulk_visit_stats(req.target_date)

    scores_df = scoring.score_doctors_batch(
        doctors_df=doctors_df,
        visit_stats=visit_stats,
        target_date=req.target_date,
    )

    # ── 2. Filter ─────────────────────────────────────────────────────────
    # Убираем уже посещённых
    if req.visited_ids:
        scores_df = scores_df[~scores_df["doctor_id"].isin(req.visited_ids)]

    # Получаем time windows
    top_ids = scores_df.head(settings.top_n_candidates * 2)["doctor_id"].tolist()
    time_windows = data.get_bulk_time_windows(top_ids, req.target_date)

    # Убираем тех, кто не работает в этот день
    working_ids = {did for did, tw in time_windows.items() if tw is not None}
    scores_df = scores_df[scores_df["doctor_id"].isin(working_ids)]

    # ── 3. Top-N ──────────────────────────────────────────────────────────
    top_n = scores_df.head(settings.top_n_candidates)

    if top_n.empty:
        raise HTTPException(
            404,
            f"No available doctors on {req.target_date}",
        )

    # ── 4. Build candidates ───────────────────────────────────────────────
    candidates: List[CandidateDoctor] = []
    for _, row in top_n.iterrows():
        doc = data.get_doctor(row["doctor_id"])
        if doc is None:
            continue
        tw = time_windows.get(row["doctor_id"])
        candidates.append(CandidateDoctor(
            doctor_id=row["doctor_id"],
            lat=float(doc["latitude"]),
            lon=float(doc["longitude"]),
            combined_score=float(row["combined_score"]),
            time_window=tw,
            visit_duration=settings.default_visit_duration_min,
        ))

    # ── 5. Optimize ───────────────────────────────────────────────────────
    result = optimizer.optimize(
        depot_lat=req.latitude,
        depot_lon=req.longitude,
        candidates=candidates,
        day_start_min=settings.day_start_hour * 60,
        day_end_min=settings.day_end_hour * 60,
        max_visits=req.max_visits,
    )

    # ── 6. Build response ─────────────────────────────────────────────────
    stops: List[RouteStop] = []
    total_distance_km = 0.0

    # Построим scores dict для быстрого доступа
    scores_dict = dict(zip(
        top_n["doctor_id"],
        zip(top_n["value_score"], top_n["prob_score"], top_n["combined_score"]),
    ))

    from src.services.geo import haversine_km

    prev_lat, prev_lon = req.latitude, req.longitude

    for stop_result in result.stops:
        doc = data.get_doctor(stop_result.doctor_id)
        if doc is None:
            continue

        vs = visit_stats.get(stop_result.doctor_id, {})
        doc_dict = doc.to_dict()

        # Scores
        val_s, prob_s, comb_s = scores_dict.get(
            stop_result.doctor_id, (0, 0, 0)
        )

        # Distance from prev
        dist = haversine_km(prev_lat, prev_lon, float(doc["latitude"]), float(doc["longitude"]))
        total_distance_km += dist
        prev_lat, prev_lon = float(doc["latitude"]), float(doc["longitude"])

        # Time window
        tw = time_windows.get(stop_result.doctor_id, (540, 1080))

        # Explanation
        contributions = ScoringService._explain(doc_dict, vs)
        explanation = ScoringService.build_explanation_text(doc_dict, vs, contributions)
        explanation_factors = [
            ExplanationFactor(factor=e["factor"], impact=e["impact"])
            for e in explanation
        ]

        # Short name
        parts = doc["full_name"].split()
        short_name = f"{parts[0]} {parts[1][0]}.{parts[2][0]}." if len(parts) >= 3 else doc["full_name"]

        stops.append(RouteStop(
            order=stop_result.order + 1,
            doctor_id=stop_result.doctor_id,
            doctor_name=short_name,
            specialty=str(doc["specialty"]),
            category=str(doc["category"]),
            address=str(doc["work_address"]),
            latitude=float(doc["latitude"]),
            longitude=float(doc["longitude"]),
            estimated_arrival=minutes_to_time_str(stop_result.arrival_min),
            estimated_departure=minutes_to_time_str(stop_result.departure_min),
            time_window_start=minutes_to_time_str(tw[0]) if tw else "09:00",
            time_window_end=minutes_to_time_str(tw[1]) if tw else "18:00",
            value_score=float(val_s),
            probability_score=float(prob_s),
            combined_score=float(comb_s),
            explanation=explanation_factors,
        ))

    # Skipped
    skipped: List[SkippedDoctor] = []
    for skip_id in result.skipped_ids[:20]:  # max 20 в ответе
        doc = data.get_doctor(skip_id)
        if doc is None:
            continue
        parts = doc["full_name"].split()
        short_name = f"{parts[0]} {parts[1][0]}.{parts[2][0]}." if len(parts) >= 3 else doc["full_name"]
        sc = scores_dict.get(skip_id, (0, 0, 0))
        skipped.append(SkippedDoctor(
            doctor_id=skip_id,
            doctor_name=short_name,
            combined_score=float(sc[2]),
            reason="Не вместился в маршрут (время / лимит визитов)",
        ))

    return RouteResponse(
        route_id=str(uuid.uuid4()),
        rep_id=req.rep_id,
        target_date=req.target_date.isoformat(),
        generated_at=datetime.now().isoformat(),
        total_score=result.total_score,
        total_distance_km=round(total_distance_km, 2),
        total_duration_minutes=result.total_duration_min,
        num_visits=len(stops),
        stops=stops,
        skipped=skipped,
        optimizer_status=result.status,
    )