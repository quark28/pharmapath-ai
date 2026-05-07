"""
PharmaPath AI — Route Optimizer (OR-Tools)
===========================================
Решает Prize-Collecting VRPTW:
  - Максимизировать суммарный Score посещённых врачей
  - Уложиться в рабочий день (8 ч.)
  - Уважать окна приёма каждого врача
  - Не превышать лимит визитов

Математика:
  OR-Tools минимизирует: Σ(travel_time) + Σ(penalty за пропущенных).
  penalty_i = combined_score_i × SCALE  →  солвер «не хочет» пропускать.
  Time Windows  →  можно приехать только в рабочие часы врача.
  Capacity      →  максимум N визитов.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.services.geo import build_time_matrix_minutes

logger = structlog.get_logger()

# Масштаб штрафа: penalty = score × PENALTY_SCALE
# Должен быть >> типичного travel_time (5–30 мин), чтобы солвер
# предпочитал включать врачей, а не экономить на переездах.
PENALTY_SCALE = 100


@dataclass
class CandidateDoctor:
    """Входные данные для оптимизатора — один врач-кандидат."""
    doctor_id: str
    lat: float
    lon: float
    combined_score: float
    time_window: Optional[Tuple[int, int]]   # (start_min, end_min) от полуночи
    visit_duration: int = 20                  # минут


@dataclass
class RouteStopResult:
    """Один пункт оптимального маршрута."""
    doctor_id: str
    order: int
    arrival_min: int       # минуты от полуночи
    departure_min: int
    travel_from_prev_min: int


@dataclass
class OptimizationResult:
    """Результат работы оптимизатора."""
    status: str                              # OPTIMAL / FEASIBLE / NO_SOLUTION
    stops: List[RouteStopResult]
    total_travel_min: int
    total_score: float
    total_duration_min: int
    skipped_ids: List[str]
    solver_wall_time_ms: int


class RouteOptimizer:
    """
    Prize-Collecting TSP with Time Windows (одно ТС, один маршрут).
    
    Используем OR-Tools Routing Library:
    1. Строим матрицу времени в пути (Haversine → мин.).
    2. Создаём Routing Model с 1 ТС и depot = текущая позиция медпреда.
    3. Добавляем Time Dimension с окнами приёма.
    4. Каждый врач — опциональный (disjunction, penalty ∝ score).
    5. Солвим → извлекаем маршрут.
    """

    def __init__(
        self,
        avg_speed_kmh: float = 18.0,
        solver_time_limit_sec: int = 5,
        max_waiting_min: int = 45,
    ):
        self.avg_speed_kmh = avg_speed_kmh
        self.solver_time_limit_sec = solver_time_limit_sec
        self.max_waiting_min = max_waiting_min

    def optimize(
        self,
        depot_lat: float,
        depot_lon: float,
        candidates: List[CandidateDoctor],
        day_start_min: int = 540,     # 09:00
        day_end_min: int = 1080,      # 18:00
        max_visits: int = 14,
    ) -> OptimizationResult:
        """
        Запуск оптимизации маршрута.

        Parameters
        ----------
        depot_lat, depot_lon : координаты старта (GPS медпреда).
        candidates : предварительно отобранные врачи (Top-N по score).
        day_start_min : начало рабочего дня (мин. от полуночи).
        day_end_min : конец рабочего дня.
        max_visits : жёсткий лимит визитов.

        Returns
        -------
        OptimizationResult
        """
        t0 = time_module.perf_counter()

        # ── Фильтруем врачей без окна приёма ──────────────────────────────
        valid = [c for c in candidates if c.time_window is not None]
        if not valid:
            return OptimizationResult(
                status="NO_SOLUTION",
                stops=[],
                total_travel_min=0,
                total_score=0.0,
                total_duration_min=0,
                skipped_ids=[c.doctor_id for c in candidates],
                solver_wall_time_ms=0,
            )

        logger.info(
            "Optimizer input",
            total_candidates=len(candidates),
            valid_candidates=len(valid),
            max_visits=max_visits,
        )

        # ── Подготовка данных ─────────────────────────────────────────────
        # index 0 = depot, indices 1..N = doctors
        num_locations = len(valid) + 1
        num_vehicles = 1
        depot_index = 0

        # Координаты: [depot, doc1, doc2, ...]
        points = [(depot_lat, depot_lon)]
        scores = [0.0]               # depot score = 0
        service_times = [0]          # depot service = 0
        time_windows = [(day_start_min, day_end_min)]   # depot window

        for c in valid:
            points.append((c.lat, c.lon))
            scores.append(c.combined_score)
            service_times.append(c.visit_duration)

            # Обрезаем окно врача границами рабочего дня
            tw_start = max(c.time_window[0], day_start_min)
            tw_end = min(c.time_window[1], day_end_min)

            # Последний визит должен закончиться до end:
            # arrival + service ≤ tw_end → arrival ≤ tw_end - service
            tw_end_adj = max(tw_start, tw_end - c.visit_duration)
            time_windows.append((tw_start, tw_end_adj))

        # Матрица времени в пути (мин., целые)
        time_matrix = build_time_matrix_minutes(points, self.avg_speed_kmh)

        # ── OR-Tools Model ────────────────────────────────────────────────
        manager = pywrapcp.RoutingIndexManager(
            num_locations, num_vehicles, depot_index,
        )
        routing = pywrapcp.RoutingModel(manager)

        # --- Transit Callback (travel + service at origin) ---
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel = int(time_matrix[from_node][to_node])
            service = service_times[from_node]
            return travel + service

        transit_cb_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

        # --- Time Dimension ---
        routing.AddDimension(
            transit_cb_index,
            self.max_waiting_min,   # slack (ожидание у кабинета)
            day_end_min + 60,       # horizon (с запасом на travel home)
            False,                  # don't fix start cumul
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # Depot: старт в [day_start, day_start] (фиксированный)
        start_index = routing.Start(0)
        time_dimension.CumulVar(start_index).SetRange(
            day_start_min, day_start_min,
        )

        # Окна для каждого врача
        for loc_idx in range(1, num_locations):
            routing_index = manager.NodeToIndex(loc_idx)
            tw = time_windows[loc_idx]
            time_dimension.CumulVar(routing_index).SetRange(tw[0], tw[1])

        # Минимизируем разброс (total span)
        time_dimension.SetGlobalSpanCostCoefficient(1)

        # --- Disjunctions (каждый врач опционален) ---
        for loc_idx in range(1, num_locations):
            routing_index = manager.NodeToIndex(loc_idx)
            penalty = int(scores[loc_idx] * PENALTY_SCALE)
            routing.AddDisjunction([routing_index], penalty)

        # --- Counter Dimension (лимит визитов) ---
        def count_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            # Каждый переход от НЕ-depot = 1 визит
            return 1 if from_node != depot_index else 0

        count_cb_index = routing.RegisterTransitCallback(count_callback)
        routing.AddDimension(
            count_cb_index,
            0,                  # no slack
            max_visits,         # capacity
            True,               # start at 0
            "Visits",
        )

        # --- Search Parameters ---
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.FromSeconds(self.solver_time_limit_sec)

        # ── Solve ─────────────────────────────────────────────────────────
        solution = routing.SolveWithParameters(search_params)
        wall_ms = int((time_module.perf_counter() - t0) * 1000)

        if solution is None:
            logger.warning("OR-Tools: no solution found")
            return OptimizationResult(
                status="NO_SOLUTION",
                stops=[],
                total_travel_min=0,
                total_score=0.0,
                total_duration_min=0,
                skipped_ids=[c.doctor_id for c in valid],
                solver_wall_time_ms=wall_ms,
            )

        # ── Extract Solution ──────────────────────────────────────────────
        status_code = routing.status()
        status_map = {
            1: "OPTIMAL",
            2: "FEASIBLE",
            3: "NO_SOLUTION",
            4: "FAIL",
        }
        status = status_map.get(status_code, f"UNKNOWN({status_code})")

        stops: List[RouteStopResult] = []
        visited_set: set = set()
        total_travel = 0
        total_score = 0.0

        index = routing.Start(0)
        order = 0
        prev_arrival = day_start_min

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            if node != depot_index:
                arrival = solution.Min(time_dimension.CumulVar(index))
                travel_from_prev = arrival - prev_arrival - (
                    service_times[manager.IndexToNode(
                        solution.Value(routing.NextVar(
                            # find previous index... complex
                            # simplified: use time matrix
                            index
                        ))
                    )] if False else 0
                )

                doc_idx = node  # node в valid list (1-indexed)
                doctor = valid[doc_idx - 1]
                departure = arrival + doctor.visit_duration

                stops.append(RouteStopResult(
                    doctor_id=doctor.doctor_id,
                    order=order,
                    arrival_min=arrival,
                    departure_min=departure,
                    travel_from_prev_min=max(0, arrival - prev_arrival),
                ))
                visited_set.add(doctor.doctor_id)
                total_score += doctor.combined_score
                prev_arrival = departure
                order += 1

            index = next_index

        # Подсчёт travel time
        total_travel = 0
        for i, stop in enumerate(stops):
            if i == 0:
                dep_node = 0
                doc_node = next(
                    j + 1 for j, c in enumerate(valid)
                    if c.doctor_id == stop.doctor_id
                )
                total_travel += int(time_matrix[dep_node][doc_node])
            else:
                prev_doc_node = next(
                    j + 1 for j, c in enumerate(valid)
                    if c.doctor_id == stops[i - 1].doctor_id
                )
                cur_doc_node = next(
                    j + 1 for j, c in enumerate(valid)
                    if c.doctor_id == stop.doctor_id
                )
                total_travel += int(time_matrix[prev_doc_node][cur_doc_node])

        total_duration = 0
        if stops:
            total_duration = stops[-1].departure_min - day_start_min

        skipped = [c.doctor_id for c in valid if c.doctor_id not in visited_set]

        logger.info(
            "Optimization complete",
            status=status,
            visits=len(stops),
            total_score=round(total_score, 1),
            total_travel_min=total_travel,
            wall_ms=wall_ms,
        )

        return OptimizationResult(
            status=status,
            stops=stops,
            total_travel_min=total_travel,
            total_score=round(total_score, 2),
            total_duration_min=total_duration,
            skipped_ids=skipped,
            solver_wall_time_ms=wall_ms,
        )


def minutes_to_time_str(minutes: int) -> str:
    """570 → '09:30'"""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"