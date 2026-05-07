"""
PharmaPath AI — Geo Utilities
==============================
Haversine + Distance/Time Matrix для OR-Tools.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

EARTH_RADIUS_KM = 6_371.0


def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Расстояние между двумя точками (км)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def build_distance_matrix_km(
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Матрица попарных расстояний (км).
    points[0] = depot, points[1:] = doctors.
    """
    n = len(points)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(
                points[i][0], points[i][1],
                points[j][0], points[j][1],
            )
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def build_time_matrix_minutes(
    points: List[Tuple[float, float]],
    avg_speed_kmh: float = 18.0,
) -> np.ndarray:
    """
    Матрица времени в пути (минуты, целые числа).
    
    avg_speed_kmh=18 — средняя скорость по Москве в рабочее время
    (включает пробки, парковку, переход).
    """
    dist_km = build_distance_matrix_km(points)
    time_hours = dist_km / avg_speed_kmh
    time_minutes = np.ceil(time_hours * 60).astype(int)
    return time_minutes