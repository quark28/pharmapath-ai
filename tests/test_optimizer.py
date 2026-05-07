"""Tests for Route Optimizer."""

import pytest

from src.services.optimizer import CandidateDoctor, RouteOptimizer


@pytest.fixture
def optimizer():
    return RouteOptimizer(
        avg_speed_kmh=18.0,
        solver_time_limit_sec=3,
        max_waiting_min=30,
    )


@pytest.fixture
def sample_candidates():
    """5 врачей вокруг центра Москвы."""
    return [
        CandidateDoctor("d1", 55.760, 37.620, 80.0, (540, 1020), 20),
        CandidateDoctor("d2", 55.750, 37.630, 65.0, (540, 1020), 20),
        CandidateDoctor("d3", 55.770, 37.610, 50.0, (600, 960), 20),
        CandidateDoctor("d4", 55.740, 37.640, 45.0, (540, 1020), 20),
        CandidateDoctor("d5", 55.780, 37.600, 30.0, (540, 1020), 20),
    ]


class TestRouteOptimizer:
    def test_basic_optimization(self, optimizer, sample_candidates):
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=sample_candidates,
            max_visits=14,
        )
        assert result.status in ("OPTIMAL", "FEASIBLE")
        assert len(result.stops) > 0
        assert len(result.stops) <= 5
        assert result.total_score > 0

    def test_max_visits_limit(self, optimizer, sample_candidates):
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=sample_candidates,
            max_visits=2,
        )
        assert len(result.stops) <= 2

    def test_empty_candidates(self, optimizer):
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=[],
        )
        assert result.status == "NO_SOLUTION"
        assert len(result.stops) == 0

    def test_no_time_window(self, optimizer):
        candidates = [
            CandidateDoctor("d1", 55.76, 37.62, 80.0, None, 20),
        ]
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=candidates,
        )
        assert result.status == "NO_SOLUTION"

    def test_stops_ordered(self, optimizer, sample_candidates):
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=sample_candidates,
        )
        orders = [s.order for s in result.stops]
        assert orders == sorted(orders)

    def test_arrivals_increasing(self, optimizer, sample_candidates):
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=sample_candidates,
        )
        arrivals = [s.arrival_min for s in result.stops]
        for i in range(1, len(arrivals)):
            assert arrivals[i] >= arrivals[i - 1]

    def test_high_score_included(self, optimizer, sample_candidates):
        """Врач с наивысшим score должен попасть в маршрут."""
        result = optimizer.optimize(
            depot_lat=55.7558,
            depot_lon=37.6173,
            candidates=sample_candidates,
            max_visits=3,
        )
        visited_ids = {s.doctor_id for s in result.stops}
        assert "d1" in visited_ids  # highest score = 80