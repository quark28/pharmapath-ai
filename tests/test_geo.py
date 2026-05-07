"""Tests for geo utilities."""

import numpy as np
import pytest

from src.services.geo import (
    build_distance_matrix_km,
    build_time_matrix_minutes,
    haversine_km,
)


class TestHaversine:
    def test_same_point(self):
        d = haversine_km(55.7558, 37.6173, 55.7558, 37.6173)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_moscow_to_sheremetyevo(self):
        # Москва центр → Шереметьево ≈ 28 км
        d = haversine_km(55.7558, 37.6173, 55.9726, 37.4146)
        assert 25 < d < 35

    def test_symmetry(self):
        d1 = haversine_km(55.0, 37.0, 56.0, 38.0)
        d2 = haversine_km(56.0, 38.0, 55.0, 37.0)
        assert d1 == pytest.approx(d2, rel=1e-9)


class TestDistanceMatrix:
    def test_shape(self):
        points = [(55.7, 37.6), (55.8, 37.7), (55.9, 37.8)]
        m = build_distance_matrix_km(points)
        assert m.shape == (3, 3)

    def test_diagonal_zero(self):
        points = [(55.7, 37.6), (55.8, 37.7)]
        m = build_distance_matrix_km(points)
        assert m[0, 0] == 0
        assert m[1, 1] == 0

    def test_symmetric(self):
        points = [(55.7, 37.6), (55.8, 37.7), (55.9, 37.8)]
        m = build_distance_matrix_km(points)
        np.testing.assert_array_almost_equal(m, m.T)


class TestTimeMatrix:
    def test_speed_relationship(self):
        points = [(55.7, 37.6), (55.8, 37.7)]
        # Быстрее ехать → меньше время
        t_fast = build_time_matrix_minutes(points, avg_speed_kmh=60.0)
        t_slow = build_time_matrix_minutes(points, avg_speed_kmh=15.0)
        assert t_fast[0, 1] < t_slow[0, 1]

    def test_integer_output(self):
        points = [(55.7, 37.6), (55.8, 37.7)]
        t = build_time_matrix_minutes(points)
        assert t.dtype in (np.int32, np.int64)