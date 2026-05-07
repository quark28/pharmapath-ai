"""Tests for ML Scoring Service."""

import pytest
from datetime import date
from pathlib import Path

from src.services.scoring import ScoringService


@pytest.fixture
def scoring_service():
    """Инициализация scoring service."""
    model_dir = Path(__file__).parent.parent / "models"
    service = ScoringService(model_dir)
    service.load_models()
    return service


class TestScoringService:
    def test_models_loaded(self, scoring_service):
        """Проверка загрузки моделей."""
        assert scoring_service.is_loaded is True

    def test_score_single_doctor(self, scoring_service):
        """Скоринг одного врача."""
        doctor_row = {
            "id": "test-doctor-001",
            "specialty": "Cardiologist",
            "category": "A",
            "avg_sales_brick": 85.0,
            "loyalty_score": 8.0,
        }
        visit_stats = {
            "days_since_last_visit": 45,
            "total_visits": 10,
            "success_rate": 0.8,
            "avg_duration": 22.0,
            "visits_last_90d": 3,
            "visits_last_30d": 1,
        }

        result = scoring_service.score_single_doctor(
            doctor_row=doctor_row,
            visit_stats=visit_stats,
            target_date=date.today(),
            target_hour=10,
        )

        assert result.doctor_id == "test-doctor-001"
        assert 0 <= result.value_score <= 100
        assert 0 <= result.probability_score <= 1
        assert result.combined_score == pytest.approx(
            result.value_score * result.probability_score, rel=0.01
        )

    def test_value_score_range(self, scoring_service):
        """Value score должен быть в диапазоне 0-100."""
        doctor_row = {
            "id": "test-doctor-002",
            "specialty": "Therapist",
            "category": "C",
            "avg_sales_brick": 20.0,
            "loyalty_score": 3.0,
        }
        visit_stats = {
            "days_since_last_visit": 999,
            "total_visits": 0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "visits_last_90d": 0,
            "visits_last_30d": 0,
        }

        result = scoring_service.score_single_doctor(
            doctor_row=doctor_row,
            visit_stats=visit_stats,
            target_date=date.today(),
            target_hour=14,
        )

        assert 0 <= result.value_score <= 100

    def test_probability_score_range(self, scoring_service):
        """Probability score должен быть в диапазоне 0-1."""
        doctor_row = {
            "id": "test-doctor-003",
            "specialty": "Neurologist",
            "category": "B",
            "avg_sales_brick": 50.0,
            "loyalty_score": 5.0,
        }
        visit_stats = {
            "days_since_last_visit": 30,
            "total_visits": 5,
            "success_rate": 0.6,
            "avg_duration": 18.0,
            "visits_last_90d": 2,
            "visits_last_30d": 1,
        }

        result = scoring_service.score_single_doctor(
            doctor_row=doctor_row,
            visit_stats=visit_stats,
            target_date=date.today(),
            target_hour=11,
        )

        assert 0 <= result.probability_score <= 1

    def test_category_a_higher_score(self, scoring_service):
        """Категория A должна давать выше score, чем C."""
        base_stats = {
            "days_since_last_visit": 30,
            "total_visits": 5,
            "success_rate": 0.7,
            "avg_duration": 20.0,
            "visits_last_90d": 2,
            "visits_last_30d": 1,
        }

        doctor_a = {
            "id": "doc-a",
            "specialty": "Cardiologist",
            "category": "A",
            "avg_sales_brick": 80.0,
            "loyalty_score": 8.0,
        }
        doctor_c = {
            "id": "doc-c",
            "specialty": "Cardiologist",
            "category": "C",
            "avg_sales_brick": 20.0,
            "loyalty_score": 3.0,
        }

        result_a = scoring_service.score_single_doctor(
            doctor_a, base_stats, date.today(), 10
        )
        result_c = scoring_service.score_single_doctor(
            doctor_c, base_stats, date.today(), 10
        )

        assert result_a.value_score > result_c.value_score

    def test_feature_contributions_not_empty(self, scoring_service):
        """Feature contributions должны быть непустыми."""
        doctor_row = {
            "id": "test-doctor-004",
            "specialty": "Therapist",
            "category": "A",
            "avg_sales_brick": 90.0,
            "loyalty_score": 9.0,
        }
        visit_stats = {
            "days_since_last_visit": 60,
            "total_visits": 15,
            "success_rate": 0.85,
            "avg_duration": 25.0,
            "visits_last_90d": 4,
            "visits_last_30d": 2,
        }

        result = scoring_service.score_single_doctor(
            doctor_row=doctor_row,
            visit_stats=visit_stats,
            target_date=date.today(),
            target_hour=10,
        )

        assert len(result.feature_contributions) > 0
        assert "category" in result.feature_contributions