"""Integration tests for FastAPI API."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthCheck:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()


class TestDoctorsAPI:
    def test_list_doctors(self, client):
        r = client.get("/api/v1/doctors/?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_filter_by_specialty(self, client):
        r = client.get("/api/v1/doctors/?specialty=Cardiologist&limit=5")
        assert r.status_code == 200
        for doc in r.json():
            assert doc["specialty"] == "Cardiologist"

    def test_filter_by_category(self, client):
        r = client.get("/api/v1/doctors/?category=A&limit=5")
        assert r.status_code == 200
        for doc in r.json():
            assert doc["category"] == "A"

    def test_doctor_not_found(self, client):
        r = client.get("/api/v1/doctors/nonexistent-id")
        assert r.status_code == 404


class TestRouteGeneration:
    def test_generate_route(self, client):
        r = client.post("/api/v1/routes/generate", json={
            "rep_id": "REP-001",
            "latitude": 55.7558,
            "longitude": 37.6173,
            "target_date": "2025-06-18",
            "max_visits": 10,
        })
        # Может быть 200 или 503 (если модели не загружены)
        assert r.status_code in (200, 503)

        if r.status_code == 200:
            data = r.json()
            assert "route_id" in data
            assert "stops" in data
            assert data["num_visits"] <= 10
            assert data["optimizer_status"] in ("OPTIMAL", "FEASIBLE", "NO_SOLUTION")


class TestReportSubmission:
    def test_submit_report(self, client):
        # Сначала получим ID врача
        docs = client.get("/api/v1/doctors/?limit=1").json()
        if not docs:
            pytest.skip("No doctors in database")

        doctor_id = docs[0]["id"]

        r = client.post("/api/v1/reports/submit", json={
            "rep_id": "REP-001",
            "doctor_id": doctor_id,
            "visit_date": "2025-06-18",
            "visit_time": "10:30",
            "duration_minutes": 20,
            "status": "Success",
            "report_text": "Визит к врачу. Обсудили Лориста. Врач заинтересован.",
        })
        assert r.status_code == 200
        data = r.json()
        assert "visit_id" in data
        assert data["sentiment"] in ("Positive", "Neutral", "Negative")
        assert "llm_backend" in data