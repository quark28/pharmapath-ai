"""
PharmaPath AI — API Client
============================
Обёртка над httpx для взаимодействия со бэкендом.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

DEFAULT_BASE = "http://localhost:8000/api/v1"
TIMEOUT = 30.0


def _base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_BASE)


def _get(path: str, params: Optional[Dict] = None) -> Any:
    """GET-запрос к бэкенду."""
    try:
        r = httpx.get(f"{_base_url()}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Бэкенд недоступен. Запустите: `uvicorn src.main:app --port 8000`")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text}")
        return None


def _post(path: str, json_data: Dict) -> Any:
    """POST-запрос к бэкенду."""
    try:
        r = httpx.post(f"{_base_url()}{path}", json=json_data, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Бэкенд недоступен. Запустите: `uvicorn src.main:app --port 8000`")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def health_check() -> Optional[Dict]:
    """Проверка работоспособности бэкенда."""
    try:
        base = _base_url().replace("/api/v1", "")
        r = httpx.get(f"{base}/health", timeout=5.0)
        return r.json()
    except Exception:
        return None


def get_doctors(
    specialty: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Optional[List[Dict]]:
    """Получить список врачей."""
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if specialty:
        params["specialty"] = specialty
    if category:
        params["category"] = category
    return _get("/doctors/", params)


def get_doctor(doctor_id: str) -> Optional[Dict]:
    """Получить одного врача."""
    return _get(f"/doctors/{doctor_id}")


def generate_route(
    rep_id: str,
    latitude: float,
    longitude: float,
    target_date: date,
    max_visits: int = 14,
    visited_ids: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Сгенерировать маршрут."""
    payload = {
        "rep_id": rep_id,
        "latitude": latitude,
        "longitude": longitude,
        "target_date": target_date.isoformat(),
        "max_visits": max_visits,
        "visited_ids": visited_ids or [],
    }
    return _post("/routes/generate", payload)


def submit_report(
    rep_id: str,
    doctor_id: str,
    visit_time: str,
    duration_minutes: int,
    status: str,
    report_text: str,
    visit_date: Optional[date] = None,
) -> Optional[Dict]:
    """Отправить отчёт о визите."""
    payload = {
        "rep_id": rep_id,
        "doctor_id": doctor_id,
        "visit_date": (visit_date or date.today()).isoformat(),
        "visit_time": visit_time,
        "duration_minutes": duration_minutes,
        "status": status,
        "report_text": report_text,
    }
    return _post("/reports/submit", payload)