"""
PharmaPath AI — Data Provider
==============================
Загружает данные из CSV в память. 
В продакшне заменяется на PostgreSQL + PostGIS.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

# ── Названия дней (для парсинга schedule_json) ────────────────────────────────
_WEEKDAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


class DataProvider:
    """
    In-memory data layer.
    
    При старте загружает CSV-файлы в DataFrames.
    Предоставляет методы для чтения и записи данных.
    
    Thread-safety: для MVP достаточно; в проде → PostgreSQL.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._doctors: pd.DataFrame = pd.DataFrame()
        self._visits: pd.DataFrame = pd.DataFrame()
        self._reps: pd.DataFrame = pd.DataFrame()
        self._load()

    # ══════════════════════════════════════════════════════════════════════
    #  LOADING
    # ══════════════════════════════════════════════════════════════════════

    def _load(self) -> None:
        """Загрузить все CSV."""
        docs_path = self._data_dir / "doctors_base.csv"
        visits_path = self._data_dir / "visits_log.csv"
        reps_path = self._data_dir / "reps.csv"

        if not docs_path.exists():
            logger.warning("doctors_base.csv not found", path=str(docs_path))
            return

        # Doctors
        self._doctors = pd.read_csv(docs_path, dtype={"id": str})
        logger.info("Doctors loaded", count=len(self._doctors))

        # Visits
        if visits_path.exists():
            self._visits = pd.read_csv(
                visits_path,
                dtype={"id": str, "doctor_id": str, "rep_id": str},
            )
            self._visits["visit_date"] = pd.to_datetime(self._visits["visit_date"])
            logger.info("Visits loaded", count=len(self._visits))

        # Reps
        if reps_path.exists():
            self._reps = pd.read_csv(reps_path, dtype={"id": str})
            logger.info("Reps loaded", count=len(self._reps))

    # ══════════════════════════════════════════════════════════════════════
    #  DOCTORS
    # ══════════════════════════════════════════════════════════════════════

    @property
    def doctors_count(self) -> int:
        return len(self._doctors)

    @property
    def visits_count(self) -> int:
        return len(self._visits)

    def get_all_doctors(self) -> pd.DataFrame:
        return self._doctors.copy()

    def get_doctor(self, doctor_id: str) -> Optional[pd.Series]:
        mask = self._doctors["id"] == doctor_id
        if mask.any():
            return self._doctors.loc[mask].iloc[0]
        return None

    def get_rep(self, rep_id: str) -> Optional[pd.Series]:
        mask = self._reps["id"] == rep_id
        if mask.any():
            return self._reps.loc[mask].iloc[0]
        return None

    # ══════════════════════════════════════════════════════════════════════
    #  VISIT STATS (агрегаты для каждого врача)
    # ══════════════════════════════════════════════════════════════════════

    def get_visit_stats(
        self,
        doctor_id: str,
        reference_date: Optional[date] = None,
    ) -> Dict:
        """
        Статистика визитов для одного врача.
        Используется scoring-сервисом для построения фичей.
        """
        if reference_date is None:
            reference_date = date.today()

        ref_dt = pd.Timestamp(reference_date)
        doc_visits = self._visits[self._visits["doctor_id"] == doctor_id]

        if doc_visits.empty:
            return {
                "days_since_last_visit": 999,
                "total_visits": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "visits_last_90d": 0,
                "visits_last_30d": 0,
            }

        is_success = doc_visits["status"] == "Success"
        total = len(doc_visits)
        success_count = int(is_success.sum())
        last_visit = doc_visits["visit_date"].max()
        durations = doc_visits.loc[
            doc_visits["duration_minutes"] > 0, "duration_minutes"
        ]

        cutoff_90 = ref_dt - pd.Timedelta(days=90)
        cutoff_30 = ref_dt - pd.Timedelta(days=30)

        return {
            "days_since_last_visit": max(0, (ref_dt - last_visit).days),
            "total_visits": total,
            "success_rate": round(success_count / total, 3) if total > 0 else 0.0,
            "avg_duration": round(float(durations.mean()), 1) if len(durations) > 0 else 0.0,
            "visits_last_90d": int((doc_visits["visit_date"] >= cutoff_90).sum()),
            "visits_last_30d": int((doc_visits["visit_date"] >= cutoff_30).sum()),
        }

    def get_bulk_visit_stats(
        self,
        reference_date: Optional[date] = None,
    ) -> Dict[str, Dict]:
        """
        Статистика визитов для ВСЕХ врачей (batch — быстрее, чем поштучно).
        """
        if reference_date is None:
            reference_date = date.today()

        ref_dt = pd.Timestamp(reference_date)
        v = self._visits.copy()

        if v.empty:
            return {}

        v["is_success"] = (v["status"] == "Success").astype(int)

        agg = v.groupby("doctor_id").agg(
            total_visits=("id", "count"),
            success_count=("is_success", "sum"),
            last_visit=("visit_date", "max"),
            avg_duration=("duration_minutes", lambda x: x[x > 0].mean()),
        )
        agg["success_rate"] = (agg["success_count"] / agg["total_visits"]).round(3)
        agg["days_since_last_visit"] = (ref_dt - agg["last_visit"]).dt.days.clip(lower=0)

        cutoff_90 = ref_dt - pd.Timedelta(days=90)
        cutoff_30 = ref_dt - pd.Timedelta(days=30)
        v90 = v[v["visit_date"] >= cutoff_90].groupby("doctor_id").size()
        v30 = v[v["visit_date"] >= cutoff_30].groupby("doctor_id").size()

        agg["visits_last_90d"] = v90.reindex(agg.index, fill_value=0).astype(int)
        agg["visits_last_30d"] = v30.reindex(agg.index, fill_value=0).astype(int)

        result = {}
        for doc_id, row in agg.iterrows():
            result[doc_id] = {
                "days_since_last_visit": int(row["days_since_last_visit"]),
                "total_visits": int(row["total_visits"]),
                "success_rate": float(row["success_rate"]),
                "avg_duration": round(float(row["avg_duration"]), 1)
                    if not pd.isna(row["avg_duration"]) else 0.0,
                "visits_last_90d": int(row["visits_last_90d"]),
                "visits_last_30d": int(row["visits_last_30d"]),
            }
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  SCHEDULE PARSING
    # ══════════════════════════════════════════════════════════════════════

    def get_doctor_time_window(
        self,
        doctor_id: str,
        target_date: date,
    ) -> Optional[Tuple[int, int]]:
        """
        Получить окно приёма врача в минутах от полуночи.
        
        Returns (start_min, end_min) или None, если врач не работает.
        """
        doc = self.get_doctor(doctor_id)
        if doc is None:
            return None

        schedule = self._parse_schedule(doc.get("schedule_json", "{}"))
        day_name = _WEEKDAY_NAMES[target_date.weekday()]

        if day_name not in schedule:
            return None

        slot = schedule[day_name]
        start_min = self._time_to_minutes(slot["start"])
        end_min = self._time_to_minutes(slot["end"])
        return (start_min, end_min)

    def get_bulk_time_windows(
        self,
        doctor_ids: List[str],
        target_date: date,
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """Окна приёма для списка врачей (batch)."""
        day_name = _WEEKDAY_NAMES[target_date.weekday()]
        result = {}

        for doc_id in doctor_ids:
            doc = self.get_doctor(doc_id)
            if doc is None:
                result[doc_id] = None
                continue

            schedule = self._parse_schedule(doc.get("schedule_json", "{}"))
            if day_name not in schedule:
                result[doc_id] = None
                continue

            slot = schedule[day_name]
            result[doc_id] = (
                self._time_to_minutes(slot["start"]),
                self._time_to_minutes(slot["end"]),
            )
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  WRITES
    # ══════════════════════════════════════════════════════════════════════

    def add_visit(
        self,
        visit_id: str,
        doctor_id: str,
        rep_id: str,
        visit_date: date,
        visit_time: str,
        status: str,
        duration_minutes: int,
        report_text: str,
    ) -> None:
        """Добавить визит в историю."""
        new_row = pd.DataFrame([{
            "id": visit_id,
            "doctor_id": doctor_id,
            "rep_id": rep_id,
            "visit_date": pd.Timestamp(visit_date),
            "visit_time": visit_time,
            "day_of_week": visit_date.weekday(),
            "status": status,
            "duration_minutes": duration_minutes,
            "report_text": report_text,
        }])
        self._visits = pd.concat([self._visits, new_row], ignore_index=True)
        logger.info("Visit added", visit_id=visit_id, doctor_id=doctor_id)

    def replace_doctors(self, df: pd.DataFrame) -> int:
        """Заменить базу врачей (загрузка нового CSV)."""
        self._doctors = df.copy()
        logger.info("Doctors base replaced", count=len(df))
        return len(df)

    # ══════════════════════════════════════════════════════════════════════
    #  PRIVATE
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_schedule(raw: str) -> dict:
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _time_to_minutes(t: str) -> int:
        """'09:30' → 570"""
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])