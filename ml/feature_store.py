#!/usr/bin/env python3
"""
PharmaPath AI — Feature Store
==============================
Единый модуль для инженерии признаков.
Используется и в ноутбуках, и в бэкенде (inference).

Принцип: «Train–Serve Skew = 0».
Один и тот же код генерит фичи при обучении и при продакшн-запросе.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_ORDER = ["C", "B", "A"]

SPECIALTY_LIST = [
    "Therapist", "Cardiologist", "Neurologist", "Endocrinologist",
    "Gastroenterologist", "Pulmonologist", "Rheumatologist", "Urologist",
]

# Коэффициенты для синтетического target (Value Model)
# В реальности — калибруются экспертами / историческими данными
CATEGORY_BASE_VALUE = {"A": 75.0, "B": 45.0, "C": 18.0}

SPECIALTY_MULTIPLIER = {
    "Cardiologist":       1.30,
    "Neurologist":        1.15,
    "Endocrinologist":    1.20,
    "Gastroenterologist": 1.10,
    "Pulmonologist":      1.05,
    "Rheumatologist":     0.95,
    "Urologist":          0.90,
    "Therapist":          1.00,
}

# Коэффициенты для синтетического target (Probability Model)
HOUR_LOGIT = {
    8: -0.3, 9: 0.5, 10: 1.0, 11: 1.2, 12: 0.3,
    13: 0.1, 14: 0.6, 15: 0.5, 16: 0.0, 17: -0.5,
    18: -1.0, 19: -1.5, 20: -2.0,
}
DAY_LOGIT = {
    0: 0.4,   # Mon
    1: 0.5,   # Tue
    2: 0.5,   # Wed
    3: 0.3,   # Thu
    4: -0.6,  # Fri
    5: -1.5,  # Sat
    6: -2.5,  # Sun
}
SPECIALTY_AVAILABILITY_LOGIT = {
    "Therapist":          0.5,
    "Cardiologist":      -0.2,
    "Neurologist":        0.0,
    "Endocrinologist":    0.1,
    "Gastroenterologist": 0.2,
    "Pulmonologist":     -0.1,
    "Rheumatologist":     0.3,
    "Urologist":         -0.3,
}

# Названия фичей для моделей
VALUE_FEATURES = [
    "specialty",
    "category",
    "avg_sales_brick",
    "loyalty_score",
    "days_since_last_visit",
    "total_visits",
    "success_rate",
    "avg_duration",
    "visits_last_90d",
    "trend_loyalty",         # loyalty * success_rate — мультипликативный сигнал
]
VALUE_CAT_FEATURES = ["specialty", "category"]

PROB_FEATURES = [
    "specialty",
    "category",
    "day_of_week",
    "hour",
    "is_morning",
    "is_friday",
    "is_weekend",
    "loyalty_score",
    "days_since_prev_visit",
    "month",
    "visits_last_30d",
    "historical_success_rate",
]
PROB_CAT_FEATURES = ["specialty", "category", "day_of_week", "month"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_doctors(path: str | Path) -> pd.DataFrame:
    """Загрузить базу врачей."""
    df = pd.read_csv(path, dtype={"id": str})
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)
    df["specialty"] = df["specialty"].astype(str)
    return df


def load_visits(path: str | Path) -> pd.DataFrame:
    """Загрузить историю визитов."""
    df = pd.read_csv(path, dtype={"id": str, "doctor_id": str, "rep_id": str})
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["visit_hour"] = df["visit_time"].str.split(":").str[0].astype(int)
    df["is_success"] = (df["status"] == "Success").astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DOCTOR-LEVEL FEATURES (для Value Model)
# ══════════════════════════════════════════════════════════════════════════════

def build_doctor_features(
    doctors: pd.DataFrame,
    visits: pd.DataFrame,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Собрать агрегатные фичи для каждого врача.

    Parameters
    ----------
    doctors : DataFrame с doctors_base.csv
    visits  : DataFrame с visits_log.csv
    reference_date : «сегодня» (по умолчанию — max(visit_date))

    Returns
    -------
    DataFrame с одной строкой на врача + все фичи + target.
    """
    if reference_date is None:
        reference_date = visits["visit_date"].max().date()

    ref_dt = pd.Timestamp(reference_date)

    # ── Агрегация визитов ─────────────────────────────────────────────────
    agg = visits.groupby("doctor_id").agg(
        total_visits=("id", "count"),
        success_count=("is_success", "sum"),
        last_visit_date=("visit_date", "max"),
        first_visit_date=("visit_date", "min"),
        avg_duration=("duration_minutes", lambda x: x[x > 0].mean()),
    ).reset_index()

    agg["success_rate"] = (agg["success_count"] / agg["total_visits"]).round(3)
    agg["days_since_last_visit"] = (ref_dt - agg["last_visit_date"]).dt.days
    agg["months_as_contact"] = (
        (ref_dt - agg["first_visit_date"]).dt.days / 30.0
    ).round(1)

    # Визиты за последние 90 дней
    cutoff_90 = ref_dt - pd.Timedelta(days=90)
    v90 = visits[visits["visit_date"] >= cutoff_90].groupby("doctor_id").size()
    v90.name = "visits_last_90d"
    agg = agg.merge(v90, left_on="doctor_id", right_index=True, how="left")
    agg["visits_last_90d"] = agg["visits_last_90d"].fillna(0).astype(int)

    # ── Мерж с базой врачей ───────────────────────────────────────────────
    df = doctors.merge(agg, left_on="id", right_on="doctor_id", how="left")

    # Врачи без визитов
    df["total_visits"] = df["total_visits"].fillna(0).astype(int)
    df["success_rate"] = df["success_rate"].fillna(0.0)
    df["avg_duration"] = df["avg_duration"].fillna(0.0)
    df["visits_last_90d"] = df["visits_last_90d"].fillna(0).astype(int)
    df["days_since_last_visit"] = df["days_since_last_visit"].fillna(999).astype(int)

    # Мультипликативный сигнал
    df["trend_loyalty"] = (df["loyalty_score"] * df["success_rate"]).round(3)

    return df


def compute_value_target(
    df: pd.DataFrame,
    noise_std: float = 3.0,
    seed: int = 42,
) -> pd.Series:
    """
    Синтетический target: potential_value ∈ [0, 100].

    Формула (имитирует экспертную разметку):
        base = category_base_value[cat]
        value = base × spec_mult × loyalty_factor × sales_factor × recency_boost + ε

    В реальном проекте это было бы:
    - Размечено менеджерами,
    - Или посчитано из CRM-данных (выписка рецептов после визита).
    """
    rng = np.random.default_rng(seed)

    cat_base = df["category"].astype(str).map(CATEGORY_BASE_VALUE)
    spec_mult = df["specialty"].map(SPECIALTY_MULTIPLIER).fillna(1.0)

    loyalty_factor = 0.4 + 0.6 * (df["loyalty_score"] / 10.0)
    sales_factor = 0.5 + 0.5 * (df["avg_sales_brick"] / 100.0).clip(0, 1.5)

    # Recency boost: если не были > 21 дня, ценность растёт (до +60 %)
    days = df["days_since_last_visit"].clip(0, 200)
    recency_boost = 1.0 + 0.6 * _sigmoid((days - 21) / 12.0)

    value = cat_base * spec_mult * loyalty_factor * sales_factor * recency_boost
    noise = rng.normal(0, noise_std, size=len(df))

    return (value + noise).clip(0, 100).round(2)


# ══════════════════════════════════════════════════════════════════════════════
#  VISIT-LEVEL FEATURES (для Probability Model)
# ══════════════════════════════════════════════════════════════════════════════

def build_visit_features(
    visits: pd.DataFrame,
    doctors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Собрать фичи для каждого визита (для обучения Probability Model).
    """
    df = visits.copy()

    # Мерж с врачами
    doc_cols = ["id", "specialty", "category", "loyalty_score", "avg_sales_brick"]
    df = df.merge(
        doctors[doc_cols], left_on="doctor_id", right_on="id",
        how="left", suffixes=("", "_doc"),
    )

    # Временные фичи
    df["hour"] = df["visit_hour"]
    df["is_morning"] = (df["hour"] < 12).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["visit_date"].dt.month.astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)

    # Дней с предыдущего визита к этому врачу
    df = df.sort_values(["doctor_id", "visit_date"])
    df["prev_visit_date"] = df.groupby("doctor_id")["visit_date"].shift(1)
    df["days_since_prev_visit"] = (
        (df["visit_date"] - df["prev_visit_date"]).dt.days
    ).fillna(999).astype(int)

    # Визиты за последние 30 дней к этому врачу (rolling count)
    df["visits_last_30d"] = df.groupby("doctor_id")["visit_date"].transform(
        lambda s: s.apply(
            lambda d: ((s >= d - pd.Timedelta(days=30)) & (s < d)).sum()
        )
    )

    # Историческая success rate этого врача (expanding, без текущего)
    df["_cum_success"] = df.groupby("doctor_id")["is_success"].cumsum() - df["is_success"]
    df["_cum_count"] = df.groupby("doctor_id").cumcount()
    df["historical_success_rate"] = (
        df["_cum_success"] / df["_cum_count"].clip(lower=1)
    ).round(3).fillna(0.5)
    df.drop(columns=["_cum_success", "_cum_count"], inplace=True)

    return df


def relabel_visit_success(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.Series:
    """
    Пересэмплировать target на основе реалистичной функции вероятности.

    Зачем: в faker_generator.py статус был ~случайным (75 % Success).
    Здесь мы инжектим реальные паттерны (час дня, день недели, специальность),
    чтобы модель могла их выучить.

    Returns
    -------
    pd.Series[int] — бинарный target (1 = success, 0 = fail)
    """
    rng = np.random.default_rng(seed)

    intercept = 0.3

    hour_effect = df["hour"].astype(int).map(HOUR_LOGIT).fillna(0.0)
    day_effect = df["day_of_week"].astype(int).map(DAY_LOGIT).fillna(0.0)
    spec_effect = df["specialty"].map(SPECIALTY_AVAILABILITY_LOGIT).fillna(0.0)

    loyalty_effect = 0.15 * (df["loyalty_score"] - 5.0)

    recency_penalty = np.where(
        df["days_since_prev_visit"] < 7,
        -0.8,    # слишком частые визиты раздражают
        np.where(
            df["days_since_prev_visit"] > 60,
            -0.3,  # слишком давно — забыли
            0.0,
        ),
    )

    logit = intercept + hour_effect + day_effect + spec_effect + loyalty_effect + recency_penalty
    prob = _sigmoid(logit)

    return (rng.random(len(df)) < prob).astype(int), prob


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE-TIME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def make_inference_features_value(
    doctor_row: Dict[str, Any],
    visit_stats: Dict[str, Any],
) -> pd.DataFrame:
    """
    Собрать фичи для одного врача в момент inference (Value Model).
    Вызывается из бэкенда при генерации маршрута.
    """
    row = {
        "specialty":              doctor_row["specialty"],
        "category":               doctor_row["category"],
        "avg_sales_brick":        doctor_row["avg_sales_brick"],
        "loyalty_score":          doctor_row["loyalty_score"],
        "days_since_last_visit":  visit_stats.get("days_since_last_visit", 999),
        "total_visits":           visit_stats.get("total_visits", 0),
        "success_rate":           visit_stats.get("success_rate", 0.0),
        "avg_duration":           visit_stats.get("avg_duration", 0.0),
        "visits_last_90d":        visit_stats.get("visits_last_90d", 0),
        "trend_loyalty":          (
            doctor_row["loyalty_score"]
            * visit_stats.get("success_rate", 0.0)
        ),
    }
    return pd.DataFrame([row])


def make_inference_features_prob(
    doctor_row: Dict[str, Any],
    visit_stats: Dict[str, Any],
    target_hour: int,
    target_date: date,
) -> pd.DataFrame:
    """
    Собрать фичи для одного врача × конкретное время (Probability Model).
    """
    dow = target_date.weekday()
    row = {
        "specialty":               doctor_row["specialty"],
        "category":                doctor_row["category"],
        "day_of_week":             str(dow),
        "hour":                    target_hour,
        "is_morning":              int(target_hour < 12),
        "is_friday":               int(dow == 4),
        "is_weekend":              int(dow >= 5),
        "loyalty_score":           doctor_row["loyalty_score"],
        "days_since_prev_visit":   visit_stats.get("days_since_last_visit", 999),
        "month":                   str(target_date.month),
        "visits_last_30d":         visit_stats.get("visits_last_30d", 0),
        "historical_success_rate": visit_stats.get("success_rate", 0.5),
    }
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
#  METADATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_feature_metadata(path: str | Path) -> None:
    """Сохранить метаданные фичей для сервинга."""
    meta = {
        "value_model": {
            "features": VALUE_FEATURES,
            "cat_features": VALUE_CAT_FEATURES,
        },
        "probability_model": {
            "features": PROB_FEATURES,
            "cat_features": PROB_CAT_FEATURES,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )