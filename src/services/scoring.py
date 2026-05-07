"""
PharmaPath AI — ML Scoring Service
====================================
Загружает CatBoost-модели, делает inference.
Импортирует feature_store из ml/ для zero train-serve skew.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# ── Импорт feature_store (shared с обучением) ────────────────────────────────
_ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from feature_store import (
    VALUE_FEATURES,
    VALUE_CAT_FEATURES,
    PROB_FEATURES,
    PROB_CAT_FEATURES,
    make_inference_features_value,
    make_inference_features_prob,
)

logger = structlog.get_logger()


@dataclass
class DoctorScore:
    """Результат скоринга одного врача."""
    doctor_id: str
    value_score: float           # 0–100
    probability_score: float     # 0.0–1.0
    combined_score: float        # value × probability
    feature_contributions: Dict[str, float]


class ScoringService:
    """
    Загружает обе модели и предоставляет scoring-интерфейс.
    """

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._value_model: Optional[CatBoostRegressor] = None
        self._prob_model: Optional[CatBoostClassifier] = None
        self._loaded = False

    # ══════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def load_models(self) -> None:
        """Загрузить модели с диска (вызывается при старте приложения)."""
        value_path = self._model_dir / "value_model.cbm"
        prob_path = self._model_dir / "probability_model.cbm"

        if not value_path.exists() or not prob_path.exists():
            logger.error(
                "Models not found. Run `python ml/train_pipeline.py` first.",
                value=value_path.exists(),
                prob=prob_path.exists(),
            )
            return

        self._value_model = CatBoostRegressor()
        self._value_model.load_model(str(value_path))

        self._prob_model = CatBoostClassifier()
        self._prob_model.load_model(str(prob_path))

        self._loaded = True
        logger.info("ML models loaded", model_dir=str(self._model_dir))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ══════════════════════════════════════════════════════════════════════
    #  BATCH SCORING (для всех врачей → generate_route)
    # ══════════════════════════════════════════════════════════════════════

    def score_doctors_batch(
        self,
        doctors_df: pd.DataFrame,
        visit_stats: Dict[str, Dict],
        target_date: date,
        target_hours: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Оценить всех врачей.

        Parameters
        ----------
        doctors_df : DataFrame с doctors_base
        visit_stats : {doctor_id: {days_since_last_visit, total_visits, ...}}
        target_date : дата маршрута
        target_hours : часы, по которым усреднять P(success).
                       По умолчанию: [9, 10, 11, 12, 13, 14, 15, 16, 17].

        Returns
        -------
        DataFrame с колонками: doctor_id, value_score, prob_score, combined_score
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded")

        if target_hours is None:
            target_hours = list(range(9, 18))

        # ── Value Model (batch) ───────────────────────────────────────────
        value_rows = []
        for _, doc in doctors_df.iterrows():
            doc_id = doc["id"]
            stats = visit_stats.get(doc_id, {
                "days_since_last_visit": 999,
                "total_visits": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "visits_last_90d": 0,
                "visits_last_30d": 0,
            })
            feat = make_inference_features_value(doc.to_dict(), stats)
            feat["doctor_id"] = doc_id
            value_rows.append(feat)

        value_df = pd.concat(value_rows, ignore_index=True)
        doctor_ids = value_df["doctor_id"].values

        X_value = value_df[VALUE_FEATURES].copy()
        for col in VALUE_CAT_FEATURES:
            X_value[col] = X_value[col].astype(str)

        cat_idx_val = [VALUE_FEATURES.index(c) for c in VALUE_CAT_FEATURES]
        value_preds = self._value_model.predict(
            Pool(X_value, cat_features=cat_idx_val)
        )
        value_preds = np.clip(value_preds, 0, 100)

        # ── Probability Model (batch × hours → mean) ─────────────────────
        prob_rows = []
        prob_doctor_ids = []
        prob_hours_list = []

        for _, doc in doctors_df.iterrows():
            doc_id = doc["id"]
            stats = visit_stats.get(doc_id, {
                "days_since_last_visit": 999,
                "success_rate": 0.5,
                "visits_last_30d": 0,
            })
            for hour in target_hours:
                feat = make_inference_features_prob(
                    doc.to_dict(), stats, hour, target_date,
                )
                prob_rows.append(feat)
                prob_doctor_ids.append(doc_id)
                prob_hours_list.append(hour)

        prob_df = pd.concat(prob_rows, ignore_index=True)
        X_prob = prob_df[PROB_FEATURES].copy()
        for col in PROB_CAT_FEATURES:
            X_prob[col] = X_prob[col].astype(str)

        cat_idx_prob = [PROB_FEATURES.index(c) for c in PROB_CAT_FEATURES]
        prob_preds = self._prob_model.predict_proba(
            Pool(X_prob, cat_features=cat_idx_prob)
        )[:, 1]

        # Усредняем вероятность по часам для каждого врача
        prob_series = pd.DataFrame({
            "doctor_id": prob_doctor_ids,
            "prob": prob_preds,
        }).groupby("doctor_id")["prob"].mean()

        # ── Объединяем ────────────────────────────────────────────────────
        result = pd.DataFrame({
            "doctor_id": doctor_ids,
            "value_score": np.round(value_preds, 2),
        })
        result["prob_score"] = result["doctor_id"].map(prob_series).fillna(0.5)
        result["prob_score"] = result["prob_score"].round(3)
        result["combined_score"] = (
            result["value_score"] * result["prob_score"]
        ).round(2)

        return result.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════
    #  SINGLE DOCTOR SCORING (для карточки врача)
    # ══════════════════════════════════════════════════════════════════════

    def score_single_doctor(
        self,
        doctor_row: Dict,
        visit_stats: Dict,
        target_date: date,
        target_hour: int,
    ) -> DoctorScore:
        """Оценить одного врача с объяснением."""
        if not self._loaded:
            raise RuntimeError("Models not loaded")

        # Value
        X_val = make_inference_features_value(doctor_row, visit_stats)
        for col in VALUE_CAT_FEATURES:
            X_val[col] = X_val[col].astype(str)

        cat_idx_v = [VALUE_FEATURES.index(c) for c in VALUE_CAT_FEATURES]
        value = float(np.clip(
            self._value_model.predict(Pool(X_val, cat_features=cat_idx_v))[0],
            0, 100,
        ))

        # Probability
        X_prob = make_inference_features_prob(
            doctor_row, visit_stats, target_hour, target_date,
        )
        for col in PROB_CAT_FEATURES:
            X_prob[col] = X_prob[col].astype(str)

        cat_idx_p = [PROB_FEATURES.index(c) for c in PROB_CAT_FEATURES]
        prob = float(
            self._prob_model.predict_proba(Pool(X_prob, cat_features=cat_idx_p))[0, 1]
        )

        combined = round(value * prob, 2)

        # Feature contributions (упрощённый explainer)
        contributions = self._explain(doctor_row, visit_stats)

        return DoctorScore(
            doctor_id=doctor_row["id"],
            value_score=round(value, 2),
            probability_score=round(prob, 3),
            combined_score=combined,
            feature_contributions=contributions,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _explain(doctor_row: Dict, visit_stats: Dict) -> Dict[str, float]:
        """
        Простой rule-based explainer (для MVP).
        В проде → SHAP per-prediction.
        """
        factors = {}

        cat = doctor_row.get("category", "C")
        factors["category"] = {"A": 0.9, "B": 0.5, "C": 0.2}.get(cat, 0.3)

        loyalty = doctor_row.get("loyalty_score", 5.0)
        factors["loyalty"] = round((loyalty - 5.0) / 5.0, 2)

        days = visit_stats.get("days_since_last_visit", 999)
        if days > 30:
            factors["recency_urgency"] = min(1.0, round((days - 30) / 60, 2))
        else:
            factors["recency_urgency"] = 0.0

        sr = visit_stats.get("success_rate", 0.5)
        factors["historical_success"] = round(sr - 0.5, 2)

        sales = doctor_row.get("avg_sales_brick", 50.0)
        factors["sales_potential"] = round((sales - 50) / 50, 2)

        return factors

    @staticmethod
    def build_explanation_text(
        doctor_row: Dict,
        visit_stats: Dict,
        contributions: Dict[str, float],
    ) -> List[Dict[str, str]]:
        """Человекочитаемое объяснение для фронтенда."""
        explanations = []

        cat = doctor_row.get("category", "C")
        if cat == "A":
            explanations.append({
                "factor": f"Категория A — ключевой клиент",
                "impact": "positive",
            })
        elif cat == "B":
            explanations.append({
                "factor": f"Категория B — стандартный потенциал",
                "impact": "neutral",
            })

        loyalty = doctor_row.get("loyalty_score", 5.0)
        if loyalty >= 7.0:
            explanations.append({
                "factor": f"Высокая лояльность ({loyalty}/10)",
                "impact": "positive",
            })
        elif loyalty <= 3.0:
            explanations.append({
                "factor": f"Низкая лояльность ({loyalty}/10) — нужна работа",
                "impact": "negative",
            })

        days = visit_stats.get("days_since_last_visit", 999)
        if days > 30:
            explanations.append({
                "factor": f"{days} дн. с последнего визита — пора навестить",
                "impact": "positive",
            })
        elif days < 7:
            explanations.append({
                "factor": f"Был недавно ({days} дн. назад)",
                "impact": "negative",
            })

        sr = visit_stats.get("success_rate", 0.5)
        if sr >= 0.8:
            explanations.append({
                "factor": f"Исторически высокая конверсия ({sr:.0%})",
                "impact": "positive",
            })

        return explanations