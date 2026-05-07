#!/usr/bin/env python3
"""
PharmaPath AI — Full Training Pipeline (CLI)
=============================================
Одна команда — полная переобучка обеих моделей.

Usage:
    python train_pipeline.py
    python train_pipeline.py --data_dir ../data/output --model_dir ../models --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)

# Relative import
sys.path.insert(0, str(Path(__file__).parent))
from feature_store import (
    load_doctors, load_visits,
    build_doctor_features, compute_value_target,
    build_visit_features, relabel_visit_success,
    save_feature_metadata,
    VALUE_FEATURES, VALUE_CAT_FEATURES,
    PROB_FEATURES, PROB_CAT_FEATURES,
)


def train_value_model(
    df: pd.DataFrame,
    model_dir: Path,
    seed: int,
) -> dict:
    """Обучить Value Model, вернуть метрики."""
    print("\n" + "=" * 60)
    print("  TRAINING: Value Model (CatBoostRegressor)")
    print("=" * 60)

    X = df[VALUE_FEATURES].copy()
    y = df["potential_value"].copy()
    for col in VALUE_CAT_FEATURES:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed,
    )

    cat_idx = [X.columns.tolist().index(c) for c in VALUE_CAT_FEATURES]

    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        loss_function="RMSE",
        eval_metric="MAE",
        early_stopping_rounds=50,
        verbose=200,
        use_best_model=True,
    )

    model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        plot=False,
    )

    y_pred = model.predict(Pool(X_test, cat_features=cat_idx))

    metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }

    model_path = model_dir / "value_model.cbm"
    model.save_model(str(model_path))
    print(f"\n  ✅ Saved: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"  📊 RMSE={metrics['rmse']}  MAE={metrics['mae']}  R²={metrics['r2']}")

    return metrics


def train_probability_model(
    df: pd.DataFrame,
    model_dir: Path,
    seed: int,
) -> dict:
    """Обучить Probability Model, вернуть метрики."""
    print("\n" + "=" * 60)
    print("  TRAINING: Probability Model (CatBoostClassifier)")
    print("=" * 60)

    X = df[PROB_FEATURES].copy()
    y = df["target"].copy()
    for col in PROB_CAT_FEATURES:
        X[col] = X[col].astype(str)

    # Temporal split
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

    cat_idx = [X.columns.tolist().index(c) for c in PROB_CAT_FEATURES]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=60,
        verbose=200,
        use_best_model=True,
        auto_class_weights="Balanced",
    )

    model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        plot=False,
    )

    y_prob = model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc":  round(float(average_precision_score(y_test, y_prob)), 4),
        "brier":   round(float(brier_score_loss(y_test, y_prob)), 4),
        "logloss": round(float(log_loss(y_test, y_prob)), 4),
    }

    model_path = model_dir / "probability_model.cbm"
    model.save_model(str(model_path))
    print(f"\n  ✅ Saved: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"  📊 AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}  Brier={metrics['brier']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="PharmaPath AI — Train Pipeline")
    parser.add_argument("--data_dir",  type=str, default="../data/output")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("  🚀 PharmaPath AI — Training Pipeline")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n📥 Loading data...")
    doctors = load_doctors(data_dir / "doctors_base.csv")
    visits  = load_visits(data_dir / "visits_log.csv")
    print(f"   {len(doctors):,} doctors, {len(visits):,} visits")

    # ── Value Model Features ──────────────────────────────────────────────
    print("\n⚙️  Building doctor-level features...")
    df_value = build_doctor_features(doctors, visits)
    df_value["potential_value"] = compute_value_target(df_value, seed=args.seed)

    # ── Probability Model Features ────────────────────────────────────────
    print("⚙️  Building visit-level features...")
    df_prob = build_visit_features(visits, doctors)
    df_prob["target"], _ = relabel_visit_success(df_prob, seed=args.seed)

    # Sort by date for temporal split
    df_prob = df_prob.sort_values("visit_date").reset_index(drop=True)

    # ── Train ─────────────────────────────────────────────────────────────
    val_metrics  = train_value_model(df_value, model_dir, args.seed)
    prob_metrics = train_probability_model(df_prob, model_dir, args.seed)

    # ── Save metadata ─────────────────────────────────────────────────────
    save_feature_metadata(model_dir / "feature_metadata.json")

    all_metrics = {
        "seed": args.seed,
        "training_time_sec": round(time.time() - t0, 1),
        "value_model": val_metrics,
        "probability_model": prob_metrics,
    }
    with open(model_dir / "training_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  ✨ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  ⏱  Total time: {elapsed:.1f}s")
    print(f"  📁 Artifacts in: {model_dir}/")
    print()
    print("  Value Model:")
    print(f"     RMSE={val_metrics['rmse']}  MAE={val_metrics['mae']}  R²={val_metrics['r2']}")
    print()
    print("  Probability Model:")
    print(f"     AUC={prob_metrics['roc_auc']}  PR-AUC={prob_metrics['pr_auc']}  Brier={prob_metrics['brier']}")
    print()

    # Listing artifacts
    print("  📦 Generated files:")
    for p in sorted(model_dir.iterdir()):
        size = p.stat().st_size
        unit = "KB" if size > 1024 else "B"
        val = size / 1024 if size > 1024 else size
        print(f"     {p.name:40s} {val:>8.1f} {unit}")
    print("=" * 60)


if __name__ == "__main__":
    main()