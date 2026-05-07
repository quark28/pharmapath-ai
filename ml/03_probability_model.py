# %% [markdown]
# # 🎲 PharmaPath AI — Probability Model (CatBoost Classifier)
#
# **Задача:** Предсказать P(Success) — вероятность успешного визита
# в конкретный день/час к конкретному врачу.
#
# **Метрики:** ROC-AUC, PR-AUC, Brier Score, Log Loss.

# %%
import sys
sys.path.insert(0, "..")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, classification_report, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from feature_store import (
    load_doctors, load_visits,
    build_visit_features, relabel_visit_success,
    PROB_FEATURES, PROB_CAT_FEATURES,
)

sns.set_theme(style="whitegrid")

DATA_DIR = Path("../data/output")
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42

# %% [markdown]
# ## 1. Подготовка данных

# %%
doctors_raw = load_doctors(DATA_DIR / "doctors_base.csv")
visits_raw  = load_visits(DATA_DIR / "visits_log.csv")

df = build_visit_features(visits_raw, doctors_raw)
df["target"], df["true_prob"] = relabel_visit_success(df, seed=SEED)

print(f"Датасет: {df.shape[0]:,} визитов")
print(f"Target balance:\n{df['target'].value_counts()}")
print(f"Success rate: {df['target'].mean():.3f}")

# %%
X = df[PROB_FEATURES].copy()
y = df["target"].copy()

for col in PROB_CAT_FEATURES:
    X[col] = X[col].astype(str)

print(f"\nФичи: {list(X.columns)}")
print(f"Категориальные: {PROB_CAT_FEATURES}")

# %% [markdown]
# ## 2. Time-Based Split (имитация продакшн-деплоя)
#
# В реальности нельзя обучать на будущих данных → делаем temporal split.

# %%
df["_sort_date"] = pd.to_datetime(df["visit_date"])
sort_idx = df["_sort_date"].argsort()
X = X.iloc[sort_idx].reset_index(drop=True)
y = y.iloc[sort_idx].reset_index(drop=True)

n = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],      y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],          y.iloc[val_end:]

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# %%
cat_idx = [X.columns.tolist().index(c) for c in PROB_CAT_FEATURES]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

# %% [markdown]
# ## 3. Обучение CatBoost Classifier

# %%
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5.0,
    random_seed=SEED,
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=60,
    verbose=100,
    use_best_model=True,
    auto_class_weights="Balanced",  # на случай дисбаланса
)

model.fit(
    train_pool,
    eval_set=val_pool,
    plot=False,
)

print(f"\nBest iteration: {model.get_best_iteration()}")

# %% [markdown]
# ## 4. Оценка на Test

# %%
y_prob = model.predict_proba(test_pool)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc  = roc_auc_score(y_test, y_prob)
pr_auc   = average_precision_score(y_test, y_prob)
brier    = brier_score_loss(y_test, y_prob)
logloss  = log_loss(y_test, y_prob)

print("=" * 50)
print("  PROBABILITY MODEL — Test Metrics")
print("=" * 50)
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  Brier     : {brier:.4f}")
print(f"  Log Loss  : {logloss:.4f}")
print("=" * 50)

print("\n" + classification_report(y_test, y_pred, target_names=["Fail", "Success"]))

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4a. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set_xlabel("FPR")
axes[0, 0].set_ylabel("TPR")
axes[0, 0].set_title("ROC Curve")
axes[0, 0].legend()

# 4b. PR Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[0, 1].plot(rec, prec, color="green", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].legend()

# 4c. Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
axes[1, 0].plot(prob_pred, prob_true, "o-", color="purple", label="Model")
axes[1, 0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
axes[1, 0].set_xlabel("Predicted probability")
axes[1, 0].set_ylabel("Actual fraction")
axes[1, 0].set_title(f"Calibration Plot (Brier={brier:.4f})")
axes[1, 0].legend()

# 4d. Feature Importance
fi = model.get_feature_importance()
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": fi,
}).sort_values("importance", ascending=True)

fi_df.plot.barh(x="feature", y="importance", ax=axes[1, 1], legend=False, color="teal")
axes[1, 1].set_title("Feature Importance")

plt.tight_layout()
plt.savefig(MODEL_DIR / "probability_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Анализ паттернов (Sanity Check)
#
# Проверяем, что модель выучила инжектированные паттерны.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True vs Predicted P(Success) по часам
test_df = X_test.copy()
test_df["y_true"]  = y_test.values
test_df["y_prob"]  = y_prob
test_df["hour_int"] = test_df["hour"].astype(int)

by_hour = test_df.groupby("hour_int").agg(
    actual=("y_true", "mean"),
    predicted=("y_prob", "mean"),
).reset_index()

axes[0].bar(by_hour["hour_int"] - 0.15, by_hour["actual"],    width=0.3, label="Actual", alpha=0.7)
axes[0].bar(by_hour["hour_int"] + 0.15, by_hour["predicted"], width=0.3, label="Predicted", alpha=0.7)
axes[0].set_title("P(Success) по часам: Actual vs Predicted")
axes[0].legend()
axes[0].set_xlabel("Hour")

# По дням
test_df["dow_int"] = test_df["day_of_week"].astype(int)
by_dow = test_df.groupby("dow_int").agg(
    actual=("y_true", "mean"),
    predicted=("y_prob", "mean"),
).reset_index()

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
x = np.arange(len(by_dow))
axes[1].bar(x - 0.15, by_dow["actual"],    width=0.3, label="Actual", alpha=0.7)
axes[1].bar(x + 0.15, by_dow["predicted"], width=0.3, label="Predicted", alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels([dow_labels[i] for i in by_dow["dow_int"]])
axes[1].set_title("P(Success) по дням недели")
axes[1].legend()

# Распределение предсказанных вероятностей
axes[2].hist(y_prob[y_test == 1], bins=30, alpha=0.6, label="Success", density=True)
axes[2].hist(y_prob[y_test == 0], bins=30, alpha=0.6, label="Fail", density=True)
axes[2].set_title("Predicted Probability Distribution")
axes[2].legend()

plt.tight_layout()
plt.savefig(MODEL_DIR / "probability_model_patterns.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Сохранение модели

# %%
model_path = MODEL_DIR / "probability_model.cbm"
model.save_model(str(model_path))
print(f"✅ Модель сохранена: {model_path}")
print(f"   Размер: {model_path.stat().st_size / 1024:.1f} KB")

metrics = {
    "model": "probability_model",
    "algorithm": "CatBoostClassifier",
    "best_iteration": model.get_best_iteration(),
    "test_metrics": {
        "roc_auc": round(roc_auc, 4),
        "pr_auc":  round(pr_auc, 4),
        "brier":   round(brier, 4),
        "logloss": round(logloss, 4),
    },
    "features": PROB_FEATURES,
    "cat_features": PROB_CAT_FEATURES,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "split": "temporal (70/15/15)",
}
with open(MODEL_DIR / "probability_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Общие метаданные
from feature_store import save_feature_metadata
save_feature_metadata(MODEL_DIR / "feature_metadata.json")

print("✅ Все артефакты сохранены")