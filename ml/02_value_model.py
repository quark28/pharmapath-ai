# %% [markdown]
# # 🎯 PharmaPath AI — Value Model (CatBoost Regressor)
#
# **Задача:** Предсказать `potential_value` — сколько «денег» принесёт визит к врачу.
#
# **Метрики:** RMSE, MAE, R².

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_store import (
    load_doctors, load_visits,
    build_doctor_features, compute_value_target,
    VALUE_FEATURES, VALUE_CAT_FEATURES,
    save_feature_metadata,
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

df = build_doctor_features(doctors_raw, visits_raw)
df["potential_value"] = compute_value_target(df, seed=SEED)

print(f"Датасет: {df.shape[0]} врачей × {df.shape[1]} колонок")
print(f"Target stats:\n{df['potential_value'].describe().round(2)}")

# %%
X = df[VALUE_FEATURES].copy()
y = df["potential_value"].copy()

# Проверяем типы
for col in VALUE_CAT_FEATURES:
    X[col] = X[col].astype(str)

print(f"\nФичи: {list(X.columns)}")
print(f"Категориальные: {VALUE_CAT_FEATURES}")
print(f"X shape: {X.shape}")

# %% [markdown]
# ## 2. Train / Validation / Test Split

# %%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176,  # 0.176 × 0.85 ≈ 0.15
    random_state=SEED,
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"Splits: {len(X_train)/len(X):.0%} / {len(X_val)/len(X):.0%} / {len(X_test)/len(X):.0%}")

# %%
cat_feature_indices = [X.columns.tolist().index(c) for c in VALUE_CAT_FEATURES]

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_feature_indices)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_feature_indices)

# %% [markdown]
# ## 3. Обучение CatBoost Regressor

# %%
model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    loss_function="RMSE",
    eval_metric="MAE",
    early_stopping_rounds=50,
    verbose=100,
    use_best_model=True,
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
y_pred = model.predict(test_pool)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=" * 50)
print("  VALUE MODEL — Test Metrics")
print("=" * 50)
print(f"  RMSE : {rmse:.3f}")
print(f"  MAE  : {mae:.3f}")
print(f"  R²   : {r2:.3f}")
print("=" * 50)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 4a. Predicted vs Actual
axes[0].scatter(y_test, y_pred, alpha=0.5, s=15, edgecolor="white")
lims = [0, 100]
axes[0].plot(lims, lims, "--", color="red", label="Ideal")
axes[0].set_xlabel("Actual Value")
axes[0].set_ylabel("Predicted Value")
axes[0].set_title(f"Predicted vs Actual (R²={r2:.3f})")
axes[0].legend()

# 4b. Residuals
residuals = y_test.values - y_pred
axes[1].hist(residuals, bins=30, edgecolor="white", color="#8e44ad")
axes[1].axvline(0, color="red", ls="--")
axes[1].set_title(f"Residuals (MAE={mae:.2f})")
axes[1].set_xlabel("Error")

# 4c. Feature Importance
fi = model.get_feature_importance()
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": fi,
}).sort_values("importance", ascending=True)

fi_df.plot.barh(x="feature", y="importance", ax=axes[2], legend=False, color="teal")
axes[2].set_title("Feature Importance")

plt.tight_layout()
plt.savefig(MODEL_DIR / "value_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. SHAP Analysis (Explainability — «Почему ИИ выбрал этого врача»)

# %%
try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP — Value Model")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "value_model_shap.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅ SHAP plot saved")
except ImportError:
    print("⚠️  SHAP не установлен (pip install shap). Пропускаем.")

# %% [markdown]
# ## 6. Сохранение модели

# %%
model_path = MODEL_DIR / "value_model.cbm"
model.save_model(str(model_path))
print(f"✅ Модель сохранена: {model_path}")
print(f"   Размер: {model_path.stat().st_size / 1024:.1f} KB")

# Сохраняем метрики
import json
metrics = {
    "model": "value_model",
    "algorithm": "CatBoostRegressor",
    "best_iteration": model.get_best_iteration(),
    "test_metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)},
    "features": VALUE_FEATURES,
    "cat_features": VALUE_CAT_FEATURES,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
}
with open(MODEL_DIR / "value_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Метрики сохранены")

# %% [markdown]
# ## ✅ Итог
#
# | Метрика | Значение |
# |---------|----------|
# | RMSE    | ~3-5     |
# | MAE     | ~2-4     |
# | R²      | ~0.95+   |
#
# Модель хорошо выучила паттерны (ожидаемо: target был синтетическим).
# В реальности R² будет ниже, но архитектура та же.