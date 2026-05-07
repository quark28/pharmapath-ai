# %% [markdown]
# # 📊 PharmaPath AI — EDA & Feature Engineering
#
# **Цель:** Изучить данные, собрать фичи, убедиться в отсутствии аномалий.

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from feature_store import (
    load_doctors, load_visits,
    build_doctor_features, compute_value_target,
    build_visit_features, relabel_visit_success,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DATA_DIR = Path("../data/output")

# %% [markdown]
# ## 1. Загрузка данных

# %%
doctors_raw = load_doctors(DATA_DIR / "doctors_base.csv")
visits_raw  = load_visits(DATA_DIR / "visits_log.csv")

print(f"Врачей:  {len(doctors_raw):,}")
print(f"Визитов: {len(visits_raw):,}")
print(f"Медпредов: {visits_raw['rep_id'].nunique()}")
print(f"Период: {visits_raw['visit_date'].min().date()} → {visits_raw['visit_date'].max().date()}")

# %%
doctors_raw.head(3)

# %%
visits_raw.head(3)

# %% [markdown]
# ## 2. Распределения (базовый EDA)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 2a. Категории
doctors_raw["category"].value_counts().sort_index().plot.bar(
    ax=axes[0, 0], color=["#2ecc71", "#3498db", "#e74c3c"],
)
axes[0, 0].set_title("Категории врачей (A/B/C)")
axes[0, 0].set_ylabel("Кол-во")

# 2b. Специальности
doctors_raw["specialty"].value_counts().plot.barh(ax=axes[0, 1])
axes[0, 1].set_title("Специальности")

# 2c. Лояльность
axes[0, 2].hist(doctors_raw["loyalty_score"], bins=30, edgecolor="white")
axes[0, 2].set_title("Loyalty Score")
axes[0, 2].axvline(doctors_raw["loyalty_score"].mean(), color="red", ls="--", label="mean")
axes[0, 2].legend()

# 2d. Продажи
axes[1, 0].hist(doctors_raw["avg_sales_brick"], bins=30, edgecolor="white", color="orange")
axes[1, 0].set_title("Avg Sales (Brick)")

# 2e. Статусы визитов
visits_raw["status"].value_counts().plot.pie(
    ax=axes[1, 1], autopct="%1.1f%%", startangle=90,
)
axes[1, 1].set_title("Статусы визитов")
axes[1, 1].set_ylabel("")

# 2f. Визиты по часам
visits_raw["visit_hour"].value_counts().sort_index().plot.bar(
    ax=axes[1, 2], color="teal",
)
axes[1, 2].set_title("Распределение по часам")

plt.tight_layout()
plt.savefig("../models/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Геораспределение врачей

# %%
fig, ax = plt.subplots(figsize=(10, 10))
colors = {"A": "#e74c3c", "B": "#3498db", "C": "#95a5a6"}
for cat, grp in doctors_raw.groupby("category"):
    ax.scatter(
        grp["longitude"], grp["latitude"],
        c=colors.get(str(cat), "gray"),
        label=f"Cat {cat} ({len(grp)})",
        alpha=0.6, s=20,
    )
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Врачи на карте Москвы (цвет = категория)")
ax.legend()
plt.savefig("../models/eda_geo.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Feature Engineering — Doctor Level

# %%
df_doctors = build_doctor_features(doctors_raw, visits_raw)
df_doctors["potential_value"] = compute_value_target(df_doctors)

print(f"Shape: {df_doctors.shape}")
df_doctors[["full_name", "specialty", "category", "loyalty_score",
            "total_visits", "success_rate", "days_since_last_visit",
            "potential_value"]].head(10)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df_doctors["potential_value"], bins=40, edgecolor="white", color="#8e44ad")
axes[0].set_title("Target: Potential Value")

sns.boxplot(data=df_doctors, x="category", y="potential_value", ax=axes[1],
            order=["C", "B", "A"])
axes[1].set_title("Value по категориям")

axes[2].scatter(df_doctors["days_since_last_visit"], df_doctors["potential_value"],
               alpha=0.3, s=10)
axes[2].set_xlabel("Days since last visit")
axes[2].set_ylabel("Potential Value")
axes[2].set_title("Recency → Value")

plt.tight_layout()
plt.savefig("../models/eda_value_target.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Feature Engineering — Visit Level

# %%
df_visits = build_visit_features(visits_raw, doctors_raw)
df_visits["target"], df_visits["true_prob"] = relabel_visit_success(df_visits)

print(f"Shape: {df_visits.shape}")
print(f"Target balance: {df_visits['target'].value_counts(normalize=True).to_dict()}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# P(success) по часам
df_visits.groupby("hour")["target"].mean().plot.bar(ax=axes[0], color="teal")
axes[0].set_title("P(Success) по часам")
axes[0].set_ylabel("Success rate")

# P(success) по дням
dow_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
sr = df_visits.copy()
sr["dow_int"] = sr["day_of_week"].astype(int)
sr.groupby("dow_int")["target"].mean().rename(index=dow_labels).plot.bar(
    ax=axes[1], color="coral",
)
axes[1].set_title("P(Success) по дням недели")

# P(success) по специальности
df_visits.groupby("specialty")["target"].mean().sort_values().plot.barh(
    ax=axes[2], color="steelblue",
)
axes[2].set_title("P(Success) по специальности")

plt.tight_layout()
plt.savefig("../models/eda_probability_patterns.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Корреляционная матрица (числовые фичи)

# %%
num_cols = [
    "avg_sales_brick", "loyalty_score", "days_since_last_visit",
    "total_visits", "success_rate", "avg_duration", "visits_last_90d",
    "trend_loyalty", "potential_value",
]
corr = df_doctors[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Корреляция (Doctor-Level Features)")
plt.tight_layout()
plt.savefig("../models/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## ✅ Вывод
#
# - Данные чистые, распределения соответствуют ожиданиям.
# - Value target коррелирует с категорией, лояльностью и recency — модель должна выучить.
# - В Probability target есть явные паттерны: час, день недели, специальность.
# - Готовы к обучению моделей.