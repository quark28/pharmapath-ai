"""
PharmaPath AI — Doctors Database Page
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.api_client import get_doctors
from components.map_builder import build_doctors_map
from components.doctor_card import render_doctor_card

st.set_page_config(page_title="Doctors", page_icon="👨‍⚕", layout="wide")
st.title("👨‍⚕️ База врачей")

# ── Фильтры ──────────────────────────────────────────────────────────────────

col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    specialty = st.selectbox(
        "Специальность",
        ["Все"] + [
            "Therapist", "Cardiologist", "Neurologist",
            "Endocrinologist", "Gastroenterologist",
            "Pulmonologist", "Rheumatologist", "Urologist",
        ],
    )

with col_filter2:
    category = st.selectbox("Категория", ["Все", "A", "B", "C"])

with col_filter3:
    view_mode = st.radio("Вид", ["📋 Список", "🗺 Карта", "📊 Таблица"], horizontal=True)

# ── Загрузка данных ───────────────────────────────────────────────────────────

doctors = get_doctors(
    specialty=specialty if specialty != "Все" else None,
    category=category if category != "Все" else None,
    limit=500,
)

if doctors is None:
    st.stop()

st.caption(f"Найдено: **{len(doctors)}** врачей")

# ── Отображение ──────────────────────────────────────────────────────────────

if view_mode == "🗺 Карта":
    m = build_doctors_map(doctors)
    st_folium(m, width=None, height=650, returned_objects=[])

elif view_mode == "📊 Таблица":
    df = pd.DataFrame(doctors)
    display_cols = [
        "full_name", "specialty", "category", "loyalty_score",
        "avg_sales_brick", "total_visits", "success_rate", "days_since_last_visit",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].rename(columns={
            "full_name": "ФИО",
            "specialty": "Специальность",
            "category": "Кат.",
            "loyalty_score": "Лояльность",
            "avg_sales_brick": "Продажи",
            "total_visits": "Визиты",
            "success_rate": "Конверсия",
            "days_since_last_visit": "Дней без визита",
        }),
        use_container_width=True,
        height=600,
    )

else:  # Список
    for doc in doctors[:50]:
        render_doctor_card(doc)
    if len(doctors) > 50:
        st.info(f"Показано 50 из {len(doctors)}. Используйте фильтры.")