"""
PharmaPath AI — Analytics Dashboard
"""

import random

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Аналитика и KPI")

st.markdown("""
> **Цель страницы:** показать бизнес-эффект внедрения PharmaPath AI.
> Данные ниже — **симуляция** (до/после внедрения).
""")

# ── Симуляция KPI ─────────────────────────────────────────────────────────────

random.seed(42)
np.random.seed(42)

st.markdown("---")
st.markdown("### 📈 Ключевые метрики (Before vs After)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Визитов / день",
        "14.2",
        delta="+3.8 (+37%)",
        help="Среднее число визитов на медпреда в день",
    )
    st.caption("Было: 10.4")

with c2:
    st.metric(
        "Конверсия визитов",
        "78%",
        delta="+15%",
        help="Доля успешных визитов",
    )
    st.caption("Было: 63%")

with c3:
    st.metric(
        "Время в пути",
        "85 мин",
        delta="-47 мин (-36%)",
        delta_color="inverse",
        help="Среднее время в транспорте",
    )
    st.caption("Было: 132 мин")

with c4:
    st.metric(
        "Coverage A/B",
        "94%",
        delta="+22%",
        help="Покрытие врачей категорий A+B за месяц",
    )
    st.caption("Было: 72%")

# ── Графики ───────────────────────────────────────────────────────────────────

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Визиты / месяц",
    "🗺 Расстояние",
    "🎯 Coverage",
    "💰 ROI",
])

with tab1:
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    before = np.random.poisson(210, 12)
    after_start = 5  # внедрение с мая
    after = before.copy()
    after[after_start:] = np.random.poisson(290, 12 - after_start)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=before, name="До PharmaPath", marker_color="#e74c3c", opacity=0.6))
    fig.add_trace(go.Bar(x=months, y=after, name="После PharmaPath", marker_color="#2ecc71", opacity=0.8))
    fig.add_vline(x=after_start, line_dash="dash", line_color="gray",
              annotation_text="Внедрение")
    fig.update_layout(barmode="group", title="Визиты команды / месяц", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    days = list(range(1, 31))
    dist_before = np.random.normal(42, 8, 30)
    dist_after = np.random.normal(27, 5, 30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=dist_before, mode="lines+markers",
        name="До", line=dict(color="#e74c3c"),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=dist_after, mode="lines+markers",
        name="После", line=dict(color="#2ecc71"),
    ))
    fig.update_layout(
        title="Расстояние (км / день) — июнь 2025",
        xaxis_title="День месяца",
        yaxis_title="км",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    categories = ["Cat A", "Cat B", "Cat C"]
    coverage_before = [68, 55, 40]
    coverage_after = [95, 88, 62]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=coverage_before, name="До", marker_color="#e74c3c"))
    fig.add_trace(go.Bar(x=categories, y=coverage_after, name="После", marker_color="#2ecc71"))
    fig.update_layout(
        title="Покрытие врачей по категориям (%)",
        barmode="group",
        yaxis_title="%",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("""
    ### 💰 Расчёт ROI

    | Показатель | Значение |
    |-----------|----------|
    | Медпредов в команде | 50 |
    | Средняя ЗП медпреда | 120 000 ₽/мес |
    | ФОТ команды / год | **72 000 000 ₽** |
    | Рост конверсии | +15% |
    | Доп. продажи (оценка) | +18% |
    | **Доп. выручка / год** | **~86 400 000 ₽** |
    | Стоимость решения / год | ~5 000 000 ₽ |
    | **ROI** | **~1 628%** |

    ---

    #### Методика

    - **+37% визитов** = больше контактов с врачами
    - **+15% конверсия** = каждый визит эффективнее (правильное время, правильный врач)
    - **−36% время в пути** = оптимальные маршруты
    - **+22% coverage** = не пропускаем ключевых врачей

    Совокупный эффект: рост продаж на **15–20%** при тех же затратах на ФОТ.
    """)

    # Waterfall chart
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Базовая выручка", "Больше визитов", "Выше конверсия",
           "Лучший coverage", "Стоимость решения", "ИТОГО"],
        y=[480, 48, 72, 38.4, -5, 633.4],
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        text=["+480M", "+48M", "+72M", "+38.4M", "−5M", "633.4M"],
        textposition="outside",
        connector={"line": {"color": "gray"}},
    ))
    fig.update_layout(title="Водопадная диаграмма: эффект (млн ₽ / год)", height=450)
    st.plotly_chart(fig, use_container_width=True)