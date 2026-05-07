"""
PharmaPath AI — Doctor Card Component
"""

from __future__ import annotations

from typing import Dict

import streamlit as st


def render_doctor_card(doc: Dict, expanded: bool = False) -> None:
    """Рендерит карточку врача."""
    cat = doc.get("category", "?")
    cat_emoji = {"A": "🔴", "B": "🔵", "C": "⚪"}.get(cat, "⚪")

    loyalty = doc.get("loyalty_score", 0)
    loyalty_bar = "❤️" * int(loyalty / 2) + "🤍" * (5 - int(loyalty / 2))

    with st.expander(
        f"{cat_emoji} **{doc['full_name']}** — {doc['specialty']} | Cat {cat}",
        expanded=expanded,
    ):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Лояльность", f"{loyalty}/10")
            st.caption(loyalty_bar)

        with col2:
            st.metric("Продажи (брик)", f"{doc.get('avg_sales_brick', 0):.0f}")

        with col3:
            days = doc.get("days_since_last_visit", 999)
            delta_color = "inverse" if days > 30 else "normal"
            st.metric(
                "Дней без визита",
                days if days < 999 else "—",
                delta=f"-{days}д" if days > 30 else None,
                delta_color=delta_color,
            )

        st.caption(f"📍 {doc.get('work_address', '—')}")
        st.caption(
            f"📊 Визитов: {doc.get('total_visits', 0)} | "
            f"Конверсия: {doc.get('success_rate', 0):.0%}"
        )


def render_route_stop_card(stop: Dict, order: int) -> None:
    """Рендерит карточку остановки маршрута."""
    cat = stop.get("category", "?")
    cat_color = {"A": "red", "B": "blue", "C": "gray"}.get(cat, "gray")

    st.markdown(
        f"""
        <div style="border-left:4px solid {cat_color}; padding:8px 12px;
                    margin-bottom:8px; background:#f8f9fa; border-radius:4px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:20px; font-weight:bold; color:{cat_color};">
                        #{order}
                    </span>
                    <b>{stop['doctor_name']}</b>
                    <span style="color:gray;">| {stop['specialty']}</span>
                </div>
                <div style="text-align:right; font-size:13px;">
                    🕐 {stop['estimated_arrival']} — {stop['estimated_departure']}<br>
                    <span style="color:gray;">Окно: {stop['time_window_start']}–{stop['time_window_end']}</span>
                </div>
            </div>
            <div style="font-size:12px; margin-top:4px;">
                📍 {stop['address'][:70]}...<br>
                Score: <b>{stop['combined_score']:.1f}</b>
                (V={stop['value_score']:.0f} × P={stop['probability_score']:.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation
    for exp in stop.get("explanation", []):
        icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️"}.get(
            exp["impact"], "ℹ️"
        )
        st.caption(f"    {icon} {exp['factor']}")