"""
PharmaPath AI — Route Planner Page
"""

import sys
from pathlib import Path

# Добавляем путь к родительской папке
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from datetime import date, timedelta
from streamlit_folium import st_folium

from components.api_client import generate_route, get_doctors
from components.map_builder import build_route_map
from components.doctor_card import render_route_stop_card

st.set_page_config(page_title="Route Planner", page_icon="🗺", layout="wide")
st.title("🗺 Планировщик маршрута")

# ── Параметры ─────────────────────────────────────────────────────────────────

col_params, col_map = st.columns([1, 3])

with col_params:
    st.markdown("### 📋 Параметры")

    rep_id = st.session_state.get("selected_rep", "REP-001")
    st.info(f"👔 Медпред: **{rep_id}**")

    target_date = st.date_input(
        "📅 Дата маршрута",
        value=date.today(),
        min_value=date.today() - timedelta(days=7),
        max_value=date.today() + timedelta(days=30),
    )

    st.markdown("**📍 Стартовая точка:**")
    start_lat = st.number_input("Широта", value=55.7558, format="%.4f", step=0.001)
    start_lon = st.number_input("Долгота", value=37.6173, format="%.4f", step=0.001)

    max_visits = st.slider("🏥 Макс. визитов", min_value=5, max_value=20, value=14)

    st.markdown("---")

    # Кнопка генерации
    generate_clicked = st.button(
        "🚀 Построить маршрут",
        type="primary",
        use_container_width=True,
    )

    # Кнопка перестроения (если маршрут уже есть)
    if st.session_state.get("current_route"):
        recalc_clicked = st.button(
            "🔄 Перестроить маршрут",
            use_container_width=True,
            help="Перестроить с учётом уже посещённых",
        )
    else:
        recalc_clicked = False

# ── Генерация маршрута ────────────────────────────────────────────────────────

if generate_clicked or recalc_clicked:
    visited = []
    if recalc_clicked and st.session_state.get("visited_doctors"):
        visited = st.session_state.visited_doctors

    with st.spinner("🧠 Генерация маршрута... (ML scoring → OR-Tools optimization)"):
        route = generate_route(
            rep_id=rep_id,
            latitude=start_lat,
            longitude=start_lon,
            target_date=target_date,
            max_visits=max_visits,
            visited_ids=visited,
        )

    if route:
        st.session_state.current_route = route
        if "visited_doctors" not in st.session_state:
            st.session_state.visited_doctors = []
        st.success(
            f"✅ Маршрут построен: **{route['num_visits']}** визитов | "
            f"Score: **{route['total_score']:.0f}** | "
            f"Расстояние: **{route['total_distance_km']:.1f} км** | "
            f"Статус: **{route['optimizer_status']}**"
        )

# ── Отображение маршрута ──────────────────────────────────────────────────────

route = st.session_state.get("current_route")

if route:
    with col_map:
        # Фоновые врачи
        all_docs = get_doctors(limit=500)

        # Карта
        m = build_route_map(
            route_data=route,
            depot_lat=start_lat,
            depot_lon=start_lon,
            all_doctors=all_docs,
        )
        st_folium(m, width=None, height=600, returned_objects=[])

    # ── Метрики ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Метрики маршрута")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🏥 Визитов", route["num_visits"])
    m2.metric("🎯 Суммарный Score", f"{route['total_score']:.0f}")
    m3.metric("🚗 Расстояние", f"{route['total_distance_km']:.1f} км")
    m4.metric("⏱ Длительность", f"{route['total_duration_minutes']} мин")
    m5.metric("🔧 Статус", route["optimizer_status"])

    # ── Список остановок ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂 Маршрутный лист")

    for stop in route["stops"]:
        render_route_stop_card(stop, stop["order"])

        # Кнопка "Визит завершён"
        col_done, col_skip = st.columns([1, 1])
        with col_done:
            if st.button(
                f"✅ Визит #{stop['order']} выполнен",
                key=f"done_{stop['doctor_id']}",
            ):
                if "visited_doctors" not in st.session_state:
                    st.session_state.visited_doctors = []
                st.session_state.visited_doctors.append(stop["doctor_id"])
                st.toast(f"Визит к {stop['doctor_name']} отмечен")

    # ── Пропущенные врачи ─────────────────────────────────────────────────
    if route.get("skipped"):
        with st.expander(f"⏭ Не вошли в маршрут ({len(route['skipped'])} врачей)"):
            for skip in route["skipped"]:
                st.caption(
                    f"• {skip['doctor_name']} (Score: {skip['combined_score']:.1f}) — "
                    f"{skip['reason']}"
                )

else:
    with col_map:
        st.info("👈 Задайте параметры и нажмите «Построить маршрут»")

        # Пустая карта Москвы
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[55.7558, 37.6173], zoom_start=11, tiles="CartoDB positron")
        st_folium(m, width=None, height=600, returned_objects=[])