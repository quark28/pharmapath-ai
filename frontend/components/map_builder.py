"""
PharmaPath AI — Map Builder
=============================
Строит Folium-карту с маршрутом и точками врачей.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import folium
from folium import plugins

# ── Цвета по категориям ──────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "A": "#e74c3c",   # красный
    "B": "#3498db",   # синий
    "C": "#95a5a6",   # серый
}

ROUTE_LINE_COLOR = "#2ecc71"   # зелёный
DEPOT_COLOR = "#f39c12"        # оранжевый


def build_route_map(
    route_data: Dict,
    depot_lat: float,
    depot_lon: float,
    all_doctors: Optional[List[Dict]] = None,
) -> folium.Map:
    """
    Построить карту с маршрутом.

    Parameters
    ----------
    route_data : ответ от /routes/generate
    depot_lat, depot_lon : стартовая точка
    all_doctors : все врачи (серые точки на фоне)

    Returns
    -------
    folium.Map
    """
    stops = route_data.get("stops", [])

    # Центр карты
    if stops:
        center_lat = sum(s["latitude"] for s in stops) / len(stops)
        center_lon = sum(s["longitude"] for s in stops) / len(stops)
    else:
        center_lat, center_lon = depot_lat, depot_lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # ── Фоновые врачи (серые) ────────────────────────────────────────────
    if all_doctors:
        visited_ids = {s["doctor_id"] for s in stops}
        bg_group = folium.FeatureGroup(name="Все врачи", show=False)

        for doc in all_doctors:
            if doc["id"] in visited_ids:
                continue
            folium.CircleMarker(
                location=[doc["latitude"], doc["longitude"]],
                radius=3,
                color="#cccccc",
                fill=True,
                fill_opacity=0.4,
                popup=f"{doc['full_name']}<br>{doc['specialty']} | Cat {doc['category']}",
            ).add_to(bg_group)

        bg_group.add_to(m)

    # ── Depot (стартовая точка) ───────────────────────────────────────────
    folium.Marker(
        location=[depot_lat, depot_lon],
        icon=folium.Icon(color="orange", icon="home", prefix="fa"),
        popup="🏠 Стартовая точка",
        tooltip="СТАРТ",
    ).add_to(m)

    # ── Маршрутные точки ──────────────────────────────────────────────────
    route_group = folium.FeatureGroup(name="Маршрут")
    route_coords = [(depot_lat, depot_lon)]

    for stop in stops:
        lat = stop["latitude"]
        lon = stop["longitude"]
        cat = stop.get("category", "B")
        color = CATEGORY_COLORS.get(cat, "#3498db")

        # Popup с информацией
        explanation_html = ""
        for exp in stop.get("explanation", []):
            icon = "✅" if exp["impact"] == "positive" else (
                "⚠️" if exp["impact"] == "negative" else "ℹ️"
            )
            explanation_html += f"<br>{icon} {exp['factor']}"

        popup_html = f"""
        <div style="width:280px; font-family:Arial,sans-serif;">
            <h4 style="margin:0 0 5px 0;">
                #{stop['order']} {stop['doctor_name']}
            </h4>
            <table style="font-size:12px;">
                <tr><td>🩺</td><td>{stop['specialty']}</td></tr>
                <tr><td>⭐</td><td>Категория {stop['category']}</td></tr>
                <tr><td>📍</td><td>{stop['address'][:60]}...</td></tr>
                <tr><td>🕐</td><td>{stop['estimated_arrival']} — {stop['estimated_departure']}</td></tr>
                <tr><td>🪟</td><td>Окно: {stop['time_window_start']} — {stop['time_window_end']}</td></tr>
            </table>
            <hr style="margin:5px 0;">
            <div style="font-size:11px;">
                <b>Score:</b> {stop['combined_score']:.1f}
                (V={stop['value_score']:.1f} × P={stop['probability_score']:.2f})
            </div>
            <div style="font-size:11px; margin-top:4px;">
                <b>Почему выбран:</b>
                {explanation_html}
            </div>
        </div>
        """

        # Маркер с номером
        folium.Marker(
            location=[lat, lon],
            icon=plugins.BeautifyIcon(
                number=stop["order"],
                border_color=color,
                background_color=color,
                text_color="white",
                icon_size=[28, 28],
            ),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{stop['order']} {stop['doctor_name']} ({stop['estimated_arrival']})",
        ).add_to(route_group)

        route_coords.append((lat, lon))

    # ── Линия маршрута ────────────────────────────────────────────────────
    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            color=ROUTE_LINE_COLOR,
            weight=3,
            opacity=0.7,
            dash_array="10 5",
        ).add_to(route_group)

        # Стрелки направления
        plugins.AntPath(
            locations=route_coords,
            color=ROUTE_LINE_COLOR,
            weight=2,
            opacity=0.5,
            delay=1500,
        ).add_to(route_group)

    route_group.add_to(m)

    # ── Легенда ───────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 15px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:12px;">
        <b>Легенда</b><br>
        <span style="color:#f39c12;">●</span> Старт<br>
        <span style="color:#e74c3c;">●</span> Категория A<br>
        <span style="color:#3498db;">●</span> Категория B<br>
        <span style="color:#95a5a6;">●</span> Категория C<br>
        <span style="color:#2ecc71;">---</span> Маршрут
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer control
    folium.LayerControl().add_to(m)

    return m


def build_doctors_map(
    doctors: List[Dict],
    center_lat: float = 55.7558,
    center_lon: float = 37.6173,
) -> folium.Map:
    """Карта всех врачей (для вкладки Doctors)."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    marker_cluster = plugins.MarkerCluster(name="Врачи")

    for doc in doctors:
        cat = doc.get("category", "B")
        color = CATEGORY_COLORS.get(cat, "#3498db")

        popup_html = f"""
        <div style="width:220px;">
            <b>{doc['full_name']}</b><br>
            🩺 {doc['specialty']}<br>
            ⭐ Категория {cat}<br>
            ❤️ Лояльность: {doc.get('loyalty_score', 0)}/10<br>
            📊 Продажи: {doc.get('avg_sales_brick', 0)}<br>
            📅 Визитов: {doc.get('total_visits', 0)}<br>
            🕐 Посл. визит: {doc.get('days_since_last_visit', '?')} дн. назад
        </div>
        """

        folium.CircleMarker(
            location=[doc["latitude"], doc["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{doc['full_name']} | {doc['specialty']}",
        ).add_to(marker_cluster)

    marker_cluster.add_to(m)
    return m