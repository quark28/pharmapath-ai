"""
PharmaPath AI — Report Submission Page
"""

import streamlit as st
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.api_client import submit_report, get_doctors

st.set_page_config(page_title="Report", page_icon="📝", layout="wide")
st.title("📝 Отчёт о визите")

# ── Выбор врача ───────────────────────────────────────────────────────────────

route = st.session_state.get("current_route")

if route and route.get("stops"):
    st.info("💡 Выберите врача из текущего маршрута или введите ID вручную.")
    route_doctors = {
        f"#{s['order']} {s['doctor_name']} ({s['specialty']})": s["doctor_id"]
        for s in route["stops"]
    }
    selected_label = st.selectbox("Врач из маршрута", list(route_doctors.keys()))
    doctor_id = route_doctors[selected_label]
else:
    st.warning("Маршрут не построен. Введите ID врача вручную.")
    doctor_id = st.text_input("Doctor ID (UUID)", "")

if not doctor_id:
    st.stop()

# ── Форма отчёта ──────────────────────────────────────────────────────────────

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    visit_time = st.time_input("🕐 Время визита")
    visit_time_str = visit_time.strftime("%H:%M")

with col2:
    duration = st.number_input("⏱ Длительность (мин)", min_value=0, max_value=120, value=20)

with col3:
    status = st.selectbox("📊 Статус", ["Success", "Cancelled", "Moved"])

st.markdown("### 📝 Текст отчёта")
report_text = st.text_area(
    "Опишите визит",
    height=200,
    placeholder=(
        "Визит к Иванову А.С. Обсудили применение Лориста при "
        "артериальной гипертензии. Врач заинтересован, но упомянул "
        "что назначает Валз. Попросил результаты КИ. Договорились "
        "о повторном визите через 2 недели."
    ),
)

# ── Шаблоны ───────────────────────────────────────────────────────────────────

with st.expander("📋 Шаблоны отчётов"):
    templates = {
        "Позитивный визит": (
            "Визит к врачу. Обсудили применение [препарат] при [диагноз]. "
            "Врач положительно отнёсся. Готов назначить 3 пациентам. "
            "Оставил образцы и буклеты."
        ),
        "Скептический врач": (
            "Визит к врачу. Презентовал [препарат]. Врач скептически отнёсся. "
            "Основное возражение: высокая цена для пациентов. "
            "Упомянул, что использует [конкурент]. Договорились предоставить "
            "сравнительные данные."
        ),
        "Короткий визит": (
            "Короткий визит. Врач был занят на приёме. Удалось обсудить "
            "[препарат]. Оставил буклет. Повторный визит через неделю."
        ),
    }
    for name, text in templates.items():
        if st.button(f"📋 {name}", key=f"tpl_{name}"):
            st.session_state["_report_template"] = text
            st.rerun()

if "_report_template" in st.session_state:
    report_text = st.session_state.pop("_report_template")

# ── Отправка ──────────────────────────────────────────────────────────────────

st.markdown("---")

if st.button("📤 Отправить отчёт", type="primary", disabled=len(report_text) < 5):
    rep_id = st.session_state.get("selected_rep", "REP-001")

    with st.spinner("🧠 Обработка отчёта (LLM-анализ)..."):
        result = submit_report(
            rep_id=rep_id,
            doctor_id=doctor_id,
            visit_time=visit_time_str,
            duration_minutes=duration,
            status=status,
            report_text=report_text,
        )

    if result:
        st.success("✅ Отчёт сохранён и обработан!")

        st.markdown("### 🤖 Результат LLM-анализа")
        st.caption(f"Backend: `{result.get('llm_backend', '?')}`")

        col_a, col_b = st.columns(2)

        with col_a:
            # Sentiment
            sent = result.get("sentiment", "Neutral")
            sent_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}.get(sent, "❓")
            st.markdown(f"**Тональность:** {sent_emoji} {sent}")

            # Competitors
            competitors = result.get("competitors", [])
            if competitors:
                st.markdown(f"**🏢 Конкуренты:** {', '.join(competitors)}")
            else:
                st.markdown("**🏢 Конкуренты:** не упомянуты")

        with col_b:
            # Objections
            objections = result.get("objections", [])
            if objections:
                st.markdown("**⚠️ Возражения:**")
                for obj in objections:
                    st.markdown(f"  - {obj}")

            # Agreements
            agreements = result.get("agreements", [])
            if agreements:
                st.markdown("**🤝 Договорённости:**")
                for agr in agreements:
                    st.markdown(f"  - {agr}")

        # Key topics
        topics = result.get("key_topics", [])
        if topics:
            st.markdown(f"**🏷 Темы:** {', '.join(topics)}")

        # Raw JSON
        with st.expander("🔍 Полный JSON-ответ"):
            st.json(result)