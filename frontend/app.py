"""
PharmaPath AI — Streamlit App (Main Page)
==========================================
Запуск:
    streamlit run frontend/app.py --server.port 8501
"""

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PharmaPath AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

with open("frontend/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000/api/v1"

if "current_route" not in st.session_state:
    st.session_state.current_route = None

if "selected_rep" not in st.session_state:
    st.session_state.selected_rep = "REP-001"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 💊 PharmaPath AI")
    st.markdown("*Умный планировщик рабочего дня медицинского представителя*")
    st.markdown("---")

    st.markdown("### ⚙️ Настройки")

    st.session_state.api_base_url = st.text_input(
        "API URL",
        value=st.session_state.api_base_url,
        help="Адрес бэкенда FastAPI",
    )

    st.session_state.selected_rep = st.selectbox(
        "Медпред",
        [f"REP-{i:03d}" for i in range(1, 11)],
        index=0,
    )

    st.markdown("---")

    # Health check
    from components.api_client import health_check
    health = health_check()
    if health:
        st.success(f"✅ Backend: {health.get('status', '?')}")
        st.caption(
            f"v{health.get('version', '?')} | "
            f"Врачей: {health.get('doctors_count', 0)} | "
            f"Визитов: {health.get('visits_count', 0)}"
        )
        if health.get("models_loaded"):
            st.caption("🧠 ML-модели загружены")
        else:
            st.warning("⚠️ ML-модели НЕ загружены")
    else:
        st.error("❌ Backend недоступен")

# ── Main Page ─────────────────────────────────────────────────────────────────

st.title("💊 PharmaPath AI")
st.markdown(
    """
    **Умный планировщик рабочего дня медицинского представителя**

    Система оптимизирует маршрут медпреда на день, используя:
    - 🧠 **ML-скоринг** — оценка ценности и вероятности успеха каждого визита
    - 🗺 **OR-Tools** — математическая оптимизация маршрута (VRPTW)
    - 📝 **LLM** — структурирование отчётов

    ---

    ### 📖 Навигация

    | Страница | Описание |
    |----------|----------|
    | 🗺 **Route Planner** | Генерация маршрута на день |
    | 👨‍⚕ **Doctors** | База врачей с фильтрами и картой |
    | 📝 **Report** | Отправка и анализ отчёта о визите |
    | 📊 **Analytics** | KPI-дашборд: до/после оптимизации |

    ---

    ### 🏗 Архитектура
    """
)

st.code("""
┌─────────────┐    ┌──────────────────────────────────────────────┐
│  Streamlit   │───▶│                FastAPI Backend                │
│  Frontend    │◀───│                                              │
└─────────────┘    │  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
                   │  │ Scoring  │ │ OR-Tools │ │ LLM Service │  │
                   │  │ (CatBoost)│ │ (VRPTW)  │ │ (Llama-3)   │  │
                   │  └──────────┘ └──────────┘ └─────────────┘  │
                   │  ┌──────────────────────────────────────────┐│
                   │  │         Data Provider (CSV / PostgreSQL) ││
                   │  └──────────────────────────────────────────┘│
                   └──────────────────────────────────────────────┘
""", language="text")

# ── Quick Stats ───────────────────────────────────────────────────────────────

if health:
    st.markdown("### 📊 Быстрая статистика")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👨‍⚕️ Врачей в базе", health.get("doctors_count", 0))
    c2.metric("📋 Визитов (история)", f"{health.get('visits_count', 0):,}")
    c3.metric("🧠 Модели", "✅" if health.get("models_loaded") else "❌")
    c4.metric("📡 API", health.get("version", "?"))