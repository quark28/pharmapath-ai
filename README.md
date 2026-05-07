<div align="center">

# 💊 PharmaPath AI

**Умный планировщик рабочего дня медицинского представителя**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-yellow)](https://catboost.ai)
[![OR--Tools](https://img.shields.io/badge/OR--Tools-Optimization-orange)](https://developers.google.com/optimization)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)](https://streamlit.io)

*Проект для курса «Модели и технологии цифровой трансформации»*

</div>

---

## 📋 Проблема

Медицинские представители (медпреды) фармкомпаний планируют свой рабочий день **вручную**:
- ❌ Посещают «удобных», а не приоритетных врачей
- ❌ Тратят 30–40% времени на переезды из-за неоптимальных маршрутов
- ❌ Пропускают ключевых врачей категории A/B
- ❌ Отчёты — неструктурированный поток сознания

**Итог:** компания теряет **15–25% потенциальной выручки** при ФОТ команды ~72 млн ₽/год.

## 💡 Решение

**PharmaPath AI** — система, которая:

1. **🧠 Оценивает** каждого врача (ML-скоринг: ценность × вероятность успеха)
2. **🗺 Строит** оптимальный маршрут на день (OR-Tools, задача VRPTW)
3. **📝 Анализирует** отчёты медпредов (LLM → структурированные данные)
4. **📊 Показывает** бизнес-эффект (KPI-дашборд)

## 🏗 Архитектура

```
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │   :8501              │
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │   FastAPI Backend    │
                    │   :8000              │
                    │                      │
                    │  ┌────────────────┐  │
                    │  │  ML Scoring    │  │
                    │  │  (CatBoost)    │  │
                    │  └────────────────┘  │
                    │  ┌────────────────┐  │
                    │  │  Optimizer     │  │
                    │  │  (OR-Tools)    │  │
                    │  └────────────────┘  │
                    │  ┌────────────────┐  │
                    │  │  LLM Service   │  │
                    │  │  (Llama-3)     │  │
                    │  └────────────────┘  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  PostgreSQL+PostGIS  │
                    │  (CSV в MVP)         │
                    └──────────────────────┘
```

## 🚀 Быстрый старт

### Вариант 1: Docker (рекомендуется)

```bash
git clone https://github.com/quark28/pharmapath-ai.git
cd pharmapath-ai

# Генерация данных и обучение моделей
make all

# Запуск всех сервисов
make docker-up
```

Открыть:
- 🖥 **Frontend:** http://localhost:8501
- 📖 **API Docs:** http://localhost:8000/docs
- ❤️ **Health:** http://localhost:8000/health

### Вариант 2: Локально

```bash
# 1. Создать окружение
python -m venv .venv && source .venv/bin/activate

# 2. Установить зависимости
pip install -r src/requirements.txt
pip install -r ml/requirements.txt
pip install -r frontend/requirements.txt

# 3. Данные + модели
cd data && python faker_generator.py && cd ..
cd ml && python train_pipeline.py && cd ..

# 4. Запуск бэкенда (терминал 1)
uvicorn src.main:app --reload --port 8000

# 5. Запуск фронтенда (терминал 2)
streamlit run frontend/app.py --server.port 8501
```

## 🧠 ML Pipeline

### Value Model (CatBoost Regressor)

Предсказывает **ценность визита** к врачу (0–100).

| Фича | Описание |
|------|----------|
| specialty | Специальность врача |
| category | Маркетинговая категория (A/B/C) |
| avg_sales_brick | Продажи по территории |
| loyalty_score | Лояльность (0–10) |
| days_since_last_visit | Рецентность |
| success_rate | Историческая конверсия |

**Метрики (test):** RMSE ≈ 3.2, R² ≈ 0.97

### Probability Model (CatBoost Classifier)

Предсказывает **P(success)** — вероятность успешного визита.

| Фича | Описание |
|------|----------|
| hour, day_of_week | Когда идём |
| is_friday, is_weekend | Паттерны доступности |
| loyalty_score | Расположенность врача |
| historical_success_rate | Как было раньше |

**Метрики (test):** ROC-AUC ≈ 0.89, Brier ≈ 0.12

### Optimizer (Google OR-Tools)

**Prize-Collecting VRPTW** — маршрут с максимальным суммарным Score:

```
Score_i = Value_i × P(success)_i
Maximize: Σ Score_visited
Subject to:
  - Time windows (часы приёма врача)
  - Max visits ≤ 14
  - Working day ≤ 8 hours
```

## 📊 Business Value

| Метрика | До | После | Δ |
|---------|-------|---------|------|
| Визитов/день | 10.4 | 14.2 | **+37%** |
| Конверсия | 63% | 78% | **+15%** |
| Время в пути | 132 мин | 85 мин | **−36%** |
| Coverage A/B | 72% | 94% | **+22%** |

**Оценка ROI: ~1600%** (доп. выручка ~86M ₽/год при затратах ~5M ₽/год)

## 📁 Структура проекта

```
pharmapath-ai/
├── data/               # Генерация синтетических данных
├── ml/                 # Feature engineering + обучение моделей
├── models/             # Обученные .cbm файлы
├── src/                # FastAPI backend
│   ├── services/       # Scoring, Optimizer, LLM
│   └── routers/        # API endpoints
├── frontend/           # Streamlit UI
│   └── pages/          # Route Planner, Doctors, Report, Analytics
├── tests/              # pytest
├── docker/             # Dockerfiles
├── docker-compose.yml
└── Makefile
```

## 🛠 Стек технологий

| Компонент | Технология | Зачем |
|-----------|-----------|-------|
| Backend | FastAPI | Async API, auto-docs, Pydantic validation |
| ML | CatBoost | Категориальные фичи без one-hot, быстрый inference |
| Optimization | Google OR-Tools | Индустриальный солвер VRPTW |
| LLM | Llama-3 (Ollama) | Локально, данные не уходят в облако |
| Frontend | Streamlit + Folium | Быстрая визуализация карт и графиков |
| DB | PostgreSQL + PostGIS | Гео-запросы (ST_DWithin), готовность к проду |
| Deploy | Docker Compose | Воспроизводимость, одна команда |

## 🧪 Тесты

```bash
python -m pytest tests/ -v
```

## 📖 API Endpoints

| Method | Path | Описание |
|--------|------|----------|
| GET | `/health` | Проверка статуса |
| GET | `/api/v1/doctors/` | Список врачей (фильтры) |
| GET | `/api/v1/doctors/{id}` | Карточка врача |
| POST | `/api/v1/doctors/upload` | Загрузка CSV |
| POST | `/api/v1/routes/generate` | 🧠 Генерация маршрута |
| POST | `/api/v1/reports/submit` | Отчёт + LLM-анализ |

## 🗺 Roadmap

- [x] MVP: данные, модели, API, UI
- [ ] PostgreSQL вместо CSV
- [ ] Auth (JWT + RBAC)
- [ ] Мобильное приложение (Flutter)
- [ ] MLflow для версионирования моделей
- [ ] Real-time GPS tracking
- [ ] A/B тестирование маршрутов
- [ ] Интеграция с CRM (Veeva, OCE)

</div>