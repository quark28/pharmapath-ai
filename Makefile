.PHONY: all data train run-backend run-frontend docker-up docker-down test clean

# ══════════════════════════════════════════════════════════════════════════════
#  PharmaPath AI — Makefile
# ══════════════════════════════════════════════════════════════════════════════

# Полная сборка с нуля
all: data train

# Генерация данных
data:
	@echo "📦 Генерация данных..."
	cd data && python faker_generator.py

# Обучение моделей
train:
	@echo "🧠 Обучение моделей..."
	cd ml && python train_pipeline.py

# Запуск бэкенда (dev)
run-backend:
	@echo "🚀 Запуск бэкенда..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Запуск фронтенда (dev)
run-frontend:
	@echo "🖥 Запуск фронтенда..."
	streamlit run frontend/app.py --server.port 8501

# Docker: собрать и поднять
docker-up: data train
	@echo "🐳 Docker Compose up..."
	docker-compose up --build -d
	@echo "✅ Backend:  http://localhost:8000/docs"
	@echo "✅ Frontend: http://localhost:8501"

# Docker: остановить
docker-down:
	docker-compose down -v

# Тесты
test:
	@echo "🧪 Запуск тестов..."
	python -m pytest tests/ -v --tb=short

# Очистка
clean:
	rm -rf data/output/*.csv data/output/*.json
	rm -rf models/*.cbm models/*.json models/*.png
	rm -rf __pycache__ **/__pycache__
	@echo "🧹 Очищено"