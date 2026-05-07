FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY src/requirements.txt /app/src/requirements.txt
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir \
    -r /app/src/requirements.txt \
    -r /app/ml/requirements.txt

# Copy code
COPY src/ /app/src/
COPY ml/ /app/ml/
COPY models/ /app/models/
COPY data/output/ /app/data/output/

# Ensure __init__.py
RUN touch /app/src/__init__.py \
    && touch /app/src/services/__init__.py \
    && touch /app/src/routers/__init__.py

ENV PYTHONPATH=/app
ENV PHARMA_DATA_DIR=/app/data/output
ENV PHARMA_MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]