FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend/ /app/frontend/

ENV PHARMA_API_URL=http://backend:8000/api/v1

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]