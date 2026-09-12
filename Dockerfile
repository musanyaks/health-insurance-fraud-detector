FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY fraud_engine ./fraud_engine
COPY api ./api
COPY config ./config
COPY synthetic ./synthetic

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]