# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

# needed by xgboost/catboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080
# module:object -> app.py defines `application`
CMD ["gunicorn","-w","2","-k","gthread","--threads","4","-b","0.0.0.0:8080","app:application"]
