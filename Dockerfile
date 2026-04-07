FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY scenarios/ ./scenarios/
COPY hoa_env/ ./hoa_env/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "hoa_env.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--timeout-keep-alive", "300", \
     "--ws-ping-interval", "5", \
     "--ws-ping-timeout", "20"]
