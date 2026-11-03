
FROM python:3.11-slim

WORKDIR /app

COPY requirments.txt .

RUN pip install --no-cache-dir -r requirments.txt && \
    pip install --no-cache-dir flask gunicorn

COPY deployment/ ./deployment/
COPY artifacts/model_registry/ ./artifacts/model_registry/
COPY models/preprocessor.pkl ./models/preprocessor.pkl

RUN mkdir -p logs/inference

ENV PYTHONPATH=/app/deployment
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "deployment.app:app"]
