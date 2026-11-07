
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application
COPY api/ ./api/

# Copy production model files
COPY deployment/model.pkl ./deployment/model.pkl
COPY deployment/preprocessor.pkl ./deployment/preprocessor.pkl
COPY deployment/model_metadata.json ./deployment/model_metadata.json

# Create inference log directory
RUN mkdir -p logs/inference

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# FastAPI port
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
