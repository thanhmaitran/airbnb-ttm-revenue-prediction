# Airbnb TTM Revenue Prediction — reproducible runtime image.
#
#   docker build -t airbnb-revenue-prediction .
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#       airbnb-revenue-prediction python train.py --model xgboost
#
# GPU note: this base image is CPU-only. On an Oracle VM GPU instance, swap
# the base for nvidia/cuda:12.1-runtime-ubuntu22.04, install Python 3.11 and
# `pip install xgboost` with CUDA support — see docs/big_data.md.
FROM python:3.11-slim

WORKDIR /app

# System deps needed by pyarrow / matplotlib at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user (good practice for a portfolio-quality image)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

ENTRYPOINT ["python"]
CMD ["train.py", "--model", "xgboost"]
