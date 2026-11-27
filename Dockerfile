# Inference image: FastAPI server serving a trained checkpoint on CPU.
# Build:  docker build -t brain-tumor-api .
# Run:    docker run -p 8000:8000 -v $(pwd)/models:/app/models brain-tumor-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CHECKPOINT=/app/models/best.pt

WORKDIR /app

# CPU-only PyTorch keeps the image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt requirements-api.txt pyproject.toml README.md LICENSE ./
COPY src ./src
# Only what inference needs: no DVC / MLflow / Optuna in the serving image.
RUN pip install -r requirements-api.txt numpy pandas scikit-learn tqdm pyyaml h5py \
    opencv-python-headless matplotlib \
    && pip install --no-deps -e .

COPY api ./api
COPY params.yaml ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
