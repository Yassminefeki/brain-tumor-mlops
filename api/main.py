"""FastAPI inference service.

Run locally::

    MODEL_CHECKPOINT=models/best.pt uvicorn api.main:app --reload

Endpoints:

* ``GET  /health``   service and model status
* ``GET  /model``    information about the loaded checkpoint
* ``POST /predict``  upload a .mat / .png / .jpg -> class probabilities
* ``POST /explain``  upload an image -> Grad-CAM overlay (PNG)

Educational project: predictions are NOT a medical diagnosis.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from brain_tumor import __version__
from brain_tumor.inference import Predictor, load_image_bytes

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
state: dict[str, Any] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    checkpoint = os.environ.get("MODEL_CHECKPOINT", "models/best.pt")
    try:
        state["predictor"] = Predictor(checkpoint)
    except Exception as exc:  # keep the service up and report the problem on /health
        state["error"] = f"Could not load {checkpoint}: {exc}"
    yield
    state["predictor"] = None


app = FastAPI(
    title="Brain Tumor MRI Classifier",
    version=__version__,
    description="Classifies T1-weighted contrast-enhanced MRI slices into meningioma, glioma "
    "or pituitary tumour. Educational project — not a medical device.",
    lifespan=lifespan,
)


def get_predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "Model not loaded")
    return state["predictor"]


async def read_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
    try:
        return load_image_bytes(data, file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = state["predictor"] is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "error": state["error"],
        "version": __version__,
    }


@app.get("/model")
def model_info() -> dict[str, Any]:
    p = get_predictor()
    return {**p.info, "classes": p.class_names, "device": str(p.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    predictor = get_predictor()
    image = await read_upload(file)
    return {
        "filename": file.filename,
        **predictor.predict(image),
        "disclaimer": "Educational use only — not a medical diagnosis.",
    }


@app.post("/explain", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def explain(file: UploadFile = File(...)) -> Response:
    predictor = get_predictor()
    image = await read_upload(file)
    rgb = (predictor.explain(image) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
