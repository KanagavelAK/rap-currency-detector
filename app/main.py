"""FastAPI service.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import logging
import time
import uuid
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from app.detector import Detector
from app.llm import LLMClient
from app.reasoning import DetectorUnavailable, ReasoningConfig, ReasoningEngine
from app.settings import Settings, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("currency_api")

STATE: dict = {}


def ensure_weights(settings: Settings) -> None:
    """Download weights on first start if they aren't on disk (keeps big files out of git)."""
    if settings.model_path.exists() or not settings.weights_url:
        return
    logger.info("Downloading weights from %s", settings.weights_url)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings.model_path.with_suffix(".part")
    with requests.get(settings.weights_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(settings.model_path)
    logger.info("Weights saved to %s", settings.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    detector = None
    try:
        ensure_weights(settings)
        detector = Detector(settings.model_path, settings.imgsz, settings.device, settings.value_overrides)
        logger.info("Model loaded: %s classes=%s", settings.model_path, detector.names)
    except Exception:
        logger.exception("Model failed to load; /detect and detector questions will return 503")
    llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                    extra_body=settings.llm_extra_body)
    logger.info("LLM phrasing %s", f"enabled ({settings.llm_model})" if llm.enabled else "disabled (templates only)")
    cfg = ReasoningConfig(settings.conf_high, settings.conf_low, settings.overlap_iou_warning,
                          settings.min_sharpness, settings.min_brightness, settings.max_brightness)
    STATE.update(settings=settings, detector=detector, llm=llm, engine=ReasoningEngine(detector, llm, cfg))
    yield
    STATE.clear()


app = FastAPI(title="Indian Banknote Detection & Reasoning API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] unhandled error on %s %s", request_id, request.method, request.url.path)
        response = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": request_id})
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("[%s] %s %s -> %s in %.0f ms", request_id, request.method, request.url.path, response.status_code, ms)
    return response


def read_image(upload: UploadFile, max_bytes: int) -> Image.Image:
    data = upload.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated or corrupt files
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos: apply the stored rotation
        return img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not a valid image (JPEG/PNG/WebP/BMP).")


@app.get("/health")
def health():
    settings: Settings = STATE["settings"]
    detector = STATE["detector"]
    return {
        "status": "ok" if detector else "degraded",
        "model_loaded": detector is not None,
        "weights": settings.model_path.name,
        "classes": detector.names if detector else None,
        "llm_enabled": STATE["llm"].enabled,
        "thresholds": {"conf_high": settings.conf_high, "conf_low": settings.conf_low},
    }


@app.post("/detect")
def detect(file: UploadFile = File(..., description="Photo containing Indian banknotes")):
    detector = STATE["detector"]
    settings: Settings = STATE["settings"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MODEL_PATH / WEIGHTS_URL.")
    img = read_image(file, settings.max_upload_bytes)
    detections, inference_ms = detector.predict(img, settings.conf_low)
    for d in detections:
        d["confident"] = d["confidence"] >= settings.conf_high
    return {
        "image": {"filename": file.filename, "width": img.width, "height": img.height},
        "count": len(detections),
        "detections": detections,
        "thresholds": {"conf_high": settings.conf_high, "conf_low": settings.conf_low},
        "model": detector.weights_name,
        "inference_ms": inference_ms,
    }


@app.post("/ask")
def ask(question: str = Form(..., min_length=1, max_length=500),
        file: UploadFile | None = File(None, description="Photo the question is about")):
    settings: Settings = STATE["settings"]
    img = read_image(file, settings.max_upload_bytes) if file is not None and file.filename else None
    try:
        return STATE["engine"].ask(question, img)
    except DetectorUnavailable:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MODEL_PATH / WEIGHTS_URL.")


@app.get("/")
def root():
    return {"message": "Indian Banknote Detection API. See /docs for interactive documentation.",
            "endpoints": ["/health", "/detect", "/ask"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
