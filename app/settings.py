"""Settings: thresholds come from configs/inference.json, secrets and paths from environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    model_path: Path
    weights_url: str | None
    device: str | None
    conf_high: float
    conf_low: float
    imgsz: int
    overlap_iou_warning: float
    min_sharpness: float
    min_brightness: float
    max_brightness: float
    max_upload_bytes: int
    value_overrides: dict = field(default_factory=dict)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_extra_body: dict = field(default_factory=dict)


def load_settings() -> Settings:
    config_path = Path(os.getenv("INFERENCE_CONFIG", "configs/inference.json"))
    cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    conf_high = float(os.getenv("CONF_HIGH", cfg.get("conf_high", 0.5)))
    conf_low = float(os.getenv("CONF_LOW", cfg.get("conf_low", 0.25)))
    if not 0 < conf_low <= conf_high < 1:
        raise ValueError(f"Need 0 < conf_low <= conf_high < 1, got low={conf_low}, high={conf_high}")
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "weights/best.pt")),
        weights_url=os.getenv("WEIGHTS_URL") or None,
        device=os.getenv("DEVICE") or None,
        conf_high=conf_high,
        conf_low=conf_low,
        imgsz=int(cfg.get("imgsz", 640)),
        overlap_iou_warning=float(cfg.get("overlap_iou_warning", 0.45)),
        min_sharpness=float(cfg.get("min_sharpness", 60.0)),
        min_brightness=float(cfg.get("min_brightness", 40.0)),
        max_brightness=float(cfg.get("max_brightness", 225.0)),
        max_upload_bytes=int(float(cfg.get("max_upload_mb", 10)) * 1024 * 1024),
        value_overrides=cfg.get("value_overrides", {}),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL") or None,
        llm_extra_body=json.loads(os.getenv("LLM_EXTRA_BODY") or "{}"),
    )
