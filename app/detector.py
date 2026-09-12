"""Thin wrapper around the trained RT-DETR model.

Loaded once at API startup (loading per request would add seconds of latency).
A lock serialises predictions because one Ultralytics model object is not safe to call
from several request threads at the same time.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image

from src.labels import value_from_class_name


class Detector:
    def __init__(self, weights: Path, imgsz: int = 640, device: str | None = None, value_overrides: dict | None = None):
        from ultralytics import RTDETR  # imported here so the rest of the app can be tested without it

        self.model = RTDETR(str(weights))
        self.imgsz = imgsz
        self.device = device
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.values = {k: value_from_class_name(v, value_overrides) for k, v in self.names.items()}
        missing = [self.names[k] for k, v in self.values.items() if v is None]
        if missing:
            raise ValueError(f"Classes without a rupee value: {missing}. Add them to value_overrides.")
        self.weights_name = Path(weights).name
        self._lock = threading.Lock()

    def predict(self, img: Image.Image, conf_floor: float) -> tuple[list[dict], float]:
        """Return detections at or above conf_floor, sorted by confidence, plus inference time in ms."""
        start = time.perf_counter()
        with self._lock:
            result = self.model.predict(source=img, conf=conf_floor, imgsz=self.imgsz,
                                        device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - start) * 1000
        detections = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(xyxy, confs, classes):
                detections.append({
                    "class_id": int(cls),
                    "class_name": self.names[int(cls)],
                    "value": self.values[int(cls)],
                    "confidence": round(float(conf), 4),
                    "box": {"x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                            "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1)},
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, round(ms, 1)
