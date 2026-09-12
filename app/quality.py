"""Cheap image-quality checks used by the guardrail.

Why: "no notes detected" on a blurry or dark photo does NOT mean "no money".
Before trusting an empty or thin result, we check whether the photo was usable.

sharpness  = variance of the Laplacian (edge strength). Blurry photos have weak edges -> low value.
brightness = mean grayscale level, 0 (black) to 255 (white).
The image is first resized to a fixed width so the numbers are comparable across photo sizes.
Calibrate the limits with scripts/quality_stats.py on your own photos.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ANALYSIS_WIDTH = 512


def measure(img: Image.Image) -> dict:
    gray = img.convert("L")
    if gray.width != ANALYSIS_WIDTH:
        new_h = max(1, round(gray.height * ANALYSIS_WIDTH / gray.width))
        gray = gray.resize((ANALYSIS_WIDTH, new_h), Image.Resampling.BILINEAR)
    g = np.asarray(gray, dtype=np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return {"sharpness": 0.0, "brightness": float(g.mean())}
    laplacian = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return {"sharpness": round(float(laplacian.var()), 2), "brightness": round(float(g.mean()), 2)}


def issues(metrics: dict, min_sharpness: float, min_brightness: float, max_brightness: float) -> list[str]:
    found = []
    if metrics["sharpness"] < min_sharpness:
        found.append("blurry")
    if metrics["brightness"] < min_brightness:
        found.append("too_dark")
    if metrics["brightness"] > max_brightness:
        found.append("overexposed")
    return found
