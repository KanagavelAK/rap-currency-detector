"""Print sharpness/brightness percentiles for a folder, to set guardrail limits from data.

Usage:
    python scripts/quality_stats.py --images datasets/currency/images/val
    python scripts/quality_stats.py --images own_photos/s4

Rule of thumb: set min_sharpness a little below the 5th percentile of photos you consider usable,
then check that your deliberately blurry test shots fall below it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.quality import measure  # noqa: E402
from src.labels import find_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path)
    args = parser.parse_args()

    paths = find_images(args.images)
    if not paths:
        sys.exit(f"No images in {args.images}")
    rows = []
    for p in paths:
        with Image.open(p) as img:
            rows.append(measure(img))
    for key in ("sharpness", "brightness"):
        arr = np.array([r[key] for r in rows])
        pct = {q: round(float(np.percentile(arr, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)}
        print(f"{key:10s} percentiles: {pct}")
    print(f"({len(rows)} images)")


if __name__ == "__main__":
    main()
