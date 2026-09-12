"""Prepare your own phone photos BEFORE labelling them.

Why: phones often save photos sideways and store the rotation in EXIF metadata.
Some tools apply that rotation and some don't, so boxes drawn in the labelling
tool can end up misaligned with the image the trainer reads. Baking the rotation
into the pixels removes that risk. We also shrink the long side to 1280 px so
uploads and labelling are faster (training uses 640 px anyway).

Usage:
    python scripts/fix_orientation.py --src raw_photos/s1 --dst own_photos/s1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import find_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args()

    images = find_images(args.src)
    if not images:
        sys.exit(f"No images found in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(images, start=1):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            out = args.dst / f"{args.src.name}_{i:03d}.jpg"
            img.save(out, "JPEG", quality=95)  # saved without EXIF, so no hidden rotation
    print(f"Saved {len(images)} images to {args.dst}")


if __name__ == "__main__":
    main()
