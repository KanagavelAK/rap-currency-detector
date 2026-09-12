"""Helpers for YOLO-format labels, class names and banknote values.

YOLO label format (one line per box, all coordinates normalised to 0-1):
    <class_id> <x_center> <y_center> <width> <height>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_DENOMINATIONS = (10, 20, 50, 100, 200, 500, 2000)

# Longest phrases first so "five hundred" wins over "hundred".
_WORD_VALUES = [
    ("two thousand", 2000),
    ("five hundred", 500),
    ("two hundred", 200),
    ("hundred", 100),
    ("fifty", 50),
    ("twenty", 20),
    ("ten", 10),
]


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def xyxy(self, img_w: float, img_h: float) -> tuple[float, float, float, float]:
        """Convert a normalised centre box to pixel corner coordinates."""
        x1 = (self.cx - self.w / 2) * img_w
        y1 = (self.cy - self.h / 2) * img_h
        x2 = (self.cx + self.w / 2) * img_w
        y2 = (self.cy + self.h / 2) * img_h
        return x1, y1, x2, y2

    @property
    def area(self) -> float:
        """Box area as a fraction of the image area."""
        return self.w * self.h


def read_yolo_labels(path: Path) -> tuple[list[Box], list[str]]:
    """Read a YOLO label file. Returns (boxes, problems). Missing file -> no boxes."""
    boxes: list[Box] = []
    problems: list[str] = []
    if not path.exists():
        return boxes, problems
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"{path.name}:{line_no} expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            problems.append(f"{path.name}:{line_no} non-numeric value")
            continue
        if w <= 0 or h <= 0:
            problems.append(f"{path.name}:{line_no} zero or negative size")
            continue
        if not all(-0.01 <= v <= 1.01 for v in (cx, cy, w, h)):
            problems.append(f"{path.name}:{line_no} coordinates outside 0-1")
            continue
        boxes.append(Box(cls, _clip(cx), _clip(cy), _clip(w), _clip(h)))
    return boxes, problems


def write_yolo_labels(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_class_names(path: Path) -> list[str]:
    """Read class names from classes.txt / obj.names (one per line) or a data.yaml."""
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
        names = data["names"]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        return [str(n) for n in names]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def find_class_file(folder: Path) -> Path | None:
    """Look for a class-name file anywhere under a folder."""
    for name in ("classes.txt", "obj.names", "data.yaml"):
        hits = sorted(folder.rglob(name))
        if hits:
            return hits[0]
    return None


def value_from_class_name(name: str, overrides: dict | None = None) -> int | None:
    """Turn a class name like '500', 'Rs_100_new' or 'fifty' into a rupee value."""
    if overrides and name in overrides:
        return int(overrides[name])
    for match in re.findall(r"\d+", name):
        if int(match) in VALID_DENOMINATIONS:
            return int(match)
    text = " " + re.sub(r"[^a-z]+", " ", name.lower()) + " "
    for phrase, value in _WORD_VALUES:
        if f" {phrase} " in text:
            return value
    return None


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: Path) -> Path:
    """Find the label file for an image.

    Supports both layouts:
      folder/img.jpg + folder/img.txt            (LabelImg, CVAT 'YOLO 1.1')
      .../images/x/img.jpg + .../labels/x/img.txt (Ultralytics layout)
    """
    same_folder = image_path.with_suffix(".txt")
    if same_folder.exists():
        return same_folder
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return same_folder


def _clip(v: float) -> float:
    return min(max(v, 0.0), 1.0)
