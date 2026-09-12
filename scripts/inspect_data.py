"""Inspect a YOLO dataset before trusting it.

Answers the questions you must settle in the first hour:
  1. What are the classes, and does every class map to a rupee value?
  2. Do boxes cover the whole note or only part of it? (box area stats + drawn samples)
  3. How many images contain more than one note?
  4. Are there broken labels or images without labels?
  5. Does the dataset's own train/validation split leak near-duplicates?

Usage:
    python scripts/inspect_data.py --data-dir /path/to/IndianBankNotes --out reports/inspect
    (add --classes path/to/classes.txt if it is not found automatically)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import (  # noqa: E402
    find_class_file, find_images, label_path_for, read_class_names, read_yolo_labels, value_from_class_name,
)
from src.phash import near_duplicate_pairs, phash_file  # noqa: E402


def draw_boxes(image_path: Path, boxes, names: list[str], out_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy(w, h)
            label = names[b.cls] if b.cls < len(names) else f"id{b.cls}"
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=max(2, w // 300))
            draw.text((x1 + 4, y1 + 4), label, fill=(0, 255, 0))
        img.save(out_path, "JPEG", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/inspect"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dup-distance", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = find_images(args.data_dir)
    if not images:
        sys.exit(f"No images found under {args.data_dir}")

    class_file = args.classes or find_class_file(args.data_dir)
    names = read_class_names(class_file) if class_file else []
    print(f"Images found: {len(images)}")
    print(f"Class file: {class_file or 'NOT FOUND - pass --classes'}")

    class_counts: Counter = Counter()
    notes_per_image: Counter = Counter()
    areas: list[float] = []
    no_label: list[str] = []
    problems: list[str] = []
    per_image = {}

    for path in images:
        lp = label_path_for(path)
        if not lp.exists():
            no_label.append(str(path))
            continue
        boxes, probs = read_yolo_labels(lp)
        problems.extend(probs)
        per_image[path] = boxes
        notes_per_image[min(len(boxes), 4)] += 1
        for b in boxes:
            class_counts[b.cls] += 1
            areas.append(b.area)

    print("\n=== Classes (box counts) ===")
    class_rows = []
    for cls_id in sorted(class_counts):
        name = names[cls_id] if cls_id < len(names) else f"<unnamed id {cls_id}>"
        value = value_from_class_name(name) if cls_id < len(names) else None
        flag = "" if value else "   <-- NO RUPEE VALUE: add to value_overrides in configs/inference.json"
        print(f"  id {cls_id:2d}  {name:20s}  value={value}  boxes={class_counts[cls_id]}{flag}")
        class_rows.append({"id": cls_id, "name": name, "value": value, "boxes": class_counts[cls_id]})
    if names and len(names) != len(class_counts):
        print(f"  NOTE: class file lists {len(names)} names but {len(class_counts)} ids are used.")

    print("\n=== Notes per image ===")
    for k in sorted(notes_per_image):
        label = f"{k}+" if k == 4 else str(k)
        print(f"  {label} notes: {notes_per_image[k]} images")

    area_stats = {}
    if areas:
        arr = np.array(areas)
        area_stats = {q: round(float(np.percentile(arr, p)), 4) for q, p in (("p10", 10), ("median", 50), ("p90", 90))}
        print("\n=== Box area as a fraction of the image ===")
        print(f"  p10={area_stats['p10']}  median={area_stats['median']}  p90={area_stats['p90']}")
        print("  Whole-note boxes in close-up photos are usually large (>0.1).")
        print("  Tiny boxes everywhere may mean only the numeral/portrait was labelled - check the samples!")

    print(f"\nImages without a label file: {len(no_label)}")
    print(f"Label problems: {len(problems)}")
    for p in problems[:10]:
        print("  ", p)

    # Drawn samples so you can SEE the box convention.
    random.seed(args.seed)
    sample_dir = args.out / "samples"
    sample_dir.mkdir(exist_ok=True)
    labelled = list(per_image)
    for i, path in enumerate(random.sample(labelled, min(args.samples, len(labelled)))):
        draw_boxes(path, per_image[path], names, sample_dir / f"sample_{i:02d}_{path.stem}.jpg")
    print(f"\nDrew {min(args.samples, len(labelled))} samples with boxes -> {sample_dir}")

    # Leakage check on the dataset's own split folders, if they exist.
    leakage = None
    split_dirs = {p.name.lower(): p for p in args.data_dir.rglob("*") if p.is_dir()
                  and p.name.lower() in {"training", "train", "validation", "val", "valid", "test"}}
    train_dir = split_dirs.get("training") or split_dirs.get("train")
    val_dir = split_dirs.get("validation") or split_dirs.get("val") or split_dirs.get("valid")
    if train_dir and val_dir:
        tr, va = find_images(train_dir), find_images(val_dir)
        print(f"\nChecking original split for near-duplicates: {len(tr)} train vs {len(va)} val ...")
        hashes = np.array([phash_file(p) for p in tr + va])
        pairs = near_duplicate_pairs(hashes, args.dup_distance)
        cross = [(i, j, d) for i, j, d in pairs if i < len(tr) <= j]
        val_hit = {j for _, j, _ in cross}
        leakage = {"cross_split_pairs": len(cross), "val_images_with_train_near_duplicate": len(val_hit),
                   "val_images": len(va), "max_hamming_distance": args.dup_distance}
        print(f"  {len(val_hit)} of {len(va)} validation images have a near-duplicate in training "
              f"(Hamming distance <= {args.dup_distance}).")

    summary = {
        "data_dir": str(args.data_dir), "images": len(images), "class_file": str(class_file) if class_file else None,
        "classes": class_rows, "notes_per_image": {str(k): v for k, v in sorted(notes_per_image.items())},
        "box_area_fraction": area_stats, "images_without_labels": len(no_label),
        "label_problems": len(problems), "original_split_leakage": leakage,
    }
    (args.out / "inspect_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {args.out / 'inspect_summary.json'}")


if __name__ == "__main__":
    main()
