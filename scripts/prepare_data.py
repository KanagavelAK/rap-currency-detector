"""Build leak-free train/val/test splits and the YAML files Ultralytics needs.

Public data:
  * near-identical images (Hamming distance <= --dup-distance) are dropped
  * near-duplicates (distance <= --group-distance) are grouped and a whole group
    always lands in ONE split, so the test set never contains a copy of a training photo
  * groups are split per dominant class, so every split has a similar class mix

Your own photos (optional):
  * organised as session folders: own_photos/s1, own_photos/s2, ...
  * sessions listed in --own-test-sessions become the locked real-world test set
  * other sessions are added to training (used by Run B)
  * class ids are remapped BY NAME to the public class list, so label order can't silently break

Outputs (under --out):
  images/{train,val,test,train_own,test_own}/  labels/{...}/
  data_public.yaml     Run A: train on public data only
  data_plus_own.yaml   Run B: public + your training sessions
  data_own_test.yaml   evaluate any run on your real-world photos
  split_report.json, split_lists/*.txt (commit these for reproducibility)

Usage:
  python scripts/prepare_data.py --public-dir /path/IndianBankNotes --out datasets/currency
  python scripts/prepare_data.py --public-dir /path/IndianBankNotes --out datasets/currency \
      --own-dir own_photos --own-classes own_photos/obj.names --own-test-sessions s4,s5
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import (  # noqa: E402
    Box, find_class_file, find_images, label_path_for, read_class_names, read_yolo_labels, write_yolo_labels,
)
from src.phash import UnionFind, near_duplicate_pairs, phash_file  # noqa: E402

SPLITS = ("train", "val", "test")


@dataclass
class Item:
    image: Path
    boxes: list[Box]
    source: str  # "public" or session name


def load_public(folder: Path, report: dict) -> list[Item]:
    items, missing, problems = [], 0, []
    for img in find_images(folder):
        lp = label_path_for(img)
        if not lp.exists():
            missing += 1
            continue
        boxes, probs = read_yolo_labels(lp)
        problems.extend(probs)
        items.append(Item(img, boxes, "public"))
    report["public_images_loaded"] = len(items)
    report["public_images_skipped_no_label"] = missing
    report["public_label_problems"] = len(problems)
    return items


def load_own(own_dir: Path, own_names: list[str], master: list[str], report: dict) -> dict[str, list[Item]]:
    remap = {}
    for i, name in enumerate(own_names):
        if name not in master:
            sys.exit(f"Own class '{name}' is not in the public class list {master}. "
                     "Rename it in your labelling tool so names match exactly.")
        remap[i] = master.index(name)
    sessions: dict[str, list[Item]] = {}
    negatives = 0
    for session in sorted(p for p in own_dir.iterdir() if p.is_dir()):
        items = []
        for img in find_images(session):
            boxes, _ = read_yolo_labels(label_path_for(img))  # missing file -> negative image (no notes)
            boxes = [Box(remap[b.cls], b.cx, b.cy, b.w, b.h) for b in boxes if b.cls in remap]
            negatives += not boxes
            items.append(Item(img, boxes, session.name))
        if items:
            sessions[session.name] = items
    report["own_sessions"] = {k: len(v) for k, v in sessions.items()}
    report["own_negative_images"] = negatives
    return sessions


def dedupe_and_group(items: list[Item], dup_distance: int, group_distance: int, report: dict,
                     examples_dir: Path) -> list[list[Item]]:
    print(f"Hashing {len(items)} images ...")
    hashes = np.array([phash_file(it.image) for it in items])
    pairs = near_duplicate_pairs(hashes, group_distance)

    uf = UnionFind(len(items))
    drop: set[int] = set()
    for i, j, d in pairs:
        uf.union(i, j)
        if d <= dup_distance and i not in drop:
            drop.add(j)  # keep the first of a near-identical pair

    examples_dir.mkdir(parents=True, exist_ok=True)
    for k, (i, j, d) in enumerate(sorted(pairs, key=lambda p: p[2])[:12]):
        save_pair(items[i].image, items[j].image, examples_dir / f"pair_{k:02d}_dist{d}.jpg")

    groups = []
    for members in uf.groups().values():
        kept = [items[m] for m in members if m not in drop]
        if kept:
            groups.append(kept)
    report["near_duplicate_pairs"] = len(pairs)
    report["dropped_near_identical"] = len(drop)
    report["groups"] = len(groups)
    report["largest_group"] = max(len(g) for g in groups)
    return groups


def save_pair(a: Path, b: Path, out: Path) -> None:
    with Image.open(a) as ia, Image.open(b) as ib:
        ia = ia.convert("RGB").resize((320, 320))
        ib = ib.convert("RGB").resize((320, 320))
        canvas = Image.new("RGB", (650, 320), "white")
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (330, 0))
        canvas.save(out, "JPEG", quality=85)


def split_groups(groups: list[list[Item]], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Item]]:
    """Assign whole groups to splits, balancing each dominant class separately."""
    rng = random.Random(seed)
    buckets: dict[int, list[list[Item]]] = defaultdict(list)
    for g in groups:
        classes = Counter(b.cls for it in g for b in it.boxes)
        dominant = classes.most_common(1)[0][0] if classes else -1
        buckets[dominant].append(g)

    out = {s: [] for s in SPLITS}
    for cls in sorted(buckets):
        bucket = buckets[cls]
        rng.shuffle(bucket)
        bucket.sort(key=len, reverse=True)  # place big groups first while there is room
        total = sum(len(g) for g in bucket)
        targets = [r * total for r in ratios]
        assigned = [0, 0, 0]
        for g in bucket:
            deficits = [targets[k] - assigned[k] for k in range(3)]
            k = max(range(3), key=lambda idx: deficits[idx])
            out[SPLITS[k]].extend(g)
            assigned[k] += len(g)
    return out


def write_split(name: str, items: list[Item], out: Path, lists_dir: Path) -> None:
    img_dir, lbl_dir = out / "images" / name, out / "labels" / name
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    lines = []
    for i, it in enumerate(sorted(items, key=lambda x: str(x.image))):
        stem = f"{it.source}_{i:05d}_{it.image.stem}"
        shutil.copy2(it.image, img_dir / f"{stem}{it.image.suffix.lower()}")
        write_yolo_labels(lbl_dir / f"{stem}.txt", it.boxes)
        lines.append(str(it.image))
    (lists_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")


def class_counts(items: list[Item], names: list[str]) -> dict[str, int]:
    c = Counter(b.cls for it in items for b in it.boxes)
    return {names[k]: c[k] for k in sorted(c)}


def write_yaml(path: Path, root: Path, train, val: str, test: str, names: list[str]) -> None:
    data = {"path": str(root.resolve()), "train": train, "val": val, "test": test,
            "names": {i: n for i, n in enumerate(names)}}
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-dir", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=None, help="public class file (auto-detected if omitted)")
    parser.add_argument("--out", type=Path, default=Path("datasets/currency"))
    parser.add_argument("--own-dir", type=Path, default=None)
    parser.add_argument("--own-classes", type=Path, default=None)
    parser.add_argument("--own-test-sessions", default="")
    parser.add_argument("--ratios", default="0.75,0.10,0.15")
    parser.add_argument("--dup-distance", type=int, default=2)
    parser.add_argument("--group-distance", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports", type=Path, default=Path("reports/data"))
    args = parser.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        sys.exit("--ratios must be three numbers that sum to 1, e.g. 0.75,0.10,0.15")

    class_file = args.classes or find_class_file(args.public_dir)
    if not class_file:
        sys.exit("No class file found. Pass --classes path/to/classes.txt")
    names = read_class_names(class_file)
    report: dict = {"seed": args.seed, "ratios": ratios, "dup_distance": args.dup_distance,
                    "group_distance": args.group_distance, "class_file": str(class_file), "classes": names}

    args.reports.mkdir(parents=True, exist_ok=True)
    public = load_public(args.public_dir, report)
    max_cls = max((b.cls for it in public for b in it.boxes), default=-1)
    if max_cls >= len(names):
        sys.exit(f"Labels use class id {max_cls} but the class file has only {len(names)} names.")

    groups = dedupe_and_group(public, args.dup_distance, args.group_distance, report,
                              args.reports / "duplicate_examples")
    splits = split_groups(groups, ratios, args.seed)

    lists_dir = args.reports / "split_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_split(name, splits[name], args.out, lists_dir)

    report["splits"] = {name: {"images": len(splits[name]), "boxes_per_class": class_counts(splits[name], names)}
                        for name in SPLITS}

    test_sessions = {s.strip() for s in args.own_test_sessions.split(",") if s.strip()}
    has_own = args.own_dir is not None
    if has_own:
        own_class_file = args.own_classes or find_class_file(args.own_dir)
        if not own_class_file:
            sys.exit("No class file for own photos. Pass --own-classes (e.g. obj.names from CVAT).")
        sessions = load_own(args.own_dir, read_class_names(own_class_file), names, report)
        unknown = test_sessions - set(sessions)
        if unknown:
            sys.exit(f"Test sessions not found: {sorted(unknown)}. Available: {sorted(sessions)}")
        if not test_sessions:
            sys.exit("Pass --own-test-sessions so your real-world test set is locked away from training.")
        own_train = [it for s, its in sessions.items() if s not in test_sessions for it in its]
        own_test = [it for s, its in sessions.items() if s in test_sessions for it in its]
        write_split("train_own", own_train, args.out, lists_dir)
        write_split("test_own", own_test, args.out, lists_dir)
        report["splits"]["train_own"] = {"images": len(own_train), "boxes_per_class": class_counts(own_train, names)}
        report["splits"]["test_own"] = {"images": len(own_test), "boxes_per_class": class_counts(own_test, names)}

    root = args.out
    write_yaml(root / "data_public.yaml", root, "images/train", "images/val", "images/test", names)
    if has_own:
        write_yaml(root / "data_plus_own.yaml", root, ["images/train", "images/train_own"],
                   "images/val", "images/test", names)
        write_yaml(root / "data_own_test.yaml", root, "images/train", "images/test_own", "images/test_own", names)

    (args.reports / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["splits"], indent=2))
    print(f"\nDropped {report['dropped_near_identical']} near-identical images; "
          f"{report['groups']} groups; largest group has {report['largest_group']} images.")
    print(f"Dataset written to {root}. Report: {args.reports / 'split_report.json'}")


if __name__ == "__main__":
    main()
