"""Evaluate a trained model on one split.

Step 1 - choose the confidence threshold on VALIDATION data (never on test):
  python scripts/evaluate.py --weights weights/run_b_best.pt \
      --data datasets/currency/data_plus_own.yaml --split val --sweep --out reports/run_b_val

Step 2 - report on the public test split and on your own photos, using that threshold:
  python scripts/evaluate.py --weights weights/run_b_best.pt \
      --data datasets/currency/data_public.yaml --split test \
      --conf-file reports/run_b_val/threshold.json --out reports/run_b_public_test
  python scripts/evaluate.py --weights weights/run_b_best.pt \
      --data datasets/currency/data_own_test.yaml --split test \
      --conf-file reports/run_b_val/threshold.json --out reports/run_b_own_test

Outputs: summary.json, per_image.csv, worst/ (drawn failure candidates), ultralytics_val/ (plots,
confusion matrix), and threshold.json + threshold_sweep.csv when --sweep is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import find_images, label_path_for, read_yolo_labels, value_from_class_name  # noqa: E402
from src.metrics import GT, ImageEval, Pred, evaluate_image, summarise, sweep_thresholds  # noqa: E402

CONF_FLOOR = 0.05  # keep low-confidence predictions so the sweep can explore thresholds


def split_image_dirs(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent))
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    dirs = [(root / e) if not Path(e).is_absolute() else Path(e) for e in entries]
    for d in dirs:
        if not d.exists():
            sys.exit(f"Split folder not found: {d}. Did you run prepare_data.py on this machine?")
    return dirs


def load_value_overrides(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text()).get("value_overrides", {})
    return {}


def collect(model, image_paths: list[Path], imgsz: int, device) -> list[ImageEval]:
    items: list[ImageEval] = []
    for start in range(0, len(image_paths), 16):
        chunk = image_paths[start:start + 16]
        results = model.predict(source=[str(p) for p in chunk], conf=CONF_FLOOR, imgsz=imgsz,
                                device=device, verbose=False, stream=True)
        for path, r in zip(chunk, results):
            h, w = r.orig_shape
            boxes, _ = read_yolo_labels(label_path_for(path))
            gts = [GT(b.cls, b.xyxy(w, h), b.area) for b in boxes]
            preds = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                preds = [Pred(int(c), float(s), tuple(float(v) for v in bb)) for bb, s, c in zip(xyxy, confs, classes)]
            items.append(ImageEval(str(path), gts, preds))
        print(f"  predicted {min(start + 16, len(image_paths))}/{len(image_paths)}")
    return items


def draw_case(item: ImageEval, report: dict, names: dict, conf: float, out_path: Path) -> None:
    with Image.open(item.image) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width = max(2, img.size[0] // 300)
        for g in item.gts:
            draw.rectangle(g.box, outline=(0, 200, 0), width=width)
            draw.text((g.box[0] + 4, g.box[1] + 4), f"GT {names.get(g.cls, g.cls)}", fill=(0, 200, 0))
        for p in item.preds:
            if p.conf >= conf:
                draw.rectangle(p.box, outline=(230, 0, 0), width=width)
                draw.text((p.box[0] + 4, p.box[3] - 14), f"{names.get(p.cls, p.cls)} {p.conf:.2f}", fill=(230, 0, 0))
        caption = (f"GT total {report['gt_total']} | pred total {report['pred_total']} | missed {report['missed']} "
                   f"| FP {report['false_positives']} | cross-value {report['cross_value_confusion']}")
        draw.rectangle([0, 0, img.size[0], 22], fill=(0, 0, 0))
        draw.text((6, 5), caption, fill=(255, 255, 255))
        img.save(out_path, "JPEG", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold for app-level metrics")
    parser.add_argument("--conf-file", type=Path, default=None, help="threshold.json produced by --sweep")
    parser.add_argument("--sweep", action="store_true", help="choose the F1-best threshold (use on val only)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a box as found")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.json"))
    args = parser.parse_args()

    if args.sweep and args.split != "val":
        sys.exit("Choose thresholds on --split val. Tuning on test would leak test information.")

    from ultralytics import RTDETR

    args.out.mkdir(parents=True, exist_ok=True)
    model = RTDETR(str(args.weights))
    names = {int(k): v for k, v in model.names.items()}
    overrides = load_value_overrides(args.inference_config)
    values = {k: value_from_class_name(v, overrides) for k, v in names.items()}
    missing = [names[k] for k, v in values.items() if v is None]
    if missing:
        sys.exit(f"Classes without a rupee value: {missing}. Add them to value_overrides in {args.inference_config}")

    # 1) Official Ultralytics metrics (mAP etc.) + confusion matrix plots.
    print("Running Ultralytics validation ...")
    m = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device,
                  plots=True, project=str(args.out), name="ultralytics_val", exist_ok=True, verbose=False)
    official = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                "precision_ultralytics": round(float(m.box.mp), 4), "recall_ultralytics": round(float(m.box.mr), 4)}
    try:
        official["per_class"] = {
            names[int(c)]: {"AP50": round(float(a50), 4), "AP50_95": round(float(a), 4)}
            for c, a50, a in zip(m.box.ap_class_index, m.box.ap50, m.box.ap)
        }
    except Exception as exc:  # attribute names can differ between Ultralytics versions
        official["per_class_error"] = str(exc)

    # 2) Our own predictions for application-level metrics.
    images = [p for d in split_image_dirs(args.data, args.split) for p in find_images(d)]
    print(f"Predicting {len(images)} images for application metrics ...")
    items = collect(model, images, args.imgsz, args.device)

    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    sweep_best = None
    if args.sweep:
        table, sweep_best = sweep_thresholds(items, thresholds, args.iou)
        with open(args.out / "threshold_sweep.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        (args.out / "threshold.json").write_text(json.dumps({"conf_high": sweep_best["conf"], **sweep_best}, indent=2))
        print(f"F1-best threshold on val: {sweep_best}")

    if args.conf is not None:
        conf = args.conf
    elif args.conf_file:
        conf = float(json.loads(args.conf_file.read_text())["conf_high"])
    elif sweep_best:
        conf = sweep_best["conf"]
    else:
        conf = 0.5
        print("No threshold given; using 0.5. Prefer --conf-file from a validation sweep.")

    reports = [evaluate_image(it, values, conf, args.iou) for it in items]
    app = summarise(reports)
    confusion_pairs = Counter((names[g], names[p]) for r in reports for g, p in r["confusions"])
    app["top_confusions"] = [{"true": g, "predicted": p, "count": c} for (g, p), c in confusion_pairs.most_common(10)]

    with open(args.out / "per_image.csv", "w", newline="") as f:
        fields = [k for k in reports[0] if k not in ("confusions", "size_hits")] if reports else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)

    worst_dir = args.out / "worst"
    worst_dir.mkdir(exist_ok=True)
    ranked = sorted(zip(items, reports), key=lambda x: (x[1]["value_abs_error"],
                    x[1]["missed"] + x[1]["false_positives"] + x[1]["cross_value_confusion"]), reverse=True)
    for i, (item, rep) in enumerate(ranked[:args.worst]):
        if rep["value_abs_error"] == 0 and rep["missed"] == 0 and rep["false_positives"] == 0:
            break
        draw_case(item, rep, names, conf, worst_dir / f"{i:02d}_err{rep['value_abs_error']}_{Path(item.image).stem}.jpg")

    summary = {"weights": str(args.weights), "data": str(args.data), "split": args.split,
               "conf_threshold": conf, "iou_match": args.iou, "official": official, "application": app}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.out}. Failure candidates are drawn in {worst_dir} (green = truth, red = prediction).")


if __name__ == "__main__":
    main()
