"""Evaluation metrics that go beyond mAP.

mAP answers: "how well are boxes ranked and placed across all thresholds?"
This module answers what the application needs at ONE chosen threshold:
- Is the rupee total right? (value error)
- When the class is wrong, does it change the money? (same-value vs cross-value)
- Which object sizes get missed? (recall by size)
"""
from __future__ import annotations

from dataclasses import dataclass, field

SIZE_BUCKETS = (("small", 0.0, 0.02), ("medium", 0.02, 0.10), ("large", 0.10, 1.01))


@dataclass
class GT:
    cls: int
    box: tuple[float, float, float, float]  # pixel x1, y1, x2, y2
    area_frac: float = 0.0  # box area / image area


@dataclass
class Pred:
    cls: int
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class ImageEval:
    image: str
    gts: list[GT]
    preds: list[Pred] = field(default_factory=list)  # raw predictions at a low confidence floor


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gts: list[GT], preds: list[Pred], iou_thr: float, class_aware: bool):
    """Match highest-confidence predictions first to their best unmatched ground truth.

    Returns (matches as (gt_idx, pred_idx), unmatched gt indices, unmatched pred indices).
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True)
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    unmatched_preds: list[int] = []
    for pi in order:
        best_gi, best_iou = -1, iou_thr
        for gi, gt in enumerate(gts):
            if gi in used_gt or (class_aware and gt.cls != preds[pi].cls):
                continue
            score = iou(gt.box, preds[pi].box)
            if score >= best_iou:
                best_gi, best_iou = gi, score
        if best_gi >= 0:
            used_gt.add(best_gi)
            matches.append((best_gi, pi))
        else:
            unmatched_preds.append(pi)
    unmatched_gts = [gi for gi in range(len(gts)) if gi not in used_gt]
    return matches, unmatched_gts, unmatched_preds


def evaluate_image(item: ImageEval, values: dict[int, int | None], conf_thr: float, iou_thr: float = 0.5) -> dict:
    """Per-image report at one confidence threshold."""
    preds = [p for p in item.preds if p.conf >= conf_thr]
    matches, missed, false_pos = greedy_match(item.gts, preds, iou_thr, class_aware=False)

    correct = same_value = cross_value = 0
    confusions: list[tuple[int, int]] = []
    for gi, pi in matches:
        g, p = item.gts[gi].cls, preds[pi].cls
        if g == p:
            correct += 1
        elif values.get(g) is not None and values.get(g) == values.get(p):
            same_value += 1
            confusions.append((g, p))
        else:
            cross_value += 1
            confusions.append((g, p))

    gt_total = sum(values.get(g.cls) or 0 for g in item.gts)
    pred_total = sum(values.get(p.cls) or 0 for p in preds)

    size_hits = {name: [0, 0] for name, _, _ in SIZE_BUCKETS}  # [found, total]
    matched_gt = {gi for gi, _ in matches}
    for gi, gt in enumerate(item.gts):
        for name, lo, hi in SIZE_BUCKETS:
            if lo <= gt.area_frac < hi:
                size_hits[name][1] += 1
                if gi in matched_gt:
                    size_hits[name][0] += 1

    return {
        "image": item.image,
        "gt_notes": len(item.gts),
        "pred_notes": len(preds),
        "correct_class": correct,
        "same_value_confusion": same_value,
        "cross_value_confusion": cross_value,
        "missed": len(missed),
        "false_positives": len(false_pos),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "value_abs_error": abs(gt_total - pred_total),
        "count_abs_error": abs(len(item.gts) - len(preds)),
        "confusions": confusions,
        "size_hits": size_hits,
    }


def summarise(reports: list[dict]) -> dict:
    n = max(len(reports), 1)
    size_totals = {name: [0, 0] for name, _, _ in SIZE_BUCKETS}
    for r in reports:
        for name, (found, total) in r["size_hits"].items():
            size_totals[name][0] += found
            size_totals[name][1] += total
    matched = sum(r["correct_class"] + r["same_value_confusion"] + r["cross_value_confusion"] for r in reports)
    return {
        "images": len(reports),
        "total_value_mae_rupees": round(sum(r["value_abs_error"] for r in reports) / n, 2),
        "exact_total_accuracy": round(sum(r["value_abs_error"] == 0 for r in reports) / n, 4),
        "count_mae": round(sum(r["count_abs_error"] for r in reports) / n, 3),
        "matched_notes": matched,
        "same_value_confusions": sum(r["same_value_confusion"] for r in reports),
        "cross_value_confusions": sum(r["cross_value_confusion"] for r in reports),
        "missed_notes": sum(r["missed"] for r in reports),
        "false_positives": sum(r["false_positives"] for r in reports),
        "recall_by_size": {
            name: (round(found / total, 4) if total else None)
            for name, (found, total) in size_totals.items()
        },
        "notes_by_size": {name: total for name, (_, total) in size_totals.items()},
    }


def precision_recall_at(items: list[ImageEval], conf_thr: float, iou_thr: float = 0.5) -> dict:
    """Class-aware precision / recall / F1 over a dataset at one threshold."""
    tp = fp = fn = 0
    for item in items:
        preds = [p for p in item.preds if p.conf >= conf_thr]
        matches, missed, extra = greedy_match(item.gts, preds, iou_thr, class_aware=True)
        tp += len(matches)
        fn += len(missed)
        fp += len(extra)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"conf": round(conf_thr, 3), "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def sweep_thresholds(items: list[ImageEval], thresholds: list[float], iou_thr: float = 0.5) -> tuple[list[dict], dict]:
    table = [precision_recall_at(items, t, iou_thr) for t in thresholds]
    best = max(table, key=lambda row: (row["f1"], row["conf"]))
    return table, best
