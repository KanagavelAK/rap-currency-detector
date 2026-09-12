"""Tests for label helpers, perceptual hashing and evaluation metrics. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import Box, label_path_for, read_yolo_labels, value_from_class_name, write_yolo_labels  # noqa: E402
from src.metrics import GT, ImageEval, Pred, evaluate_image, iou, summarise, sweep_thresholds  # noqa: E402
from src.phash import UnionFind, near_duplicate_pairs, phash_image  # noqa: E402


def test_value_from_class_name():
    assert value_from_class_name("500") == 500
    assert value_from_class_name("Rs_100_new") == 100
    assert value_from_class_name("10old") == 10
    assert value_from_class_name("two_hundred") == 200
    assert value_from_class_name("five hundred") == 500
    assert value_from_class_name("gandhi") is None
    assert value_from_class_name("weird", {"weird": 50}) == 50


def test_label_round_trip_and_validation(tmp_path):
    path = tmp_path / "a.txt"
    write_yolo_labels(path, [Box(1, 0.5, 0.5, 0.2, 0.1)])
    boxes, problems = read_yolo_labels(path)
    assert len(boxes) == 1 and boxes[0].cls == 1 and not problems
    path.write_text("0 0.5 0.5 0 0.1\n1 2.0 0.5 0.1 0.1\nbad line\n")
    boxes, problems = read_yolo_labels(path)
    assert boxes == [] and len(problems) == 3


def test_label_path_for_both_layouts(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    img = tmp_path / "images" / "train" / "x.jpg"
    assert label_path_for(img) == tmp_path / "labels" / "train" / "x.txt"
    side = tmp_path / "flat" / "y.jpg"
    side.parent.mkdir()
    (tmp_path / "flat" / "y.txt").write_text("")
    assert label_path_for(side) == tmp_path / "flat" / "y.txt"


def _scene(seed):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rng.integers(0, 350), rng.integers(0, 250)
        draw.rectangle([x, y, x + 60, y + 40], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return img


def test_phash_finds_near_duplicates_not_different_images():
    a = _scene(1)
    a_resized = a.resize((320, 240))
    b = _scene(2)
    hashes = np.array([phash_image(a), phash_image(a_resized), phash_image(b)])
    pairs = near_duplicate_pairs(hashes, max_distance=8)
    linked = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in linked
    assert (0, 2) not in linked and (1, 2) not in linked


def test_union_find_groups():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    groups = sorted(sorted(g) for g in uf.groups().values())
    assert groups == [[0, 1, 2], [3], [4]]


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_same_value_vs_cross_value_confusion():
    values = {0: 100, 1: 100, 2: 10}  # class 0 = old ₹100, class 1 = new ₹100, class 2 = ₹10
    item = ImageEval("x", gts=[GT(0, (0, 0, 100, 50), 0.2), GT(0, (200, 0, 300, 50), 0.2)],
                     preds=[Pred(1, 0.9, (0, 0, 100, 50)), Pred(2, 0.8, (200, 0, 300, 50))])
    r = evaluate_image(item, values, conf_thr=0.5)
    assert r["same_value_confusion"] == 1
    assert r["cross_value_confusion"] == 1
    assert r["gt_total"] == 200 and r["pred_total"] == 110 and r["value_abs_error"] == 90


def test_missed_and_false_positive_and_threshold():
    values = {0: 500}
    item = ImageEval("x", gts=[GT(0, (0, 0, 100, 50), 0.005)],
                     preds=[Pred(0, 0.3, (0, 0, 100, 50)), Pred(0, 0.9, (300, 300, 400, 350))])
    r = evaluate_image(item, values, conf_thr=0.5)
    assert r["missed"] == 1 and r["false_positives"] == 1
    s = summarise([r])
    assert s["recall_by_size"]["small"] == 0.0
    table, best = sweep_thresholds([item], [0.2, 0.5])
    assert best["conf"] == 0.2  # at 0.2 the true box is found -> higher F1
