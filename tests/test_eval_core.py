"""eval_core unit tests: geometry + entry helpers (stub model, CPU)."""
import numpy as np
import torch

from iwpod.eval_core import (
    IOU_THRESHOLDS,
    area_split,
    average_precision,
    entries_from_annotated_dir,
    entries_from_loader_entries,
    evaluate_quads,
    format_block,
    format_summary,
    map_scores,
    quad_iou,
    threshold_sweep,
)


def test_iou_curve_spans_050_to_095_coco_style():
    assert list(IOU_THRESHOLDS) == [round(0.5 + 0.05 * k, 2) for k in range(10)]
    assert IOU_THRESHOLDS[0] == 0.5 and IOU_THRESHOLDS[-1] == 0.95


def test_quad_iou_identical_and_disjoint():
    box = np.array([[0.1, 0.3, 0.3, 0.1], [0.1, 0.1, 0.3, 0.3]])
    assert quad_iou(box, box) > 0.99
    far = box + 0.6  # 0.7..0.9: fully disjoint from 0.1..0.3
    assert quad_iou(box, far) == 0.0


def test_entries_from_loader_entries_real_and_fake(tmp_path):
    from iwpod.label import Shape

    real = Shape(np.array([[0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]]))
    fake = Shape(np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]]))
    entries = entries_from_loader_entries([("a.jpg", [real]), ("b.jpg", [fake])])
    assert entries[0][1] is not None and entries[0][1].shape == (2, 4)
    assert entries[1][1] is None


def test_entries_from_annotated_dir(tmp_path):
    import cv2
    jpg = str(tmp_path / "img.jpg")
    cv2.imwrite(jpg, np.zeros((32, 32, 3), np.uint8))
    with open(str(tmp_path / "img.txt"), "w") as f:
        f.write("4,0.2,0.2,0.8,0.2,0.8,0.8,0.2,0.8,\n")
    entries = entries_from_annotated_dir(str(tmp_path))
    assert len(entries) == 1 and entries[0][1].shape == (2, 4)


class _NegModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        return torch.full((b, 7, h // 16, w // 16), -10.0)


def test_evaluate_quads_all_negative_stub(tmp_path):
    import cv2
    jpg = str(tmp_path / "img.jpg")
    cv2.imwrite(jpg, np.zeros((64, 64, 3), np.uint8))
    gt = np.array([[0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
    r = evaluate_quads(_NegModel(), [(jpg, gt)], size_px=64, threshold=0.3,
                       device=torch.device("cpu"))
    assert r.n == 1 and r.dets == 0
    assert r.recall == 0.0 and r.mean_iou == 0.0
    assert r.iou50 == 0.0 and r.iou70 == 0.0
    assert "recall=0.000" in format_summary(r)
    # sweep on all-miss: zero F1, area buckets still partition the positive
    assert r.sweep["best_f1"] == 0.0
    assert sum(b["n"] for b in r.area.values()) == 1
    block = format_block(r, epoch=1, epochs=2, size_px=64, threshold=0.3, with_sweep=True)
    assert "EVAL epoch 1/2" in block and "area:" in block and "thr sweep" in block


def test_threshold_sweep_recall_monotone_and_best():
    confs = np.array([0.9, 0.8, 0.4, 0.2, 0.95])
    raw = np.array([0.8, 0.3, 0.7, 0.0, 0.0])
    is_pos = np.array([True, True, True, True, False])
    s = threshold_sweep(confs, raw, is_pos, operating=0.3)
    recs = [r["rec"] for r in s["table"]]
    assert all(a >= b for a, b in zip(recs, recs[1:], strict=False))  # recall non-increasing
    assert 0.0 <= s["best_f1"] <= 1.0
    assert s["op_thr"] == 0.3


def test_area_split_balanced_terciles():
    rng = np.random.RandomState(0)
    n = 30
    confs = np.ones(n)
    ious = rng.rand(n)
    areas = np.linspace(0.01, 0.3, n)
    is_pos = np.ones(n, dtype=bool)
    out = area_split(confs, ious, areas, is_pos, threshold=0.3)
    assert sorted(out) == ["large", "medium", "small"]
    assert all(b["n"] == 10 for b in out.values())
    assert out["small"]["recall"] == 1.0


def test_average_precision_perfect_and_empty():
    confs = np.array([0.9, 0.8, 0.7])
    assert average_precision(confs, np.array([True, True, True]), 3) == 1.0
    assert average_precision(np.array([]), np.array([], dtype=bool), 0) == 0.0
    assert average_precision(confs, np.array([False, False, False]), 3) == 0.0


def test_average_precision_ranked_example():
    # TP, FP, TP, FP over 2 positives -> AP = (1 + 2/3) / 2
    confs = np.array([0.9, 0.8, 0.7, 0.6])
    matched = np.array([True, False, True, False])
    assert abs(average_precision(confs, matched, 2) - (1.0 + 2 / 3) / 2) < 1e-9


def test_map_scores_keys_and_range():
    confs = np.array([0.9, 0.4, 0.8, 0.1])
    raw = np.array([0.8, 0.3, 0.7, 0.0])
    is_pos = np.array([True, True, True, False])
    m = map_scores(confs, raw, is_pos)
    assert set(m["curve"]) == set(IOU_THRESHOLDS)
    assert m["map50"] == m["curve"][0.5]
    assert 0.0 <= m["map"] <= 1.0
    assert m["map"] <= m["map50"] + 1e-9  # stricter thresholds can only drop


def test_map_scores_no_positives_is_zero():
    m = map_scores(np.array([0.9]), np.array([0.0]), np.array([False]))
    assert m["map"] == 0.0 and m["map50"] == 0.0
