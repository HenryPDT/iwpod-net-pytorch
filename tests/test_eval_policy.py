"""Eval policy: negatives, malformed skip, exact sweep, recall@0.5."""
import numpy as np
import torch

from iwpod.eval_core import (
    entries_from_annotated_dir,
    evaluate_quads,
    format_summary,
    threshold_sweep,
)


def test_entries_missing_txt_is_negative(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((16, 16, 3), np.uint8))
    entries = entries_from_annotated_dir(str(tmp_path))
    assert len(entries) == 1 and entries[0][1] is None


def test_entries_malformed_skipped_not_negative(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "bad.jpg"), np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "bad.txt").write_text("not,a,quad\n")
    cv2.imwrite(str(tmp_path / "ok.jpg"), np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "ok.txt").write_text("4,0.2,0.8,0.8,0.2,0.2,0.2,0.8,0.8,\n")
    entries = entries_from_annotated_dir(str(tmp_path))
    assert len(entries) == 1 and entries[0][1] is not None


def test_entries_empty_txt_is_negative(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "e.jpg"), np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "e.txt").write_text("")
    entries = entries_from_annotated_dir(str(tmp_path))
    assert len(entries) == 1 and entries[0][1] is None


def test_threshold_sweep_exact_extra():
    confs = np.array([0.9, 0.33, 0.1])
    raw = np.array([0.8, 0.7, 0.0])
    is_pos = np.array([True, True, False])
    s = threshold_sweep(confs, raw, is_pos, operating=0.3, extra_thresholds=[0.33])
    thrs = [r["thr"] for r in s["table"]]
    assert 0.33 in thrs
    row = next(r for r in s["table"] if r["thr"] == 0.33)
    assert row["rec"] == 1.0  # both positives >= 0.33 and IoU>0.5


def test_evaluate_recall50_not_det_rate(tmp_path):
    import cv2

    class _Hit(torch.nn.Module):
        def forward(self, x):
            b, _, h, w = x.shape
            out = torch.full((b, 7, h // 16, w // 16), -10.0)
            out[:, 0, 1, 1] = 8.0
            out[:, 1, 1, 1] = 0.4
            out[:, 5, 1, 1] = 0.4
            return out

    jpg = str(tmp_path / "img.jpg")
    cv2.imwrite(jpg, np.zeros((64, 64, 3), np.uint8))
    gt = np.array([[0.8, 0.95, 0.95, 0.8], [0.8, 0.8, 0.95, 0.95]])  # far from cell
    r = evaluate_quads(_Hit(), [(jpg, gt)], size_px=64, threshold=0.3,
                       device=torch.device("cpu"))
    assert "det_rate" in r and "recall50" in r
    assert r["rmse_n"] >= 0
    assert "RMSE_det" in format_summary(r)
