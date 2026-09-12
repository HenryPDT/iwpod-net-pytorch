"""Sampler/data correctness: centroid-per-plate, winding, HSV, fake resize, mixed-case."""
import numpy as np

from iwpod.constants import HSV_H_DELTA, HSV_S_DELTA, HSV_V_DELTA, REAL_PLATE_DIAG, is_real_plate
from iwpod.label import Label, Shape
from iwpod.sampler import (
    LinePolygonEdges,
    ShrinkQuadrilateral,
    augment_sample,
    insidePolygon,
    labels2output_map,
)
from iwpod.utils import hsv_transform, image_files_from_folder


def _quad(offset=0.0):
    return np.array([[0.2, 0.7, 0.7, 0.2], [0.3, 0.3, 0.6, 0.6]], dtype=float) + offset


def test_centroid_fallback_per_plate():
    q1 = _quad()
    q2 = _quad(0.05)
    # CCW copy of q1: polygon scan may miss; centroid must still fire per plate.
    q1_ccw = q1[:, [0, 3, 2, 1]]
    labels = [Label(0, q.min(1), q.max(1)) for q in (q1_ccw, q2)]
    out_map = labels2output_map(labels, [q1_ccw, q2], dim=256, stride=16)
    pos = np.argwhere(out_map[..., 0] > 0.5)
    assert len(pos) >= 2


def test_inside_polygon_both_windings():
    cw = _quad()
    ccw = cw[:, [0, 3, 2, 1]]
    grid = np.array([16.0, 16.0])
    for pts in (cw, ccw):
        pts2 = (ShrinkQuadrilateral(pts, 0.75).T * grid).T
        lines = LinePolygonEdges(pts2)
        c = pts.mean(1) * 16
        assert insidePolygon(c, lines)


def test_fake_resize_non_square_outputs_dim():
    fake = Shape(np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]]))
    import random
    for shape in ((200, 80, 3), (80, 200, 3)):  # portrait and landscape
        img = (np.random.rand(*shape) * 255).astype(np.uint8)
        np.random.seed(0)
        random.seed(0)
        aug, _, pts = augment_sample(img, [fake], 128)
        assert aug.shape == (128, 128, 3)
        assert len(pts) == 1


def test_hsv_bounds_opencv_units():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    rng = np.random.RandomState(0)
    hs = []
    for _ in range(200):
        mod = np.array([
            (rng.rand() - 0.5) * 2 * HSV_H_DELTA,
            (rng.rand() - 0.5) * 2 * HSV_S_DELTA,
            (rng.rand() - 0.5) * 2 * HSV_V_DELTA,
        ], dtype=np.float32)
        hs.append(abs(mod[0]))
        out = hsv_transform(img, mod)
        assert out.shape == img.shape and np.isfinite(out).all()
    assert max(hs) <= HSV_H_DELTA + 1e-6
    assert np.mean(hs) < 40  # well below the old ±72 OpenCV-H bug


def test_image_files_mixed_case(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "b.PNG"), np.zeros((8, 8, 3), np.uint8))
    files = image_files_from_folder(str(tmp_path))
    assert len(files) == 2


def test_real_plate_diag_shared():
    fake = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
    small = np.array([[0.1, 0.12, 0.12, 0.1], [0.1, 0.1, 0.12, 0.12]])  # diag ~0.028
    assert not is_real_plate(fake)
    assert is_real_plate(small)
    assert REAL_PLATE_DIAG == 1e-3
    mid = np.array([[0.0, 0.004, 0.004, 0.0], [0.0, 0.0, 0.004, 0.004]])
    # diag ~0.0056 > 1e-3 → real for both train and eval
    assert is_real_plate(mid)


def test_normalize_quad_order_ccw():
    from iwpod.constants import normalize_quad_order
    ccw = np.array([[0.2, 0.2, 0.7, 0.7], [0.3, 0.6, 0.6, 0.3]])  # TL,BL,BR,TR
    ordered = normalize_quad_order(ccw)
    # TL has min x+y
    s = ordered[0] + ordered[1]
    assert np.argmin(s) == 0
