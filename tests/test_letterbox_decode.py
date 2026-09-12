"""Letterbox / decode contract tests."""
import numpy as np

from iwpod.constants import SIDE
from iwpod.decode import decode_single
from iwpod.preprocess import letterbox, unletterbox_quad


def test_letterbox_topleft_vs_center():
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, :10] = 255  # left stripe
    tl, s, dx, dy = letterbox(img, 64, color=0, anchor="topleft")
    assert dx == 0 and dy == 0
    assert tl[0, 0, 0] > 200
    c, _, cdx, cdy = letterbox(img, 64, color=0, anchor="center")
    assert cdx > 0 or cdy > 0
    # half-pad shift on a 2:1 crop
    assert (cdx, cdy) != (dx, dy)


def test_unletterbox_topleft_identity_on_square():
    img = np.zeros((64, 64, 3), np.uint8)
    canvas, s, dx, dy = letterbox(img, 64)
    q = np.array([[10., 50., 50., 10.], [10., 10., 50., 50.]])
    back = unletterbox_quad(q, 64, s, dx, dy, 64, 64)
    assert np.allclose(back, q)


def _grid_pred(y=12, x=12, logit=8.0, aff=None, gh=24, gw=24):
    pred = np.zeros((7, gh, gw), np.float32)
    pred[0, y, x] = logit
    aff = [0.3, 0.0, 0.0, 0.0, 0.3, 0.0] if aff is None else aff
    pred[1:7, y, x] = np.asarray(aff, dtype=np.float32)
    return pred


def test_aabb_gate_keeps_sheared_quad_that_corner_pair_drops():
    # Diamond affine: opposite corners share x (|x0-x2|~0) but AABB is wide.
    # a0=1, a1=-1, a3=a4=0.8 (C++ clamps only a0/a4 to >=0).
    pred = _grid_pred(aff=[1.0, -1.0, 0.0, 0.8, 0.8, 0.0])
    found = decode_single(pred, 384, 384, threshold=0.3, from_logits=True)
    assert found
    pts = found[0][0]
    tl, br = pts.min(1), pts.max(1)
    s = 384 / 400.0
    assert (br[0] - tl[0]) >= 30 * s
    assert abs(pts[0, 0] - pts[0, 2]) < 30 * s


def test_aabb_vs_corner_pair_divergence_case():
    """Document the shear case: AABB width >> |x0-x2|."""
    pts = np.array([[10., 100., 12., 102.], [10., 10., 80., 80.]])
    tl, br = pts.min(1), pts.max(1)
    aabb_w = br[0] - tl[0]
    corner_w = abs(pts[0, 0] - pts[0, 2])
    assert aabb_w > 30 and corner_w < 30


def _cpp_style_quads(pred, in_w, in_h, threshold=0.3, from_logits=True, side=SIDE):
    """NumPy port of reconstructIwpod candidate math (no warp / NMS)."""
    conf_map = 1.0 / (1.0 + np.exp(-pred[0])) if from_logits else pred[0]
    aff = pred[1:7]
    gh, gw = conf_map.shape
    v = 0.5
    base = np.array([[-v, v, v, -v], [-v, -v, v, v], [1.0, 1.0, 1.0, 1.0]])
    scale_gate = max(in_w, in_h) / 400.0
    out = []
    for y in range(gh):
        for x in range(gw):
            if conf_map[y, x] < threshold:
                continue
            a = aff[:, y, x].astype(float)
            aff_mat = np.array([[max(a[0], 0.0), a[1], a[2]], [a[3], max(a[4], 0.0), a[5]]])
            pts_raw = aff_mat @ base
            mn = np.array([[x + 0.5], [y + 0.5]])
            pts = (pts_raw * side + mn) / np.array([[gw], [gh]]) * np.array([[in_w], [in_h]])
            if (pts < 0).any():
                continue
            tl, br = pts.min(1), pts.max(1)
            if (br[0] - tl[0]) < 30 * scale_gate or (br[1] - tl[1]) < 10 * scale_gate:
                continue
            out.append((pts, float(conf_map[y, x])))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def test_python_matches_cpp_golden_quad():
    pred = _grid_pred(y=12, x=10, aff=[0.3, 0.15, 0.0, 0.05, 0.3, 0.0])
    py = decode_single(pred, 384, 384, threshold=0.3, from_logits=True)
    cpp = _cpp_style_quads(pred, 384, 384, threshold=0.3)
    assert py and cpp
    assert np.allclose(py[0][0], cpp[0][0], atol=1e-5)
    assert abs(py[0][1] - cpp[0][1]) < 1e-6


def test_rotated_size_gate_keeps_wide_aabb():
    pred = _grid_pred(aff=[0.4, 1.2, 0.0, 1.2, 0.4, 0.0])
    found = decode_single(pred, 384, 384, threshold=0.3, from_logits=True)
    assert found
    pts = found[0][0]
    tl, br = pts.min(1), pts.max(1)
    assert (br[0] - tl[0]) >= 30 * (384 / 400.0)
    assert (br[1] - tl[1]) >= 10 * (384 / 400.0)
