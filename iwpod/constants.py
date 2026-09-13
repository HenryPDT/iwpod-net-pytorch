"""Central IWPOD-v2 constants (single source of truth).

Must stay in sync with:
- DeepStream parser `reconstructIwpod` (stride/side, compile-time)
- `iwpod export` (stride assert, %32 rule)
- training sampler (`side` used in labels2output_map)
"""
import numpy as np

NET_STRIDE = 16
# Empirically chosen in Silva & Jung 2021: ((208 + 40) / 2) / 16
SIDE = ((208.0 + 40.0) / 2.0) / NET_STRIDE  # = 7.75
ANCHOR_HALF = 0.5

# Export / DeepStream alignment: multiples of 32 cover the stride-16 grid
# plus TensorRT alignment headroom.
EXPORT_ALIGNMENT = 32
EXPORT_MIN_SIZE = 256
EXPORT_OPT_SIZE = 416  # Phase-1 Xavier NX infer-dims
EXPORT_MAX_SIZE = 512
DEPLOY_SIZE = 416

# Default rectified plate (OCR contract). Lives in nvdspreprocess
# config, NOT in ONNX graph, so it can change without re-export.
DEFAULT_PLATE_W = 256
DEFAULT_PLATE_H = 96

# Image discovery (suffix-lowercased; case-insensitive on every platform).
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Degenerate "fake plate" used as a background placeholder has diagonal ~1.4e-4.
# Anything just above that is a real annotation (including distant/small plates).
REAL_PLATE_DIAG = 1e-3

# HSV jitter in OpenCV units (H in [0, 179], S/V in [0, 255]).
# ±8 H ≈ ±16°; ±25 S/V is a meaningful photometric nudge without washing the crop.
HSV_H_DELTA = 8.0
HSV_S_DELTA = 25.0
HSV_V_DELTA = 25.0

# Decode size gate: 30x10 px at 400-px input, scaled by max(in_w, in_h)/400.
DECODE_GATE_REF = 400.0
DECODE_MIN_WIDTH = 30.0
DECODE_MIN_HEIGHT = 10.0


def is_real_plate(pts, thresh=None):
    """True iff `pts` is a (2,4) quad whose 0–2 diagonal exceeds REAL_PLATE_DIAG."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (2, 4):
        return False
    t = REAL_PLATE_DIAG if thresh is None else thresh
    return float(np.linalg.norm(pts[:, 0] - pts[:, 2])) > t


def normalize_quad_order(pts):
    """Reorder a (2,4) quad to TL, TR, BR, BL (clockwise from top-left).

    TL = min(x+y), BR = max(x+y), TR = max(x-y), BL = min(x-y).
    """
    xy = np.asarray(pts, dtype=float)
    if xy.shape != (2, 4):
        return xy
    corners = xy.T
    s = corners[:, 0] + corners[:, 1]
    d = corners[:, 0] - corners[:, 1]
    tl = corners[int(np.argmin(s))]
    br = corners[int(np.argmax(s))]
    tr = corners[int(np.argmax(d))]
    bl = corners[int(np.argmin(d))]
    return np.column_stack([tl, tr, br, bl])
