"""NumPy reference decode for IWPOD-v2 (mirrors DeepStream parser).

ONNX v2 output: [B,7,Gh,Gw] NCHW float32, ch0 = RAW LOGITS, ch1..6 = affine.
Legacy: ch0 = sigmoid prob. `from_logits` flag handles both.

Decode per cell (x=col, y=row):
  mn = [x+0.5, y+0.5]; MN = [Gw, Gh]
  A = [[max(a0,0), a1, a2],[a3, max(a4,0), a5]]
  pts_raw = A @ Base  (Base = 3x4 canonical square, half=0.5)
  pts = (pts_raw * side + mn) / MN * [W_in, H_in]
Returns quads in INPUT pixel space (vehicle crop, pre-letterbox if any).
"""
import numpy as np

from .constants import ANCHOR_HALF, DECODE_GATE_REF, DECODE_MIN_HEIGHT, DECODE_MIN_WIDTH, NET_STRIDE, SIDE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_single(pred, in_w, in_h, threshold=0.3, from_logits=True,
                  topk=1, nms_iou=0.25, side=SIDE, stride=NET_STRIDE):
    """pred: [7,Gh,Gw] single batch. Returns list of (quad(2,4), conf)."""
    assert pred.shape[0] in (7, 8), f'unexpected channels {pred.shape}'
    if pred.shape[0] == 8:  # legacy WPOD [obj,bg,affine6]
        conf_map = pred[0]
        aff = pred[2:8]
    else:
        raw = pred[0]
        conf_map = sigmoid(raw) if from_logits else raw
        aff = pred[1:7]
    gh, gw = conf_map.shape
    grid_mn = np.array([gw, gh], dtype=float)
    v = ANCHOR_HALF
    base = np.array([[-v, v, v, -v], [-v, -v, v, v], [1., 1., 1., 1.]])

    ys, xs = np.where(conf_map > threshold)
    cands = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        a = aff[:, y, x].astype(float)
        aff_mat = np.array([[max(a[0], 0), a[1], a[2]], [a[3], max(a[4], 0), a[5]]])
        pts_raw = aff_mat @ base
        mn = np.array([x + 0.5, y + 0.5])
        pts = (pts_raw * side + mn.reshape(2, 1)) / grid_mn.reshape(2, 1) * np.array([[in_w], [in_h]])
        if (pts < 0).any():
            continue
        # degenerate size gate: AABB (matches C++ reconstructIwpod)
        tl = pts.min(axis=1)
        br = pts.max(axis=1)
        s = max(in_w, in_h) / DECODE_GATE_REF
        if (br[0] - tl[0]) < DECODE_MIN_WIDTH * s or (br[1] - tl[1]) < DECODE_MIN_HEIGHT * s:
            continue
        cands.append((pts, float(conf_map[y, x])))
    cands.sort(key=lambda t: t[1], reverse=True)
    # quad-NMS via enclosing AABB IoU (fast) — full polygon IoU in eval only
    kept = []
    for pts, conf in cands:
        tl = pts.min(axis=1)
        br = pts.max(axis=1)
        overlap = False
        for kpts, _ in kept:
            ktl = kpts.min(axis=1)
            kbr = kpts.max(axis=1)
            iw = max(0, min(br[0], kbr[0]) - max(tl[0], ktl[0]))
            ih = max(0, min(br[1], kbr[1]) - max(tl[1], ktl[1]))
            inter = iw * ih
            union = max(1e-6, (br[0] - tl[0]) * (br[1] - tl[1]) + (kbr[0] - ktl[0]) * (kbr[1] - ktl[1]) - inter)
            if inter / union > nms_iou:
                overlap = True
                break
        if not overlap:
            kept.append((pts, conf))
        if len(kept) >= topk:
            break
    return kept


def warp_plate(crop_bgr, quad_xy, out_w=256, out_h=96):
    """Perspective-rectify vehicle crop to flat plate (default 256x96)."""
    import cv2
    src = np.asarray(quad_xy, dtype=np.float32).T.reshape(4, 2)  # (2,4)->(4,2)
    # order: TL,TR,BR,BL assumed from training convention
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    try:
        hom = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return cv2.warpPerspective(crop_bgr, hom, (out_w, out_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
