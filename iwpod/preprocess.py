"""Shared image preprocessing: letterbox resize preserving aspect ratio.

DeepStream feeds this model letterboxed square crops
(`maintain-aspect-ratio=1` without `symmetric-padding`), so offline eval/infer
must do the same: longest-side scale, **top-left** placement, black pad.
A named `center` anchor is kept only for legacy experiments.
"""
import cv2
import numpy as np


def letterbox(img, size, color=0, anchor="topleft"):
    """Resize longest side to `size`, pad to `size`x`size`.

    Returns (canvas_uint8, scale, pad_x, pad_y) with
    canvas = resize(img, scale) placed at offset (pad_x, pad_y).

    `anchor='topleft'` matches DeepStream asymmetric letterbox (image at
    origin, pad right/bottom). `anchor='center'` is the old offline path.
    """
    h0, w0 = img.shape[:2]
    s = size / max(h0, w0)
    nh, nw = int(round(h0 * s)), int(round(w0 * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    if anchor == "center":
        dx, dy = (size - nw) // 2, (size - nh) // 2
    else:
        dx, dy = 0, 0
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, s, dx, dy


def unletterbox_quad(quad_s, size, scale, dx, dy, w0, h0):
    """Map a (2,4) quad from letterboxed `size` space back to original pixels."""
    q = (quad_s - np.array([[dx], [dy]])) / scale
    q[0] = np.clip(q[0], 0, w0)
    q[1] = np.clip(q[1], 0, h0)
    return q
