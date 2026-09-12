"""Decode unit tests: synthetic positives only (no weights needed)."""
import numpy as np

from iwpod.decode import decode_single, warp_plate


def _pos7(gh=24, gw=24, y=12, x=10, logit=8.0, a=0.3):
    pred = np.zeros((7, gh, gw), np.float32)
    pred[0, y, x] = logit
    pred[1, y, x] = a
    pred[5, y, x] = a
    return pred


def test_decode_finds_synthetic_positive():
    found = decode_single(_pos7(), 384, 384, threshold=0.3, from_logits=True, topk=1)
    assert len(found) == 1
    quad, conf = found[0]
    assert quad.shape == (2, 4)
    assert conf > 0.99
    assert (quad >= 0).all() and (quad[0] < 384).all() and (quad[1] < 384).all()


def test_decode_empty_grid_finds_none():
    pred = np.full((7, 24, 24), -10.0, np.float32)
    assert decode_single(pred, 384, 384, threshold=0.3, from_logits=True) == []


def test_decode_legacy_8ch():
    pred = np.zeros((8, 24, 24), np.float32)
    pred[0, 12, 10] = 0.99
    pred[2, 12, 10] = 0.3
    pred[6, 12, 10] = 0.3
    found = decode_single(pred, 384, 384, threshold=0.3, from_logits=False, topk=1)
    assert len(found) == 1


def test_decode_logits_vs_probs_consistent():
    import math
    p = _pos7(logit=2.0)
    f_logits = decode_single(p, 384, 384, threshold=0.3, from_logits=True)
    f_probs = decode_single(1 / (1 + np.exp(-p)), 384, 384, threshold=0.3, from_logits=False)
    assert len(f_logits) == len(f_probs) == 1
    assert math.isclose(f_logits[0][1], f_probs[0][1], rel_tol=1e-5)


def test_warp_plate_shape():
    crop = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    quad = np.array([[50., 250., 250., 50.], [40., 40., 160., 160.]])
    plate = warp_plate(crop, quad, out_w=256, out_h=96)
    assert plate.shape == (96, 256, 3)
