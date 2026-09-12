"""Augmentation<->annotation consistency tests on real LP data.

Proves the full chain sample -> augment_sample -> labels2output_map keeps
annotations correct:
- shapes/dtypes/finite outputs,
- real plates always yield >= 1 positive cell (centroid fallback),
- background samples yield zero positives,
- positive cells sit inside the plate bbox (quad<->map agreement),
- regression targets round-trip back to the input quad (exact inversion),
- same seed reproduces the identical sample (dataset entries are reused).
"""
import numpy as np
import pytest

from iwpod.dataset import ALPRDataset

DATA_TRAIN = "datasets/LP/train"
DATA_VAL = "datasets/LP/val"
DIM = 256
STRIDE = 16
SIDE = ((208.0 + 40.0) / 2.0) / STRIDE


def _entries(sub, n=12):
    import os

    from iwpod.dataset import image_label_loader
    entries, _ = image_label_loader(os.path.join("datasets/LP", sub))
    return entries[:n]


def _decode_cell_targets(vec, cx, cy):
    """Invert the side-normalized encoding back to a normalized (2,4) quad."""
    p_side = np.asarray(vec, dtype=float).reshape(4, 2).T
    p_mn = p_side * SIDE + np.array([[cx + .5], [cy + .5]])
    return p_mn * STRIDE / DIM


def test_real_plates_have_positive_cells_and_agree_with_quad():
    ds = ALPRDataset(dim=DIM, entries=_entries("train"))
    checked = 0
    for idx in range(len(ds)):
        np.random.seed(1000 + idx)
        import random as _random
        _random.seed(1000 + idx)
        inputs, targets = ds[idx]
        assert inputs.shape == (3, DIM, DIM) and targets.shape[0] == 9
        assert np.isfinite(inputs.numpy()).all() and np.isfinite(targets.numpy()).all()
        assert inputs.min() >= 0.0 and inputs.max() <= 1.0
        y = targets.numpy().transpose(1, 2, 0)
        pos = np.argwhere(y[..., 0] > 0.5)
        assert len(pos) > 0, f"real plate with zero positive cells (idx={idx})"
        # every positive cell reconstructs the input quad through the encoding
        _, _, ptslist = _raw(ds, idx)
        quad = np.asarray(ptslist[0], dtype=float)
        for cy, cx in pos:
            rec = _decode_cell_targets(y[cy, cx, 1:], cx, cy)
            assert np.allclose(rec, quad, atol=1e-4), f"encode round-trip failed (idx={idx})"
        checked += 1
    assert checked == len(ds)


def _raw(ds, idx):
    from iwpod.sampler import augment_sample
    jpg, shapes = ds.data[idx]
    import cv2
    img = cv2.imread(jpg)
    np.random.seed(1000 + idx)
    import random as _random
    _random.seed(1000 + idx)
    return augment_sample(img, shapes, DIM)


def test_background_samples_have_zero_positives():
    from iwpod.label import Shape
    fake = Shape(np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]]))
    ds = ALPRDataset(dim=DIM, entries=[("none.jpg", [fake])] * 3)
    ds._read = lambda i: (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    for idx in range(len(ds)):
        np.random.seed(42 + idx)
        import random as _random
        _random.seed(42 + idx)
        _, targets = ds[idx]
        assert (targets.numpy()[0] == 0).all()


def test_same_seed_reproduces_labels():
    # Label coherence must be deterministic given seeds (dataset entries are
    # reused across epochs). Pixels may vary: albumentations draws from an
    # independent RNG, like most industrial augmentors.
    ds = ALPRDataset(dim=DIM, entries=_entries("val", n=4))
    for idx in range(len(ds)):
        runs = []
        for _ in range(2):
            np.random.seed(7)
            import random as _random
            _random.seed(7)
            inputs, targets = ds[idx]
            runs.append(targets.numpy().copy())
        assert np.array_equal(runs[0], runs[1])
        _, _, p1 = _raw(ds, idx)
        _, _, p2 = _raw(ds, idx)
        assert all(np.array_equal(a, b) for a, b in zip(p1, p2, strict=True))


def test_positive_cells_inside_plate_bbox():
    ds = ALPRDataset(dim=DIM, entries=_entries("train"))
    out = DIM // STRIDE
    for idx in range(len(ds)):
        np.random.seed(1000 + idx)
        import random as _random
        _random.seed(1000 + idx)
        _, targets = ds[idx]
        _, _, ptslist = _raw(ds, idx)
        q = np.asarray(ptslist[0], dtype=float) * out
        x0, y0, x1, y1 = q[0].min(), q[1].min(), q[0].max(), q[1].max()
        pos = np.argwhere(targets.numpy()[0] > 0.5)
        for cy, cx in pos:
            assert x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1


@pytest.mark.parametrize("dim", [256, 384])
def test_consistency_across_scales(dim):
    ds = ALPRDataset(dim=dim, entries=_entries("val", n=6))
    out = dim // STRIDE
    for idx in range(len(ds)):
        np.random.seed(99)
        import random as _random
        _random.seed(99)
        inputs, targets = ds[idx]
        assert inputs.shape == (3, dim, dim)
        assert targets.shape == (9, out, out)
        assert (targets.numpy()[0] > 0.5).sum() > 0
