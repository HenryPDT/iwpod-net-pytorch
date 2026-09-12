"""Horizontal-flip augmentation tests (no weights/data needed)."""
import numpy as np

import iwpod.sampler as sampler
from iwpod.label import Shape
from iwpod.sampler import augment_sample, flip_image_and_ptslist


def _quad():
    return np.array([[0.2, 0.7, 0.7, 0.2], [0.3, 0.3, 0.6, 0.6]])


def test_flip_twice_is_identity():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8).astype(float) / 255
    pts = [_quad()]
    i1, p1 = flip_image_and_ptslist(img, [q.copy() for q in pts])
    i2, p2 = flip_image_and_ptslist(i1, p1)
    assert np.array_equal(i2, img)
    assert np.allclose(p2[0], pts[0])


def test_flip_mirrors_x_and_reorders_corners():
    img = np.zeros((8, 8, 3), dtype=float)
    img[:, 6:, :] = 1.0  # bright right edge -> must end up on the left
    pts = [_quad()]
    flipped, p = flip_image_and_ptslist(img, [q.copy() for q in pts])
    assert (flipped[:, 0, :].mean() > 0.99) and (flipped[:, -1, :].mean() < 0.01)
    # x mirrored + TL<->TR, BL<->BR reorder keeps the corner convention
    assert np.allclose(p[0][0], 1.0 - pts[0][0][[1, 0, 3, 2]])
    assert np.allclose(p[0][1], pts[0][1][[1, 0, 3, 2]])


def test_augment_sample_with_forced_flip():
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    shapes = [Shape(_quad())]
    old = sampler.HFLIP_PROB
    sampler.HFLIP_PROB = 1.0
    try:
        aug, llp, ptslist = augment_sample(img, shapes, 256)
    finally:
        sampler.HFLIP_PROB = old
    assert aug.shape == (256, 256, 3)
    assert len(ptslist) == 1 and ptslist[0].shape == (2, 4)
    # warped pts stay inside the normalized frame
    assert ptslist[0].min() >= -0.05 and ptslist[0].max() <= 1.05
