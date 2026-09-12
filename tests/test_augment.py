"""New-augmentation tests: crop jitter coherence, dropout guard, pixel stage."""
import numpy as np

import iwpod.sampler as sampler
from iwpod.sampler import (
    MAX_ROT_DEG,
    MAX_ROT_SUM,
    _sample_angles,
    guarded_cutout,
    jitter_crop_window,
    limit_angles,
)


def test_sampled_angles_within_production_limits():
    np.random.seed(0)
    for _ in range(1000):
        a = _sample_angles()
        assert np.isfinite(a).all()
        assert (np.abs(a) <= MAX_ROT_DEG + 1e-9).all(), a
        assert np.abs(a).sum() <= MAX_ROT_SUM + 1e-9, a


def test_rotation_limits_are_production_sane():
    # Calibrated per axis: yaw headroom for gate turns, pitch near mount
    # range, roll cheap (no foreshortening); sum admits the joint corner.
    assert list(MAX_ROT_DEG) == [45.0, 60.0, 30.0]
    assert MAX_ROT_SUM == 135.0


def test_limit_angles_keeps_normal_vectors():
    a = np.array([30.0, -20.0, 10.0])
    assert np.array_equal(limit_angles(a, 140), a)


def test_limit_angles_caps_proportionally():
    a = np.array([60.0, 60.0, 60.0])
    out = limit_angles(a, 90.0)
    assert abs(np.abs(out).sum() - 90.0) < 1e-9
    assert np.allclose(out / out.sum(), a / a.sum())  # direction preserved


def test_limit_angles_safe_on_sign_cancellation():
    # Legacy code divided by the signed sum (~0 here) and exploded.
    a = np.array([65.0, -65.0, 10.0])
    out = limit_angles(a, 140.0)
    assert np.isfinite(out).all()
    assert abs(np.abs(out).sum() - 140.0) < 1e-9


def _img(h=64, w=80):
    return (np.random.rand(h, w, 3)).astype(np.float32)


def test_jitter_moves_image_and_pts_together():
    img = np.zeros((64, 80, 3), dtype=np.float32)
    img[10, 20, :] = 1.0  # marker pixel
    px, py = 20 / 80, 10 / 64
    d = 0.02
    pts = [np.array([[px - d, px + d, px + d, px - d],
                     [py - d, py + d, py - d, py + d]])]
    np.random.seed(1)
    jimg, shifted = jitter_crop_window(img, pts, max_shift=0.15)
    my, mx = np.unravel_index(jimg[..., 0].argmax(), jimg.shape[:2])
    cx, cy = shifted[0].mean(axis=1) * np.array([80, 64])
    assert abs(mx - cx) < 1.5 and abs(my - cy) < 1.5


def test_jitter_falls_back_when_plate_leaves_frame():
    img = _img()
    corner = [np.array([[0.0, 0.08, 0.08, 0.0], [0.0, 0.0, 0.08, 0.08]])]
    np.random.seed(0)
    jimg, shifted = jitter_crop_window(img, corner, max_shift=0.9)
    in_frame = (shifted[0].min() >= 0.0 and shifted[0].max() <= 1.0)
    if not in_frame:
        assert np.array_equal(jimg, img)  # fallback: original frame untouched
    else:
        assert shifted[0].mean() != corner[0].mean() or np.array_equal(jimg, img)


def test_jitter_never_mutates_inputs():
    img = _img()
    pts = [np.array([[0.2, 0.7, 0.7, 0.2], [0.3, 0.3, 0.6, 0.6]])]
    before = pts[0].copy()
    np.random.seed(3)
    jitter_crop_window(img, pts, max_shift=0.15)
    assert np.array_equal(pts[0], before)  # dataset entries are reused across epochs


def test_cutout_skips_plate_swallowing_holes():
    # Oversized holes cover the whole frame, hence the whole plate:
    # guard must skip every hole -> output identical.
    dim = 64
    roi = np.full((dim, dim, 3), 0.5, dtype=np.float32)
    plate = [np.array([[0.3, 0.6, 0.6, 0.3], [0.3, 0.3, 0.6, 0.6]])]
    np.random.seed(0)
    out = guarded_cutout(roi, plate, dim, p=1.0, max_holes=4, hole=200)
    assert np.array_equal(out, roi)


def test_cutout_applies_on_background():
    dim = 128
    roi = np.zeros((dim, dim, 3), dtype=np.float32)
    np.random.seed(0)
    out = guarded_cutout(roi, [], dim, p=1.0, max_holes=4, hole=30)
    assert not np.array_equal(out, roi)


def test_pixel_stage_preserves_shape_dtype():
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    for _ in range(5):
        out = sampler._pixel_stage_fn()(image=img)["image"]
        assert out.shape == img.shape and out.dtype == np.uint8


def test_pixel_stage_covers_grayscale():
    # IR/night cameras output grayscale: the stage must produce chroma-free
    # samples often enough to matter (p=0.15 -> ~all runs hit it in 100 draws).
    import albumentations as alb

    assert any(isinstance(t, alb.ToGray) for t in sampler._pixel_stage_fn().transforms)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    gray = alb.ToGray(p=1.0)(image=img)["image"]
    assert (gray[..., 0] == gray[..., 1]).all() and (gray[..., 1] == gray[..., 2]).all()
    hits = 0
    for _ in range(100):
        out = sampler._pixel_stage_fn()(image=img)["image"]
        if (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all():
            hits += 1
    assert hits > 5, f"grayscale too rare: {hits}/100"
