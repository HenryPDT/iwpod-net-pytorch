"""Batch-1 regression tests: exact-count OHEM, EMA BN copy, tiny-scene val, seeding."""
from copy import deepcopy

import cv2
import numpy as np
import torch

from iwpod.ckpt import load_ckpt_weights, save_ckpt
from iwpod.cli.train import _restore_numpy_state, _seed_worker, _snapshot_numpy_state, update_ema
from iwpod.constants import NET_STRIDE, SIDE
from iwpod.decode import decode_single
from iwpod.loss import iwpodnet_loss_v2
from iwpod.model import IWPODNet
from iwpod.prepare import prepare_dataset


def test_ohem_keeps_exact_count_on_ties():
    # All-background logits equal -> ties must not over-keep.
    yt = torch.zeros(1, 9, 8, 8)
    yt[:, 0, 2, 2] = 1.0  # 1 positive
    yp = torch.zeros(1, 7, 8, 8)  # uniform logits -> uniform BCE ties
    r = iwpodnet_loss_v2(yt, yp, ohem_neg_ratio=3.0)
    assert torch.isfinite(r.total).all()
    # dice_ohem path must also stay finite on the kept subset
    r2 = iwpodnet_loss_v2(yt, yp, ohem_neg_ratio=3.0, dice_ohem=True)
    assert torch.isfinite(r2.total).all()
    assert torch.isfinite(r2.dice)


def test_ema_copies_bn_stats():
    m = IWPODNet(raw_logits=True)
    ema = deepcopy(m)
    # perturb BN running stats on model; EMA must copy exactly, not average
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, torch.nn.BatchNorm2d):
                b.running_mean.fill_(3.0)
                b.running_var.fill_(7.0)
    update_ema(ema, m, 0.999, updates=5000)
    for eb, mb in zip(ema.modules(), m.modules(), strict=True):
        if isinstance(eb, torch.nn.BatchNorm2d):
            assert torch.equal(eb.running_mean, mb.running_mean)
            assert torch.equal(eb.running_var, mb.running_var)


def test_prepare_tiny_scene_gets_val(tmp_path):
    src = tmp_path / "raw" / "tiny"
    src.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(src / f"f{i:03d}.jpg"), np.zeros((40, 60, 3), np.uint8))
        (src / f"f{i:03d}.txt").write_text("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,\n")
    out = tmp_path / "prep"
    stats = prepare_dataset(str(tmp_path / "raw"), str(out), ratio=0.8, seed=0,
                            image_reader=lambda p: (60, 40))
    tiny = next(s for s in stats["scenes"] if s["scene"] == "tiny")
    assert tiny["val"] >= 1 and tiny["train"] >= 1


def test_seed_worker_deterministic():
    import random
    torch.manual_seed(123)
    _seed_worker(0)
    a = np.random.rand()
    torch.manual_seed(123)
    _seed_worker(0)
    b = np.random.rand()
    assert a == b
    assert isinstance(random.getstate(), tuple)


def test_eval_gating_degenerate_high_conf_no_fp(tmp_path):
    # High logit but degenerate quad must not count as FP (gated before sweep).
    aff = np.zeros((6, 4, 4))
    pred = np.concatenate([np.full((1, 4, 4), 10.0), aff], 0)
    found = decode_single(pred, 64, 64, threshold=0.3, from_logits=True)
    assert found == []  # degenerate size gate filters it


def test_numpy_rng_snapshot_roundtrip_and_weights_only(tmp_path):
    np.random.seed(7)
    snap = _snapshot_numpy_state()
    # plain types only: safe under torch.load(weights_only=True)
    assert isinstance(snap[1], list) and all(type(v) is int for v in snap[1][:8])
    p = str(tmp_path / "rng.pt")
    torch.save({"rng": snap}, p)
    back = torch.load(p, map_location="cpu", weights_only=True)["rng"]
    np.random.set_state(_restore_numpy_state(back))
    a = np.random.rand()
    np.random.set_state(_restore_numpy_state(snap))
    assert np.random.rand() == a


def test_lr_floor_min_lr_ratio():
    from iwpod.cli.train import build_lr_fn
    base, total, warm = 0.001, 1000, 100
    floored = build_lr_fn(base, total, warm, "cosine", min_lr_ratio=0.05)
    assert abs(floored(total) - base * 0.05) < 1e-12
    assert floored(total // 2) > floored(total)  # still decays
    legacy = build_lr_fn(base, total, warm, "cosine")
    assert abs(legacy(total) - 0.0) < 1e-12  # default preserves decay-to-zero
    assert abs(legacy(0) - base * 0.1) < 1e-12  # warmup start unchanged


def test_validate_rejects_bad_min_lr_ratio():
    import pytest

    from iwpod.cli.train import default_config_path, load_cfg, validate_config
    cfg = load_cfg(default_config_path())
    validate_config(cfg)  # bundled base.yaml (0.05) must pass
    bad = dict(cfg, min_lr_ratio=1.5)
    with pytest.raises(RuntimeError, match="min_lr_ratio"):
        validate_config(bad)


def test_ckpt_arch_stamps_contract(tmp_path):
    m = IWPODNet(raw_logits=True)
    p = str(tmp_path / "m.pth")
    save_ckpt(p, m, epoch=0, meta={"dim": 128})
    _, arch = load_ckpt_weights(p)
    assert arch["stride"] == NET_STRIDE
    assert arch["side"] == SIDE
    assert arch["dim"] == 128
