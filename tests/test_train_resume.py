"""Train CLI/resume helpers (no GPU, no full training)."""
import argparse

import torch

from iwpod.cli.train import (
    apply_cli_overrides,
    build_continue_lr_fn,
    build_lr_fn,
    default_config_path,
    is_better,
    load_cfg,
    resolve_resume,
    tb_epoch_tags,
    validate_config,
)


def _parser():
    from iwpod.cli import train as train_cli
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    return ap


def test_lr_fn_zero_is_warmup_start():
    fn = build_lr_fn(0.001, 1000, 100, "cosine", min_lr_ratio=0.05)
    assert abs(fn(0) - 0.001 * 0.1) < 1e-12


def test_grad_accum_uses_opt_step_index():
    # First optimizer step must see lr_fn(0), not lr_fn(1).
    fn = build_lr_fn(0.001, 10, 2, "cosine", min_lr_ratio=0.05)
    opt_iters_done = 0
    lrs = []
    for micro in range(8):
        accum = 4
        is_step = ((micro + 1) % accum == 0)
        if is_step:
            lrs.append(fn(opt_iters_done))
            opt_iters_done += 1
    assert abs(lrs[0] - fn(0)) < 1e-12
    assert abs(lrs[1] - fn(1)) < 1e-12


def test_continue_lr_does_not_jump_up():
    old = build_lr_fn(0.001, 100, 10, "cosine", min_lr_ratio=0.05)
    start_it, new_total = 80, 150
    current = old(start_it)
    cont = build_continue_lr_fn(current, start_it, new_total, 0.001 * 0.05)
    assert cont(start_it) <= current + 1e-12
    assert cont(new_total) <= cont(start_it) + 1e-12


def test_resolve_resume_prefers_last(tmp_path):
    (tmp_path / "run_last.pth").write_bytes(b"x")
    (tmp_path / "run_best.pth").write_bytes(b"y")
    cand, d = resolve_resume(str(tmp_path))
    assert cand.endswith("_last.pth") and d == str(tmp_path)


def test_resolve_resume_falls_back_to_best(tmp_path):
    (tmp_path / "run_best.pth").write_bytes(b"y")
    cand, d = resolve_resume(str(tmp_path))
    assert cand.endswith("_best.pth")


def test_save_every_enables_history():
    cfg = load_cfg(default_config_path())
    assert cfg["save_history_ckpt"] is False
    args = _parser().parse_args(["--save-every", "5"])
    apply_cli_overrides(cfg, args)
    assert cfg["save_every"] == 5 and cfg["save_history_ckpt"] is True


def test_patience_from_yaml_and_cli():
    cfg = load_cfg(default_config_path())
    validate_config(cfg)
    assert int(cfg.get("patience", 0)) == 0
    args = _parser().parse_args(["--patience", "20"])
    apply_cli_overrides(cfg, args)
    assert cfg["patience"] == 20


def test_is_better_iou50_tiebreak():
    base = dict(map=0.5, iou70=0.4, iou50=0.6, recall=0.8, vloss=1.0)
    assert is_better({**base, "iou50": 0.7}, base)
    assert not is_better({**base, "iou50": 0.5}, base)


def test_tb_tags_include_det_rate_and_rmse_detected():
    tags = tb_epoch_tags()
    assert "val/det_rate" in tags
    assert "val/rmse_detected" in tags
    assert "val/rmse" not in tags
    assert "val/iou50" not in tags


def test_as_byte_tensor_roundtrip():
    from iwpod.cli.train import _as_byte_tensor
    t = torch.get_rng_state()
    back = _as_byte_tensor(t.tolist())
    assert back.dtype == torch.uint8 and back.numel() == t.numel()


def test_dry_run_flag_registered():
    args = _parser().parse_args(["--dry-run"])
    assert args.dry_run is True


def test_improved_computed_before_history_and_last():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iwpod" / "cli" / "train.py").read_text()
    assert src.index("improved = is_better") < src.index("extra = _extra")
    assert src.index("extra = _extra") < src.index("if save_history")
    assert src.index("extra = _extra") < src.index('f"{run_name}_last.pth"')


def test_evaluate_size_defaults_to_ckpt_dim():
    from argparse import Namespace

    from iwpod.cli.evaluate import _resolve_size
    from iwpod.constants import DEPLOY_SIZE
    assert _resolve_size(Namespace(size=None), {"dim": 384}) == 384
    assert _resolve_size(Namespace(size=None), {}) == DEPLOY_SIZE
    try:
        _resolve_size(Namespace(size=100), {})
    except RuntimeError as e:
        assert "multiple of" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
