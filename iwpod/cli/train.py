"""`iwpod train`: CLI-first config over the bundled base.yaml, run dirs under out/train.

Data: --data ROOT with ROOT/train/ + ROOT/val/ (both required; see docs/DATASET.md).
Training never splits for you. Checkpoints + train_command.txt +
config.yaml + train.log land in out/train/<name>[_2..]. Resume with --resume
(ckpt path or run dir; continues in place). Single-GPU only.

Best checkpoint: single `<name>_best.pth` tracking held-out mAP@50-95
(tie-break IoU@0.7 → IoU@0.5 → recall@0.5 → val loss). With no val set it falls back to
train loss with a loud warning.
"""
import argparse
import math
import os
import random
import time

import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from iwpod import ckpt as ckptlib
from iwpod import runs
from iwpod.constants import NET_STRIDE, SIDE
from iwpod.dataset import ALPRDataset, image_label_loader
from iwpod.eval_core import entries_from_loader_entries, evaluate_quads, format_block
from iwpod.loss import iwpodnet_loss_v2
from iwpod.meters import MeterBuffer, gpu_mem_usage, host_mem_usage
from iwpod.utils import image_files_from_folder


def register(p):
    p.epilog = (
        "Examples:\n"
        "  iwpod train --data datasets/LP --epochs 200 --batch-size 32 --lr 0.001\n"
        "  iwpod train --data datasets/LP --epochs 200 --batch-size 32 --lr 0.001 "
        "--size 384 --seed 42 --name exp1\n"
        "  iwpod train --data datasets/LP --name exp1  # rerun -> out/train/exp1_2\n"
        "  iwpod train --resume out/train/exp1 --epochs 300  # continue in place\n"
        "  iwpod train --data datasets/LP --batch-size -1  # probe VRAM for max batch\n"
        "  iwpod train --data datasets/LP --batch-size 8 --grad-accum 4  # eff. batch 32"
    )
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    data_g = p.add_argument_group("data", "Dataset selection")
    data_g.add_argument("--data", default=None,
                        help="Dataset root with train/ + val/ subdirs, e.g. datasets/LP "
                             "(default: datasets/<only-name> if unambiguous, else --train-dir)")
    data_g.add_argument("--train-dir", default="train_dir",
                        help="Legacy alias: pairs dir used as train/ with no val set (ignored if --data given)")
    data_g.add_argument("--cache", nargs="?", const="ram", default=None, choices=["ram"],
                        help="Preload images into RAM: bare --cache means ram (fastest, "
                             "needs ~1.2x raw pixels free). Default: lazy reads")
    train_g = p.add_argument_group("training", "Common training parameters (override the YAML config)")
    train_g.add_argument("--epochs", type=int, default=None)
    train_g.add_argument("--batch-size", type=int, default=None,
                         help="Micro-batch size; -1 = probe VRAM for max (yolox-style)")
    train_g.add_argument("--auto-batch-target", type=float, default=0.7,
                         help="VRAM fraction target for --batch-size -1 probing")
    train_g.add_argument("--grad-accum", type=int, default=None,
                         help="Gradient accumulation steps; effective batch = batch-size x steps")
    train_g.add_argument("--patience", type=int, default=None,
                         help="Early-stopping patience in epochs without improvement (0 = off)")
    train_g.add_argument("--lr", "--learning-rate", type=float, default=None)
    train_g.add_argument("--weight-decay", type=float, default=None)
    train_g.add_argument("--seed", type=int, default=None)
    train_g.add_argument("--amp", dest="amp", action="store_true", default=None,
                         help="Enable AMP autocast (default from config)")
    train_g.add_argument("--ema-decay", type=float, default=None)
    train_g.add_argument("--warmup-epochs", type=int, default=None)
    train_g.add_argument("--dry-run", action="store_true",
                         help="Validate config+data and exit without training (single-GPU)")
    train_g.add_argument("--size", type=int, default=None,
                         help="Square train resolution (sets dim; default from config)")
    train_g.add_argument("--no-multiscale", action="store_true",
                         help="Disable multi-scale (train at --size/config dim only)")
    train_g.add_argument("--scheduler", type=str, default=None, help="cosine | none")
    train_g.add_argument("--no-amp", action="store_true", help="Disable AMP autocast")
    train_g.add_argument("--no-ema", action="store_true", help="Disable EMA weights")
    train_g.add_argument("--save-every", type=int, default=None, help="Epoch checkpoint cadence (implies history on)")
    train_g.add_argument("--save-history", dest="save_history", action="store_true", default=None,
                         help="Keep <name>_epoch<N>.pth history (default from config)")
    train_g.add_argument("--num-workers", type=int, default=None)
    out_g = p.add_argument_group("output", "Run directory")
    out_g.add_argument("--model-dir", default="out/train",
                       help="Base run dir (default out/train); run = <base>/<name>[_2..]")
    out_g.add_argument("--name", default="iwpodv2")
    out_g.add_argument("--config", default=None,
                       help="YAML config (default: bundled iwpod/configs/base.yaml)")
    out_g.add_argument("--resume", default=None, help="Checkpoint path or run dir; continues in place")
    p.set_defaults(func=run)


def default_config_path():
    from importlib import resources
    return str(resources.files("iwpod") / "configs" / "base.yaml")


def load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _has_images(d):
    return bool(image_files_from_folder(d))


def _autodetect_data_root():
    """Single unambiguous datasets/<name>/ with train/ → use it, else None."""
    if not os.path.isdir("datasets"):
        return None
    cands = [d for d in sorted(os.listdir("datasets"))
             if os.path.isdir(os.path.join("datasets", d, "train")) and
             _has_images(os.path.join("datasets", d, "train"))]
    return os.path.join("datasets", cands[0]) if len(cands) == 1 else None


def resolve_data(args):
    """Returns (train_entries, val_entries).

    The train/val split is dataset preparation, not training: both splits must
    exist explicitly (ROOT/train/ + ROOT/val/, e.g. --data datasets/LP).
    """
    data_root = args.data or _autodetect_data_root()
    if data_root:
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        if not os.path.isdir(train_dir):
            raise RuntimeError(f"No train/ subdir in --data root '{data_root}'")
        if not os.path.isdir(val_dir):
            raise RuntimeError(
                f"No val/ subdir in --data root '{data_root}': prepare the "
                f"split in the dataset (see docs/DATASET.md), training will not split for you.")
        train_entries, _ = image_label_loader(train_dir)
        val_entries, _ = image_label_loader(val_dir)
        logger.info(f"Train: {train_dir} ({len(train_entries)} images) | "
                    f"Val: {val_dir} ({len(val_entries)} images)")
    else:
        train_entries, _ = image_label_loader(args.train_dir)
        logger.info(f"Train: {args.train_dir} ({len(train_entries)} images) | no val set")
        val_entries = []
    if not train_entries:
        raise RuntimeError("Empty train set: check --data / --train-dir paths")
    return train_entries, val_entries


def apply_cli_overrides(cfg, args):
    """CLI flags win over YAML. Returns resolved (epochs, bs, lr, seed)."""
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.size is not None:
        if args.size % 32:
            raise RuntimeError(f"--size must be multiple of 32, got {args.size}")
        cfg["dim"] = args.size
    if args.no_multiscale:
        cfg["multi_scale"] = [cfg["dim"]]
    if args.scheduler is not None:
        cfg["scheduler"] = args.scheduler
    if args.no_amp:
        cfg["use_amp"] = False
    if args.no_ema:
        cfg["use_ema"] = False
    if args.save_every is not None:
        if args.save_every <= 0:
            raise RuntimeError("--save-every must be positive")
        cfg["save_every"] = args.save_every
        # Explicit cadence implies history; otherwise the flag is silently dead.
        cfg["save_history_ckpt"] = True
    if getattr(args, "save_history", None):
        cfg["save_history_ckpt"] = True
    if getattr(args, "amp", None) is not None:
        cfg["use_amp"] = bool(args.amp)
    if getattr(args, "ema_decay", None) is not None:
        cfg["ema_decay"] = float(args.ema_decay)
    if getattr(args, "warmup_epochs", None) is not None:
        cfg["warmup_epochs"] = int(args.warmup_epochs)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.patience is not None:
        cfg["patience"] = args.patience
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    return cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg.get("seed", 42)


def validate_config(cfg):
    """Fail fast on inconsistent configs (mirrors yolox config.validate)."""
    dim = cfg["dim"]
    if dim % 32:
        raise RuntimeError(f"dim must be a multiple of 32, got {dim}")
    scales = cfg.get("multi_scale", [dim])
    if any(s % 32 for s in scales):
        raise RuntimeError(f"multi_scale entries must be multiples of 32, got {scales}")
    if cfg.get("scheduler", "cosine") not in ("cosine", "none"):
        raise RuntimeError(f"scheduler must be cosine|none, got {cfg.get('scheduler')}")
    if not 0.0 <= float(cfg.get("min_lr_ratio", 0.0)) <= 1.0:
        raise RuntimeError(f"min_lr_ratio must be in [0, 1], got {cfg.get('min_lr_ratio')}")
    if cfg.get("print_interval", 10) <= 0:
        raise RuntimeError("print_interval must be positive")
    if int(cfg.get("warmup_epochs", 0)) < 0:
        raise RuntimeError("warmup_epochs must be >= 0")
    if int(cfg.get("warmup_epochs", 0)) > int(cfg.get("epochs", 0)):
        raise RuntimeError(
            f"warmup_epochs ({cfg.get('warmup_epochs')}) > epochs ({cfg.get('epochs')}): "
            "LR would never leave warmup")
    if int(cfg.get("save_every", 10)) <= 0:
        raise RuntimeError("save_every must be positive")
    if int(cfg.get("patience", 0)) < 0:
        raise RuntimeError("patience must be >= 0")
    if cfg.get("stride", NET_STRIDE) != NET_STRIDE:
        raise RuntimeError(f"stride is a label-encoding constant, must be {NET_STRIDE}")
    if abs(cfg.get("side", SIDE) - SIDE) > 1e-9:
        raise RuntimeError(f"side is a label-encoding constant, must be {SIDE}")
    loss = cfg.get("loss", {})
    for k in ("w_cls", "w_dice", "w_loc"):
        if loss.get(k, 1.0) < 0:
            raise RuntimeError(f"loss.{k} must be non-negative")


def _harden_cv2_threads():
    """Disable OpenCV internal threading (forked DataLoader workers + threaded
    cv2/OpenMP can deadlock or oversubscribe alongside albumentations)."""
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(0)
    except Exception:
        pass


def _seed_worker(worker_id):
    """Seed numpy/random per DataLoader worker (sampler is numpy-heavy)."""
    _harden_cv2_threads()
    worker_seed = torch.initial_seed() % 2**32
    import numpy as _np
    _np.random.seed(worker_seed)
    random.seed(worker_seed)


def _snapshot_numpy_state():
    """Numpy RNG state as plain lists (torch>=2.6 weights_only-safe)."""
    import numpy as _np
    name, arr, pos, has_gauss, cached = _np.random.get_state()
    return (name, [int(v) for v in arr], int(pos), int(has_gauss), float(cached))


def _restore_numpy_state(state):
    """Inverse of _snapshot_numpy_state; tolerates legacy raw-tuple states."""
    import numpy as _np
    name, arr, pos, has_gauss, cached = state
    return (name, _np.array(arr, dtype=_np.uint32), int(pos), int(has_gauss), float(cached))


def _as_byte_tensor(state):
    """Restore a torch RNG ByteTensor from a tensor or a serialized list."""
    if isinstance(state, torch.Tensor):
        return state.cpu().contiguous().to(torch.uint8)
    return torch.tensor(state, dtype=torch.uint8)


def _set_seed(seed):
    """Python/torch/numpy (+ CUDA) seed and deterministic cuDNN policy."""
    random.seed(seed)
    torch.manual_seed(seed)
    import numpy as _np
    _np.random.seed(seed)
    _harden_cv2_threads()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_resume(resume):
    """Accept a checkpoint path or a run dir (prefers last, falls back to best)."""
    if resume and os.path.isdir(resume):
        files = os.listdir(resume)
        lasts = sorted(f for f in files if f.endswith("_last.pth"))
        bests = sorted(f for f in files if f.endswith("_best.pth"))
        if lasts:
            return os.path.join(resume, lasts[0]), resume
        cand = os.path.join(resume, bests[0]) if bests else None
        return cand, resume
    return resume, None


# --------------------------------------------------------------------------
# Training helpers (kept small; run() below is orchestration only)
# --------------------------------------------------------------------------

def setup_logging(run_dir):
    logger.remove()
    fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(lambda m: print(m, end=""), level="INFO",
               format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(os.path.join(run_dir, "train.log"), level="INFO", format=fmt)


def log_model_summary(model, dim, device):
    n_params = sum(p.numel() for p in model.parameters())
    try:
        import torch.utils.flop_counter as _fc
        _dummy = torch.zeros(1, 3, dim, dim, device=device)
        with _fc.FlopCounterMode(model, display=False) as _fcp:
            model(_dummy)
        gflops = _fcp.get_total_flops() / 1e9
        logger.info(f"Model: IWPODNet  params={n_params / 1e6:.2f}M  GFLOPs={gflops:.2f}")
    except Exception:
        logger.info(f"Model: IWPODNet  params={n_params / 1e6:.2f}M  GFLOPs=n/a")


def build_lr_fn(base_lr, total_iters, warmup_iters, scheduler, min_lr_ratio=0.0):
    """Per-iteration LR (YOLOX update_lr pattern): linear warmup + cosine.

    min_lr_ratio floors the cosine tail at base_lr*ratio (YOLOX min_lr_ratio,
    default 0.05 in base.yaml) so late epochs keep updating instead of
    decaying to zero.
    """
    min_lr = base_lr * min_lr_ratio

    def _lr(it):
        if scheduler == "none" or total_iters <= 0:
            return base_lr
        if it < warmup_iters:
            return base_lr * (0.1 + 0.9 * it / max(1, warmup_iters))
        t = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
        return min_lr + (base_lr - min_lr) * cos
    return _lr


def build_continue_lr_fn(start_lr, start_it, total_iters, min_lr):
    """Anneal from the current LR over the remaining horizon (no cosine jump)."""
    remaining = max(1, int(total_iters) - int(start_it))

    def _lr(it):
        t = (it - start_it) / remaining
        t = min(1.0, max(0.0, t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr + (start_lr - min_lr) * cos
    return _lr


def update_ema(ema, model, base_decay, updates):
    """EMA over params with YOLOX-style decay ramp; BN stats are copied.

    d = base_decay * (1 - exp(-updates/2000)): ~0 early (responsive) -> base
    late. Same ramp as pixeltable-yolox ModelEMA. running_mean/var and
    num_batches_tracked follow the model directly instead of being averaged.
    """
    d = base_decay * (1 - math.exp(-updates / 2000))
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if "running_" in k or "num_batches" in k:
                v.copy_(msd[k])
            elif v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


_DEFAULT_LOSS = dict(w_cls=1.0, w_dice=0.5, w_loc=1.0,
                     focal_alpha=0.25, focal_gamma=2.0, ohem_neg_ratio=3.0,
                     dice_ohem=False)


def loss_kwargs(loss_cfg):
    merged = {**_DEFAULT_LOSS, **(loss_cfg or {})}
    return dict(w_cls=merged["w_cls"], w_dice=merged["w_dice"], w_loc=merged["w_loc"],
                focal_alpha=merged["focal_alpha"], focal_gamma=merged["focal_gamma"],
                ohem_neg_ratio=merged["ohem_neg_ratio"], dice_ohem=bool(merged["dice_ohem"]))


def train_one_epoch(model, ema, ema_state, loader, opt, scaler, lr_fn, cfg, device,
                    use_cuda, epoch, epochs, cur_scale, iters_done, total_iters, tb,
                    opt_iters_done=0, total_micro=None):
    """One epoch with smoothed per-iter logging. Returns (means dict, iters_done, opt_iters_done).

    LR is applied at the *current* optimizer-step index, then the counter
    increments after `scaler.step`, so the first update uses `lr_fn(0)`.
    ETA uses remaining micro-batches; TensorBoard `train/*` uses micro-batch
    global step. Optimizer-step count is tracked separately.
    """
    print_interval = cfg.get("print_interval", 10)
    accum = max(1, int(cfg.get("_grad_accum", 1)))
    lkw = loss_kwargs(cfg.get("loss"))
    meter = MeterBuffer(window_size=print_interval)
    model.train()
    opt.zero_grad()
    tot = tot_cls = tot_dice = tot_loc = 0.0
    epoch_t0 = time.perf_counter()
    data_t0 = time.perf_counter()
    n_iters = max(1, len(loader))
    micro_budget = int(total_micro) if total_micro is not None else n_iters * epochs
    lr = lr_fn(min(opt_iters_done, total_iters))
    for pg in opt.param_groups:
        pg["lr"] = lr
    for step_i, (inputs, labels) in enumerate(loader):
        data_time = time.perf_counter() - data_t0
        iter_t0 = time.perf_counter()
        iters_done += 1
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=bool(cfg.get("use_amp") and use_cuda)):
            out = model(inputs)
            result = iwpodnet_loss_v2(labels, out, **lkw)
            loss = result.total.mean() / accum
        scaler.scale(loss).backward()
        is_step = ((step_i + 1) % accum == 0) or ((step_i + 1) == n_iters)
        if is_step:
            lr = lr_fn(min(opt_iters_done, total_iters))
            for pg in opt.param_groups:
                pg["lr"] = lr
            if cfg.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            opt_iters_done += 1
            if ema is not None:
                ema_state["updates"] += 1
                update_ema(ema, model, cfg.get("ema_decay", 0.999), ema_state["updates"])
        iter_time = time.perf_counter() - iter_t0
        batch_loss = loss.item() * accum
        tot += batch_loss
        tot_cls += result.cls.item()
        tot_dice += result.dice.item()
        tot_loc += result.loc.item()
        meter.update(iter_time=iter_time, data_time=data_time, lr=lr,
                     total_loss=batch_loss, cls_loss=result.cls.item(),
                     dice_loss=result.dice.item(), loc_loss=result.loc.item())
        if (step_i + 1) % print_interval == 0 or (step_i + 1) == n_iters:
            eta_s = meter["iter_time"].global_avg * max(0, micro_budget - iters_done)
            eta_str = time.strftime("%H:%M", time.gmtime(int(eta_s)))
            mem_str = (f"gpu_mem={gpu_mem_usage():.0f}Mb host={host_mem_usage():.1f}Gb"
                       if use_cuda else f"host={host_mem_usage():.1f}Gb")
            logger.info(
                f"epoch: {epoch+1}/{epochs}, iter: {step_i+1}/{n_iters}, "
                f"{mem_str}, "
                f"iter_time={meter['iter_time'].avg:.3f}s data_time={meter['data_time'].avg:.3f}s, "
                f"total={meter['total_loss'].latest:.3f} "
                f"(cls={meter['cls_loss'].latest:.3f} dice={meter['dice_loss'].latest:.3f} "
                f"loc={meter['loc_loss'].latest:.3f}), "
                f"lr={lr:.3e}, size={cur_scale}x{cur_scale}, ETA={eta_str}")
            if tb is not None:
                tb.add_scalar("train/total_loss", meter["total_loss"].latest, iters_done)
                tb.add_scalar("train/cls_loss", meter["cls_loss"].latest, iters_done)
                tb.add_scalar("train/dice_loss", meter["dice_loss"].latest, iters_done)
                tb.add_scalar("train/loc_loss", meter["loc_loss"].latest, iters_done)
                tb.add_scalar("train/lr", lr, opt_iters_done)
            meter.clear_meters()
        data_t0 = time.perf_counter()
    means = dict(total=tot / n_iters, cls=tot_cls / n_iters,
                 dice=tot_dice / n_iters, loc=tot_loc / n_iters,
                 epoch_time=time.perf_counter() - epoch_t0)
    return means, iters_done, opt_iters_done


@torch.no_grad()
def validate_loss(eval_model, loader, cfg, device):
    """Val loss components (same weights as geometric eval input)."""
    lkw = loss_kwargs(cfg.get("loss"))
    use_cuda = device.type == "cuda" if isinstance(device, torch.device) else torch.cuda.is_available()
    vtot = vcls = vdice = vloc = 0.0
    vn = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=bool(cfg.get("use_amp") and use_cuda)):
            r = iwpodnet_loss_v2(labels, eval_model(inputs), **lkw)
        vtot += r.total.mean().item()
        vcls += r.cls.item()
        vdice += r.dice.item()
        vloc += r.loc.item()
        vn += 1
    n = max(1, vn)
    return dict(total=vtot / n, cls=vcls / n, dice=vdice / n, loc=vloc / n)


def is_better(candidate, best):
    """Single-best policy: mAP@50-95, tie-break IoU@0.7 → IoU@0.5 → recall@0.5 → val loss.

    `recall` is recall@IoU>0.5 at the operating threshold (not conf-only det rate).
    mAP integrates over detection confidence, so it is threshold-free and more
    stable across epochs than recall at a fixed operating threshold.
    """
    for k in ("map", "iou70", "iou50", "recall"):
        if candidate[k] != best[k]:
            return candidate[k] > best[k]
    return candidate["vloss"] < best["vloss"]


def tb_epoch_tags():
    """Single source of truth for per-epoch TensorBoard tags.

    Geometric tags are the COCO trio (AP, AP50, AP75 from ap_curve) — the
    10-point recall@IoU curve stays in the console EVAL block, not in TB;
    the EvalResult fields of the same names still drive best selection.
    """
    return [
        "loss/train", "loss/train_cls", "loss/train_dice", "loss/train_loc",
        "loss/val", "loss/val_cls", "loss/val_dice", "loss/val_loc",
        "train/total_loss", "train/cls_loss", "train/dice_loss", "train/loc_loss", "train/lr",
        "val/recall", "val/det_rate", "val/miou", "val/rmse_detected",
        "val/map50", "val/map75", "val/map50-95",
        "val/best_thr", "val/best_f1",
    ]


def run(args):
    cfg = load_cfg(args.config or default_config_path())
    yaml_bs = cfg["batch_size"]
    epochs, bs, lr, seed = apply_cli_overrides(cfg, args)
    validate_config(cfg)

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_ckpt, resume_dir = resolve_resume(args.resume)
    if resume_dir and args.model_dir != "out/train":
        raise RuntimeError(
            "--resume <run-dir> continues in place inside that dir, so "
            "--model-dir is silently ignored in that mode. Drop --model-dir "
            "to continue in place, or pass --resume <checkpoint-file> to "
            "fork a fresh run under --model-dir. Refusing to avoid writing "
            "checkpoints into an unintended run directory.")
    if resume_dir:
        run_dir, run_name = resume_dir, os.path.basename(os.path.normpath(resume_dir))
        if args.name != "iwpodv2":
            logger.warning(f"--name '{args.name}' ignored with --resume <run-dir>; continuing in place as '{run_name}'")
    else:
        os.makedirs(args.model_dir, exist_ok=True)
        run_dir, run_name = runs.resolve_run_dir(args.model_dir, args.name)

    setup_logging(run_dir)
    runs.write_train_command(run_dir)
    if resume_dir:
        # Keep original config.yaml for provenance; record resume deltas separately.
        import yaml as _yaml
        try:
            with open(os.path.join(run_dir, "config_resume.yaml"), "w") as f:
                _yaml.safe_dump(cfg, f, sort_keys=False)
        except OSError:
            pass
    else:
        runs.write_config_snapshot(run_dir, cfg)
    logger.info(f"Run dir: {run_dir}")

    from iwpod.model import IWPODNet

    model = IWPODNet(raw_logits=cfg["model"].get("raw_logits", True)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 5e-4))

    train_entries, val_entries = resolve_data(args)
    dim = cfg["dim"]
    loss_cfg = cfg.get("loss") or {}
    scales = [s for s in cfg.get("multi_scale", [dim]) if s % 32 == 0] or [dim]
    train_ds = ALPRDataset(dim=dim, entries=train_entries, cache=args.cache)
    use_cuda = device.type == "cuda"
    nw = cfg.get("num_workers", 8)

    if getattr(args, "dry_run", False):
        logger.info(f"Dry-run OK: {len(train_entries)} train / {len(val_entries)} val, "
                    f"dim={dim} scales={scales} bs={bs} epochs={epochs}")
        return

    if bs == -1:
        from iwpod.autobatch import auto_batch_size

        train_ds.scale = max(scales)
        probe_in, probe_tg = train_ds[0]
        train_ds.scale = None
        def _model_fn():
            from iwpod.model import IWPODNet as _Net
            return _Net(raw_logits=cfg["model"].get("raw_logits", True))

        def _loss_fn(out, tg):
            return iwpodnet_loss_v2(tg, out, **loss_kwargs(loss_cfg)).total.mean()

        probed = auto_batch_size(
            _model_fn, _loss_fn, (probe_in, probe_tg), device,
            target_fraction=args.auto_batch_target,
            amp_enabled=bool(cfg.get("use_amp") and use_cuda))
        if probed:
            bs = probed
            cfg["batch_size"] = bs
            logger.info(f"Using probed batch size: {bs}")
        else:
            bs = yaml_bs
            cfg["batch_size"] = bs
            logger.info(f"Probing unavailable; keeping configured batch size: {bs}")
        runs.write_config_snapshot(run_dir, cfg)  # record resolved batch size

    accum = max(1, int(args.grad_accum if args.grad_accum is not None else cfg.get("grad_accum", 1)))
    cfg["_grad_accum"] = accum
    patience = max(0, int(cfg.get("patience", 0)))
    if accum > 1:
        logger.info(f"Gradient accumulation: {accum} steps "
                    f"(effective batch = {bs} x {accum} = {bs * accum})")

    log_model_summary(model, dim, device)
    import pprint
    logger.info("Config:\n" + pprint.pformat(cfg, width=120))

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=use_cuda,
                              worker_init_fn=_seed_worker, generator=train_gen)
    val_loader = None
    val_eval_entries = []
    if val_entries:
        val_ds = ALPRDataset(dim=dim, entries=val_entries, cache=args.cache)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=use_cuda,
                                worker_init_fn=_seed_worker)
        val_eval_entries = entries_from_loader_entries(val_entries)
    else:
        logger.warning("No val set: `_best.pth` tracks TRAIN loss. "
                       "Prepare an explicit val/ split for anything you ship.")

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("use_amp") and use_cuda))
    ema = None
    ema_state = {"updates": 0}
    if cfg.get("use_ema"):
        from copy import deepcopy
        ema = deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

    # LR schedule is per-optimizer-step; total opt-steps account for grad-accum.
    opt_steps_per_epoch = max(1, (len(train_loader) + accum - 1) // accum)
    total_iters = epochs * opt_steps_per_epoch
    warmup_iters = cfg.get("warmup_epochs", 5) * opt_steps_per_epoch
    lr_fn = build_lr_fn(lr, total_iters, warmup_iters, cfg.get("scheduler", "cosine"),
                        cfg.get("min_lr_ratio", 0.0))

    start, iters_done, opt_iters_done, no_improve = 0, 0, 0, 0
    best = dict(map=-1.0, iou70=-1.0, iou50=-1.0, recall=-1.0, vloss=float("inf"), epoch=-1)
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ck = ckptlib.load_ckpt(resume_ckpt, map_location=device)
        _sd, _arch = ckptlib.weights_and_arch(ck)
        model.load_state_dict(_sd)
        opt.load_state_dict(ck.get("optimizer_state_dict", opt.state_dict()))
        # CLI --weight-decay must survive resume (load_state_dict overwrites it).
        # YAML-only weight_decay is left as stored in the optimizer.
        if args.weight_decay is not None:
            for pg in opt.param_groups:
                pg["weight_decay"] = float(args.weight_decay)
        start = ck.get("epoch", -1) + 1
        iters_done = ck.get("iters_done", start * max(1, len(train_loader)))
        opt_iters_done = ck.get("opt_iters_done", start * opt_steps_per_epoch)
        _stored_total = ck.get("total_iters")
        min_lr = lr * cfg.get("min_lr_ratio", 0.0)
        if args.epochs is None and _stored_total:
            total_iters = int(_stored_total)
            warmup_iters = int(ck.get("warmup_iters", warmup_iters))
            lr_fn = build_lr_fn(lr, total_iters, warmup_iters, cfg.get("scheduler", "cosine"),
                                cfg.get("min_lr_ratio", 0.0))
        elif args.epochs is not None and _stored_total and int(_stored_total) != total_iters:
            logger.warning(
                f"Resume with --epochs {epochs}: extending LR horizon "
                f"{_stored_total} -> {total_iters} by annealing from the current LR "
                f"(no cosine jump).")
            current_lr = lr_fn(min(opt_iters_done, int(_stored_total)))
            lr_fn = build_continue_lr_fn(current_lr, opt_iters_done, total_iters, min_lr)
        if ema is not None and ck.get("ema_state_dict"):
            ema.load_state_dict(ck["ema_state_dict"])
        ema_state["updates"] = ck.get("ema_updates", 0)
        if ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        for k in ("map", "iou70", "iou50", "recall", "vloss"):
            if ck.get(f"best_{k}") is not None:
                best[k] = ck[f"best_{k}"]
        best["epoch"] = ck.get("best_epoch", -1)
        no_improve = ck.get("no_improve", 0)
        rng = ck.get("rng_state") or {}
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(_as_byte_tensor(rng["torch"]))
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in rng["torch_cuda"]])
            except Exception as e:
                logger.warning(f"Could not restore CUDA RNG state: {e}")
        if rng.get("numpy") is not None:
            import numpy as _np2
            _np2.random.set_state(_restore_numpy_state(rng["numpy"]))
        gen_state = rng.get("train_gen", ck.get("train_gen_state"))
        if gen_state is not None:
            try:
                train_gen.set_state(_as_byte_tensor(gen_state))
            except Exception as e:
                logger.warning(f"Could not restore DataLoader generator: {e}")
        logger.info(f"Resumed {resume_ckpt} @ epoch {start} "
                    f"(best mAP@50-95={best['map']:.3f} @ epoch {best['epoch']})")
    if epochs <= start:
        raise RuntimeError(f"--epochs ({epochs}) <= resume start ({start}): nothing to train")

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run_dir, "tensorboard"))

    eval_size = cfg.get("eval_size") or dim
    eval_threshold = cfg.get("eval_threshold", 0.3)
    save_history = bool(cfg.get("save_history_ckpt", False))
    total_micro = epochs * max(1, len(train_loader))

    logger.info(
        f"Training start — {epochs - start} epochs, "
        f"{len(train_loader)} iters/epoch, "
        f"batch={bs}, workers={nw}, "
        f"amp={bool(cfg.get('use_amp') and use_cuda)}, "
        f"ema={cfg.get('use_ema', False)}, "
        f"scheduler={cfg.get('scheduler', 'none')} (per-optimizer-step), "
        f"best-metric=val mAP@50-95 (single-GPU)"
    )

    for epoch in range(start, epochs):
        # Dataset-level multi-scale: one scale per epoch so batches stack and
        # label encoding stays consistent (no post-hoc grid interpolation).
        # Per-iter resize like YOLOX isn't feasible: augmentation bakes the
        # resolution inside DataLoader workers, which don't see main-process
        # scale updates mid-epoch.
        cur_scale = random.choice(scales) if len(scales) > 1 else dim
        train_ds.scale = cur_scale if len(scales) > 1 else None

        means, iters_done, opt_iters_done = train_one_epoch(
            model, ema, ema_state, train_loader, opt, scaler, lr_fn, cfg,
            device, use_cuda, epoch, epochs, cur_scale, iters_done, total_iters, tb,
            opt_iters_done, total_micro=total_micro)
        model.eval()
        eval_model = ema if ema is not None else model

        if val_loader is not None:
            v = validate_loss(eval_model, val_loader, cfg, device)
            g = evaluate_quads(eval_model, val_eval_entries, size_px=eval_size,
                               threshold=eval_threshold, device=device)
            logger.info(
                f"Epoch {epoch+1}/{epochs}  "
                f"train={means['total']:.4f} (cls={means['cls']:.3f} "
                f"dice={means['dice']:.3f} loc={means['loc']:.3f})  "
                f"val={v['total']:.4f} (cls={v['cls']:.3f} "
                f"dice={v['dice']:.3f} loc={v['loc']:.3f})  "
                f"lr={opt.param_groups[0]['lr']:.3e}  "
                f"epoch_time={means['epoch_time']:.1f}s  "
                f"best_map={best['map']:.3f}@e{best['epoch']}")
            tb.add_scalar("loss/train", means["total"], epoch + 1)
            tb.add_scalar("loss/train_cls", means["cls"], epoch + 1)
            tb.add_scalar("loss/train_dice", means["dice"], epoch + 1)
            tb.add_scalar("loss/train_loc", means["loc"], epoch + 1)
            tb.add_scalar("loss/val", v["total"], epoch + 1)
            tb.add_scalar("loss/val_cls", v["cls"], epoch + 1)
            tb.add_scalar("loss/val_dice", v["dice"], epoch + 1)
            tb.add_scalar("loss/val_loc", v["loc"], epoch + 1)
            tb.add_scalar("val/recall", g["recall"], epoch + 1)
            tb.add_scalar("val/det_rate", g.get("det_rate", g["recall"]), epoch + 1)
            tb.add_scalar("val/miou", g["mean_iou"], epoch + 1)
            tb.add_scalar("val/map50", g["map50"], epoch + 1)
            tb.add_scalar("val/map75", g["ap_curve"][0.75], epoch + 1)
            tb.add_scalar("val/map50-95", g["map"], epoch + 1)
            tb.add_scalar("val/rmse_detected", g.get("rmse_detected", g["rmse"]), epoch + 1)
            do_sweep = (epoch + 1) % cfg.get("eval_sweep_every", 5) == 0
            logger.info("\n" + format_block(g, epoch + 1, epochs, eval_size,
                                           eval_threshold, with_sweep=do_sweep))
            if do_sweep:
                tb.add_scalar("val/best_thr", g["sweep"]["best_thr"], epoch + 1)
                tb.add_scalar("val/best_f1", g["sweep"]["best_f1"], epoch + 1)
            candidate = dict(map=g["map"], iou70=g["iou70"], iou50=g["iou50"],
                             recall=g["recall"], vloss=v["total"])
        else:
            v = dict(total=means["total"], cls=0.0, dice=0.0, loc=0.0)
            logger.info(
                f"Epoch {epoch+1}/{epochs}  train={means['total']:.4f}  "
                f"val=n/a (train)  lr={opt.param_groups[0]['lr']:.3e}  "
                f"epoch_time={means['epoch_time']:.1f}s  best={best['vloss']:.4f}")
            tb.add_scalar("loss/train", means["total"], epoch + 1)
            # no-val fallback: single best tracks train loss
            candidate = dict(map=-1.0, iou70=-1.0, iou50=-1.0, recall=-1.0,
                             vloss=means["total"])

        def _extra(iters_done, opt_iters_done, no_improve):
            _rng = {
                "python": random.getstate(),
                "torch": torch.get_rng_state().cpu(),
                "numpy": _snapshot_numpy_state(),
                "train_gen": train_gen.get_state().cpu(),
            }
            if torch.cuda.is_available():
                try:
                    _rng["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
                except Exception:
                    pass
            return {
                "optimizer_state_dict": opt.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "ema_updates": ema_state["updates"],
                "scaler_state_dict": scaler.state_dict(),
                "iters_done": iters_done,
                "opt_iters_done": opt_iters_done,
                "total_iters": total_iters,
                "warmup_iters": warmup_iters,
                "rng_state": _rng,
                "train_gen_state": _rng.get("train_gen"),
                "best_map": best["map"], "best_iou70": best["iou70"], "best_iou50": best["iou50"],
                "best_recall": best["recall"], "best_vloss": best["vloss"],
                "best_epoch": best["epoch"], "no_improve": no_improve,
            }

        improved = is_better(candidate, best)
        if improved:
            best.update({**candidate, "epoch": epoch + 1})
            no_improve = 0
        else:
            no_improve += 1

        extra = _extra(iters_done, opt_iters_done, no_improve)
        if save_history and (epoch + 1) % cfg.get("save_every", 10) == 0:
            ckptlib.save_ckpt(os.path.join(run_dir, f"{run_name}_epoch{epoch+1}.pth"),
                              model, None, epoch, None, extra=extra, meta={"dim": dim})
        ckptlib.save_ckpt(os.path.join(run_dir, f"{run_name}_last.pth"),
                          model, None, epoch, None, extra=extra, meta={"dim": dim})
        if improved:
            ckptlib.save_ckpt(os.path.join(run_dir, f"{run_name}_best.pth"),
                              ema if ema is not None else model, None, epoch, None,
                              extra=extra, meta={"dim": dim})
            if val_loader is not None:
                logger.info(f"New best: mAP@50-95={best['map']:.3f} "
                            f"(IoU@0.7={best['iou70']:.3f} IoU@0.5={best['iou50']:.3f} "
                            f"recall@0.5={best['recall']:.3f})")
            else:
                logger.info(f"New best: train loss={best['vloss']:.4f}")
        elif patience > 0 and no_improve >= patience:
            if val_loader is not None:
                logger.info(f"Early stopping: no improvement for {patience} epochs "
                            f"(best mAP@50-95={best['map']:.3f}@e{best['epoch']})")
            else:
                logger.info(f"Early stopping: no improvement for {patience} epochs "
                            f"(best train loss={best['vloss']:.4f}@e{best['epoch']})")
            tb.close()
            break
    tb.close()
    if val_loader is not None:
        logger.info(f"Done. Best mAP@50-95: {best['map']:.3f} @ epoch {best['epoch']} -> {run_dir}")
    else:
        logger.info(f"Done. Best train loss: {best['vloss']:.4f} @ epoch {best['epoch']} -> {run_dir}")
