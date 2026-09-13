# Training

`legacy/train.py` is frozen for reproduction. All real training uses
`iwpod train`, which is CLI-first: every material knob is a flag, the YAML is
the fallback layer. Precedence: **CLI > YAML > built-in defaults.**

## Quick start (copy-pasteable, all relevant keys)

```bash
iwpod train --data datasets/LP --epochs 200 --batch-size 32 --lr 0.001 \
  --size 384 --seed 42 --name exp1
# equivalent: python -m iwpod train --data datasets/LP ...
```

More examples (`iwpod train --help` prints these too):

```bash
# single scale, no AMP/EMA (e.g. small GPU / debugging)
iwpod train --data datasets/LP --epochs 50 --batch-size 16 --no-multiscale \
  --no-amp --no-ema --name debug
# resume a run in place (no new dir) and extend the budget
iwpod train --resume out/train/exp1 --epochs 300 --data datasets/LP
```

## Run directories (never overwritten)

Runs live under `--model-dir` (default `out/train`): `out/train/<name>`,
auto-incremented to `<name>_2`, `<name>_3` on collision. `--resume` takes a
checkpoint path or a run dir and continues **in place** (no increment).
Each run dir contains:

| File | Content |
|---|---|
| `<name>_epoch<N>.pth` | full resume state (model, optimizer, scaler, EMA, RNG, best), every `save_every` epochs (disable with `save_history_ckpt: false`) |
| `<name>_best.pth` | **single best** by val mAP@50-95 (tie-break IoU@0.7 → IoU@0.5 → recall@0.5 → val loss), EMA weights — export/eval this |
| `<name>_last.pth` | latest epoch (for `--patience` stops and crashes) |
| `train_command.txt` | exact CLI invocation for reproduction |
| `config.yaml` | resolved config snapshot (CLI overrides applied) |
| `train.log` | full log (mirrors stdout) |
| `tensorboard/` | TensorBoard events, always on: per-iter `train/*` (micro-batch x-axis) + `train/lr` (optimizer-step x-axis), per-epoch `loss/*`, `val/*` incl. COCO trio `val/map50-95` (mAP@50-95) + `val/map50` + `val/map75`, `val/recall` (recall@IoU>0.5), `val/det_rate`, `val/rmse_detected`, plus `val/best_thr` + `val/best_f1` on sweep epochs (the full recall@IoU curve stays in the console EVAL block) |

`weights/` is reserved for shipped/pretrained checkpoints, not runs.

## CLI flags (override the YAML)

Data: `--data` (root with `train/`+`val/`, which may contain scene
subfolders; defaults to `./datasets` when
populated, else legacy `--train-dir` (trains with no val set). Training: `--epochs`, `--batch-size`
(`-1` = probe VRAM for the max fitting batch, yolox-style; CPU keeps the
configured size), `--grad-accum` (effective batch = batch-size × steps),
`--patience` (early-stop epochs without val gain, 0 = off),
`--lr`/`--learning-rate`,
`--weight-decay`, `--seed`, `--size` (square, sets `dim`), `--no-multiscale`,
`--scheduler`, `--amp` / `--no-amp`, `--no-ema`, `--ema-decay`, `--warmup-epochs`,
`--save-every` (implies history), `--save-history`, `--dry-run`, `--num-workers`.
Output: `--model-dir`, `--name`, `--config`
(default: bundled `iwpod/configs/base.yaml`, works from any CWD),
`--resume` (ckpt path or run dir).

## Configuration (bundled `iwpod/configs/base.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | Python + torch + numpy seed (DataLoader workers seeded per-worker) |
| `dim` | 384 | Base train resolution (square, must be %32) |
| `multi_scale` | [320, 352, 384, 416, 448, 480] | One `%32` scale sampled per epoch at dataset level (label encoding stays consistent); dense 32px steps spanning the 416 deployment size for scale robustness; `[dim]`-only or `--no-multiscale` disables it |
| `stride` / `side` | 16 / 7.75 | Label encoding constants — **do not change** (must match `iwpod/constants.py` + DeepStream parser) |
| `epochs` | 200 | Full training budget |
| `batch_size` | 32 | Per-step batch (tune to GPU memory) |
| `num_workers` | 8 | DataLoader workers |
| `lr` | 0.001 | AdamW base LR |
| `weight_decay` | 0.0005 | AdamW decay |
| `warmup_epochs` | 5 | Linear warmup before cosine (per-iteration stepping) |
| `scheduler` | cosine | `cosine` \| `none` (stepped per-iteration, YOLOX-style) |
| `min_lr_ratio` | 0.05 | Cosine tail floor (`lr × ratio`); late epochs keep updating instead of decaying to zero |
| `use_amp` | true | Autocast FP16 on CUDA (no-op on CPU) |
| `use_ema` | true | EMA over params (BN stats copied) with decay ramp; the EMA copy is evaluated and saved to `_best.pth` |
| `ema_decay` | 0.999 | EMA decay |
| `grad_clip` | 1.0 | Gradient norm clip (null disables) |
| `loss.w_cls` | 1.0 | Focal-loss weight (objectness) |
| `loss.w_dice` | 0.5 | Dice-loss weight (objectness overlap) |
| `loss.w_loc` | 1.0 | Wing-loss weight (corner regression) |
| `loss.focal_alpha` / `focal_gamma` | 0.25 / 2.0 | Focal imbalance tuning (≈1:150 pos:neg cells) |
| `loss.ohem_neg_ratio` | 3.0 | Hardest negatives kept per positive |
| `model.raw_logits` | true | Train the logits head (required for export; sigmoid runs in the DS parser) |
| `save_every` | 10 | Epoch checkpoint cadence |
| `save_history_ckpt` | false | Keep `<name>_epoch<N>.pth` history; false = `_last` + `_best` only |
| `print_interval` | 10 | Iters between per-iter log lines (losses smoothed over the window) |
| `eval_threshold` | 0.3 | Decode confidence for the in-train geometric val |
| `eval_size` | null | Val decode resolution (null = `dim`); fixed so IoU is comparable across epochs |
| `eval_sweep_every` | 5 | Epochs between threshold/F1 sweep reports (eval block logs every epoch) |
| `patience` | 0 | Early-stop epochs without improvement (0 = off); `--patience` overrides |

## Best-checkpoint policy (single `_best.pth`)

`_best.pth` tracks the **held-out mAP@50-95** (tie-break: IoU@0.7 → IoU@0.5 →
recall@IoU>0.5 → val loss),
evaluated every epoch on the EMA weights at fixed `eval_size`. mAP integrates
over detection confidence, so unlike recall at a fixed threshold it stays a
stable selector throughout training. Loss alone is not
the selector: in our 5-epoch gate run, epoch 5 had the lowest train loss but
epoch 3 had the best IoU@0.7 (0.521 vs 0.202) — loss-best would have shipped the
worse localizer. With no `val/` set, best falls back to train loss with a warning.
Early stopping (`--patience` / `patience` in YAML) watches the same metric.

## What a healthy run looks like

```
epoch: 1/200, iter: 10/232, gpu_mem=812Mb host=1.8Gb, iter_time=0.015s data_time=0.003s,
  total=1.305 (cls=0.048 dice=0.970 loc=0.772), lr=2.800e-04, size=384x384, ETA=00:03
Epoch 1/200  train=1.3051 (cls=0.048 dice=0.970 loc=0.772)
  val=1.0299 (cls=0.043 dice=0.984 loc=0.495)
  lr=2.800e-04  epoch_time=13.1s  best_iou70=-1.000@e-1
```
followed by the per-epoch `EVAL` block (geometry lives only there — no duplication):

Per-iter lines log smoothed timings + latest windowed losses with a global-avg ETA
(YOLOX-style `MeterBuffer`); TensorBoard gets per-iter `train/*` + `lr` (lr on
optimizer-step x-axis) and per-epoch `loss/*`, `val/*` (COCO trio + recall@0.5,
det_rate, rmse_detected). The 0.5–0.95 IoU curve stays in the console EVAL
block, not as `val/iou/*` scalars.
Every epoch also logs a multi-line eval block with the plate-area split
(small/medium/large GT terciles — distance is the main variance axis) and, every
`eval_sweep_every` epochs, the operating-threshold F1 sweep with its best-threshold
recommendation:

```
==================== EVAL epoch 2/2 (size=256 thr=0.30) ====================
recall=0.943  mIoU=0.475  RMSE=0.0724  dets=1743  fp_neg=0  infer=1.5ms/img
IoU: @0.50=0.640  @0.60=0.439  @0.70=0.169  @0.80=0.016  @0.90=0.000  @0.95=0.000
area: small(n=616): recall=0.878 miou=0.305 | medium(n=616): recall=0.966 miou=0.509 | ...
==============================================================================
```

Train and val loss should fall together. Train falling while val stalls for
>20 epochs = overfitting: add data (see DATASET.md), strengthen augmentation,
or stop early and take `_best.pth`.

## Augmentation (what the model actually sees)

Per sample (`iwpod/sampler.py:augment_sample`): upstream-detector box jitter
(crop-window translate ±15%, quads shifted coherently), plate rectified to a
random width/aspect (label-dependent ranges), random 3D perspective rotation
capped to production views (pitch ±40°, yaw ±45°, roll ±15°, total ≤90° —
cameras are roughly level and never see edge-on plates),
`warpPerspective` + background fill, horizontal flip (p=0.5, mirror-safe),
5% invert, HSV jitter (clipped to valid ranges), then a pixel-only sensor
stage (albumentations, no label plumbing needed — the output map is built
afterwards): grayscale for IR/night cameras (p=0.15, chroma-free detection),
motion/gaussian blur, brightness/contrast, gamma, sensor noise /
JPEG compression, plus guarded cutout (small holes, never swallowing the
plate). Multi-scale is dataset-level: one `%32` scale from `multi_scale` per
epoch, so batches stack and label encoding stays consistent
(no post-hoc grid interpolation).

## Resource guidance: `--cache` (yolox-style)

Default is lazy reads (images stay on disk — safe on any RAM size). On a
training server with headroom, preload for speed:

```bash
iwpod train --data datasets/LP --cache ...    # bare --cache means ram; needs ~1.2x raw pixels free
```

Measured reference: 7,395 vehicle crops = **5.4 GB raw pixels** (~0.7 MB mean).
`--cache ram` prints its estimate first and refuses to start if it exceeds
available RAM (fail fast beats an OOM-kill at epoch 3).

- x86 + discrete GPU: start with the defaults; if you OOM, halve
  `batch_size` (effective batch matters less here than epoch count — the loss
  is per-cell averaged).
- Jetson (Xavier/Orin): train on x86, deploy the engine. If training on-device,
  halve `batch_size`, set `multi_scale: [256, 320]`, `num_workers: 4`, and
  expect ~5–10× slower epochs.
- CPU-only: works (smoke-tested) but slow; use `--epochs 2 --batch-size 4`
  for plumbing checks only.

## Legacy `legacy/train.py`

Frozen for reproduction (imports updated to the `iwpod` package). Prefer
`iwpod train` for anything you ship.
