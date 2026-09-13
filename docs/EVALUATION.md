# Evaluation

```bash
iwpod eval --weights out/train/exp1/exp1_best.pth --data datasets/LP/val \
  --threshold 0.3 --size 416
```

(`--data` uses the
same image+sibling-`.txt` format as training, including scene subfolders
under `val/` — see DATASET.md. Inputs are
letterboxed like DeepStream (top-left, black pad; `maintain-aspect-ratio=1`
without `symmetric-padding`) and compared in original normalized coords.
Results append to `val_log.txt` under `out/eval/<weights-stem>/` (override
with `--log-dir`); the same geometry runs
inside training every epoch — see TRAINING.md.)

## Metrics reported

| Metric | Definition | What it tells you |
|---|---|---|
| `recall` | fraction of annotated plates with conf ≥ threshold **and** IoU > 0.5 | find rate at the operating point |
| `det_rate` | fraction of annotated plates with any detection above threshold (no IoU) | end-to-end detection rate |
| `mean IoU` | polygon IoU between predicted and GT quads (rasterized at 256px), averaged with misses as 0 | localization quality |
| `IoU@0.5` / `IoU@0.7` | fraction of plates above each IoU | 0.5 = loose (paper AOLP protocol), 0.7 = strict (CCPD protocol) |
| `mAP@50` / `mAP@50-95` | area under the precision-recall curve at IoU>0.5 / mean over 0.5–0.95 (single class, so AP = mAP) | threshold-free ranking quality; **mAP@50-95 selects `_best.pth`** |
| `IoU curve` | same fraction at 0.5–0.95 (console EVAL block; TB carries the COCO trio `val/map50-95` + `val/map50` + `val/map75`) | where accuracy falls off with strictness |
| `RMSE_det` | RMS corner error in normalized [0,1] coords **over detections only**, with `rmse_n` | geometric precision for the warp (not overall accuracy) |
| `infer ms/img` | forward + decode time per image | latency signal next to accuracy |

## How to read the numbers

- **Train-set recall ≈ 1.0 is a correctness proof, not a result.** Our reference:
  the legacy 10k-epoch weights score recall 0.969 / mean IoU 0.769 /
  IoU@0.7 0.875 on `train_dir`. That proves model + decode + harness are wired
  correctly. Real accuracy claims require a held-out set (fixed `val/`,
  never trained on).
- **Misses vs sloppy quads:** low recall + decent IoU@0.7 on detections =
  threshold too high (or plates out of trained scale range). Decent recall +
  low mean IoU = localization weak (check `w_loc`, augmentation ranges, and
  that crop labels are in the right aspect bucket — DATASET.md).
- **Corner RMSE ≈ 0.016** (reference) warps cleanly; above ~0.03 expect visibly
  skewed plates and OCR degradation.

## Threshold sweep procedure

```bash
iwpod eval --weights out/train/exp1/exp1_best.pth --data datasets/LP/val \
  --threshold 0.3 --size 416 --sweep 0.15 0.25 0.4 0.5 0.6
```

Every eval also prints an `EVAL` block with the plate-area split
(small/medium/large) and a threshold/F1 sweep derived from a single decode
pass — `Best threshold: 0.35 (F1=…)` is the YOLOX-confidence-analysis analog
for picking the operating point:

Pick the knee where recall saturates before false-positives-per-crop climb
(count `*_quad.txt` files with detections on known-negative crops if you keep
any). Mirror the chosen value in runtime `wpod_threshold`. DeepStream decode
threshold is compile-time `kMinConfidence` — rebuild the plugin to change it.

## Gating retraining and quantization

- A new `_best.pth` ships only if it beats the incumbent on **held-out**
  `val/` mAP@50-95 (not train loss, not single-threshold recall).
- An FP16/INT8 engine ships only if eval on the **same ONNX-exported model**
  shows <1pp (FP16) / <0.5pp (INT8) recall drop vs FP32 at the operating
  threshold.
