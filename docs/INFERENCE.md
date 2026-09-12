# Inference

## Which entry point?

| Tool | Use when |
|---|---|
| `python legacy/detect.py -i <img> -v car\|bike\|fullimage` | Legacy interactive demo (OpenCV windows; needs a display) |
| `iwpod infer --weights W --input DIR [--output OUT]` | Headless folder inference: quads + rectified plates |
| `iwpod eval --weights W --data DIR` | Scored eval on annotated data (see EVALUATION.md) |
| DeepStream SGIE | Production (see `deepstream/DOWNSTREAM_GUIDE.md`) |

## Folder inference

```bash
iwpod infer --weights out/train/exp1/exp1_best.pth \
  --input vehicle_crops/ \
  --size 416 --threshold 0.3 --plate-size 256 96
# outputs land in out/detect/exp1_best/ (override with --output OUT)
```

Inputs are letterboxed to square like DeepStream (`maintain-aspect-ratio=1`
without `symmetric-padding`: **top-left** placement, black pad) and quads are
mapped back to original pixels before warping.

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | (required) | `.pth` checkpoint. Legacy sigmoid checkpoints load too — the linear weights are identical, decode applies sigmoid once |
| `--input` / `--output` | input required; output default `out/detect/<weights-stem>/` | Input dir (case-insensitive `.jpg/.jpeg/.png`; `--recursive` walks subdirs) / output dir (created) |
| `--size` | ckpt `dim`, else 416 | Letterbox longest side to `size×size` before forward |
| `--threshold` | 0.3 | Objectness threshold (post-sigmoid). Lower → more recall, more false plates; mirrors compile-time `kMinConfidence` in the DS plugin |
| `--plate-size` | `256 96` | Rectified plate as `W H`. **Keep 256 96** unless you retrain OCR (see below) |
| `--recursive` | off | Walk `--input` recursively |

Outputs per input `<base>.<ext>`:
- `<base>_quad.txt` — `x0 x1 x2 x3 y0 y1 y2 y3 conf` in **original crop pixels**
  (`none` if no detection).
- `<base>_plate.png` — perspective-rectified plate (`plate-w × plate-h`).

## Input guidance

Feed **vehicle crops**, not full frames — the model was trained on crops and
the DeepStream stage runs as an SGIE on vehicle objects. Full-frame images
work mechanically but recall drops (plates fall below the trained scale range).

## Threshold tuning

1. Run `iwpod eval` over a threshold sweep (0.15–0.6) on val crops.
2. Pick the knee of recall vs false-positives-per-crop (see EVALUATION.md).
3. Mirror the chosen value in `wpod_threshold` (runtime) and rebuild the DS
   plugin if you change compile-time `kMinConfidence` so offline and online agree.

## Changing the rectified plate size

`--plate-size W H` only changes the warp target. The OCR model was trained on
256×96 — changing the size without retraining OCR **will** degrade string
accuracy. Treat plate size as an OCR contract, not an inference knob.
