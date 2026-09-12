# Dataset Format

This page specifies the exact on-disk format the training and eval tools expect.
If your annotations don't match this, convert them first — the loader does no
auto-detection.

## Layout

A dataset directory contains image + sibling annotation pairs sharing a basename:

```
train_dir/
  000001.jpg
  000001.txt
  000002.jpg
  000002.txt
  ...
```

Supported image extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive; see
`iwpod/utils.py:image_files_from_folder`). Every image should have a `.txt`
sibling. Images *without* a sibling are kept as **background** samples
(trained as all-negative output maps; eval counts them as known negatives).

## Annotation format (quad, normalized)

Each `.txt` file holds **one line per plate**. A line is comma-separated:

```
<n>,<x0>,<x1>,<x2>,<x3>,<y0>,<y1>,<y2>,<y3>,<label>,
```

- `<n>` — number of corners (always `4` for plates).
- `<x0..x3>`, `<y0..y3>` — corner coordinates **normalized to [0, 1]** by image
  width/height. Order: top-left, top-right, bottom-right, bottom-left
  (clockwise from the top-left corner).
- `<label>` — vehicle type string, `car` (or empty) vs `bike`. It selects the
  augmentation aspect-ratio range (see below). A trailing comma is typical.

Example (plate quad in a 1160×720 image):

```
4,0.213889,0.504167,0.536111,0.245833,0.330172,0.346552,0.407759,0.391379,,
```

The filename-embedded CCPD-style metadata some files carry (e.g.
`...-90_86-...jpg`) is **ignored** — only the image pixels and the `.txt`
content are used.

Parsed by `iwpod/label.py:readShapes` (the `Shape` class: `pts` is a `(2, n)`
array, `text` is the label string).

## Vehicle-crop convention (production training data)

The deployed model runs on **vehicle crops** (output of your YOLO vehicle
detector), not full frames. Train on what you deploy:

1. Crop each annotated image with your vehicle box **plus ~8% padding**.
2. Recompute the quad coordinates relative to the crop (still [0, 1]).
3. Save as `<crop_id>.jpg` + `<crop_id>.txt` in the format above.

Aspect-ratio augmentation is driven by the label string
(`iwpod/sampler.py:augment_sample`):

| label | target plate aspect `w/h` sampled uniformly from |
|---|---|
| `bike` | 1.25 – 2.5 |
| anything else (`car`/empty) | 2.5 – 4.5 |

Label crops correctly — a bike plate trained with the car range (or vice versa)
warps to the wrong canonical proportions.

## Background images (`bgimages/`)

During augmentation the homography warp leaves unfilled borders; these are
filled with random crops from `bgimages/*.jpg|*.jpeg|*.png` (resolved relative
to the repo root, cross-platform). Requirements:

- At least one image; more variety = fewer false positives.
- Each image's **short side must be ≥ your train `dim`** (e.g. ≥384 px for the
  default config). Smaller images are auto-upscaled, but tiny backgrounds
  produce blurry fill — prefer large road/street scenes *without* plates.

## Resolution and the `%32` rule

Training resolution comes from the bundled config (`dim`, default 384).
`dim` and every entry of `multi_scale` must be multiples of 32 (covers the
stride-16 network, TensorRT alignment, and the export contract). The output
grid is always `dim/16` per side (e.g. 384 → 24×24).

## Preparing raw scene folders (`iwpod prepare-data`)

Raw collections usually arrive as scene folders (camera runs, collection
batches) with no train/val split:

```
datasets/LP/
├── train_wpod_01_12_23/   # scene: *.jpg + sibling *.txt
├── train_wpod_12_10_23/   # scene
└── ...
```

Convert once into the layout above:

```bash
iwpod prepare-data --input datasets/LP --output datasets/LPR --ratio 0.8 --seed 42
iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --name lpr_exp1
```

What it does:
- Walks `--input` recursively — every directory holding images is one scene
  (flat or nested, doesn't matter).
- Splits **each scene independently** (seeded shuffle; default 80/20 train/val
  via `--ratio 0.8`), so small scenes land in both splits proportionally instead
  of being swallowed whole into one side. Scene ids are the path relative to
  `--input` with `/` replaced by `__` (avoids `a/X` vs `b/X` collisions).
  Tiny scenes (N≥2) keep at least one val sample by design.
- Validates every pair: image readable; quad parseable (4 corners; pixel
  coords auto-normalized; **reordered to TL,TR,BR,BL**; out-of-range rejected
  and counted, not fatal).
  Empty/missing `.txt` is kept as a background sample. Output `.txt` files
  are canonicalized (normalized, 6 decimals, vehicle labels preserved) and
  the split applies to validated pairs only.
- Writes `<output>/{train,val}/` + `split_manifest.csv` (`file,scene,split`).
  Files are always **copied**; the source tree is never modified.
  Refuses an existing `--output` unless `--overwrite`.
- No `--output` given → `<input>_prepared` next to the input.

## Train/val layout (the convention)

The loader is lazy (images stay on disk until sampled), so 9k+ image
datasets load in seconds with flat memory — verified on 7,395 crops.

Datasets live as **named folders** inside `datasets/`, each with `train/` and
`val/` splits you prepare yourself (80/20 is the convention):

```bash
iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --lr 0.001 --name exp1
```

with:

```
datasets/
└── LPR/
    ├── train/
    │   ├── sample_001.jpg
    │   ├── sample_001.txt
    │   └── ...
    └── val/
        ├── sample_101.jpg
        ├── sample_101.txt
        └── ...
```

- If `datasets/` holds exactly one such named dataset, `--data` can be
  omitted — it is picked up automatically.
- If that dataset's `val/` is missing, training **refuses to start** — the
  train/val split is dataset preparation, not training.
- The legacy `--train-dir` bare-pairs path trains with no val set at all
  (warns; `_best.pth` tracks train loss).
- `iwpod eval --data datasets/LPR/val` and `iwpod infer --input ...` accept
  any directory of pairs, so the val split doubles as the eval set.
