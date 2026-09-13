# Dataset Format

This page specifies the exact on-disk format the training and eval tools expect.
If your annotations don't match this, convert them first — the loader does no
auto-detection.

## Layout

A dataset directory contains image + sibling annotation pairs sharing a
basename. Discovery is **recursive**: pairs may sit directly in the folder or
one scene-subfolder down (or deeper). Scene folder names are ignored — only
the image and the `.txt` next to it matter.

```
train_dir/
  Access_Control/
    Access_Control_..._car_0.jpg
    Access_Control_..._car_0.txt
  Cahill_Entry_Front/
    Cahill_Entry_Front_..._detected_0.jpg
    Cahill_Entry_Front_..._detected_0.txt
  ...
```

A flat folder (`train_dir/*.jpg` + sibling `*.txt`) is also valid.

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

## Train/val layout (the convention)

The loader is lazy (images stay on disk until sampled), so 9k+ image
datasets load in seconds with flat memory — verified on 7,395 crops.

Prepare the train/val split **externally** (this repo does not split for you).
Datasets live as **named folders** inside `datasets/`, each with `train/` and
`val/` trees. Scene subfolders under those splits are the usual layout:

```bash
iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --lr 0.001 --name exp1
```

with:

```
datasets/
└── LPR/
    ├── train/
    │   ├── Access_Control/
    │   │   ├── sample_001.jpg
    │   │   ├── sample_001.txt
    │   │   └── ...
    │   └── Cahill_Entry_Front/
    │       └── ...
    └── val/
        ├── Access_Control/
        └── ...
```

Pairs may also sit directly in `train/` / `val/` with no scene folders.

- If `datasets/` holds exactly one such named dataset, `--data` can be
  omitted — it is picked up automatically.
- If that dataset's `val/` is missing, training **refuses to start** — the
  train/val split is dataset preparation, not training.
- The legacy `--train-dir` bare-pairs path trains with no val set at all
  (warns; `_best.pth` tracks train loss).
- `iwpod eval --data datasets/LPR/val` walks scene subfolders the same way,
  so the val split doubles as the eval set. `iwpod infer --input ...` is
  flat by default (`--recursive` to walk nested dumps).
