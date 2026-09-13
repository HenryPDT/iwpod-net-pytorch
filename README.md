# IWPOD-NET in Pytorch

This repository is Pytorch implementation of [A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios](https://doi.org/10.1109/TITS.2021.3055946). The referenced tensorflow and keras codes can be found [here](https://github.com/claudiojung/iwpod-net).

![20230602_171136](https://github.com/blastak/iwpod-net-pytorch/assets/12149098/fb96f4b9-49a8-4b70-a99c-88db969708a2 "Results are shown in green, ground-truth annotations in red.")

## Setup

### Option A: uv (recommended)

```bash
uv venv
source .venv/bin/activate
# CPU/default torch, then editable install:
uv pip install -e .
# ...or CUDA torch first, then editable install without deps:
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-deps -e .
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate iwpodnet
```

This installs the `iwpod` command (`train | eval | infer | export`
subcommands; `python -m iwpod ...` works too).

## Quickstart (end to end)

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
# train/val split is prepared externally (scene subfolders under train/ and val/; see docs/DATASET.md)
iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --lr 0.001 \
  --size 384 --seed 42 --name exp1
iwpod eval --weights out/train/exp1/exp1_best.pth --data datasets/LPR/val \
  --threshold 0.3 --size 384
iwpod export -w out/train/exp1/exp1_best.pth -s 384 \
  --dynamic --dynamic-shape --simplify
iwpod infer --weights out/train/exp1/exp1_best.pth \
  --input vehicle_crops/ --output out/ --size 384 --plate-size 256 96
```

Runs land in `out/train/<name>[_2..]` with checkpoints, `train_command.txt`,
`config.yaml`, and `train.log` (see docs/TRAINING.md).

## Docs

| Page | Covers |
|---|---|
| `docs/DATASET.md` | Annotation format, vehicle crops, `bgimages/`, `%32` rule |
| `docs/TRAINING.md` | CLI flags, bundled config reference, run dirs, augmentation |
| `docs/INFERENCE.md` | Entry points, flags, threshold tuning, plate size |
| `docs/EXPORT.md` | Flag matrix, per-target recipes, validation, TensorRT |
| `docs/EVALUATION.md` | Metric definitions, reading numbers, gating |
| `docs/TROUBLESHOOTING.md` | GPU env, training, export, detection symptoms |
| `deepstream/DOWNSTREAM_GUIDE.md` | DeepStream IWPOD export, plugin contract, rollback |

> **NoMachine/NX + cuDNN (fixed permanently):** NX injects
> `LD_PRELOAD=/usr/NX/lib/libnxegl.so` into every session, which breaks cuDNN
> for *all* repos (`Invalid handle. Cannot load symbol cudnnGetVersion`).
> Fixed in `~/.bashrc` — any new terminal unsets the NX hook (verified:
> cuDNN 92400 loads, GPU convs run, no prefixes needed). NX menu-launched GUI
> apps don't source bashrc, so remote-desktop GL is unaffected. For
> non-interactive contexts (systemd/cron/DeepStream services) use
> `env -u LD_PRELOAD <cmd>` or `UnsetEnvironment=LD_PRELOAD`.

## Training

```bash
iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --lr 0.001 --name exp1
```

`--data` points at a root containing `train/` **and** `val/` (scene
subfolders under those splits are walked recursively; training will not
split for you). Runs go to
`out/train/<name>[_2..]`. Details and the bundled-config reference:
`docs/TRAINING.md`. Legacy scripts live frozen in `legacy/` (reproduction only).

## Inferencing

```bash
iwpod infer --weights out/train/exp1/exp1_best.pth --input vehicle_crops/ --output out/
iwpod eval --weights out/train/exp1/exp1_best.pth --data datasets/LPR/val
iwpod export -w out/train/exp1/exp1_best.pth -s 384 --dynamic --dynamic-shape --simplify
```

Details: `docs/INFERENCE.md`, `docs/EVALUATION.md`, `docs/EXPORT.md`.

## NOTE

`weights/` holds shipped checkpoints (incl. the 10,000-epoch legacy weights).
Resume one with `iwpod train --resume weights/<ckpt>.pth --data datasets/LPR ...`
(note: a bare ckpt resumes into a *new* run dir; pass a run dir to continue it).

