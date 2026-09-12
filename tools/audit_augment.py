"""Visual audit for augmentation<->annotation coherence.

Renders N augmented training samples as overlay PNGs: warped image, GT quad
polygon (green), positive label-map cells (red squares), plate-area bucket and
positive count in the corner. Also prints aggregate stats (positive rate,
out-of-frame rate, per-scale means).

Usage:
    uv run python tools/audit_augment.py --data datasets/LP/train --dim 384 \\
        --num 24 --out /tmp/aug_audit --seed 0
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/LP/train")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--out", default="/tmp/aug_audit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from iwpod.dataset import image_label_loader
    from iwpod.sampler import augment_sample

    entries, _ = image_label_loader(args.data)
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(entries), size=min(args.num, len(entries)), replace=False)
    os.makedirs(args.out, exist_ok=True)

    from iwpod.sampler import labels2output_map
    stride = 16
    n_pos_all, oob = 0, 0
    for k, idx in enumerate(picks):
        jpg, shapes = entries[int(idx)]
        img = cv2.imread(jpg)
        if img is None:
            continue
        np.random.seed(args.seed * 1000 + k)
        import random as _random
        _random.seed(args.seed * 1000 + k)
        aug, llp, ptslist = augment_sample(img, shapes, args.dim)
        y = labels2output_map(llp, ptslist, args.dim, stride, alpha=0.5)
        pos = np.argwhere(y[..., 0] > 0.5)
        n_pos_all += len(pos)
        canvas = (np.clip(aug, 0, 1) * 255).astype(np.uint8).copy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        for pts in ptslist:
            q = (np.asarray(pts, dtype=float) * args.dim).astype(np.int32).T.reshape(-1, 1, 2)
            if (q < 0).any() or (q >= args.dim).any():
                oob += 1
            cv2.polylines(canvas, [q], True, (0, 255, 0), 2)
        cell = args.dim // y.shape[0]
        for cy, cx in pos:
            cv2.rectangle(canvas, (cx * cell, cy * cell),
                          ((cx + 1) * cell, (cy + 1) * cell), (255, 0, 0), 1)
        tag = f"pos={len(pos)}"
        cv2.putText(canvas, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(args.out, f"aug_{k:02d}.png"), canvas)
    print(f"wrote {len(picks)} overlays to {args.out}")
    print(f"mean positives/sample: {n_pos_all / max(1, len(picks)):.1f}  "
          f"samples with quad out-of-frame: {oob}")


if __name__ == "__main__":
    main()
