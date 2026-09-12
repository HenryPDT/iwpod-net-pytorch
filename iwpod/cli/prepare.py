"""`iwpod prepare-data`: raw scene folders -> train/val dataset layout.

Walks --input recursively (every dir with images = one scene), splits EACH
scene into train/val (seeded shuffle, default 0.8 train via --ratio) so small
scenes land in both splits, validates quad annotations, and writes
<output>/{train,val}/ with <scene>__-prefixed filenames plus split_manifest.csv.

Example:
  iwpod prepare-data --input datasets/LP --output datasets/LPR --ratio 0.8
  iwpod train --data datasets/LPR --epochs 200 --batch-size 32 --name lpr_exp1
"""


def register(p):
    p.add_argument("--input", required=True, help="Raw root, e.g. datasets/LP (scene subdirs)")
    p.add_argument("--output", default=None,
                   help="Prepared root (default: <input>_prepared, e.g. datasets/LP_prepared)")
    p.add_argument("--ratio", type=float, default=0.8, help="Train fraction per scene (default 0.8)")
    p.add_argument("--seed", type=int, default=42, help="Split seed (default 42)")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into existing --output")
    p.set_defaults(func=run)


def run(args):
    import os

    from iwpod.prepare import prepare_dataset

    output = args.output or (args.input.rstrip(os.sep) + "_prepared")
    stats = prepare_dataset(args.input, output, ratio=args.ratio, seed=args.seed,
                            overwrite=args.overwrite)
    print(f"Scenes: {len(stats['scenes'])} | train: {stats['train']} | "
          f"val: {stats['val']} | invalid skipped: {stats['invalid']} | "
          f"background: {stats['background']}")
    for s in stats["scenes"]:
        print(f"  {s['scene']}: train={s['train']} val={s['val']} invalid={s['invalid']}")
    print(f"Wrote {output}/{{train,val}}/ + split_manifest.csv")
