"""`iwpod eval`: quad-IoU recall + corner RMSE (headless).

Letterboxes inputs to square like DeepStream (maintain-aspect-ratio=1,
top-left pad) and compares quads in original normalized coords. Geometry
lives in `iwpod.eval_core` (shared with training); this module is CLI plumbing.
"""
import os

from loguru import logger

from iwpod import ckpt as ckptlib
from iwpod.constants import DEPLOY_SIZE, EXPORT_ALIGNMENT
from iwpod.eval_core import entries_from_annotated_dir, evaluate_quads, format_block, format_summary


def register(p):
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="Annotated dir (image + sibling .txt pairs)")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--size", type=int, default=None,
                   help="Decode resolution (default: ckpt dim stamp, else 416)")
    p.add_argument("--sweep", nargs="*", type=float, default=None,
                   help="Extra thresholds to report, e.g. --sweep 0.15 0.25 0.4 0.5 0.6")
    p.add_argument("--log-dir", default=None,
                     help="Write val_log.txt here (default: out/eval/<weights-stem>/)")
    p.set_defaults(func=run)


def _resolve_size(args, arch=None):
    if args.size is not None:
        size = int(args.size)
        if size % EXPORT_ALIGNMENT:
            raise RuntimeError(f"--size must be multiple of {EXPORT_ALIGNMENT}, got {size}")
        return size
    if arch and arch.get("dim") is not None:
        return int(arch["dim"])
    return DEPLOY_SIZE


def run(args):
    import datetime as _dt

    import torch

    from iwpod.model import IWPODNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = ckptlib.load_ckpt(args.weights, map_location="cpu")
    sd, arch = ckptlib.weights_and_arch(ck)
    size_px = _resolve_size(args, arch)
    _mkw = {}
    for _k in ("backbone", "head", "use_simam", "arch_version"):
        if arch.get(_k) is not None:
            _mkw[_k] = arch[_k]
    model = IWPODNet(raw_logits=True, **_mkw).to(device).eval()
    ckptlib.load_state_dict_compat(model, sd, source=args.weights)

    entries = entries_from_annotated_dir(args.data)
    if not entries:
        raise RuntimeError(f"No images in '{args.data}'")
    extra = list(dict.fromkeys(args.sweep or []))
    r = evaluate_quads(model, entries, size_px=size_px, threshold=args.threshold,
                       device=device, progress=True, extra_thresholds=extra)
    table = {row["thr"]: row for row in r["sweep"]["table"]}
    logger.info(f"thr={args.threshold:.2f} " + format_summary(r))
    for th in extra:
        if abs(th - args.threshold) < 1e-12:
            continue
        row = table.get(round(float(th), 4))
        if row is None:
            row = min(table.values(), key=lambda x: abs(x["thr"] - th))
        logger.info(f"thr={th:.2f} prec={row['prec']:.3f} "
                    f"rec={row['rec']:.3f} F1={row['f1']:.3f} (single-pass)")
    logger.info("\n" + format_block(r, size_px=size_px, threshold=args.threshold, with_sweep=True))
    from iwpod import runs as _runs
    log_dir = args.log_dir or _runs.default_task_dir("eval", args.weights)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "val_log.txt"), "a") as f:
            f.write(f"# {_dt.datetime.now().isoformat(timespec='seconds')} "
                    f"weights={args.weights} data={args.data} size={size_px}\n")
            f.write(f"thr={args.threshold:.2f} " + format_summary(r) + "\n")
            for th in extra:
                row = table.get(round(float(th), 4))
                if row is None:
                    continue
                f.write(f"thr={th:.2f} prec={row['prec']:.3f} rec={row['rec']:.3f} "
                        f"F1={row['f1']:.3f}\n")
            f.write(format_block(r, size_px=size_px, threshold=args.threshold,
                                 with_sweep=True) + "\n")
    except OSError as e:
        logger.warning(f"Could not write val_log.txt: {e}")
