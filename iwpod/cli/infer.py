"""`iwpod infer`: headless folder inference, vehicle-crop in -> quads + plates.

Letterboxes inputs (like DeepStream: top-left, black pad) and warps plates
from the ORIGINAL crop.
Writes <base>_quad.txt (x0..x3 y0..y3 conf in original px, or `none`) and
<base>_plate.png (plate-size warp, default 256x96 = the OCR contract).
"""
import os

import cv2
import numpy as np
import torch

from iwpod import ckpt as ckptlib
from iwpod.constants import DEPLOY_SIZE, EXPORT_ALIGNMENT, IMAGE_EXTS
from iwpod.decode import decode_single, warp_plate
from iwpod.preprocess import letterbox, unletterbox_quad


def register(p):
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None,
                   help="Output dir (default: out/detect/<weights-stem>/)")
    p.add_argument("--size", type=int, default=None,
                   help="Letterbox size (default: ckpt dim stamp, else 416)")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--plate-size", nargs=2, type=int, default=[256, 96], metavar=("W", "H"),
                   help="Rectified plate size as W H (default 256 96; keep for trained OCR)")
    p.add_argument("--recursive", action="store_true",
                   help="Walk --input recursively")
    p.set_defaults(func=run)


def _list_images(root, recursive=False):
    files = []
    if recursive:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                if name.lower().endswith(IMAGE_EXTS):
                    files.append(os.path.join(dirpath, name))
    elif os.path.isdir(root):
        for name in os.listdir(root):
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(root, name))
    elif os.path.isfile(root) and os.path.basename(root).lower().endswith(IMAGE_EXTS):
        files.append(root)
    return sorted(files)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = ckptlib.load_ckpt(args.weights, map_location="cpu")
    sd, arch = ckptlib.weights_and_arch(ck)
    from iwpod.model import IWPODNet
    _mkw = {}
    for _k in ("backbone", "head", "use_simam", "arch_version"):
        if arch.get(_k) is not None:
            _mkw[_k] = arch[_k]
    model = IWPODNet(raw_logits=True, **_mkw).to(device).eval()
    ckptlib.load_state_dict_compat(model, sd, source=args.weights)

    if args.size is not None:
        if args.size % EXPORT_ALIGNMENT:
            raise RuntimeError(f"--size must be multiple of {EXPORT_ALIGNMENT}, got {args.size}")
        size_px = int(args.size)
    elif arch.get("dim") is not None:
        size_px = int(arch["dim"])
    else:
        size_px = DEPLOY_SIZE
    pw, ph = int(args.plate_size[0]), int(args.plate_size[1])
    from iwpod import runs as _runs
    out_dir = args.output or _runs.default_task_dir("detect", args.weights)
    os.makedirs(out_dir, exist_ok=True)
    files = _list_images(args.input, recursive=args.recursive)
    if not files:
        raise RuntimeError(f"No images in '{args.input}'")
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        canvas, s, dx, dy = letterbox(img, size_px)
        inp = canvas.astype(np.float32) / 255.0
        t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(t).squeeze(0).cpu().numpy()
        found = decode_single(pred, size_px, size_px, threshold=args.threshold,
                              from_logits=True, topk=1)
        base = os.path.splitext(os.path.basename(f))[0]
        if not found:
            open(os.path.join(out_dir, base + "_quad.txt"), "w").write("none\n")
            continue
        quad, conf = found[0]
        q0 = unletterbox_quad(quad, size_px, s, dx, dy, w0, h0)
        np.savetxt(os.path.join(out_dir, base + "_quad.txt"),
                   np.append(q0.flatten(), conf)[None], fmt="%.2f")
        plate = warp_plate(img, q0, out_w=pw, out_h=ph)
        if plate is not None:
            cv2.imwrite(os.path.join(out_dir, base + "_plate.png"), plate)
        print(f"{base}: conf={conf:.2f}")
