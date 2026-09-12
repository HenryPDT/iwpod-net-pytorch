"""`iwpod export`: ONNX export, DeepStream-ready (dynamic batch + shape).

v2 default: single NCHW output `lpd_pred [B,7,Gh,Gw]` (raw logits + affine).
`-s N` = NxN square; `-s H W` = non-square (allowed, accuracy-untested —
training is square; the 256x96 plate is a separate warp stage, not this flag).
Default size is 416 (Xavier NX Phase-1 infer-dims).
"""
import os

import torch
import torch.nn as nn

from iwpod.constants import DEPLOY_SIZE, NET_STRIDE, SIDE
from iwpod.model import IWPODNet


def parse_size(values, default=DEPLOY_SIZE):
    """One number -> square (N,N); two numbers -> (H, W) as given (torch order)."""
    if not values:
        return default, default
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError("--size takes one (square) or two (H W) numbers")


def register(p):
    req = p.add_argument_group("required")
    req.add_argument("-w", "--weights", required=True,
                     help="Input .pt/.pth checkpoint (state_dict or {model_state_dict})")
    adv = p.add_argument_group("sizing")
    adv.add_argument("-s", "--size", nargs="+", type=int, default=[DEPLOY_SIZE],
                     help=f"Inference size: one number = square, two = H W (default [{DEPLOY_SIZE}])")
    adv.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    adv.add_argument("--dynamic-shape", action="store_true", help="Dynamic H/W (multiples of --align)")
    adv.add_argument("--batch", type=int, default=1, help="Static batch (ignored with --dynamic)")
    adv.add_argument("--align", type=int, default=32, help="H/W alignment requirement (default 32)")
    adv.add_argument("--opset", type=int, default=17, help="ONNX opset (17 default; 13 for max TRT8.5 compat)")
    adv.add_argument("--simplify", action="store_true", help="Run onnxslim")
    adv.add_argument("--check-parity", action="store_true",
                     help="Compare torch vs ORT on a representative (non-zero) crop")
    adv.add_argument("--fuse", action="store_true",
                     help="Fold Conv-BN before export (opt-in; re-runs parity if --check-parity)")
    adv.add_argument("-o", "--output", type=str, default=None, help="Output .onnx path")
    leg = p.add_argument_group("legacy / compat")
    leg.add_argument("--with-sigmoid", action="store_true", help="Bake sigmoid into graph (default: raw logits)")
    leg.add_argument("--with-passthrough", action="store_true",
                     help="LEGACY Phase-1: add Identity 2nd output (doubles host traffic; "
                          "required by the deployed Phase-1 pipeline, omit for clean Phase-2)")
    p.set_defaults(func=run)


def _parity_input(batch, h, w, seed=0):
    """Representative [0,1] NCHW crop (not zeros). Non-square-friendly."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, h, w, generator=g)


class _Passthrough(nn.Module):
    """Phase-1 helper: forward the input crop alongside the prediction so
    the deployed pipeline can keep reading pixels from tensor output 1."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x), x


def run(args):
    from iwpod import ckpt as ckptlib

    print(f"\nStarting: {args.weights}")
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set dynamic batch-size and static batch-size at same time")

    exp_h, exp_w = parse_size(args.size)
    if exp_h % args.align or exp_w % args.align:
        raise RuntimeError(f"Export size must be multiple of {args.align}: got {exp_h}x{exp_w}")

    raw_logits = not args.with_sigmoid
    model = IWPODNet(raw_logits=raw_logits)
    sd, arch = ckptlib.load_ckpt_weights(args.weights, map_location="cpu")
    for k, want in (("stride", NET_STRIDE), ("side", SIDE)):
        if arch.get(k) is not None and abs(float(arch[k]) - want) > 1e-9:
            raise RuntimeError(
                f"Checkpoint {k}={arch[k]} != code {k}={want}: label-encoding "
                f"contract broken, quads would silently shift. Retrain or fix constants.")
    if arch.get("dim") is not None and (exp_h != arch["dim"] or exp_w != arch["dim"]):
        print(f"Note: exporting at {exp_h}x{exp_w}, trained at dim={arch['dim']} "
              f"(grid scales as H/16 x W/16; deploy size is {DEPLOY_SIZE})")
    model.load_state_dict(sd, strict=True)
    for p_ in model.parameters():
        p_.requires_grad = False
    model.eval().float()
    if args.fuse:
        model.fuse()
        print("Fused Conv-BN (opt-in)")

    out_name = "lpd_pred"
    out_names = [out_name]
    export_model = model
    if args.with_passthrough:
        export_model = _Passthrough(model)
        out_names.append("pass_through_output")
    export_model.eval()
    dummy = _parity_input(args.batch, exp_h, exp_w)
    onnx_path = args.output or (os.path.splitext(args.weights)[0] + ".onnx")

    dyn = None
    if args.dynamic or args.dynamic_shape:
        dyn = {"input": {}}
        if args.dynamic:
            dyn["input"][0] = "batch"
        if args.dynamic_shape:
            dyn["input"][2] = "height"
            dyn["input"][3] = "width"
        dyn[out_name] = {}
        if args.dynamic:
            dyn[out_name][0] = "batch"
        if args.dynamic_shape:
            dyn[out_name][2] = "h_out"
            dyn[out_name][3] = "w_out"
        if args.with_passthrough:
            dyn["pass_through_output"] = dict(dyn["input"])

    print(f"Exporting {exp_h}x{exp_w} batch={args.batch} dynamic={bool(dyn)} opset={args.opset}")
    torch.onnx.export(
        export_model, dummy, onnx_path, verbose=False, opset_version=args.opset,
        do_constant_folding=True, input_names=["input"], output_names=out_names,
        dynamic_axes=dyn, dynamo=False,
    )

    if args.with_passthrough:
        print("Passthrough second output exported natively (input crop copy)")

    if args.simplify:
        print("Simplifying with onnxslim")
        import onnx
        import onnxslim
        m = onnx.load(onnx_path)
        m = onnxslim.slim(m)
        onnx.save(m, onnx_path)

    import onnx as _onnx
    from onnx import shape_inference
    _onnx.checker.check_model(onnx_path)
    inferred = shape_inference.infer_shapes(_onnx.load(onnx_path))
    names = [o.name for o in inferred.graph.output]
    print(f"Done: {onnx_path}\n  outputs: {names}")
    if names[0] != "lpd_pred":
        raise RuntimeError(f"Expected first output lpd_pred, got {names}")
    if args.with_passthrough and "pass_through_output" not in names:
        raise RuntimeError("Phase-1 export missing pass_through_output")
    if args.check_parity:
        _check_parity(export_model, dummy, onnx_path)
        if args.dynamic:
            _check_parity_batch2(export_model, dummy, onnx_path)


def _check_parity(model, dummy, onnx_path, tol=1e-4):
    """Torch vs ORT max-abs diff on a representative input (lpd_pred)."""
    import numpy as _np
    model.eval()
    with torch.no_grad():
        ref = model(dummy)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        ref = ref.detach().cpu().numpy()
    import onnxruntime as _ort
    sess = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["lpd_pred"], {"input": dummy.cpu().numpy()})[0]
    diff = float(_np.abs(ref - got).max())
    print(f"Parity torch-vs-ORT max-abs: {diff:.3e} (tol {tol:.0e})")
    if diff > tol:
        raise RuntimeError(f"Export parity failed: max-abs {diff:.3e} > {tol:.0e}")


def _check_parity_batch2(model, dummy, onnx_path, tol=1e-4):
    """Dynamic-batch contract: batch=2 must run in ORT from the same file."""
    import numpy as _np
    import onnxruntime as _ort
    x = dummy[:1].repeat(2, 1, 1, 1)
    sess = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["lpd_pred"], {"input": x.cpu().numpy()})[0]
    if got.shape[0] != 2:
        raise RuntimeError(f"Dynamic batch-2 failed: got shape {got.shape}")
    with torch.no_grad():
        ref = model(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        ref = ref.detach().cpu().numpy()
    diff = float(_np.abs(ref - got).max())
    print(f"Parity batch-2 max-abs: {diff:.3e}")
    if diff > tol:
        raise RuntimeError(f"Export batch-2 parity failed: max-abs {diff:.3e} > {tol:.0e}")
